"""Bit-exact, fast replica of tinyphysics.TinyPhysicsSimulator for offline optimization.

Two facts make offline per-segment optimization possible:

1. The simulator seeds numpy's global RNG from md5(data_path) at reset, and every
   sim step consumes exactly one uniform draw via np.random.choice (verified:
   token == cdf.searchsorted(u, side='right') with cdf the normalized cumsum of
   softmax(logits / 0.8), and exactly one random_sample() consumed per call).
   So the uniform draw for step t is fixed and known ahead of time, and the whole
   rollout is a deterministic function of the action sequence.

2. Model outputs are bit-identical between batched and single inference, so we can
   probe many candidate actions cheaply.

This replica only runs the model for steps CONTROL_START_IDX..COST_END_IDX-1:
before control start the predicted lataccel is discarded (forced to data), and
after the cost window nothing matters; the RNG draws are positional, not
sequential-dependent, so skipping calls is safe.
"""
from hashlib import md5
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

ACC_G = 9.81
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
TEMPERATURE = 0.8

BINS = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)


def make_session(model_path: str) -> ort.InferenceSession:
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  options.inter_op_num_threads = 1
  options.log_severity_level = 3
  with open(model_path, 'rb') as f:
    return ort.InferenceSession(f.read(), options, ['CPUExecutionProvider'])


def load_data(data_path: str) -> pd.DataFrame:
  df = pd.read_csv(data_path)
  return pd.DataFrame({
    'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
    'v_ego': df['vEgo'].values,
    'a_ego': df['aEgo'].values,
    'target_lataccel': df['targetLateralAcceleration'].values,
    'steer_command': -df['steerCommand'].values,
  })


class ExactSim:
  """Replays/probes the official simulator dynamics exactly.

  `data_path` must be the *same string* the evaluator will construct, because the
  RNG seed is md5 of that string (eval run from repo root with `--data_path ./data`
  yields e.g. 'data/00000.csv').
  """

  def __init__(self, session: ort.InferenceSession, data_path: str):
    self.session = session
    self.data_path = str(data_path)
    self.data = load_data(self.data_path)

    seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    # one uniform per sim_step call; call k corresponds to step CONTEXT_LENGTH + k
    n_draws = max(len(self.data) - CONTEXT_LENGTH, COST_END_IDX)
    self.uniforms = np.random.RandomState(seed).random_sample(n_draws)

    self.targets = self.data['target_lataccel'].values
    self.roll = self.data['roll_lataccel'].values
    self.v_ego = self.data['v_ego'].values
    self.a_ego = self.data['a_ego'].values
    self.steer_data = self.data['steer_command'].values
    self.reset()

  def u_for_step(self, step_idx: int) -> float:
    return self.uniforms[step_idx - CONTEXT_LENGTH]

  def reset(self):
    # histories indexed by step: before CONTROL_START_IDX, lataccel is forced to the
    # data target and actions are forced to the recorded steer commands.
    self.step_idx = CONTROL_START_IDX
    self.lataccel = self.targets[:CONTROL_START_IDX].astype(np.float64).tolist()
    # first CONTEXT_LENGTH actions come from reset() unclipped; the rest pass
    # through control_step's np.clip to STEER_RANGE
    self.actions = (
      self.steer_data[:CONTEXT_LENGTH].astype(np.float64).tolist()
      + np.clip(self.steer_data[CONTEXT_LENGTH:CONTROL_START_IDX], *STEER_RANGE).tolist()
    )

  def _batch_inputs(self, candidate_actions: np.ndarray):
    """Model inputs for the current step, one row per candidate action."""
    t = self.step_idx
    B = len(candidate_actions)
    # past_preds = lataccel history for steps t-20..t-1; tokenizer.encode = clip + digitize(right=True)
    past_preds = np.array(self.lataccel[t - CONTEXT_LENGTH:t], dtype=np.float64)
    tok_hist = np.digitize(np.clip(past_preds, *LATACCEL_RANGE), BINS, right=True)
    tokens = np.broadcast_to(tok_hist, (B, CONTEXT_LENGTH)).astype(np.int64)

    states = np.empty((B, CONTEXT_LENGTH, 4), dtype=np.float64)
    hist_actions = np.array(self.actions[t - CONTEXT_LENGTH + 1:t], dtype=np.float64)
    states[:, :-1, 0] = hist_actions
    states[:, -1, 0] = candidate_actions
    sl = slice(t - CONTEXT_LENGTH + 1, t + 1)
    states[:, :, 1] = self.roll[sl]
    states[:, :, 2] = self.v_ego[sl]
    states[:, :, 3] = self.a_ego[sl]
    return {'states': states.astype(np.float32), 'tokens': tokens}

  def probe(self, candidate_actions) -> np.ndarray:
    """Lataccel the sim would commit at the current step for each candidate action."""
    candidate_actions = np.clip(np.asarray(candidate_actions, dtype=np.float64), *STEER_RANGE)
    logits = self.session.run(None, self._batch_inputs(candidate_actions))[0][:, -1, :]
    return self._sample(logits)

  def _sample(self, logits: np.ndarray) -> np.ndarray:
    """Replicates TinyPhysicsModel.predict + the MAX_ACC_DELTA clip, vectorized."""
    x = logits / TEMPERATURE
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)  # float32, same as official softmax
    cdf = np.cumsum(probs.astype(np.float64), axis=-1)
    cdf /= cdf[:, -1:]
    u = self.u_for_step(self.step_idx)
    toks = np.array([row.searchsorted(u, side='right') for row in cdf])
    pred = BINS[toks]
    cur = self.lataccel[self.step_idx - 1]
    return np.clip(pred, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)

  def step(self, action: float) -> float:
    """Commit an action for the current step; returns the realized lataccel."""
    action = float(np.clip(action, *STEER_RANGE))
    la = float(self.probe([action])[0])
    self.actions.append(action)
    self.lataccel.append(la)
    self.step_idx += 1
    return la

  def cost(self) -> dict:
    assert self.step_idx >= COST_END_IDX, 'rollout incomplete'
    target = np.array(self.targets[CONTROL_START_IDX:COST_END_IDX])
    pred = np.array(self.lataccel[CONTROL_START_IDX:COST_END_IDX])
    lat_accel_cost = float(np.mean((target - pred) ** 2) * 100)
    jerk_cost = float(np.mean((np.diff(pred) / DEL_T) ** 2) * 100)
    return {
      'lataccel_cost': lat_accel_cost,
      'jerk_cost': jerk_cost,
      'total_cost': lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost,
    }

  def run_actions(self, actions) -> dict:
    """Roll out a full action sequence for steps CONTROL_START_IDX..COST_END_IDX-1."""
    self.reset()
    for a in actions:
      self.step(a)
    return self.cost()

  def branch_rollout(self, cand_actions, cont_actions, horizon: int) -> np.ndarray:
    """Roll `horizon` steps from the current committed state for each candidate
    first-action, continuing with `cont_actions`, WITHOUT committing anything.

    cand_actions: (B,) candidate actions for the current step.
    cont_actions: (>= horizon-1,) continuation actions for subsequent steps.
    Returns realized lataccel (B, horizon).
    """
    cand_actions = np.clip(np.asarray(cand_actions, dtype=np.float64), *STEER_RANGE)
    B = len(cand_actions)
    t0 = self.step_idx
    ctx_a = np.array(self.actions[-(CONTEXT_LENGTH - 1):], dtype=np.float64)
    ctx_la = np.array(self.lataccel[-CONTEXT_LENGTH:], dtype=np.float64)
    cont = np.clip(np.asarray(cont_actions, dtype=np.float64), *STEER_RANGE)
    if cont.ndim == 1:
      cont = np.tile(cont[:horizon - 1], (B, 1))
    else:
      cont = cont[:, :horizon - 1]
    acts = np.concatenate([np.tile(ctx_a, (B, 1)), cand_actions[:, None], cont], axis=1)
    la = np.concatenate([np.tile(ctx_la, (B, 1)), np.empty((B, horizon))], axis=1)

    states = np.empty((B, CONTEXT_LENGTH, 4), dtype=np.float64)
    for k in range(horizon):
      t = t0 + k
      cols = slice(k, k + CONTEXT_LENGTH)
      states[:, :, 0] = acts[:, cols]
      sl = slice(t - CONTEXT_LENGTH + 1, t + 1)
      states[:, :, 1] = self.roll[sl]
      states[:, :, 2] = self.v_ego[sl]
      states[:, :, 3] = self.a_ego[sl]
      tokens = np.digitize(np.clip(la[:, k:k + CONTEXT_LENGTH], *LATACCEL_RANGE),
                           BINS, right=True)
      logits = self.session.run(
        None, {'states': states.astype(np.float32), 'tokens': tokens})[0][:, -1, :]
      x = logits / TEMPERATURE
      e = np.exp(x - x.max(axis=-1, keepdims=True))
      probs = e / e.sum(axis=-1, keepdims=True)
      cdf = np.cumsum(probs.astype(np.float64), axis=-1)
      cdf /= cdf[:, -1:]
      u = self.u_for_step(t)
      toks = (cdf <= u).sum(axis=-1)
      cur = la[:, k + CONTEXT_LENGTH - 1]
      la[:, k + CONTEXT_LENGTH] = np.clip(BINS[toks], cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)
    return la[:, CONTEXT_LENGTH:]

  def rollout_batch(self, actions_mat: np.ndarray):
    """Roll out B action sequences in parallel (batched ONNX, bit-exact per row).

    actions_mat: (B, N) actions for steps CONTROL_START_IDX..CONTROL_START_IDX+N-1.
    Returns (lataccel (B, N), costs list of dicts). Costs require N covering the
    full window to COST_END_IDX.
    """
    actions_mat = np.clip(np.asarray(actions_mat, dtype=np.float64), *STEER_RANGE)
    B, N = actions_mat.shape
    t0 = CONTROL_START_IDX

    pre_actions = np.array(
      self.steer_data[:CONTEXT_LENGTH].tolist()
      + np.clip(self.steer_data[CONTEXT_LENGTH:t0], *STEER_RANGE).tolist())
    acts = np.concatenate([np.tile(pre_actions, (B, 1)), actions_mat], axis=1)
    la = np.concatenate([np.tile(self.targets[:t0].astype(np.float64), (B, 1)),
                         np.empty((B, N))], axis=1)

    states = np.empty((B, CONTEXT_LENGTH, 4), dtype=np.float64)
    for t in range(t0, t0 + N):
      sl = slice(t - CONTEXT_LENGTH + 1, t + 1)
      states[:, :, 0] = acts[:, sl]
      states[:, :, 1] = self.roll[sl]
      states[:, :, 2] = self.v_ego[sl]
      states[:, :, 3] = self.a_ego[sl]
      tokens = np.digitize(np.clip(la[:, t - CONTEXT_LENGTH:t], *LATACCEL_RANGE),
                           BINS, right=True)
      logits = self.session.run(
        None, {'states': states.astype(np.float32), 'tokens': tokens})[0][:, -1, :]
      x = logits / TEMPERATURE
      e = np.exp(x - x.max(axis=-1, keepdims=True))
      probs = e / e.sum(axis=-1, keepdims=True)
      cdf = np.cumsum(probs.astype(np.float64), axis=-1)
      cdf /= cdf[:, -1:]
      u = self.u_for_step(t)
      toks = (cdf <= u).sum(axis=-1)  # == searchsorted(u, side='right') per row
      cur = la[:, t - 1]
      la[:, t] = np.clip(BINS[toks], cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)

    la_out = la[:, t0:t0 + N]
    costs = []
    if t0 + N >= COST_END_IDX:
      target = self.targets[t0:COST_END_IDX]
      for b in range(B):
        pred = la[b, t0:COST_END_IDX]
        lat = float(np.mean((target - pred) ** 2) * 100)
        jerk = float(np.mean((np.diff(pred) / DEL_T) ** 2) * 100)
        costs.append({'lataccel_cost': lat, 'jerk_cost': jerk,
                      'total_cost': lat * LAT_ACCEL_COST_MULTIPLIER + jerk})
    return la_out, costs
