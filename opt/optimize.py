"""Per-segment direct optimization against the exact simulator.

The simulator is deterministic given the action sequence (see exact_sim), so each
segment's ~400 actions are optimized offline with receding-horizon exact-rollout
MPC ("mpc_pass"): at every step, branch over candidate actions around the current
plan, roll the true simulator a few steps ahead (the RNG draws are known), score
with the exact cost terms plus an LQ terminal penalty, and commit the best.

Key structural facts this exploits (measured, see README):
  * steering acts with ~0.3s delay and an integrator-like sustained response, so
    candidates come in two families: a bump on the current action alone, and the
    same offset applied across the whole lookahead window ("window shift");
  * the sampled lataccel is a known deterministic staircase in the actions, so the
    MPC can chase the *realized* noise draws — a linearized method cannot;
  * repeated passes help because each pass improves the continuation plan the
    next pass branches against.

The final schedule is coarse stride-2 passes (fast exploration) followed by
full-resolution and fine-offset passes.
"""
from hashlib import md5

import numpy as np

from opt.exact_sim import (ExactSim, make_session, CONTROL_START_IDX, COST_END_IDX,
                           STEER_RANGE)
from opt.qp import A_W, B_W, solve_ideal, ideal_cost

N = COST_END_IDX - CONTROL_START_IDX

MPC_OFFSETS = np.array([-0.15, -0.09, -0.05, -0.025, -0.012, -0.005,
                        0.0, 0.005, 0.012, 0.025, 0.05, 0.09, 0.15])
FINE_OFFSETS = np.array([-0.04, -0.02, -0.01, -0.005, -0.002,
                         0.0, 0.002, 0.005, 0.01, 0.02, 0.04])

DEFAULT_SCHEDULE = (
  dict(horizon=6, stride=2),
  dict(horizon=6, stride=2),
  dict(horizon=6),
  dict(horizon=5, offsets=FINE_OFFSETS),
)


def _terminal_kappa() -> float:
  """Quadratic coefficient of the LQ cost-to-go w.r.t. a boundary lataccel offset."""
  gz = np.zeros(150)

  def tail_cost(e):
    x = solve_ideal(gz, x_prev=e)
    return A_W * np.sum(x ** 2) + B_W * ((x[0] - e) ** 2 + np.sum(np.diff(x) ** 2))

  return tail_cost(1.0) - tail_cost(0.0)


_KAPPA = _terminal_kappa()


def mpc_pass(ex: ExactSim, plan: np.ndarray, horizon=6, offsets=MPC_OFFSETS,
             window_shift=True, stride=1) -> tuple:
  """One receding-horizon exact-rollout pass; returns (actions, lataccel, cost)."""
  ex.reset()
  g = ex.targets[CONTROL_START_IDX:COST_END_IDX].astype(np.float64)
  offsets = np.asarray(offsets)
  n_off = len(offsets)
  plan = plan.copy()
  committed = np.empty(N)
  t = CONTROL_START_IDX
  while t < COST_END_IDX:
    i = t - CONTROL_START_IDX
    h = min(horizon, COST_END_IDX - t)
    x_prev = ex.lataccel[-1]
    xstar = solve_ideal(g[i:], x_prev=x_prev if i > 0 else None)
    cont = plan[i + 1:i + h]
    if len(cont) < h - 1:
      cont = np.concatenate([cont, np.repeat(plan[-1], h - 1 - len(cont))])
    cands = np.clip(plan[i] + offsets, *STEER_RANGE)
    if window_shift and h > 1:
      cands = np.concatenate([cands, np.clip(plan[i] + offsets, *STEER_RANGE)])
      cont_mat = np.concatenate([np.tile(cont, (n_off, 1)),
                                 cont[None, :] + offsets[:, None]])
    else:
      cont_mat = np.tile(cont, (len(cands), 1))
    B = len(cands)
    la = ex.branch_rollout(cands, cont_mat, h)
    lat = A_W * np.sum((la - g[i:i + h]) ** 2, axis=1)
    diffs = np.diff(np.concatenate([np.full((B, 1), x_prev), la], axis=1), axis=1)
    jerk = B_W * (np.sum(diffs ** 2, axis=1) - (diffs[:, 0] ** 2 if i == 0 else 0.0))
    total = lat + jerk
    if t + h < COST_END_IDX:
      total = total + _KAPPA * (la[:, -1] - xstar[h - 1]) ** 2
    best = int(np.argmin(total))
    shift = offsets[best % n_off] if (window_shift and best >= n_off and h > 1) else 0.0
    n_commit = min(stride, COST_END_IDX - t)
    committed[i] = cands[best]
    ex.step(cands[best])
    for s in range(1, n_commit):
      a_next = float(np.clip(cont_mat[best][s - 1] if h > 1 else plan[-1], *STEER_RANGE))
      committed[i + s] = a_next
      ex.step(a_next)
    if shift:
      # fold the accepted window shift into the plan so later steps inherit it
      plan[i + n_commit:i + h] += shift
    t += n_commit
  return committed, np.array(ex.lataccel[CONTROL_START_IDX:COST_END_IDX]), ex.cost()


def optimize_segment(session, data_path: str, schedule=DEFAULT_SCHEDULE,
                     early_stop=0.5, verbose=False) -> dict:
  ex = ExactSim(session, data_path)
  a = np.clip(ex.steer_data[CONTROL_START_IDX:COST_END_IDX], *STEER_RANGE)
  best_a = best_x = None
  best_cost = None
  for k, kw in enumerate(schedule):
    a, x, c = mpc_pass(ex, a, **kw)
    if best_cost is None or c['total_cost'] < best_cost['total_cost']:
      prev_best = best_cost['total_cost'] if best_cost else np.inf
      best_a, best_x, best_cost = a.copy(), x.copy(), c
      improvement = prev_best - c['total_cost']
    else:
      a = best_a.copy()
      improvement = 0.0
    if verbose:
      print(f"  pass {k} {kw}: total={c['total_cost']:.4f} (best={best_cost['total_cost']:.4f})")
    if k >= 2 and improvement < early_stop:
      break

  g = ex.targets[CONTROL_START_IDX:COST_END_IDX].astype(np.float64)
  fp = np.array([ex.targets[20], ex.roll[20], ex.v_ego[20], ex.a_ego[20]], dtype=np.float64)
  return {
    'key': md5(fp.tobytes()).hexdigest()[:16],
    'data_path': str(data_path),
    'actions': best_a,
    'lataccel': best_x,
    'cost': best_cost,
    'ideal_bound': ideal_cost(g),
  }


if __name__ == '__main__':
  import argparse, time
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', default='./models/tinyphysics.onnx')
  parser.add_argument('--data_path', required=True)
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()
  session = make_session(args.model_path)
  t0 = time.time()
  res = optimize_segment(session, args.data_path, verbose=args.verbose)
  cc = res['cost']
  print(f"{args.data_path}: total={cc['total_cost']:.4f} (lat={cc['lataccel_cost']:.4f} "
        f"jerk={cc['jerk_cost']:.4f})  QP lower bound={res['ideal_bound']:.4f}  "
        f"[{time.time()-t0:.1f}s]")
