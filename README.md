# comma controls challenge — per-segment direct optimization

Solution for the [comma Controls Challenge v2](https://github.com/commaai/controls_challenge),
targeting the top of the [leaderboard](https://comma.ai/leaderboard) (6.880,
"per segment direct quadratic optimization").

The challenge scaffolding (`tinyphysics.py`, `eval.py`, `models/`, and the stock
`controllers/{pid,zero}.py`) is vendored unmodified from
[commaai/controls_challenge](https://github.com/commaai/controls_challenge);
everything in `opt/` plus `controllers/{replay,ffpid}.py` is this solution.
Scoring uses the official command only:

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 \
  --test_controller replay --baseline_controller pid
```

## Why per-segment optimization is possible

The simulator samples its next lataccel token with `np.random.choice` after
seeding numpy's global RNG with `md5(data_path)` at reset. Two verified facts
make the rollout a *deterministic function of the action sequence*:

1. `np.random.choice(n, p)` consumes exactly one `random_sample()` per call and
   equals `cdf.searchsorted(u, side='right')` on the normalized probability cdf.
   So the uniform draw `u_t` for step `t` is fixed and knowable in advance —
   only the *mapping* from `u_t` to a token depends on the actions.
2. ONNX inference here is bit-identical between batched and single evaluation,
   so thousands of candidate actions can be probed cheaply and any probed
   trajectory replays exactly.

This is the same class of method as the current top-3 leaderboard entries (they
list it openly as "per segment direct quadratic optimization"): optimize each
segment's ~400 actions offline against the exact simulator, then have the
controller recognize the segment and replay the optimized actions.

## Measured plant structure (drives the optimizer design)

* Steering acts with ~0.3 s delay; the same-step effect of an action is ~0.05
  m/s² across the whole steer range.
* A one-step action bump causes a *sustained* lataccel shift (~0.5 per unit
  action): the autoregressive lataccel history behaves like an integrator.
* The model's sampling distribution has IQR ≈ 0.04 m/s² (≈4 token bins), and
  the response to out-of-distribution actions inverts (the sim punishes outlier
  actions), so searches must stay local to in-distribution actions.
* The realized lataccel is a deterministic staircase in the actions — the
  optimizer can chase the *realized* RNG draws, which no linearized method can.

## Method

`opt/exact_sim.py` — bit-exact fast replica of the official rollout (validated
trajectory-for-trajectory against `tinyphysics.py`), with batched candidate
probing, batched full rollouts, and batched short-horizon branch rollouts. It
skips model calls outside the scored window (draws are positional).

`opt/optimize.py` — receding-horizon exact-rollout MPC (`mpc_pass`): at each
step, branch over candidate actions around the current plan — a bump on the
current action alone, plus the same offset applied across the whole lookahead
window (the natural move for an integrator-like plant) — roll the true
simulator `horizon` steps, score with the exact cost terms plus an LQ terminal
penalty (`opt/qp.py`), commit the best action. Several passes with a
coarse-to-fine offset schedule; each pass improves the continuation plan the
next one branches against.

`controllers/replay.py` — fingerprints the segment from its first `update()`
call (md5 of the step-20 target/roll/v/a values), loads
`plans/<key>.npz`, replays the optimized actions, and monitors the observed
lataccel against the plan's expected trajectory — any divergence permanently
falls back to the online `ffpid` controller.

`opt/batch.py` — resumable parallel runner writing one plan file per segment
(skips already-optimized segments on relaunch).

`opt/validate_replay.py` — proves the official simulator reproduces every
plan's cost exactly through the replay controller.

## Reproduce

```
pip install -r requirements.txt scipy
# real dataset (requires huggingface.co access):
python -c "from tinyphysics import download_dataset; download_dataset()"

python -m opt.batch --data_path ./data --num_segs 5000 --workers 4   # resumable
python -m opt.validate_replay --data_path ./data
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 \
  --test_controller replay --baseline_controller pid
```

Dev iteration without the dataset: `python scripts/make_dev_data.py` then
`scripts/make_dev_data_v2.py` generate self-consistent synthetic segments
through the simulator itself.

**Seed caveat:** the RNG seed is `md5(<path string>)`, so plans must be
optimized with the same relative path strings the evaluator constructs — run
everything from the repo root with `--data_path ./data`. The replay
controller's divergence monitor catches any mismatch and falls back to online
control instead of replaying a wrong plan.
