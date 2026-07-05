"""Self-consistent synthetic dev segments, generated through the simulator.

v1 segments had steer commands from a made-up gain formula, leaving the
(state, action, lataccel) joint distribution far from what tinyphysics was trained
on — the model becomes unconfident and its sampling noise explodes. Here we roll
the simulator itself with a PID controller chasing a smooth random reference, then
write a CSV where targetLateralAcceleration is the *achieved* lataccel and
steerCommand is the *taken* action. The result is in-distribution and
self-consistent, like the real dataset (where the target is what the car did).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.make_dev_data import make_segment  # noqa: E402
import tinyphysics  # noqa: E402
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator  # noqa: E402
from controllers import BaseController  # noqa: E402


class PidPreview(BaseController):
  def __init__(self):
    self.i = 0.0
    self.prev_err = 0.0

  def update(self, target, current, state, future_plan):
    n = min(6, len(future_plan.lataccel))
    tgt = np.mean(future_plan.lataccel[:n]) if n else target
    err = tgt - current
    self.i = np.clip(self.i + err * 0.1, -5, 5)
    d = err - self.prev_err
    self.prev_err = err
    return 0.12 * err + 0.6 * self.i * 0.1 + 0.0 * d


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', default='dev_data')
  parser.add_argument('--num', type=int, default=20)
  parser.add_argument('--rows', type=int, default=600)
  parser.add_argument('--model_path', default='./models/tinyphysics.onnx')
  args = parser.parse_args()

  out = Path(args.out)
  out.mkdir(exist_ok=True)
  model = TinyPhysicsModel(args.model_path, debug=False)
  tmp = Path('.tmp_seed_segment.csv')

  for i in range(args.num):
    df = make_segment(seed=2000 + i, rows=args.rows)
    df.to_csv(tmp, index=False)
    sim = TinyPhysicsSimulator(model, str(tmp), controller=PidPreview(), debug=False)
    sim.rollout()
    df['targetLateralAcceleration'] = np.array(sim.current_lataccel_history)[:args.rows]
    # sim uses right-positive; the CSV convention is left-positive (negated on load)
    df['steerCommand'] = -np.array(sim.action_history)[:args.rows]
    df.to_csv(out / f'{i:05d}.csv', index=False)
  tmp.unlink(missing_ok=True)
  print(f'wrote {args.num} self-consistent segments to {out}/')
