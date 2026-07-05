"""Validate that ExactSim reproduces the official simulator bit-for-bit.

Runs the official TinyPhysicsSimulator with the pid controller on each segment,
then replays the recorded actions through ExactSim and asserts the lataccel
trajectory and costs are identical.
"""
import argparse
import importlib
from pathlib import Path

import numpy as np

from opt.exact_sim import ExactSim, make_session, CONTROL_START_IDX, COST_END_IDX


def validate(model_path: str, data_paths, verbose=True) -> bool:
  import tinyphysics
  session = make_session(model_path)
  model = tinyphysics.TinyPhysicsModel(model_path, debug=False)
  ok = True
  for p in data_paths:
    controller = importlib.import_module('controllers.pid').Controller()
    sim = tinyphysics.TinyPhysicsSimulator(model, str(p), controller=controller, debug=False)
    official_cost = sim.rollout()

    ex = ExactSim(session, str(p))
    actions = sim.action_history[CONTROL_START_IDX:COST_END_IDX]
    ex.reset()
    for a in actions:
      ex.step(a)
    replica = np.array(ex.lataccel[:COST_END_IDX])
    official = np.array(sim.current_lataccel_history[:COST_END_IDX])
    exact = bool(np.array_equal(replica, official))
    cost = ex.cost()
    cost_match = all(np.isclose(cost[k], official_cost[k], rtol=0, atol=0) for k in cost)
    ok &= exact and cost_match
    if verbose:
      print(f'{p}: trajectories bit-exact={exact} cost match={cost_match} '
            f"(official total={official_cost['total_cost']:.4f}, replica total={cost['total_cost']:.4f})")
  return ok


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', default='./models/tinyphysics.onnx')
  parser.add_argument('--data_path', default='./dev_data')
  parser.add_argument('--num_segs', type=int, default=5)
  args = parser.parse_args()
  paths = sorted(Path(args.data_path).iterdir())[:args.num_segs]
  print('ALL OK' if validate(args.model_path, paths) else 'FAILED')
