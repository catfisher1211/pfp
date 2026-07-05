"""Round-trip validation: the official simulator + replay controller must
reproduce each optimized plan's cost exactly (bit-for-bit trajectory match).

Usage: PLANS_DIR=plans_dev python -m opt.validate_replay --data_path ./dev_data
"""
import argparse
import importlib
import os
from pathlib import Path

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', default='./models/tinyphysics.onnx')
  parser.add_argument('--data_path', default='./dev_data')
  parser.add_argument('--num_segs', type=int, default=1000000)
  args = parser.parse_args()

  import tinyphysics
  from opt.batch import fingerprint_key

  plans_dir = Path(os.environ.get('PLANS_DIR', 'plans'))
  model = tinyphysics.TinyPhysicsModel(args.model_path, debug=False)
  files = sorted(Path(args.data_path).iterdir())[:args.num_segs]

  n_ok = n_missing = n_bad = 0
  totals = []
  for f in files:
    plan_file = plans_dir / f'{fingerprint_key(f)}.npz'
    if not plan_file.exists():
      n_missing += 1
      continue
    plan = np.load(plan_file)
    controller = importlib.import_module('controllers.replay').Controller()
    sim = tinyphysics.TinyPhysicsSimulator(model, str(f), controller=controller, debug=False)
    cost = sim.rollout()
    expected = float(plan['total_cost'])
    match = abs(cost['total_cost'] - expected) < 1e-12
    n_ok += match
    n_bad += (not match)
    totals.append(cost['total_cost'])
    flag = 'OK ' if match else 'BAD'
    print(f"{flag} {f}: replayed={cost['total_cost']:.4f} planned={expected:.4f}")
  print(f'\n{n_ok} exact, {n_bad} mismatched, {n_missing} missing plans')
  if totals:
    print(f'mean replayed total_cost: {np.mean(totals):.4f}')


if __name__ == '__main__':
  main()
