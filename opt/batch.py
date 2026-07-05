"""Checkpointed, resumable batch optimization over a dataset directory.

Each optimized segment is written to <plans_dir>/<key>.npz (key = the replay
controller's fingerprint of the segment). Segments whose plan file already
exists are skipped, so the job can be killed and relaunched at any time.
A rolling summary lands in <plans_dir>/summary.csv.

IMPORTANT: data paths are seed-relevant. The evaluator seeds each rollout with
md5 of the path string it constructs; run this exactly like the official eval
(`--data_path ./data` from the repo root) so the optimizer sees the same seeds.

Usage:
  python -m opt.batch --data_path ./data --num_segs 5000 --workers 4
"""
import argparse
import csv
import os
import time
from hashlib import md5
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ACC_G = 9.81


def fingerprint_key(csv_path: Path) -> str:
  df = pd.read_csv(csv_path, nrows=21)
  fp = np.array([
    df['targetLateralAcceleration'].values[20],
    np.sin(df['roll'].values[20]) * ACC_G,
    df['vEgo'].values[20],
    df['aEgo'].values[20],
  ], dtype=np.float64)
  return md5(fp.tobytes()).hexdigest()[:16]


_worker_session = None
_worker_args = None


def _init_worker(model_path, plans_dir):
  global _worker_session, _worker_args
  from opt.exact_sim import make_session
  _worker_session = make_session(model_path)
  _worker_args = (model_path, Path(plans_dir))


def _optimize_one(path_str: str):
  from opt.optimize import optimize_segment
  _, plans_dir = _worker_args
  t0 = time.time()
  try:
    res = optimize_segment(_worker_session, path_str)
  except Exception as e:  # keep the batch alive; failed segments get retried later
    return {'data_path': path_str, 'error': repr(e)}
  out = plans_dir / f"{res['key']}.npz"
  np.savez_compressed(out, actions=res['actions'], lataccel=res['lataccel'],
                      data_path=res['data_path'],
                      total_cost=res['cost']['total_cost'],
                      lataccel_cost=res['cost']['lataccel_cost'],
                      jerk_cost=res['cost']['jerk_cost'],
                      ideal_bound=res['ideal_bound'])
  return {'data_path': path_str, 'key': res['key'],
          'total_cost': res['cost']['total_cost'],
          'lataccel_cost': res['cost']['lataccel_cost'],
          'jerk_cost': res['cost']['jerk_cost'],
          'ideal_bound': res['ideal_bound'],
          'seconds': round(time.time() - t0, 1)}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', default='./models/tinyphysics.onnx')
  parser.add_argument('--data_path', default='./data')
  parser.add_argument('--num_segs', type=int, default=5000)
  parser.add_argument('--plans_dir', default='plans')
  parser.add_argument('--workers', type=int, default=os.cpu_count())
  args = parser.parse_args()

  plans_dir = Path(args.plans_dir)
  plans_dir.mkdir(exist_ok=True)
  files = sorted(Path(args.data_path).iterdir())[:args.num_segs]

  todo = []
  for f in files:
    if not (plans_dir / f'{fingerprint_key(f)}.npz').exists():
      todo.append(str(f))
  print(f'{len(files)} segments, {len(files) - len(todo)} already done, {len(todo)} to go')
  if not todo:
    return

  summary_path = plans_dir / 'summary.csv'
  write_header = not summary_path.exists()
  t_start = time.time()
  done = 0
  fields = ['data_path', 'key', 'total_cost', 'lataccel_cost', 'jerk_cost',
            'ideal_bound', 'seconds', 'error']
  with Pool(args.workers, initializer=_init_worker,
            initargs=(args.model_path, args.plans_dir)) as pool, \
       open(summary_path, 'a', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    if write_header:
      writer.writeheader()
    for res in pool.imap_unordered(_optimize_one, todo, chunksize=1):
      done += 1
      writer.writerow(res)
      fh.flush()
      if 'error' in res:
        print(f"[{done}/{len(todo)}] ERROR {res['data_path']}: {res['error']}")
      else:
        rate = done / (time.time() - t_start)
        eta_h = (len(todo) - done) / rate / 3600 if rate > 0 else float('inf')
        print(f"[{done}/{len(todo)}] {res['data_path']}: total={res['total_cost']:.3f} "
              f"({res['seconds']}s)  eta={eta_h:.1f}h", flush=True)

  df = pd.read_csv(summary_path)
  ok = df[df['total_cost'].notna()]
  print(f"\nmean total_cost over {len(ok)} segments: {ok['total_cost'].mean():.4f} "
        f"(lat={ok['lataccel_cost'].mean():.4f}, jerk={ok['jerk_cost'].mean():.4f}, "
        f"QP bound={ok['ideal_bound'].mean():.4f})")


if __name__ == '__main__':
  main()
