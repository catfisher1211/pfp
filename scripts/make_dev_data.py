"""Generate synthetic dev segments in the SYNTHETIC_V0 CSV format.

The real challenge dataset (huggingface.co/datasets/commaai/commaSteeringControl,
SYNTHETIC_V0.zip) is unreachable from this environment, so these segments exist to
develop and validate the optimization pipeline end-to-end. They mimic the column
schema and rough signal statistics of the real data: smooth highway-ish speeds,
small road roll, and smooth target lateral acceleration profiles.

Usage: python scripts/make_dev_data.py [--out dev_data] [--num 20] [--rows 600]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACC_G = 9.81


def smooth_noise(rng, n, scales=((60, 1.0), (20, 0.4), (7, 0.15))):
  """Sum of box-smoothed white noise at several time scales, unit-ish variance."""
  out = np.zeros(n)
  for win, amp in scales:
    x = rng.standard_normal(n + win)
    kernel = np.hanning(win)
    kernel /= kernel.sum()
    out += amp * np.convolve(x, kernel, mode='same')[:n]
  return out


def make_segment(seed: int, rows: int) -> pd.DataFrame:
  rng = np.random.default_rng(seed)

  v_base = rng.uniform(8, 32)
  v_ego = np.clip(v_base + 3.0 * smooth_noise(rng, rows), 1.0, 40.0)
  a_ego = np.clip(np.gradient(v_ego) / 0.1, -2.5, 2.5) + 0.05 * rng.standard_normal(rows)

  roll = 0.03 * smooth_noise(rng, rows)  # radians, ±~0.05 typical

  # Target lataccel: smooth curvy driving, bigger swings at lower speed.
  target = 1.2 * smooth_noise(rng, rows)
  target = np.clip(target, -3.5, 3.5)

  # Plausible steer log: roughly proportional to (target - roll component),
  # attenuated with speed; logged with left-positive convention (hence minus).
  roll_lataccel = np.sin(roll) * ACC_G
  steer = -(target - roll_lataccel) * (13.5 / np.maximum(v_ego, 5.0) ** 2) * 25.0
  steer = np.clip(steer + 0.02 * rng.standard_normal(rows), -2, 2)

  return pd.DataFrame({
    't': np.arange(rows) * 0.1,
    'vEgo': v_ego,
    'aEgo': a_ego,
    'roll': roll,
    'targetLateralAcceleration': target,
    'steerCommand': steer,
  })


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', default='dev_data')
  parser.add_argument('--num', type=int, default=20)
  parser.add_argument('--rows', type=int, default=600)
  args = parser.parse_args()

  out = Path(args.out)
  out.mkdir(exist_ok=True)
  for i in range(args.num):
    make_segment(seed=1000 + i, rows=args.rows).to_csv(out / f'{i:05d}.csv', index=False)
  print(f'wrote {args.num} segments to {out}/')
