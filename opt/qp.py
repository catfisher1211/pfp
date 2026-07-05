"""Quadratic program for the ideal lataccel trajectory.

The official cost over the window [CONTROL_START_IDX, COST_END_IDX) is

  total = 50 * 100/N * sum (x_t - g_t)^2  +  100/(dt^2 (N-1)) * sum (x_t - x_{t-1})^2

with N = 400, dt = 0.1, and the jerk sum only over consecutive pairs *inside* the
window (x_99 is not coupled — np.diff of the window slice). Minimizing over x is an
unconstrained tridiagonal QP:

  (A + B D^T D) x = A g   (+ boundary term when re-solving mid-rollout)

Re-solving the suffix each step with the committed previous value as boundary makes
the greedy rollout a receding-horizon LQ tracker, which absorbs quantization errors
optimally.
"""
import numpy as np
from scipy.linalg import solve_banded

N_WINDOW = 400
A_W = 100.0 * 50.0 / N_WINDOW          # weight per (x_t - g_t)^2
B_W = 100.0 / (0.01 * (N_WINDOW - 1))  # weight per (x_t - x_{t-1})^2


def solve_ideal(targets: np.ndarray, x_prev: float | None) -> np.ndarray:
  """Minimize sum A_W*(x-g)^2 + B_W*jerk-pairs over the suffix.

  targets: remaining targets g_{t..T-1}.
  x_prev: committed lataccel at t-1 if the jerk pair (x_t - x_{t-1}) is in the cost
          (i.e. t > CONTROL_START_IDX), else None.
  """
  n = len(targets)
  if n == 1:
    if x_prev is None:
      return targets.astype(np.float64).copy()
    return np.array([(A_W * targets[0] + B_W * x_prev) / (A_W + B_W)])

  diag = np.full(n, A_W + 2.0 * B_W)
  diag[0] = A_W + B_W + (B_W if x_prev is not None else 0.0)
  diag[-1] = A_W + B_W
  off = np.full(n - 1, -B_W)
  rhs = A_W * targets.astype(np.float64)
  if x_prev is not None:
    rhs[0] += B_W * x_prev

  ab = np.zeros((3, n))
  ab[0, 1:] = off
  ab[1, :] = diag
  ab[2, :-1] = off
  return solve_banded((1, 1), ab, rhs)


def ideal_cost(targets: np.ndarray) -> float:
  """Unconstrained lower bound on total cost for a full window of targets."""
  x = solve_ideal(np.asarray(targets, dtype=np.float64), x_prev=None)
  lat = np.mean((x - targets) ** 2) * 100
  jerk = np.mean((np.diff(x) / 0.1) ** 2) * 100
  return float(lat * 50 + jerk)
