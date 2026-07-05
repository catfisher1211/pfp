from . import BaseController
import numpy as np


class Controller(BaseController):
  """Preview + PI feedback fallback controller (no lookup tables).

  Averages the next ~0.6s of the future plan as the tracking reference (which
  matches the plant's actuation delay and keeps commanded jerk low) and applies
  PI feedback on it.
  """
  PREVIEW = 6
  KP = 0.12
  KI = 0.06

  def __init__(self):
    self.i = 0.0
    self.prev_err = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    n = min(self.PREVIEW, len(future_plan.lataccel))
    tgt = np.mean(future_plan.lataccel[:n]) if n else target_lataccel
    err = tgt - current_lataccel
    self.i = np.clip(self.i + err * 0.1, -5.0, 5.0)
    return float(self.KP * err + self.KI * self.i)
