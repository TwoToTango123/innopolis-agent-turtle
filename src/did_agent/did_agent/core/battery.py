"""Battery drain model used by the judge. No ROS dependencies.

drain = per_meter * distance * terrain_multiplier + per_radian * |turn| + idle_per_second * dt
(QUESTIONS.md#3: units and rates are not in TASK.md; values come from config/judge.yaml)
"""
import math
from dataclasses import dataclass

from .frames import wrap_angle


@dataclass
class BatteryConfig:
    initial: float = 60.0         # TASK.md
    per_meter: float = 1.0
    per_radian: float = 0.05
    idle_per_second: float = 0.005


class Battery:
    def __init__(self, cfg: BatteryConfig):
        self.cfg = cfg
        self.level = cfg.initial
        self.distance = 0.0           # total ground-truth path length, m

    @property
    def empty(self) -> bool:
        return self.level <= 0.0

    def drain(self, dist: float, dyaw: float, dt: float, multiplier: float = 1.0) -> float:
        """Apply one step; returns the amount drained."""
        if self.empty:
            return 0.0
        c = self.cfg
        d = c.per_meter * dist * multiplier + c.per_radian * abs(dyaw) + c.idle_per_second * max(dt, 0.0)
        d = min(d, self.level)
        self.level -= d
        self.distance += dist
        return d

    def step(self, prev: tuple[float, float, float], cur: tuple[float, float, float], dt: float,
             multiplier: float = 1.0) -> float:
        """prev/cur = (x, y, yaw) ground-truth poses."""
        dist = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        return self.drain(dist, wrap_angle(cur[2] - prev[2]), dt, multiplier)
