"""/did/sample_sensor model. No ROS dependencies.

TASK.md: proximity to the nearest uncollected sample in [0, 1], noisy, no direction.
Model (QUESTIONS.md#7): exp(-d / sigma) + N(0, noise_std), clipped to [0, 1].
"""
import math
import random
from dataclasses import dataclass


@dataclass
class SensorConfig:
    sigma: float = 0.7
    noise_std: float = 0.03


class SampleSensor:
    def __init__(self, cfg: SensorConfig, rng: random.Random | None = None):
        self.cfg = cfg
        self.rng = rng or random.Random()

    def clean(self, d: float | None) -> float:
        """Noise-free reading for distance d (None = nothing left to find)."""
        return 0.0 if d is None else math.exp(-max(d, 0.0) / self.cfg.sigma)

    def read(self, d: float | None, noise_std: float | None = None) -> float:
        std = self.cfg.noise_std if noise_std is None else noise_std
        v = self.clean(d) + (self.rng.gauss(0.0, std) if std > 0 else 0.0)
        return min(1.0, max(0.0, v))
