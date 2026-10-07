import math
import random
import statistics

import pytest

from did_agent.core.battery import Battery, BatteryConfig
from did_agent.core.sensor import SampleSensor, SensorConfig


def test_battery_starts_at_60():
    assert Battery(BatteryConfig()).level == 60.0


def test_battery_drain_per_meter_and_terrain():
    b = Battery(BatteryConfig(per_meter=1.0, per_radian=0.0, idle_per_second=0.0))
    b.step((0, 0, 0), (1, 0, 0), 1.0)
    assert b.level == pytest.approx(59.0)
    b.step((1, 0, 0), (2, 0, 0), 1.0, multiplier=3.0)
    assert b.level == pytest.approx(56.0)
    assert b.distance == pytest.approx(2.0)


def test_battery_turn_and_idle():
    b = Battery(BatteryConfig(per_meter=1.0, per_radian=0.1, idle_per_second=0.01))
    b.step((0, 0, 3.0), (0, 0, -3.0), 10.0)       # wraps: |dyaw| = 2*pi - 6
    assert b.level == pytest.approx(60.0 - 0.1 * (2 * math.pi - 6.0) - 0.1)


def test_battery_never_negative():
    b = Battery(BatteryConfig(initial=1.0))
    b.step((0, 0, 0), (5, 0, 0), 1.0)
    assert b.level == 0.0 and b.empty
    assert b.step((5, 0, 0), (6, 0, 0), 1.0) == 0.0


def test_sensor_monotonic_without_noise():
    s = SampleSensor(SensorConfig(noise_std=0.0))
    vals = [s.read(d) for d in (0.0, 0.3, 1.0, 2.0, 4.0)]
    assert vals[0] == 1.0
    assert vals == sorted(vals, reverse=True)
    assert s.read(None) == 0.0


def test_sensor_noise_is_seeded_and_bounded():
    a = SampleSensor(SensorConfig(), random.Random(5))
    b = SampleSensor(SensorConfig(), random.Random(5))
    ra = [a.read(0.5) for _ in range(200)]
    assert ra == [b.read(0.5) for _ in range(200)]
    assert all(0.0 <= v <= 1.0 for v in ra)
    assert statistics.mean(ra) == pytest.approx(math.exp(-0.5 / 0.7), abs=0.01)
    assert 0.02 < statistics.stdev(ra) < 0.04
