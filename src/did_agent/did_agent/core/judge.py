"""Judge logic: battery, sample sensor, collect/finish, penalties, score, hidden
environment events. No ROS dependencies; judge_node is a thin wrapper.

The judge works on the GROUND-TRUTH robot pose from Gazebo, not /odom: wheel
odometry keeps integrating when the robot pushes against a wall (see DEVLOG).
"""
import math
import random
from dataclasses import dataclass, field, fields

import yaml

from .battery import Battery, BatteryConfig
from .costmap import dilate
from .grid_map import FREE, GridMap
from .scenario import Scenario, Zone
from .sensor import SampleSensor, SensorConfig

PUBLIC_EVENTS = ('collision', 'false_collect', 'hazard_hit', 'sample_collected')


@dataclass
class ScoreConfig:                       # QUESTIONS.md#4
    sample: float = 10.0
    false_collect: float = -2.0
    collision: float = -3.0
    hazard_hit: float = -5.0
    return_bonus_per_battery: float = 0.1


@dataclass
class JudgeConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    collect_radius: float = 0.30         # TASK.md
    base_radius: float = 0.30            # QUESTIONS.md#6
    robot_radius: float = 0.105          # Burger, QUESTIONS.md#5
    collision_margin: float = 0.02
    hazard_noise_factor: float = 3.0     # QUESTIONS.md#8

    @classmethod
    def from_dict(cls, d: dict | None) -> 'JudgeConfig':
        d = dict(d or {})
        sub = {'battery': BatteryConfig, 'sensor': SensorConfig, 'score': ScoreConfig}
        kw = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            kw[f.name] = sub[f.name](**d[f.name]) if f.name in sub else d[f.name]
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown judge config keys: {sorted(unknown)}')
        return cls(**kw)

    @classmethod
    def load(cls, path: str) -> 'JudgeConfig':
        with open(path) as f:
            return cls.from_dict((yaml.safe_load(f) or {}).get('judge'))


class Judge:
    def __init__(self, scenario: Scenario, cfg: JudgeConfig, grid: GridMap, seed: int | None = None):
        self.sc = scenario
        self.cfg = cfg
        self.grid = grid
        self.battery = Battery(cfg.battery)
        self.sensor = SampleSensor(cfg.sensor, random.Random(seed))
        self.terrain = [Zone(**z.__dict__) for z in scenario.terrain]   # mutable copies
        self.hazards = [Zone(**z.__dict__) for z in scenario.hazards]
        self.pending = list(scenario.events)
        self.collected: set[str] = set()
        self.penalties = {'collision': 0, 'false_collect': 0, 'hazard_hit': 0}
        self.points = 0.0
        self.state = 'running'               # running | finished | depleted
        self.returned = False
        self.t = 0.0
        self.pose: tuple[float, float, float] | None = None
        self.sensor_fault_until = -1.0
        self.sensor_fault_std = 0.0
        self._in_contact = False
        self._in_hazard: set[str] = set()
        self.log: list[dict] = []            # private: includes hidden env changes
        self.trajectory: list[tuple[float, float, float]] = []   # ground truth (t, x, y), every 0.5 s
        res = grid.resolution
        self._contact = dilate(grid.occ != FREE, (cfg.robot_radius + cfg.collision_margin) / res)

    # ---- geometry helpers -------------------------------------------------
    def terrain_multiplier(self, x: float, y: float) -> float:
        return max([z.multiplier for z in self.terrain if z.contains(x, y)], default=1.0)

    def nearest_sample(self) -> tuple[str, float] | None:
        if self.pose is None:
            return None
        best = None
        for s in self.sc.samples:
            if s.id in self.collected:
                continue
            d = math.hypot(s.x - self.pose[0], s.y - self.pose[1])
            if best is None or d < best[1]:
                best = (s.id, d)
        return best

    def at_base(self) -> bool:
        return self.pose is not None and \
            math.hypot(self.pose[0] - self.sc.base[0], self.pose[1] - self.sc.base[1]) <= self.cfg.base_radius

    def _touching(self, x: float, y: float) -> bool:
        ix, iy = self.grid.world_to_cell(x, y)
        return not self.grid.in_bounds(ix, iy) or bool(self._contact[iy, ix])

    # ---- events -----------------------------------------------------------
    def _event(self, etype: str, penalty: float = 0.0, **extra) -> dict:
        e = {'type': etype, 't': round(self.t, 2)}
        if self.pose is not None:
            e.update(x=round(self.pose[0], 2), y=round(self.pose[1], 2))
        if penalty:
            e['penalty'] = penalty
            self.points += penalty
        e.update(extra)
        self.log.append(e)
        return e

    def _apply_env_event(self, ev) -> None:
        p = ev.params
        if ev.type == 'terrain_change':
            for z in self.terrain:
                if z.id == p['zone']:
                    z.multiplier = p['multiplier']
        elif ev.type == 'new_hazard':
            self.hazards.append(Zone(**p['zone']))
        elif ev.type == 'sensor_fault':
            self.sensor_fault_std = p.get('noise_std', 0.2)
            self.sensor_fault_until = self.t + p.get('duration', 60.0)
        else:
            raise ValueError(f'unknown scenario event {ev.type!r}')
        self.log.append({'type': 'env_' + ev.type, 't': round(self.t, 2), 'hidden': True, **p})

    # ---- main API -----------------------------------------------------------
    def update(self, t: float, x: float, y: float, yaw: float) -> list[dict]:
        """Feed a ground-truth pose at sim time t (s since judge start). Returns public events."""
        out = []
        prev, self.pose = self.pose, (x, y, yaw)
        dt = t - self.t if prev is not None else 0.0
        self.t = t
        if not self.trajectory or t - self.trajectory[-1][0] >= 0.5:
            self.trajectory.append((round(t, 2), round(x, 3), round(y, 3)))
        if self.state != 'running':
            return out
        while self.pending and self.pending[0].t <= t:
            self._apply_env_event(self.pending.pop(0))
        if prev is not None:
            mid = ((prev[0] + x) / 2, (prev[1] + y) / 2)
            self.battery.step(prev, self.pose, dt, self.terrain_multiplier(*mid))
            if self.battery.empty:
                self.state = 'depleted'
                self.log.append({'type': 'battery_depleted', 't': round(t, 2)})
        touching = self._touching(x, y)
        if touching and not self._in_contact:
            self.penalties['collision'] += 1
            out.append(self._event('collision', self.cfg.score.collision))
        self._in_contact = touching
        inside = {h.id for h in self.hazards if h.contains(x, y)}
        for hid in sorted(inside - self._in_hazard):
            self.penalties['hazard_hit'] += 1
            out.append(self._event('hazard_hit', self.cfg.score.hazard_hit, zone=hid))
        self._in_hazard = inside
        return out

    def sensor_reading(self) -> float:
        n = self.nearest_sample()
        std = self.cfg.sensor.noise_std
        if self.t < self.sensor_fault_until:
            std = max(std, self.sensor_fault_std)
        if self._in_hazard:
            std *= self.cfg.hazard_noise_factor
        return self.sensor.read(None if n is None else n[1], std)

    def collect(self) -> tuple[bool, str, list[dict]]:
        if self.state != 'running':
            return False, f'run is over ({self.state})', []
        n = self.nearest_sample()
        if n is not None and n[1] < self.cfg.collect_radius:
            self.collected.add(n[0])
            self.points += self.cfg.score.sample
            e = self._event('sample_collected', sample=n[0])
            return True, f'collected {n[0]} ({len(self.collected)}/{len(self.sc.samples)})', [e]
        self.penalties['false_collect'] += 1
        e = self._event('false_collect', self.cfg.score.false_collect)
        return False, 'no sample within %.2f m' % self.cfg.collect_radius, [e]

    def finish(self) -> tuple[bool, str]:
        """Ends the run. success = robot is at base (return counted)."""
        if self.state == 'finished':
            return self.returned, 'already finished'
        if self.state == 'running':
            self.returned = self.at_base()
            if self.returned:
                self.points += self.cfg.score.return_bonus_per_battery * self.battery.level
        self.state = 'finished'
        self.log.append({'type': 'finish', 't': round(self.t, 2), 'returned': self.returned})
        return self.returned, ('returned to base' if self.returned else 'finished away from base, no return bonus') + \
            f'; score {self.points:.2f}'

    def score(self) -> dict:
        """Public state for /did/score. Never reveals sample/zone positions."""
        return {
            'scenario': self.sc.name,
            't': round(self.t, 2),
            'state': self.state,
            'battery': round(self.battery.level, 3),
            'distance': round(self.battery.distance, 3),
            'collected': len(self.collected),
            'samples_total': len(self.sc.samples),
            'penalties': dict(self.penalties),
            'at_base': self.at_base(),
            'returned': self.returned,
            'score': round(self.points, 2),
        }
