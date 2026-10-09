"""Scenario = hidden samples, terrain zones, hazard zones, timed events. No ROS dependencies.

Stored as YAML; can also be generated deterministically from (difficulty, seed).
All coordinates are in the Gazebo world frame (= map frame).
"""
import math
import random
from collections import deque
from dataclasses import asdict, dataclass, field

import yaml

from .costmap import CostMap
from .grid_map import GridMap

BASE = (-2.0, -0.5)

# TASK.md table: samples / slow terrain zones / events.
# Hazard counts are not in TASK.md -> QUESTIONS.md#11 (temporary: hard only).
DIFFICULTY = {
    'easy':   {'samples': 3, 'terrain': 1, 'hazards': 0, 'events': False},
    'medium': {'samples': 5, 'terrain': 3, 'hazards': 0, 'events': False},
    'hard':   {'samples': 7, 'terrain': 4, 'hazards': 1, 'events': True},
}


@dataclass
class Sample:
    id: str
    x: float
    y: float


@dataclass
class Zone:
    """Circle (cx, cy, r) or axis-aligned rect (x0, y0, x1, y1)."""
    id: str
    shape: str
    multiplier: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    def contains(self, x: float, y: float) -> bool:
        if self.shape == 'circle':
            return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r * self.r
        if self.shape == 'rect':
            return min(self.x0, self.x1) <= x <= max(self.x0, self.x1) and \
                min(self.y0, self.y1) <= y <= max(self.y0, self.y1)
        raise ValueError(f'unknown zone shape {self.shape!r}')

    def to_dict(self) -> dict:
        keys = ('cx', 'cy', 'r') if self.shape == 'circle' else ('x0', 'y0', 'x1', 'y1')
        d = {'id': self.id, 'shape': self.shape, 'multiplier': self.multiplier}
        d.update({k: getattr(self, k) for k in keys})
        return d


@dataclass
class Event:
    """Unannounced environment change at sim time `t` (seconds since judge start).
    type: terrain_change {zone, multiplier} | new_hazard {zone: Zone dict} | sensor_fault {noise_std, duration}"""
    t: float
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    difficulty: str
    seed: int | None
    base: tuple[float, float]
    samples: list[Sample]
    terrain: list[Zone]
    hazards: list[Zone] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            'name': self.name, 'difficulty': self.difficulty, 'seed': self.seed,
            'base': list(self.base),
            'samples': [asdict(s) for s in self.samples],
            'terrain': [z.to_dict() for z in self.terrain],
            'hazards': [z.to_dict() for z in self.hazards],
            'events': [asdict(e) for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Scenario':
        return cls(
            name=d.get('name', 'unnamed'), difficulty=d.get('difficulty', 'custom'), seed=d.get('seed'),
            base=tuple(d.get('base', BASE)),
            samples=[Sample(**s) for s in d.get('samples', [])],
            terrain=[Zone(**z) for z in d.get('terrain', [])],
            hazards=[Zone(**z) for z in d.get('hazards', [])],
            events=sorted((Event(**e) for e in d.get('events', [])), key=lambda e: e.t),
        )

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str) -> 'Scenario':
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def resolve_scenario(name: str, seed: int, grid: GridMap, scenarios_dir: str) -> Scenario:
    """`name` is a YAML path, or easy/medium/hard: seed >= 0 -> generate, else the bundled YAML."""
    import os
    if name.endswith('.yaml'):
        return Scenario.load(os.path.expanduser(name))
    if name not in DIFFICULTY:
        raise ValueError(f'scenario must be a .yaml path or one of {sorted(DIFFICULTY)}, got {name!r}')
    if seed >= 0:
        return generate(name, seed, grid)
    return Scenario.load(os.path.join(scenarios_dir, f'{name}.yaml'))


# ---- generator --------------------------------------------------------------

def reachable_points(grid: GridMap, base=BASE, clearance: float = 0.25) -> list[tuple[float, float]]:
    """World centres of cells reachable from `base` while keeping `clearance` from obstacles."""
    cm = CostMap(grid, inflation_radius=clearance, soft_radius=0.0)
    start = cm.nearest_free(*grid.world_to_cell(*base))
    seen = {start}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (nx, ny) not in seen and cm.is_free(nx, ny):
                seen.add((nx, ny))
                q.append((nx, ny))
    return [grid.cell_to_world(ix, iy) for ix, iy in sorted(seen)]


def _pick(rng: random.Random, pts, ok, tries: int = 2000):
    for _ in range(tries):
        p = rng.choice(pts)
        if ok(p):
            return p
    raise RuntimeError('scenario generator: could not place an object, constraints too tight')


def generate(difficulty: str, seed: int, grid: GridMap, base=BASE) -> Scenario:
    spec = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    pts = reachable_points(grid, base)
    dist = math.dist

    samples: list[Sample] = []
    for i in range(spec['samples']):
        p = _pick(rng, pts, lambda p: dist(p, base) >= 0.8 and all(dist(p, (s.x, s.y)) >= 0.6 for s in samples))
        samples.append(Sample(f's{i + 1}', round(p[0], 2), round(p[1], 2)))

    terrain: list[Zone] = []
    for i in range(spec['terrain']):
        r = round(rng.uniform(0.35, 0.6), 2)
        p = _pick(rng, pts, lambda p: dist(p, base) >= r + 0.4 and
                  all(dist(p, (z.cx, z.cy)) >= z.r + r for z in terrain))
        terrain.append(Zone(chr(ord('A') + i), 'circle', round(rng.uniform(2.0, 4.0), 1),
                            cx=round(p[0], 2), cy=round(p[1], 2), r=r))

    def hazard_ok(p, r):
        return dist(p, base) >= r + 0.6 and all(dist(p, (s.x, s.y)) >= r + 0.3 for s in samples)

    hazards: list[Zone] = []
    for i in range(spec['hazards']):
        r = 0.3
        p = _pick(rng, pts, lambda p: hazard_ok(p, r))
        hazards.append(Zone(f'H{i + 1}', 'circle', 1.0, cx=round(p[0], 2), cy=round(p[1], 2), r=r))

    events: list[Event] = []
    if spec['events']:
        z = rng.choice(terrain)
        new_mult = round(rng.uniform(1.0, 1.5), 1) if z.multiplier >= 2.5 else round(rng.uniform(3.0, 4.0), 1)
        # timed to happen mid-mission: the agent clears hard in ~2 min offline (QUESTIONS.md#14)
        events.append(Event(round(rng.uniform(20, 40), 1), 'terrain_change', {'zone': z.id, 'multiplier': new_mult}))
        r = 0.35
        p = _pick(rng, pts, lambda p: hazard_ok(p, r))
        events.append(Event(round(rng.uniform(35, 60), 1), 'new_hazard',
                            {'zone': Zone(f'H{len(hazards) + 1}', 'circle', 1.0,
                                          cx=round(p[0], 2), cy=round(p[1], 2), r=r).to_dict()}))
        events.append(Event(round(rng.uniform(45, 75), 1), 'sensor_fault',
                            {'noise_std': 0.2, 'duration': 40.0}))

    return Scenario(f'{difficulty}_seed{seed}', difficulty, seed, tuple(base),
                    samples, terrain, hazards, events)
