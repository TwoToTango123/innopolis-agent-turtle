"""One offline run of the scientific agent with its full journals (for debugging a batch outlier).
    cd src/did_agent && python3 ../../scripts/science_debug.py hard 14 --battery 60 [--no-terrain]
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from science_batch import PKG  # noqa: E402,F401  (sets sys.path)

from did_agent.core.costmap import CostMap  # noqa: E402
from did_agent.core.grid_map import GridMap  # noqa: E402
from did_agent.core.judge import Judge, JudgeConfig  # noqa: E402
from did_agent.core.navigator import Navigator  # noqa: E402
from did_agent.core.offline_sim import run_mission  # noqa: E402
from did_agent.core.scenario import generate  # noqa: E402
from did_agent.core.science import ScientificPlanner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('level')
ap.add_argument('seed', type=int)
ap.add_argument('--battery', type=float, default=60.0)
ap.add_argument('--no-terrain', action='store_true')
ap.add_argument('--tail', type=int, default=25)
a = ap.parse_args()
grid = GridMap.from_yaml(os.path.join(PKG, 'maps', 'map.yaml'))
sc = generate(a.level, a.seed, grid)
cfg = JudgeConfig.load(os.path.join(PKG, 'config', 'judge.yaml'))
cfg.battery.initial = a.battery
judge = Judge(sc, cfg, grid, seed=a.seed)
cm = CostMap(grid, inflation_radius=0.2)
pl = ScientificPlanner(grid, cm, sc.base, Navigator(cm).path_cost, learn_terrain=not a.no_terrain)
res = run_mission(grid, judge, pl, cm=cm)
print('samples', [(s.id, s.x, s.y) for s in sc.samples])
print('terrain', [(z.id, z.cx, z.cy, z.r, z.multiplier) for z in sc.terrain])
print('hazards', [(z.id, z.cx, z.cy, z.r) for z in sc.hazards])
print('hidden', [e for e in judge.log if e.get('hidden') or e['type'] in ('hazard_hit', 'battery_depleted', 'finish')])
for j in res.journal[-a.tail:]:
    print({k: j[k] for k in ('t', 'kind', 'target', 'success', 'message', 'battery')})
print('\n'.join(pl.log[-a.tail:]))
print(pl.lab.to_markdown())
print('score', res.score)
