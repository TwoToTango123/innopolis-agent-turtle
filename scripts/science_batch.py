"""Batch experiment for the scientific agent (offline, no ROS):
H1 - learning the terrain cost from the battery vs planning by the shortest path.

    cd src/did_agent && python3 ../../scripts/science_batch.py --levels easy medium hard --seeds 10 --battery 60 25
Writes runs/science_batch_<time>.json and prints a Markdown table.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PKG = os.path.join(ROOT, 'src', 'did_agent')
sys.path.insert(0, PKG)

from did_agent.core.costmap import CostMap  # noqa: E402
from did_agent.core.grid_map import GridMap  # noqa: E402
from did_agent.core.judge import Judge, JudgeConfig  # noqa: E402
from did_agent.core.navigator import Navigator  # noqa: E402
from did_agent.core.offline_sim import run_mission  # noqa: E402
from did_agent.core.scenario import generate  # noqa: E402
from did_agent.core.science import ScientificPlanner  # noqa: E402


def one(job):
    level, seed, battery, learn = job
    grid = GridMap.from_yaml(os.path.join(PKG, 'maps', 'map.yaml'))
    sc = generate(level, seed, grid)
    cfg = JudgeConfig.load(os.path.join(PKG, 'config', 'judge.yaml'))
    cfg.battery.initial = battery
    judge = Judge(sc, cfg, grid, seed=seed)
    cm = CostMap(grid, inflation_radius=0.2)
    planner = ScientificPlanner(grid, cm, sc.base, Navigator(cm).path_cost, learn_terrain=learn)
    t0 = time.time()
    res = run_mission(grid, judge, planner, cm=cm, max_time=900.0)
    sco = res.score
    hyp = planner.lab.hypotheses.values()
    return {'level': level, 'seed': seed, 'battery0': battery, 'learn': learn,
            'collected': sco['collected'], 'total': sco['samples_total'], 'returned': sco['returned'],
            'battery': sco['battery'], 'distance': sco['distance'], 'score': sco['score'],
            'penalties': sum(sco['penalties'].values()), 'sim_time': round(res.sim_time, 1),
            'hypotheses': len(hyp), 'confirmed': sum(h.status == 'подтверждена' for h in hyp),
            'adapted': sum(h.status == 'изменилась' for h in hyp), 'wall': round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', nargs='+', default=['easy', 'medium', 'hard'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--battery', nargs='+', type=float, default=[60.0])
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = [(lv, sd, b, learn) for lv in a.levels for b in a.battery for sd in range(a.seeds) for learn in (True, False)]
    with Pool(a.jobs) as pool:
        rows = pool.map(one, jobs)
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    out = os.path.join(ROOT, 'runs', time.strftime('science_batch_%Y%m%d-%H%M%S.json'))
    json.dump(rows, open(out, 'w'), indent=1, ensure_ascii=False)
    print('| уровень | заряд | агент | образцы | возврат | штрафы | остаток заряда | путь, м | счёт |')
    print('|---|---|---|---|---|---|---|---|---|')
    for lv in a.levels:
        for b in a.battery:
            for learn in (True, False):
                g = [r for r in rows if r['level'] == lv and r['battery0'] == b and r['learn'] == learn]
                n = len(g)
                col = sum(r['collected'] for r in g) / sum(r['total'] for r in g)
                print(f"| {lv} | {b:.0f} | {'B: учит грунты' if learn else 'A: кратчайший путь'} | {100 * col:.0f} % | "
                      f"{sum(r['returned'] for r in g)}/{n} | {sum(r['penalties'] for r in g)} | "
                      f"{sum(r['battery'] for r in g) / n:.1f} | {sum(r['distance'] for r in g) / n:.1f} | {sum(r['score'] for r in g) / n:.1f} |")
    print('saved', out)


if __name__ == '__main__':
    main()
