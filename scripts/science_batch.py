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
    level, seed, battery, learn, llm = job
    if llm == 'K':          # H2: run the mission, then the same scenario again starting from what it learned
        first = mission(level, seed, battery, True, False)
        second = mission(level, seed, battery, True, False, knowledge=first.pop('_knowledge'))
        second.pop('_knowledge')
        return [dict(first, variant='B'), dict(second, variant='K')]
    row = mission(level, seed, battery, learn, llm)
    row.pop('_knowledge')
    return [dict(row, variant='C' if llm else 'B' if learn else 'A')]


def mission(level, seed, battery, learn, llm, knowledge=None):
    grid = GridMap.from_yaml(os.path.join(PKG, 'maps', 'map.yaml'))
    sc = generate(level, seed, grid)
    cfg = JudgeConfig.load(os.path.join(PKG, 'config', 'judge.yaml'))
    cfg.battery.initial = battery
    judge = Judge(sc, cfg, grid, seed=seed + (1000 if knowledge else 0))
    cm = CostMap(grid, inflation_radius=0.2)
    advisor = None
    if llm:
        from did_agent.core.llm_client import LLMClient
        from did_agent.core.llm_scientist import LLMScientist
        advisor = LLMScientist(LLMClient('deepseek-v4.1-flash', env_file=os.path.join(ROOT, '.env')), async_mode=False)
    planner = ScientificPlanner(grid, cm, sc.base, Navigator(cm).path_cost, learn_terrain=learn, advisor=advisor,
                                knowledge=knowledge)
    t0 = time.time()
    res = run_mission(grid, judge, planner, cm=cm, max_time=900.0)
    sco = res.score
    hyp = planner.lab.hypotheses.values()
    calls = advisor.journal if advisor else []
    return {'level': level, 'seed': seed, 'battery0': battery, 'learn': learn, 'llm': llm,
            'llm_calls': sum(len(r['attempts']) for r in calls), 'llm_fallbacks': sum(r['fallback'] for r in calls),
            'llm_latency': round(sum(a.get('latency') or 0 for r in calls for a in r['attempts']), 1),
            'collected': sco['collected'], 'total': sco['samples_total'], 'returned': sco['returned'],
            'battery': sco['battery'], 'distance': sco['distance'], 'score': sco['score'],
            'penalties': sum(sco['penalties'].values()), 'sim_time': round(res.sim_time, 1),
            'hypotheses': len(hyp), 'confirmed': sum(h.status == 'подтверждена' for h in hyp),
            'adapted': sum(h.status == 'изменилась' for h in hyp), 'wall': round(time.time() - t0, 1),
            '_knowledge': planner.export_knowledge()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', nargs='+', default=['easy', 'medium', 'hard'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--battery', nargs='+', type=float, default=[60.0])
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--llm', action='store_true', help='compare B (algorithm) with C (algorithm + LLM advisor) instead of A/B')
    ap.add_argument('--knowledge', action='store_true', help='H2: compare B (fresh) with K (same scenario, knowledge from the previous mission)')
    a = ap.parse_args()
    if a.knowledge:
        variants, names = [(True, 'K')], {'B': 'B: с нуля', 'K': 'K: со знаниями прошлой миссии'}
    elif a.llm:
        variants, names = [(True, False), (True, True)], {'B': 'B: учит грунты', 'C': 'C: + LLM-советник'}
    else:
        variants, names = [(True, False), (False, False)], {'B': 'B: учит грунты', 'A': 'A: кратчайший путь'}
    jobs = [(lv, sd, b, learn, llm) for lv in a.levels for b in a.battery for sd in range(a.seeds) for learn, llm in variants]
    with Pool(a.jobs) as pool:
        rows = [r for rs in pool.map(one, jobs) for r in rs]
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    out = os.path.join(ROOT, 'runs', time.strftime('science_batch_%Y%m%d-%H%M%S.json'))
    json.dump(rows, open(out, 'w'), indent=1, ensure_ascii=False)
    print('| уровень | заряд | агент | образцы | возврат | штрафы | остаток заряда | путь, м | счёт |')
    print('|---|---|---|---|---|---|---|---|---|')
    for lv in a.levels:
        for b in a.battery:
            for v, name in names.items():
                g = [r for r in rows if r['level'] == lv and r['battery0'] == b and r['variant'] == v]
                n = len(g)
                col = sum(r['collected'] for r in g) / sum(r['total'] for r in g)
                print(f"| {lv} | {b:.0f} | {name} | {100 * col:.0f} % | "
                      f"{sum(r['returned'] for r in g)}/{n} | {sum(r['penalties'] for r in g)} | "
                      f"{sum(r['battery'] for r in g) / n:.1f} | {sum(r['distance'] for r in g) / n:.1f} | {sum(r['score'] for r in g) / n:.1f} |")
    print('saved', out)


if __name__ == '__main__':
    main()
