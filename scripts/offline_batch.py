"""Offline batch: scripted mission on N generated scenarios per difficulty (no ROS).
    cd src/did_agent && python3 ../../scripts/offline_batch.py [N]
"""
import json
import statistics as st
import sys
import time

from did_agent.core.grid_map import GridMap
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.mission import ScriptedPlanner
from did_agent.core.offline_sim import run_mission
from did_agent.core.scenario import generate

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
g = GridMap.from_yaml('maps/map.yaml')
cfg = JudgeConfig.load('config/judge.yaml')
t0 = time.time()
summary = {}
for d in ('easy', 'medium', 'hard'):
    rows = []
    for seed in range(n):
        sc = generate(d, seed, g)
        j = Judge(sc, cfg, g, seed=seed)
        r = run_mission(g, j, ScriptedPlanner([(s.x, s.y) for s in sc.samples]))
        s = r.score
        rows.append({'collected': s['collected'], 'total': s['samples_total'], 'returned': s['returned'],
                     'collision': s['penalties']['collision'], 'hazard_hit': s['penalties']['hazard_hit'],
                     'battery': s['battery'], 'score': s['score'], 'time': r.sim_time})
    summary[d] = {
        'runs': n,
        'samples_pct': round(100 * sum(r['collected'] for r in rows) / sum(r['total'] for r in rows), 1),
        'returned': sum(r['returned'] for r in rows),
        'collisions': sum(r['collision'] for r in rows),
        'hazard_hits': sum(r['hazard_hit'] for r in rows),
        'battery_left': round(st.mean(r['battery'] for r in rows), 1),
        'score': round(st.mean(r['score'] for r in rows), 1),
        'sim_time_s': round(st.mean(r['time'] for r in rows)),
    }
    print(d, summary[d], flush=True)
print(f'wall time {time.time() - t0:.0f}s')
json.dump(summary, open('../../runs/offline_batch.json', 'w'), indent=1)
