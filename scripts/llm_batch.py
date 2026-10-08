"""Scripted (fixed order) vs LLM planner on generated scenarios, offline (no ROS, real model).
    cd src/did_agent && PYTHONPATH=. python3 ../../scripts/llm_batch.py [seeds] [model]
"""
import json
import os
import statistics as st
import sys
import time

from did_agent.core.costmap import CostMap
from did_agent.core.grid_map import GridMap
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.llm_client import LLMClient
from did_agent.core.llm_planner import LLMPlanner, Target
from did_agent.core.mission import ScriptedPlanner
from did_agent.core.navigator import Navigator
from did_agent.core.offline_sim import run_mission
from did_agent.core.scenario import generate

seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
model = sys.argv[2] if len(sys.argv) > 2 else 'deepseek-v4.1-flash'
g = GridMap.from_yaml('maps/map.yaml')
cm = CostMap(g, inflation_radius=0.2)
client = LLMClient(model, env_file=os.path.expanduser('~/innopolis_proj/.env'))
rows = []
for difficulty, battery in (('medium', 60.0), ('hard', 60.0), ('medium', 12.0), ('hard', 15.0)):
    for seed in range(seeds):
        sc = generate(difficulty, seed, g)
        for kind in ('scripted', 'llm'):
            cfg = JudgeConfig.load('config/judge.yaml')
            cfg.battery.initial = battery
            judge = Judge(sc, cfg, g, seed=seed)
            if kind == 'llm':
                planner = LLMPlanner(client, [Target(s.id, s.x, s.y) for s in sc.samples], sc.base,
                                     Navigator(cm).path_cost, async_mode=False)
            else:
                planner = ScriptedPlanner([(s.x, s.y) for s in sc.samples])
            t0 = time.time()
            r = run_mission(g, judge, planner, cm=cm)
            calls = [a for rec in getattr(planner, 'journal', []) for a in rec['attempts']]
            rows.append({'difficulty': difficulty, 'battery0': battery, 'seed': seed, 'planner': kind,
                         'collected': r.score['collected'], 'total': r.score['samples_total'],
                         'returned': r.score['returned'], 'battery_left': r.score['battery'],
                         'distance': r.score['distance'], 'score': r.score['score'],
                         'collisions': r.score['penalties']['collision'], 'hazard_hits': r.score['penalties']['hazard_hit'],
                         'llm_calls': len(calls), 'llm_latency': round(sum(a.get('latency', 0) for a in calls), 1),
                         'rejected': sum(1 for a in calls if a.get('error')),
                         'fallback': sum(1 for rec in getattr(planner, 'journal', []) if rec.get('fallback'))})
            print(rows[-1], flush=True)

print('\nSUMMARY')
summary = {}
for (d, b) in sorted({(r['difficulty'], r['battery0']) for r in rows}):
    for kind in ('scripted', 'llm'):
        rs = [r for r in rows if r['difficulty'] == d and r['battery0'] == b and r['planner'] == kind]
        s = {'samples_pct': round(100 * sum(r['collected'] for r in rs) / sum(r['total'] for r in rs), 1),
             'returned': f"{sum(r['returned'] for r in rs)}/{len(rs)}",
             'distance': round(st.mean(r['distance'] for r in rs), 2),
             'battery_left': round(st.mean(r['battery_left'] for r in rs), 1),
             'score': round(st.mean(r['score'] for r in rs), 1)}
        if kind == 'llm':
            calls = [r for r in rs]
            s.update(llm_calls=sum(r['llm_calls'] for r in calls), rejected=sum(r['rejected'] for r in calls),
                     fallback=sum(r['fallback'] for r in calls),
                     mean_latency_per_call=round(sum(r['llm_latency'] for r in calls) / max(1, sum(r['llm_calls'] for r in calls)), 1))
        summary[f'{d} battery={b:g} {kind}'] = s
        print(f'{d:6s} battery={b:4g} {kind:8s} {s}')
json.dump({'rows': rows, 'summary': summary, 'model': model}, open('../../runs/llm_batch.json', 'w'), indent=1, ensure_ascii=False)
