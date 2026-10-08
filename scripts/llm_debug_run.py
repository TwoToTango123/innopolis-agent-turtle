"""Replay one generated scenario with the LLM planner offline and print the LLM journal.
    cd src/did_agent && PYTHONPATH=. python3 ../../scripts/llm_debug_run.py medium 4 12
"""
import os
import sys

from did_agent.core.costmap import CostMap
from did_agent.core.grid_map import GridMap
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.llm_client import LLMClient
from did_agent.core.llm_planner import LLMPlanner, Target
from did_agent.core.navigator import Navigator
from did_agent.core.offline_sim import run_mission
from did_agent.core.scenario import generate

difficulty, seed, battery = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
g = GridMap.from_yaml('maps/map.yaml')
cm = CostMap(g, inflation_radius=0.2)
sc = generate(difficulty, seed, g)
cfg = JudgeConfig.load('config/judge.yaml')
cfg.battery.initial = battery
judge = Judge(sc, cfg, g, seed=seed)
client = LLMClient(sys.argv[4] if len(sys.argv) > 4 else 'deepseek-v4.1-flash', env_file=os.path.expanduser('~/innopolis_proj/.env'))
p = LLMPlanner(client, [Target(s.id, s.x, s.y) for s in sc.samples], sc.base, Navigator(cm).path_cost, async_mode=False)
r = run_mission(g, judge, p, cm=cm)
for rec in p.journal:
    print(f'\n[t={rec["t"]}] TRIGGER: {rec["trigger"]}')
    for o in rec.get('options', []):
        print(f'   option {o["id"]}: {o["order"]} cost {o["cost"]} margin {o["margin"]}')
    for a in rec['attempts']:
        print(f'   attempt {a.get("latency")}s' + (f' REJECTED: {a["error"]}' if a.get('error') else ''))
    print(f'   PLAN {"(FALLBACK) " if rec.get("fallback") else ""}{rec.get("plan")} forecast {rec.get("predicted_battery")}')
    print(f'   THOUGHT: {rec.get("thought")}')
print('\nSUBGOALS:')
for j in r.journal:
    print(f'  t={j["t"]:6.1f} {j["kind"]:8s} {str(j["target"]):18s} ok={j["success"]} used={j["battery_used"]:.2f} left={j["battery"]:.2f} {j["message"][:60]}')
print('\nEVENTS:', r.events)
print('SCORE:', r.score)
