"""Offline mission simulator: unicycle kinematics + the real Judge + the real map.
No ROS, no Gazebo; a whole mission runs in seconds. For tests and for developing
planners (e.g. the LLM planner) on any OS:

    python3 -m did_agent.core.offline_sim easy
"""
import math
from dataclasses import dataclass, field

from .costmap import CostMap
from .executor import MissionExecutor
from .grid_map import FREE, GridMap
from .judge import Judge, JudgeConfig
from .mission import AgentState, Planner
from .navigator import Navigator


def raycast_front(grid: GridMap, x: float, y: float, yaw: float, half_angle: float = 0.5,
                  max_range: float = 1.0, rays: int = 11) -> float:
    """Poor man's lidar: nearest non-free map cell within the front cone."""
    best = math.inf
    step = grid.resolution / 2
    for k in range(rays):
        a = yaw - half_angle + 2 * half_angle * k / (rays - 1)
        r = 0.0
        while r < min(best, max_range):
            r += step
            ix, iy = grid.world_to_cell(x + r * math.cos(a), y + r * math.sin(a))
            if not grid.in_bounds(ix, iy) or grid.occ[iy, ix] != FREE:
                best = min(best, r)
                break
    return best


@dataclass
class OfflineResult:
    score: dict
    journal: list[dict]
    events: list[dict]
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    sim_time: float = 0.0


def run_mission(grid: GridMap, judge: Judge, planner: Planner, cm: CostMap | None = None,
                dt: float = 0.05, max_time: float = 900.0, start=(-2.0, -0.5, 0.0)) -> OfflineResult:
    cm = cm or CostMap(grid, inflation_radius=0.2)
    ex = MissionExecutor(planner, Navigator(cm), judge.sc.base)
    x, y, yaw = start
    t = 0.0
    events, traj = [], [(x, y)]
    judge.update(t, x, y, yaw)
    seq = 0
    while t < max_time and ex.state != MissionExecutor.DONE:
        seq += 1
        r = judge.sensor_reading()
        s = AgentState(t, x, y, yaw, judge.battery.level, r, judge.sc.base,
                       collected=len(judge.collected), score=judge.score(), sensor_raw=r, sensor_seq=seq)
        cmd = ex.step(s, raycast_front(grid, x, y, yaw))
        if cmd.call == 'collect':
            ok, msg, ev = judge.collect()
            events += ev
            ex.add_events(ev)
            ex.service_result(ok, msg, s)
        elif cmd.call == 'finish':
            ok, msg = judge.finish()
            ex.service_result(ok, msg, s)
        if judge.state == 'depleted':
            cmd.v = cmd.w = 0.0
        x += cmd.v * math.cos(yaw) * dt
        y += cmd.v * math.sin(yaw) * dt
        yaw = math.atan2(math.sin(yaw + cmd.w * dt), math.cos(yaw + cmd.w * dt))
        t += dt
        ev = judge.update(t, x, y, yaw)
        events += ev
        ex.add_events(ev)
        traj.append((x, y))
    return OfflineResult(judge.score(), ex.journal, events, traj, t)


def main(argv=None):
    import argparse
    import os

    from .mission import ScriptedPlanner

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument('scenario', help='easy | medium | hard | path.yaml')
    ap.add_argument('--planner', choices=['scripted', 'llm', 'science'], default='scripted')
    ap.add_argument('--no-terrain', action='store_true', help='science: do not learn terrain (H1 baseline)')
    ap.add_argument('--llm', action='store_true', help='science: the LLM chooses the strategy at key moments')
    ap.add_argument('--seed', type=int, default=-1, help='>= 0: generate the scenario from this seed')
    ap.add_argument('--battery', type=float, default=0.0, help='> 0: starting battery instead of 60')
    ap.add_argument('--model', default='deepseek-v4.1-flash')
    ap.add_argument('--env', default=os.path.expanduser('~/innopolis_proj/.env'), help='file with MAI_API_KEY')
    args = ap.parse_args(argv)
    from .scenario import resolve_scenario
    grid = GridMap.from_yaml(os.path.join(here, 'maps', 'map.yaml'))
    sc = resolve_scenario(args.scenario, args.seed, grid, os.path.join(here, 'scenarios'))
    cfg = JudgeConfig.load(os.path.join(here, 'config', 'judge.yaml'))
    if args.battery > 0:
        cfg.battery.initial = args.battery
    judge = Judge(sc, cfg, grid, seed=0)
    cm = CostMap(grid, inflation_radius=0.2)
    if args.planner == 'llm':
        from .llm_client import LLMClient
        from .llm_planner import LLMPlanner, Target
        planner = LLMPlanner(LLMClient(args.model, env_file=args.env), [Target(s.id, s.x, s.y) for s in sc.samples],
                             sc.base, Navigator(cm).path_cost, async_mode=False)
    elif args.planner == 'science':
        from .science import ScientificPlanner
        advisor = None
        if args.llm:
            from .llm_client import LLMClient
            from .llm_scientist import LLMScientist
            advisor = LLMScientist(LLMClient(args.model, env_file=args.env), async_mode=False)   # sim time waits for the model
        planner = ScientificPlanner(grid, cm, sc.base, Navigator(cm).path_cost, learn_terrain=not args.no_terrain, advisor=advisor)
    else:
        planner = ScriptedPlanner([(s.x, s.y) for s in sc.samples])
    res = run_mission(grid, judge, planner, cm=cm)
    for rec in getattr(planner, 'journal', []):
        print(f'\nLLM [{rec["trigger"]}] {"FALLBACK " if rec.get("fallback") else ""}{rec.get("plan")}  forecast {rec.get("predicted_battery")}')
        for a in rec['attempts']:
            print(f'   attempt: {a.get("latency")}s, {a.get("tokens")} tokens' + (f', REJECTED: {a["error"]}' if a.get('error') else ''))
        print('   thought:', rec.get('thought'))
    print()
    for j in res.journal:
        print({k: j[k] for k in ('t', 'kind', 'target', 'success', 'message', 'battery_used', 'battery')})
    if hasattr(planner, 'lab'):
        print(planner.lab.to_markdown())
    print('events:', res.events)
    print('hidden env changes:', [e for e in judge.log if e.get('hidden')])
    print('score:', res.score, f'sim time {res.sim_time:.1f}s')


if __name__ == '__main__':
    main()
