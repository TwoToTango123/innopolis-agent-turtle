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
    while t < max_time and ex.state != MissionExecutor.DONE:
        s = AgentState(t, x, y, yaw, judge.battery.level, judge.sensor_reading(), judge.sc.base,
                       collected=len(judge.collected), score=judge.score())
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
    from .scenario import Scenario
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument('scenario', help='easy | medium | hard | path.yaml')
    args = ap.parse_args(argv)
    path = args.scenario if args.scenario.endswith('.yaml') else os.path.join(here, 'scenarios', args.scenario + '.yaml')
    sc = Scenario.load(path)
    grid = GridMap.from_yaml(os.path.join(here, 'maps', 'map.yaml'))
    judge = Judge(sc, JudgeConfig.load(os.path.join(here, 'config', 'judge.yaml')), grid, seed=0)
    res = run_mission(grid, judge, ScriptedPlanner([(s.x, s.y) for s in sc.samples]))
    for j in res.journal:
        print(j)
    print('events:', res.events)
    print('score:', res.score, f'sim time {res.sim_time:.1f}s')


if __name__ == '__main__':
    main()
