import math
import os

import pytest

from did_agent.core.controller import BURGER_MAX_V, BURGER_MAX_W, WaypointFollower, front_clearance
from did_agent.core.costmap import CostMap
from did_agent.core.frames import Pose2D
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.mission import ScriptedPlanner, Subgoal, parse_targets
from did_agent.core.navigator import Navigator
from did_agent.core.offline_sim import run_mission
from did_agent.core.scenario import Scenario, generate

PKG = os.path.join(os.path.dirname(__file__), '..')
CONFIG = os.path.join(PKG, 'config', 'judge.yaml')


def unicycle(nav, pose, goal, dt=0.05, max_t=200.0):
    nav.start(pose, goal, 0.0)
    x, y, yaw = pose.x, pose.y, pose.yaw
    t, vmax, wmax, traj = 0.0, 0.0, 0.0, []
    while nav.status == Navigator.ACTIVE and t < max_t:
        t += dt
        v, w = nav.step(Pose2D(x, y, yaw), t)
        vmax, wmax = max(vmax, abs(v)), max(wmax, abs(w))
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        traj.append((x, y))
    return (x, y), traj, vmax, wmax


def test_follower_respects_burger_limits():
    f = WaypointFollower()
    f.set_path([(0, 0), (5, 5)])
    for _ in range(200):
        v, w, _ = f.step(Pose2D(0, 0, math.pi), 0.05)
        assert abs(v) <= BURGER_MAX_V + 1e-9 and abs(w) <= BURGER_MAX_W + 1e-9
    assert v == 0.0      # facing away: turn in place first


@pytest.mark.parametrize('goal', [(1.6, 1.6), (0.55, -0.55), (1.8, -1.6), (-1.6, 1.7), (2.0, 0.55)])
def test_navigates_on_real_map_without_collisions(world_map, goal):
    cm = CostMap(world_map, inflation_radius=0.2)
    nav = Navigator(cm)
    end, traj, vmax, wmax = unicycle(nav, Pose2D(-2.0, -0.5, 0.0), goal)
    assert nav.status == Navigator.ARRIVED, nav.reason
    assert math.dist(end, goal) < 0.08
    judge = Judge(Scenario('t', 'custom', None, (-2.0, -0.5), [], []), JudgeConfig.load(CONFIG), world_map)
    assert not any(judge._touching(x, y) for x, y in traj)
    assert vmax <= BURGER_MAX_V and wmax <= BURGER_MAX_W


def test_unreachable_goal_fails_cleanly(world_map):
    nav = Navigator(CostMap(world_map, inflation_radius=0.2))
    assert not nav.start(Pose2D(-2.0, -0.5, 0.0), (8.0, 8.0))
    assert nav.status == Navigator.FAILED and nav.reason


def test_stop_when_obstacle_in_front(world_map):
    nav = Navigator(CostMap(world_map, inflation_radius=0.2))
    nav.start(Pose2D(-2.0, -0.5, 0.0), (0.55, -0.55), 0.0)
    for k in range(40):
        v, w = nav.step(Pose2D(-2.0, -0.5, 0.0), 0.05 * (k + 1), front=0.1)
        assert v == 0.0


def test_front_clearance():
    ranges = [3.0] * 360
    ranges[5] = 0.4          # 5 deg left: inside the cone
    ranges[90] = 0.1         # 90 deg: outside
    ranges[355] = float('inf')
    assert front_clearance(ranges, 0.0, math.radians(1)) == pytest.approx(0.4)


def test_path_cost_counts_terrain(world_map):
    cm = CostMap(world_map, inflation_radius=0.2)
    nav = Navigator(cm)
    plain = nav.path_cost((-2.0, -0.5), (-2.0, 0.6))
    cm.set_terrain_rect(-3, -3, 3, 3, 2.0)       # whole arena x2
    assert nav.path_cost((-2.0, -0.5), (-2.0, 0.6)) == pytest.approx(2 * plain, rel=0.05)


def test_subgoal_validation():
    with pytest.raises(ValueError):
        Subgoal('fly')
    with pytest.raises(ValueError):
        Subgoal('goto')


def test_parse_targets():
    assert parse_targets('1,2; -0.5, 3.25') == [(1.0, 2.0), (-0.5, 3.25)]


# ---- end-to-end offline (level 1 mission, no ROS) ---------------------------

def _offline(world_map, sc, **planner_kw):
    judge = Judge(sc, JudgeConfig.load(CONFIG), world_map, seed=0)
    planner = ScriptedPlanner([(s.x, s.y) for s in sc.samples], **planner_kw)
    return run_mission(world_map, judge, planner), planner


def test_easy_mission_end_to_end(world_map):
    sc = Scenario.load(os.path.join(PKG, 'scenarios', 'easy.yaml'))
    res, _ = _offline(world_map, sc)
    s = res.score
    assert s['collected'] == 3 and s['returned'] and s['state'] == 'finished'
    assert s['penalties'] == {'collision': 0, 'false_collect': 0, 'hazard_hit': 0}
    assert [e['type'] for e in res.events] == ['sample_collected'] * 3
    assert [j['kind'] for j in res.journal] == ['goto', 'collect'] * 3 + ['return', 'finish']


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_medium_generated_missions(world_map, seed):
    res, _ = _offline(world_map, generate('medium', seed, world_map))
    assert res.score['collected'] == 5 and res.score['returned']
    assert res.score['penalties']['collision'] == 0


def test_low_battery_goes_home_early(world_map):
    sc = Scenario.load(os.path.join(PKG, 'scenarios', 'easy.yaml'))
    judge = Judge(sc, JudgeConfig.from_dict({'battery': {'initial': 9.0}}), world_map, seed=0)
    planner = ScriptedPlanner([(s.x, s.y) for s in sc.samples])
    res = run_mission(world_map, judge, planner)
    assert res.score['returned'], res.score
    assert res.score['collected'] < 3
    assert any('going home' in line for line in planner.log)
    assert res.score['battery'] > 0
