"""Operator mode (RViz goals / route) through the real executor + judge, offline."""
import math
import os

from did_agent.core.costmap import CostMap
from did_agent.core.executor import MissionExecutor
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.mission import AgentState, GoalQueuePlanner
from did_agent.core.navigator import Navigator
from did_agent.core.scenario import Scenario

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'judge.yaml')
BASE = (-2.0, -0.5)


class Sim:
    def __init__(self, world_map):
        self.judge = Judge(Scenario('t', 'custom', None, BASE, [], []), JudgeConfig.load(CONFIG), world_map)
        self.planner = GoalQueuePlanner(BASE)
        self.ex = MissionExecutor(self.planner, Navigator(CostMap(world_map, inflation_radius=0.2)), BASE)
        self.x, self.y, self.yaw, self.t = -2.0, -0.5, 0.0, 0.0
        self.judge.update(0.0, self.x, self.y, self.yaw)

    def run(self, seconds, dt=0.05):
        for _ in range(int(seconds / dt)):
            s = AgentState(self.t, self.x, self.y, self.yaw, self.judge.battery.level, 0.0, BASE)
            cmd = self.ex.step(s)
            if cmd.call == 'finish':
                ok, msg = self.judge.finish()
                self.ex.service_result(ok, msg, s)
            self.x += cmd.v * math.cos(self.yaw) * dt
            self.y += cmd.v * math.sin(self.yaw) * dt
            self.yaw += cmd.w * dt
            self.t += dt
            self.judge.update(self.t, self.x, self.y, self.yaw)


def test_idle_until_goal_then_drive(world_map):
    sim = Sim(world_map)
    sim.run(3)
    assert sim.ex.state == MissionExecutor.RUNNING and math.dist((sim.x, sim.y), BASE) < 0.01
    sim.planner.set_goal(0.55, -0.55)
    sim.run(40)
    assert math.dist((sim.x, sim.y), (0.55, -0.55)) < 0.08
    assert sim.ex.state == MissionExecutor.RUNNING        # still waiting for the next goal


def test_route_points_in_order(world_map):
    sim = Sim(world_map)
    route = [(-1.6, 0.55), (-0.55, 0.55), (-0.55, -0.55)]
    for p in route:
        sim.planner.add_point(*p)
    sim.run(90)
    gotos = [j for j in sim.ex.journal if j['kind'] == 'goto']
    assert [tuple(j['target']) for j in gotos] == route and all(j['success'] for j in gotos)
    assert sim.judge.score()['penalties']['collision'] == 0


def test_new_goal_preempts_current_trip(world_map):
    sim = Sim(world_map)
    sim.planner.set_goal(1.6, 1.6)
    sim.run(4)
    sim.planner.set_goal(-1.6, 0.55)
    sim.run(40)
    assert any(j['message'] == 'preempted by a new goal' for j in sim.ex.journal)
    assert math.dist((sim.x, sim.y), (-1.6, 0.55)) < 0.08


def test_goal_at_base_returns_and_finishes(world_map):
    sim = Sim(world_map)
    sim.planner.set_goal(-1.6, 0.55)
    sim.run(20)
    sim.planner.set_goal(-2.0, -0.5)
    sim.run(30)
    assert sim.ex.state == MissionExecutor.DONE
    assert sim.judge.score()['returned']
