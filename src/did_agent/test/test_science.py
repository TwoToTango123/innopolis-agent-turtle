"""Levels 3-4: the scientific agent finds samples by the sensor alone, learns the terrain
from the battery, and adapts to unannounced changes. Offline (real judge + map, no ROS)."""
import math
import os

import numpy as np

from did_agent.core.costmap import CostMap
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.mission import AgentState
from did_agent.core.navigator import Navigator
from did_agent.core.offline_sim import run_mission
from did_agent.core.scenario import Event, Sample, Scenario, Zone
from did_agent.core.science import Obs, SampleSearch, ScientificPlanner, SensorTracker, TerrainLearner

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'judge.yaml')
BASE = (-2.0, -0.5)


def run(world_map, sc, battery=60.0, learn=True, max_time=600.0):
    cfg = JudgeConfig.load(CONFIG)
    cfg.battery.initial = battery
    judge = Judge(sc, cfg, world_map, seed=3)
    cm = CostMap(world_map, inflation_radius=0.2)
    planner = ScientificPlanner(world_map, cm, sc.base, Navigator(cm).path_cost, learn_terrain=learn)
    res = run_mission(world_map, judge, planner, cm=cm, max_time=max_time)
    return res, planner, judge


def test_sensor_is_a_range_finder(world_map):
    s = SampleSearch(world_map, BASE)
    sample = (0.55, -0.55)
    for x, y in [(-1.6, -0.5), (-0.3, -0.55), (0.55, 0.35), (1.3, -0.6)]:
        d = math.dist((x, y), sample)
        s.add(Obs(x, y, math.exp(-d / 0.7), 5, 0.013, 0.0))
    (ex, ey), info = s.estimate()
    assert math.dist((ex, ey), sample) < 0.15, info
    # a weak reading rules out the area around it, but never the true sample position
    assert not s.is_excluded(*sample)
    assert s.is_excluded(-1.6, -0.5)


def test_noise_estimate_detects_a_sensor_fault():
    rng = np.random.default_rng(0)
    tr = SensorTracker()
    for i in range(200):
        tr.add(0.01 * i, 0.0, max(0.0, 0.1 + rng.normal(0, 0.03)), i * 0.1)
    calm = tr.noise
    for i in range(200):
        tr.add(0.01 * i, 0.0, min(1.0, max(0.0, 0.1 + rng.normal(0, 0.2))), 20 + i * 0.1)
    assert calm < 0.05 and tr.noise > 3 * calm


def test_terrain_learner_finds_an_expensive_strip(world_map):
    cm = CostMap(world_map, inflation_radius=0.2)
    tl = TerrainLearner(cm)
    battery, x = 60.0, -1.9
    while x < -0.3:                                # drive along y = -0.5; x in [-1.2, -0.9] costs x3
        nx = x + 0.01
        battery -= 0.01 * (3.0 if -1.2 <= (x + nx) / 2 <= -0.9 else 1.0)
        x = nx
        tl.add(AgentState(0.0, x, -0.55, 0.0, battery, 0.0, BASE))
    assert abs(tl.k - 1.0) < 0.05
    comps = tl.components()
    assert len(comps) == 1
    mu, n, (cx, cy) = tl.zone_stats(comps[0])
    assert mu > 1.7 and -1.3 < cx < -0.8


def test_finds_all_samples_without_knowing_where_they_are(world_map):
    sc = Scenario('t', 'custom', None, BASE, [Sample('s1', 0.55, -0.55), Sample('s2', -0.55, 1.6), Sample('s3', 1.6, 0.55)], [])
    res, planner, judge = run(world_map, sc)
    assert res.score['collected'] == 3 and res.score['returned']
    assert res.score['penalties'] == {'collision': 0, 'false_collect': 0, 'hazard_hit': 0}
    samples = [h for h in planner.lab.hypotheses.values() if h.kind == 'sample']
    assert sum(h.status == 'подтверждена' for h in samples) == 3
    assert '# Журнал эксперимента' in planner.lab.to_markdown()


def test_learns_a_zone_and_sees_it_change(world_map):
    """A sample sits inside zone A, which gets more expensive mid-mission (hard-style event)."""
    zone = Zone('A', 'circle', 2.5, cx=-0.55, cy=-0.55, r=0.5)
    sc = Scenario('t', 'custom', None, BASE, [Sample('s1', -0.55, -0.55), Sample('s2', 0.55, 0.55)], [zone],
                  events=[Event(30.0, 'terrain_change', {'zone': 'A', 'multiplier': 4.0})])
    res, planner, _ = run(world_map, sc)
    assert res.score['returned']
    zones = [h for h in planner.lab.hypotheses.values() if h.kind == 'terrain']
    assert zones, planner.lab.to_markdown()
    near = [h for h in zones if math.dist((h.data['x'], h.data['y']), (zone.cx, zone.cy)) < 0.6]
    assert near and near[0].status in ('подтверждена', 'изменилась')
    # the cost map now prices the zone: A* would rather go around it
    ix, iy = planner.cm.grid.world_to_cell(zone.cx, zone.cy)
    assert planner.cm.terrain[iy, ix] > 1.8


def test_hazard_is_learned_from_the_penalty_and_avoided(world_map):
    hz = Zone('H1', 'circle', 1.0, cx=-0.55, cy=-0.55, r=0.3)
    sc = Scenario('t', 'custom', None, BASE, [Sample('s1', 0.55, -0.55), Sample('s2', -0.55, 1.6)], [], hazards=[hz])
    res, planner, _ = run(world_map, sc)
    assert res.score['returned'] and res.score['collected'] == 2
    assert res.score['penalties']['hazard_hit'] <= 1           # hit once, then avoided
    if res.score['penalties']['hazard_hit']:
        assert any(h.kind == 'hazard' for h in planner.lab.hypotheses.values())


def test_low_battery_returns_home(world_map):
    sc = Scenario('t', 'custom', None, BASE, [Sample('s1', 1.6, 1.6), Sample('s2', 1.6, -1.6)], [])
    res, planner, _ = run(world_map, sc, battery=8.0)
    assert res.score['returned'] and res.score['battery'] > 0
