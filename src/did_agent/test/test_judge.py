import json
import os

import pytest

from did_agent.core.judge import PUBLIC_EVENTS, Judge, JudgeConfig
from did_agent.core.scenario import Event, Sample, Scenario, Zone

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'judge.yaml')


def _scenario(**kw):
    d = dict(name='t', difficulty='custom', seed=None, base=(-2.0, -0.5),
             samples=[Sample('s1', -1.0, -0.5), Sample('s2', 0.55, 0.55)],
             terrain=[Zone('A', 'circle', 3.0, cx=-0.5, cy=-0.55, r=0.3)])
    d.update(kw)
    return Scenario(**d)


@pytest.fixture
def judge(world_map):
    return Judge(_scenario(), JudgeConfig.load(CONFIG), world_map, seed=1)


def drive(j, pts, t0=0.0, dt=0.1):
    """Feed a straight-line trajectory sampled every 2 cm."""
    events, t = [], t0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / 0.02))
        for i in range(1, n + 1):
            t += dt
            events += j.update(t, x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, 0.0)
    return events, t


def test_config_file_loads():
    cfg = JudgeConfig.load(CONFIG)
    assert cfg.battery.initial == 60.0 and cfg.collect_radius == 0.30


def test_config_rejects_typos():
    with pytest.raises(ValueError):
        JudgeConfig.from_dict({'colect_radius': 0.3})


def test_battery_drains_when_driving(judge):
    judge.update(0.0, -2.0, -0.5, 0.0)
    drive(judge, [(-2.0, -0.5), (-1.5, -0.5)])
    b = judge.score()['battery']
    assert 59.2 < b < 59.6        # 0.5 m * 1.0 + idle


def test_expensive_terrain_costs_more(world_map):
    cfg = JudgeConfig.load(CONFIG)
    a, b = Judge(_scenario(), cfg, world_map), Judge(_scenario(terrain=[]), cfg, world_map)
    for j in (a, b):
        j.update(0.0, -1.0, -0.55, 0.0)
        drive(j, [(-1.0, -0.55), (0.0, -0.55)])
    used_a, used_b = 60 - a.battery.level, 60 - b.battery.level
    # 0.6 m of the 1.0 m path is inside zone A (x3): 0.4 + 0.6*3 = 2.2 vs 1.0
    assert used_a == pytest.approx(2.2, abs=0.1)
    assert used_b == pytest.approx(1.0, abs=0.1)


def test_collect_success_and_false_collect(judge):
    judge.update(0.0, -1.2, -0.5, 0.0)          # 0.2 m from s1
    ok, msg, ev = judge.collect()
    assert ok and ev[0]['type'] == 'sample_collected' and ev[0]['sample'] == 's1'
    ok, msg, ev = judge.collect()               # s1 already taken, s2 far away
    assert not ok and ev[0]['type'] == 'false_collect' and ev[0]['penalty'] < 0
    s = judge.score()
    assert s['collected'] == 1 and s['penalties']['false_collect'] == 1
    assert s['score'] == 10 - 2


def test_collect_radius_is_strict(judge):
    judge.update(0.0, -1.0 - 0.31, -0.5, 0.0)
    assert not judge.collect()[0]


def test_sensor_rises_near_sample(judge):
    judge.update(0.0, -2.0, -0.5, 0.0)
    far = sum(judge.sensor_reading() for _ in range(50)) / 50
    judge.update(1.0, -1.05, -0.5, 0.0)
    near = sum(judge.sensor_reading() for _ in range(50)) / 50
    assert near > far + 0.3


def test_collision_once_per_contact(judge):
    judge.update(0.0, -1.5, 0.0, 0.0)
    ev, _ = drive(judge, [(-1.5, 0.0), (-1.15, 0.0), (-1.5, 0.0), (-1.15, 0.0)])  # into pillar (-1.1, 0) twice
    assert [e['type'] for e in ev] == ['collision', 'collision']
    assert judge.score()['penalties']['collision'] == 2


def test_no_collision_in_free_space(judge):
    judge.update(0.0, -2.0, -0.5, 0.0)
    # corridors between pillar rows/columns (straight (-.55,-.55)->(.55,.55) would hit the (0, 0) pillar)
    ev, _ = drive(judge, [(-2.0, -0.5), (-0.55, -0.55), (0.55, -0.55), (0.55, 0.55)])
    assert not [e for e in ev if e['type'] == 'collision']


def test_hazard_and_hidden_events(world_map):
    sc = _scenario(events=[
        Event(5.0, 'new_hazard', {'zone': Zone('H1', 'circle', 1.0, cx=-1.5, cy=-0.5, r=0.2).to_dict()}),
        Event(6.0, 'terrain_change', {'zone': 'A', 'multiplier': 1.0}),
        Event(7.0, 'sensor_fault', {'noise_std': 0.3, 'duration': 10.0}),
    ])
    j = Judge(sc, JudgeConfig.load(CONFIG), world_map, seed=2)
    ev, t = drive(j, [(-2.0, -0.5), (-1.95, -0.5)])          # before the events
    assert ev == []
    ev, t = drive(j, [(-1.95, -0.5), (-1.5, -0.5)], t0=10.0)  # enters the new hazard
    assert [e['type'] for e in ev] == ['hazard_hit']
    assert all(e['type'] in PUBLIC_EVENTS for e in ev)
    assert j.terrain[0].multiplier == 1.0
    assert {e['type'] for e in j.log if e.get('hidden')} == {'env_new_hazard', 'env_terrain_change', 'env_sensor_fault'}


def test_finish_at_base_gives_bonus(judge):
    judge.update(0.0, -2.0, -0.5, 0.0)
    ok, msg = judge.finish()
    assert ok and judge.score()['returned']
    assert judge.score()['score'] == pytest.approx(0.1 * judge.battery.level, abs=0.01)
    assert not judge.collect()[0]               # run is over


def test_finish_away_from_base(judge):
    judge.update(0.0, 0.55, -0.55, 0.0)
    ok, msg = judge.finish()
    assert not ok and judge.score()['state'] == 'finished' and judge.score()['score'] == 0


def test_depleted_battery_stops_the_run(world_map):
    cfg = JudgeConfig.from_dict({'battery': {'initial': 0.3}})
    j = Judge(_scenario(), cfg, world_map)
    j.update(0.0, -2.0, -0.5, 0.0)
    drive(j, [(-2.0, -0.5), (-1.5, -0.5)])
    assert j.score()['state'] == 'depleted' and j.score()['battery'] == 0.0
    assert not j.collect()[0]


def test_score_is_json_and_hides_positions(judge):
    judge.update(0.0, -2.0, -0.5, 0.0)
    s = json.dumps(judge.score())
    for secret in ('-1.0', '0.55', 's1', 's2', 'cx'):
        assert secret not in s
