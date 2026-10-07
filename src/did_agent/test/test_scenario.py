import math

import pytest

from did_agent.core.costmap import CostMap
from did_agent.core.scenario import BASE, DIFFICULTY, Scenario, Zone, generate


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_counts_match_task_table(world_map, difficulty):
    sc = generate(difficulty, 42, world_map)
    spec = DIFFICULTY[difficulty]
    assert len(sc.samples) == spec['samples']
    assert len(sc.terrain) == spec['terrain']
    assert bool(sc.events) == spec['events']


def test_task_table_values():
    # TASK.md: easy 3/1/no events, medium 5/3/no events, hard 7/4/events
    assert [(DIFFICULTY[d]['samples'], DIFFICULTY[d]['terrain'], DIFFICULTY[d]['events'])
            for d in ('easy', 'medium', 'hard')] == [(3, 1, False), (5, 3, False), (7, 4, True)]


def test_same_seed_same_scenario(world_map):
    assert generate('hard', 7, world_map).to_dict() == generate('hard', 7, world_map).to_dict()
    assert generate('hard', 7, world_map).to_dict() != generate('hard', 8, world_map).to_dict()


@pytest.mark.parametrize('seed', range(10))
def test_samples_are_reachable_and_collectable(world_map, seed):
    sc = generate('hard', seed, world_map)
    cm = CostMap(world_map, inflation_radius=0.2)   # what the agent plans with
    for s in sc.samples:
        assert cm.is_free_world(s.x, s.y), s
        assert math.dist((s.x, s.y), BASE) >= 0.8
    for a in sc.samples:
        for b in sc.samples:
            if a is not b:
                assert math.dist((a.x, a.y), (b.x, b.y)) >= 0.6


@pytest.mark.parametrize('seed', range(10))
def test_hazards_do_not_cover_samples_or_base(world_map, seed):
    sc = generate('hard', seed, world_map)
    hz = list(sc.hazards) + [Zone(**e.params['zone']) for e in sc.events if e.type == 'new_hazard']
    assert hz
    for h in hz:
        assert not h.contains(*BASE)
        assert not any(h.contains(s.x, s.y) for s in sc.samples)


def test_hard_has_the_three_event_kinds(world_map):
    sc = generate('hard', 1, world_map)
    assert [e.type for e in sc.events] == ['terrain_change', 'new_hazard', 'sensor_fault']
    assert sc.events == sorted(sc.events, key=lambda e: e.t)


def test_yaml_roundtrip(world_map, tmp_path):
    sc = generate('hard', 3, world_map)
    p = tmp_path / 'sc.yaml'
    sc.save(str(p))
    assert Scenario.load(str(p)).to_dict() == sc.to_dict()


def test_zone_shapes():
    c = Zone('A', 'circle', 2.0, cx=0, cy=0, r=1)
    r = Zone('B', 'rect', 2.0, x0=1, y0=1, x1=0, y1=0)
    assert c.contains(0.5, 0.5) and not c.contains(0.8, 0.8)
    assert r.contains(0.5, 0.5) and not r.contains(1.5, 0.5)
