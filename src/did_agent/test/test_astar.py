import math
import time

from conftest import make_grid
from did_agent.core.astar import astar, plan_world
from did_agent.core.costmap import CostMap


def _wall_with_gap(w=40, h=40, wall_x=20, gap=(30, 36)):
    occ = [(wall_x, y) for y in range(h) if not gap[0] <= y < gap[1]]
    return make_grid(w, h, occupied=occ)


def _assert_valid(cm, path):
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        assert max(abs(ax - bx), abs(ay - by)) == 1
    assert all(cm.is_free(x, y) for x, y in path)


def test_straight_line_on_empty_grid():
    cm = CostMap(make_grid(20, 20), soft_radius=0.0)
    path = astar(cm, (2, 2), (12, 2))
    assert path[0] == (2, 2) and path[-1] == (12, 2)
    assert len(path) == 11


def test_goes_through_gap():
    g = _wall_with_gap()
    cm = CostMap(g, inflation_radius=0.05, soft_radius=0.0)
    path = astar(cm, (5, 5), (35, 5))
    assert path is not None
    _assert_valid(cm, path)
    assert any(x == 20 for x, _ in path)
    assert all(30 <= y < 36 for x, y in path if x == 20)


def test_unreachable_returns_none():
    g = _wall_with_gap(gap=(0, 0))  # solid wall
    cm = CostMap(g, inflation_radius=0.05, soft_radius=0.0)
    assert astar(cm, (5, 5), (35, 5)) is None


def test_start_in_obstacle_returns_none():
    g = make_grid(20, 20, occupied=[(5, 5)])
    cm = CostMap(g, inflation_radius=0.1)
    assert astar(cm, (5, 5), (15, 15)) is None


def test_avoids_expensive_terrain_when_detour_is_cheap():
    g = make_grid(60, 60)
    cm = CostMap(g, soft_radius=0.0)
    # 0.5 m wide band across the direct line, cost x5
    cm.set_terrain_rect(1.25, 1.0, 1.75, 2.0, 5.0)
    path = astar(cm, g.world_to_cell(0.5, 1.5), g.world_to_cell(2.5, 1.5))
    terrain = [cm.terrain[y, x] for x, y in path]
    assert max(terrain) == 1.0


def test_crosses_expensive_terrain_when_detour_is_long():
    g = make_grid(60, 60)
    cm = CostMap(g, soft_radius=0.0)
    cm.set_terrain_rect(1.25, 0.0, 1.75, 3.0, 1.5)   # band covers the whole height
    path = astar(cm, g.world_to_cell(0.5, 1.5), g.world_to_cell(2.5, 1.5))
    assert path is not None


def test_world_map_plan_from_spawn(world_map):
    cm = CostMap(world_map, inflation_radius=0.2)
    t0 = time.perf_counter()
    path = plan_world(cm, (-2.0, -0.5), (1.6, 1.6))
    dt = time.perf_counter() - t0
    assert path is not None
    _assert_valid(cm, path)
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])) * world_map.resolution
    straight = math.hypot(1.6 + 2.0, 1.6 + 0.5)
    assert straight <= length < straight * 1.6
    assert dt < 3.0, f'A* too slow: {dt:.2f}s'


def test_world_map_goal_in_margin_is_snapped(world_map):
    cm = CostMap(world_map, inflation_radius=0.2)
    # 0.25 m from a pillar centre: inside the margin, the planner snaps to a free cell
    path = plan_world(cm, (-2.0, -0.5), (1.1 + 0.25, 0.0))
    assert path is not None
    gx, gy = world_map.cell_to_world(*path[-1])
    assert math.hypot(gx - 1.35, gy) < 0.3
