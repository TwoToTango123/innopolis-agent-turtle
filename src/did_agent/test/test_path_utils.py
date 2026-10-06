import math

from conftest import make_grid
from did_agent.core.astar import astar, plan_world
from did_agent.core.costmap import CostMap
from did_agent.core.path_utils import _segment_cost, path_length, waypoints_from_cells


def test_straight_path_collapses_to_two_points():
    g = make_grid(40, 40)
    cm = CostMap(g, soft_radius=0.0)
    cells = astar(cm, (2, 3), (30, 20))
    wps = waypoints_from_cells(cm, cells)
    assert len(wps) == 2


def test_exact_goal_point_is_kept():
    g = make_grid(40, 40)
    cm = CostMap(g, soft_radius=0.0)
    cells = astar(cm, (2, 2), g.world_to_cell(1.23, 1.07))
    wps = waypoints_from_cells(cm, cells, goal_xy=(1.23, 1.07))
    assert wps[-1] == (1.23, 1.07)


def test_waypoints_do_not_cut_expensive_terrain():
    g = make_grid(60, 60)
    cm = CostMap(g, soft_radius=0.0)
    cm.set_terrain_rect(1.25, 1.0, 1.75, 2.0, 5.0)
    cells = astar(cm, g.world_to_cell(0.5, 1.5), g.world_to_cell(2.5, 1.5))
    wps = waypoints_from_cells(cm, cells)
    assert len(wps) > 2
    terrain = cm.terrain
    for a, b in zip(wps, wps[1:]):
        n = 50
        for i in range(n + 1):
            x = a[0] + (b[0] - a[0]) * i / n
            y = a[1] + (b[1] - a[1]) * i / n
            ix, iy = g.world_to_cell(x, y)
            assert terrain[iy, ix] == 1.0


def test_world_map_waypoints_are_collision_free(world_map):
    cm = CostMap(world_map, inflation_radius=0.2)
    cost = cm.cost
    for goal in [(1.6, 1.6), (0.55, -0.55), (1.8, -1.6), (-1.6, 1.7)]:
        cells = plan_world(cm, (-2.0, -0.5), goal)
        assert cells is not None, goal
        wps = waypoints_from_cells(cm, cells, goal_xy=goal)
        assert len(wps) < len(cells) / 3
        for a, b in zip(wps, wps[1:]):
            assert math.isfinite(_segment_cost(cost, cm, a, b)), (goal, a, b)
        assert path_length(wps) <= len(cells) * world_map.resolution * 1.5
