import math

import numpy as np

from conftest import PILLARS, make_grid
from did_agent.core.costmap import CostMap


def test_inflation_radius_exact():
    g = make_grid(21, 21, occupied=[(10, 10)])
    cm = CostMap(g, inflation_radius=0.2, soft_radius=0.0)  # 0.2 m = 4 cells
    assert not cm.is_free(10, 10)
    assert not cm.is_free(14, 10)      # 4 cells away: inside the margin
    assert cm.is_free(15, 10)          # 5 cells away: outside
    assert not cm.is_free(12, 12)      # diagonal sqrt(8) < 4
    assert cm.is_free(13, 13)          # diagonal sqrt(18) > 4


def test_soft_margin_costs_more():
    g = make_grid(31, 31, occupied=[(15, 15)])
    cm = CostMap(g, inflation_radius=0.2, soft_radius=0.35, soft_cost=2.0)
    c = cm.cost
    assert math.isinf(c[15, 19])
    assert c[15, 21] == 2.0            # 6 cells = 0.30 m: soft zone
    assert c[15, 25] == 1.0            # 10 cells = 0.50 m: normal floor


def test_terrain_layer_is_updatable():
    g = make_grid(40, 40)
    cm = CostMap(g, soft_radius=0.0)
    cm.set_terrain_rect(0.5, 0.5, 1.0, 1.0, 3.0)
    ix, iy = g.world_to_cell(0.75, 0.75)
    assert cm.cost[iy, ix] == 3.0
    cm.set_terrain_circle(0.25, 0.25, 0.1, 0.5)   # clamped to >= 1
    ix, iy = g.world_to_cell(0.25, 0.25)
    assert cm.cost[iy, ix] == 1.0
    cm.reset_terrain()
    assert np.all(cm.terrain == 1.0)


def test_nearest_free():
    g = make_grid(21, 21, occupied=[(10, 10)])
    cm = CostMap(g, inflation_radius=0.2)
    fx, fy = cm.nearest_free(10, 10)
    assert cm.is_free(fx, fy)
    assert math.hypot(fx - 10, fy - 10) <= 5.0


def test_world_map_pillars_have_margin(world_map):
    cm = CostMap(world_map, inflation_radius=0.2)
    for x, y in PILLARS:
        assert not cm.is_free_world(x, y)
        # pillar radius ~0.17 m + 0.2 m margin: 0.3 m from centre is still forbidden
        assert not cm.is_free_world(x + 0.3, y)
    assert cm.is_free_world(-2.0, -0.5)
    assert cm.is_free_world(0.55, 0.55)
