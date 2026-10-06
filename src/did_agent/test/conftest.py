import os

import numpy as np
import pytest

from did_agent.core.grid_map import FREE, OCCUPIED, GridMap

MAP_YAML = os.path.join(os.path.dirname(__file__), '..', 'maps', 'map.yaml')

# turtlebot3_world pillar centres (verified on map.pgm, see DEVLOG)
PILLARS = [(x, y) for x in (-1.1, 0.0, 1.1) for y in (-1.1, 0.0, 1.1)]


def make_grid(w: int, h: int, occupied=(), res: float = 0.05) -> GridMap:
    """Free w x h grid with origin (0, 0); `occupied` is an iterable of (ix, iy)."""
    occ = np.full((h, w), FREE, dtype=np.int8)
    for ix, iy in occupied:
        occ[iy, ix] = OCCUPIED
    return GridMap(occ, res, 0.0, 0.0)


@pytest.fixture(scope='session')
def world_map() -> GridMap:
    return GridMap.from_yaml(MAP_YAML)
