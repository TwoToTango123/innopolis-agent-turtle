import numpy as np
import pytest

from conftest import PILLARS
from did_agent.core.grid_map import FREE, OCCUPIED, UNKNOWN, read_pgm


def test_map_metadata(world_map):
    assert (world_map.width, world_map.height) == (384, 384)
    assert world_map.resolution == pytest.approx(0.05)
    assert (world_map.origin_x, world_map.origin_y) == pytest.approx((-10.0, -10.0))


def test_map_matches_world_frame(world_map):
    # pillars are not free, spawn point and gaps between pillars are free
    for x, y in PILLARS:
        assert world_map.value_at(x, y) != FREE, (x, y)
    assert world_map.value_at(-2.0, -0.5) == FREE
    assert world_map.value_at(0.55, 0.55) == FREE
    assert world_map.value_at(-0.55, -0.55) == FREE


def test_cell_world_roundtrip(world_map):
    for x, y in [(-2.0, -0.5), (1.234, -0.987), (0.0, 0.0)]:
        ix, iy = world_map.world_to_cell(x, y)
        cx, cy = world_map.cell_to_world(ix, iy)
        assert abs(cx - x) <= world_map.resolution / 2 + 1e-9
        assert abs(cy - y) <= world_map.resolution / 2 + 1e-9


def test_y_axis_points_up(world_map):
    ix, iy0 = world_map.world_to_cell(0.0, -1.0)
    _, iy1 = world_map.world_to_cell(0.0, 1.0)
    assert iy1 > iy0


def test_read_ascii_pgm(tmp_path):
    p = tmp_path / 'tiny.pgm'
    p.write_text('P2\n# comment\n3 2\n255\n0 128 255\n255 255 0\n')
    img = read_pgm(str(p))
    assert img.shape == (2, 3)
    assert img.tolist() == [[0, 128, 255], [255, 255, 0]]


def test_out_of_bounds_is_unknown(world_map):
    assert world_map.value_at(100.0, 100.0) == UNKNOWN
