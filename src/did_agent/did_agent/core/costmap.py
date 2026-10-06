"""Planning cost map on top of a GridMap. No ROS dependencies.

Two layers:
  * lethal  - obstacles (and unknown) inflated by `inflation_radius`; never entered.
              The radius is measured from the robot CENTER to the obstacle edge.
  * cost    - per-cell traversal multiplier >= 1.0 (1.0 = normal floor). It is
              meant to be updated at runtime (expensive terrain learned from the
              battery drain, hazard zones, ...). A soft "wall proximity" term is
              baked into a separate base layer so terrain updates don't erase it.
"""
from collections import deque

import numpy as np

from .grid_map import FREE, GridMap


def _disk_offsets(radius_cells: float) -> list[tuple[int, int]]:
    r = int(np.ceil(radius_cells))
    return [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
            if dx * dx + dy * dy <= radius_cells * radius_cells]


def _dilate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.nonzero(mask)
    for dx, dy in _disk_offsets(radius_cells):
        ty, tx = ys + dy, xs + dx
        ok = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
        out[ty[ok], tx[ok]] = True
    return out


class CostMap:
    def __init__(self, grid: GridMap, inflation_radius: float = 0.2,
                 soft_radius: float = 0.35, soft_cost: float = 2.0,
                 unknown_is_lethal: bool = True):
        self.grid = grid
        self.inflation_radius = inflation_radius
        obstacle = grid.occ != FREE if unknown_is_lethal else grid.occ > 0
        res = grid.resolution
        self.lethal = _dilate(obstacle, inflation_radius / res)
        # Soft margin: cells between inflation_radius and soft_radius cost extra,
        # so paths keep to the middle of corridors when there is room.
        self.base = np.ones(grid.occ.shape, dtype=np.float64)
        if soft_radius > inflation_radius and soft_cost > 1.0:
            near = _dilate(obstacle, soft_radius / res) & ~self.lethal
            self.base[near] = soft_cost
        self.terrain = np.ones(grid.occ.shape, dtype=np.float64)

    # ---- queries -------------------------------------------------------
    @property
    def cost(self) -> np.ndarray:
        """Effective multiplier per cell (inf for lethal cells)."""
        c = self.base * self.terrain
        c[self.lethal] = np.inf
        return c

    def is_free(self, ix: int, iy: int) -> bool:
        return self.grid.in_bounds(ix, iy) and not self.lethal[iy, ix]

    def is_free_world(self, x: float, y: float) -> bool:
        return self.is_free(*self.grid.world_to_cell(x, y))

    def nearest_free(self, ix: int, iy: int, max_cells: int = 40) -> tuple[int, int] | None:
        """BFS to the closest non-lethal cell (used when start/goal sit in the margin)."""
        if self.is_free(ix, iy):
            return ix, iy
        seen = {(ix, iy)}
        q = deque([(ix, iy, 0)])
        while q:
            cx, cy, d = q.popleft()
            if d >= max_cells:
                continue
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (nx, ny) in seen or not self.grid.in_bounds(nx, ny):
                    continue
                if not self.lethal[ny, nx]:
                    return nx, ny
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
        return None

    # ---- terrain updates (levels 3-4 will call these) -------------------
    def _region_mask(self, kind: str, params) -> np.ndarray:
        h, w = self.grid.occ.shape
        xs = self.grid.origin_x + (np.arange(w) + 0.5) * self.grid.resolution
        ys = self.grid.origin_y + (np.arange(h) + 0.5) * self.grid.resolution
        X, Y = np.meshgrid(xs, ys)
        if kind == 'circle':
            cx, cy, r = params
            return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        if kind == 'rect':
            x0, y0, x1, y1 = params
            return (X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))
        raise ValueError(f'unknown region kind {kind!r}')

    def set_terrain_circle(self, cx: float, cy: float, r: float, multiplier: float) -> None:
        self.terrain[self._region_mask('circle', (cx, cy, r))] = max(1.0, multiplier)

    def set_terrain_rect(self, x0: float, y0: float, x1: float, y1: float, multiplier: float) -> None:
        self.terrain[self._region_mask('rect', (x0, y0, x1, y1))] = max(1.0, multiplier)

    def reset_terrain(self) -> None:
        self.terrain[:] = 1.0
