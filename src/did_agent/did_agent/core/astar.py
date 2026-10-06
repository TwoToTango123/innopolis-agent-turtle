"""A* on a CostMap (8-connected, no corner cutting). No ROS dependencies.

Edge cost = step length (cells) * mean multiplier of the two cells, so the
path minimises sum(distance * terrain multiplier). All multipliers are >= 1,
which keeps the octile-distance heuristic admissible.
"""
import heapq
import math

from .costmap import CostMap

SQRT2 = math.sqrt(2.0)
_MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def _octile(ax: int, ay: int, bx: int, by: int) -> float:
    dx, dy = abs(ax - bx), abs(ay - by)
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)


def astar(cm: CostMap, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Return the list of cells from start to goal (inclusive), or None if unreachable."""
    if not (cm.is_free(*start) and cm.is_free(*goal)):
        return None
    w, h = cm.grid.width, cm.grid.height
    cost = cm.cost.tolist()  # python lists: much faster per-element access than numpy
    inf = math.inf
    gx, gy = goal
    g = {start: 0.0}
    parent = {start: None}
    open_heap = [(_octile(*start, gx, gy), 0.0, start)]
    closed = set()
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        closed.add(cur)
        cx, cy = cur
        c_cur = cost[cy][cx]
        for dx, dy, step in _MOVES:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            c_n = cost[ny][nx]
            if c_n == inf:
                continue
            if dx and dy and (cost[cy][nx] == inf or cost[ny][cx] == inf):
                continue  # don't squeeze diagonally past a lethal corner
            ng = gc + step * 0.5 * (c_cur + c_n)
            nb = (nx, ny)
            if ng < g.get(nb, inf):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_heap, (ng + _octile(nx, ny, gx, gy), ng, nb))
    return None


def plan_world(cm: CostMap, start_xy: tuple[float, float], goal_xy: tuple[float, float],
               snap_cells: int = 40) -> list[tuple[int, int]] | None:
    """Plan between world points; start/goal inside the safety margin are snapped
    to the nearest free cell (the robot may legitimately end up close to a wall)."""
    grid = cm.grid
    s = cm.nearest_free(*grid.world_to_cell(*start_xy), max_cells=snap_cells)
    g = cm.nearest_free(*grid.world_to_cell(*goal_xy), max_cells=snap_cells)
    if s is None or g is None:
        return None
    return astar(cm, s, g)
