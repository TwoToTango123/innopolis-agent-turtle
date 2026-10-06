"""Grid path -> sparse world waypoints. No ROS dependencies.

A* returns a staircase of cells; the robot should drive straight lines instead.
A segment between two path points is used as a shortcut only if it stays out of
lethal cells AND is not more expensive than the original sub-path (so shortcuts
never cut through costly terrain).
"""
import math

import numpy as np

from .costmap import CostMap


def _segment_cost(cost: np.ndarray, cm: CostMap, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Integral of cost along the straight segment a->b, in metres*multiplier (inf if blocked)."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(math.ceil(length / (cm.grid.resolution * 0.5))))
    total = 0.0
    for i in range(n + 1):
        t = i / n
        ix, iy = cm.grid.world_to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if not cm.grid.in_bounds(ix, iy):
            return math.inf
        c = cost[iy, ix]
        if not math.isfinite(c):
            return math.inf
        total += c
    return total / (n + 1) * length


def cells_to_world(cm: CostMap, cells: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [cm.grid.cell_to_world(ix, iy) for ix, iy in cells]


def simplify(cm: CostMap, points: list[tuple[float, float]], tolerance: float = 1.02) -> list[tuple[float, float]]:
    """Greedy line-of-sight simplification. Keeps the first and last point."""
    if len(points) <= 2:
        return list(points)
    cost = cm.cost
    # prefix cost along the original polyline
    prefix = [0.0]
    for p, q in zip(points, points[1:]):
        prefix.append(prefix[-1] + _segment_cost(cost, cm, p, q))
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1:
            straight = _segment_cost(cost, cm, points[i], points[j])
            if straight <= (prefix[j] - prefix[i]) * tolerance:
                break
            j -= 1
        out.append(points[j])
        i = j
    return out


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def waypoints_from_cells(cm: CostMap, cells: list[tuple[int, int]],
                         goal_xy: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """Cells -> simplified waypoints. If the exact goal point is free, end there
    instead of at the goal cell's center."""
    pts = cells_to_world(cm, cells)
    if goal_xy is not None and cm.is_free_world(*goal_xy):
        pts[-1] = goal_xy
    return simplify(cm, pts)
