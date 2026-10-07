"""Navigator: plan on the cost map (A*) and follow the waypoints. No ROS dependencies.

Executes one goal at a time; the mission layer decides which goal comes next.
"""
import math
from dataclasses import dataclass

from .astar import plan_world
from .controller import FollowerConfig, WaypointFollower
from .costmap import CostMap
from .frames import Pose2D
from .path_utils import path_length, waypoints_from_cells


@dataclass
class NavConfig:
    stop_distance: float = 0.16      # m: lidar range in front at which forward motion stops
    slow_distance: float = 0.35      # m: start slowing down
    stuck_timeout: float = 15.0      # s without progress -> failed
    progress_eps: float = 0.05       # m of remaining-distance decrease that counts as progress
    max_replans: int = 3


class Navigator:
    IDLE, ACTIVE, ARRIVED, FAILED = 'idle', 'active', 'arrived', 'failed'

    def __init__(self, costmap: CostMap, cfg: NavConfig | None = None, follower: FollowerConfig | None = None):
        self.cm = costmap
        self.cfg = cfg or NavConfig()
        self.follower = WaypointFollower(follower)
        self.status = self.IDLE
        self.goal: tuple[float, float] | None = None
        self.path: list[tuple[float, float]] = []
        self.reason = ''
        self._t = 0.0
        self._best_remaining = math.inf
        self._last_progress_t = 0.0
        self._replans = 0

    # ---- planning ---------------------------------------------------------
    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        cells = plan_world(self.cm, start, goal)
        if cells is None:
            return None
        wps = waypoints_from_cells(self.cm, cells, goal_xy=goal)
        wps[0] = start
        return wps

    def path_cost(self, start, goal) -> float | None:
        """Planned length in metres weighted by the cost map's terrain layer
        (what the battery will roughly pay). None if unreachable."""
        wps = self.plan(start, goal)
        if wps is None:
            return None
        cost = 0.0
        for a, b in zip(wps, wps[1:]):
            n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / self.cm.grid.resolution))
            for i in range(n):
                x = a[0] + (b[0] - a[0]) * (i + 0.5) / n
                y = a[1] + (b[1] - a[1]) * (i + 0.5) / n
                ix, iy = self.cm.grid.world_to_cell(x, y)
                cost += math.hypot(b[0] - a[0], b[1] - a[1]) / n * self.cm.terrain[iy, ix]
        return cost

    # ---- execution --------------------------------------------------------
    def start(self, pose: Pose2D, goal: tuple[float, float], t: float = 0.0) -> bool:
        self.goal = goal
        self._replans = 0
        return self._start_path(pose, t)

    def _start_path(self, pose: Pose2D, t: float) -> bool:
        wps = self.plan((pose.x, pose.y), self.goal)
        if wps is None:
            self.status, self.reason, self.path = self.FAILED, 'no path to goal', []
            self.follower.stop()
            return False
        self.path = wps
        self.follower.set_path(wps)
        self.status, self.reason = self.ACTIVE, ''
        self._best_remaining = math.inf
        self._last_progress_t = t
        return True

    def cancel(self) -> None:
        self.follower.stop()
        self.status, self.reason = self.IDLE, 'cancelled'

    def step(self, pose: Pose2D, t: float, front: float = math.inf) -> tuple[float, float]:
        """Returns (v, w). Check `status` afterwards."""
        dt = t - self._t if self._t else 0.05
        self._t = t
        if self.status != self.ACTIVE:
            return 0.0, 0.0
        c = self.cfg
        if front <= c.stop_distance:
            max_fwd = 0.0
        elif front < c.slow_distance:
            max_fwd = 0.22 * (front - c.stop_distance) / (c.slow_distance - c.stop_distance)
        else:
            max_fwd = None
        v, w, done = self.follower.step(pose, dt, max_fwd)
        if done:
            self.status = self.ARRIVED
            return 0.0, 0.0
        remaining = self.follower.remaining(pose)
        if remaining < self._best_remaining - c.progress_eps:
            self._best_remaining = remaining
            self._last_progress_t = t
        elif t - self._last_progress_t > c.stuck_timeout:
            if self._replans < c.max_replans:
                self._replans += 1
                self._start_path(pose, t)
            else:
                self.follower.stop()
                self.status, self.reason = self.FAILED, f'no progress for {c.stuck_timeout:.0f}s'
                return 0.0, 0.0
        return v, w

    @property
    def planned_length(self) -> float:
        return path_length(self.path)
