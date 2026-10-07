"""Waypoint follower for a differential-drive robot. No ROS dependencies.

Rotate in place when the heading error is large, otherwise drive forward with a
proportional heading correction; slow down near the final waypoint. Output is
clipped to the TurtleBot3 Burger limits and rate-limited (acceleration).
"""
import math
from dataclasses import dataclass

from .frames import Pose2D, wrap_angle

BURGER_MAX_V = 0.22    # m/s
BURGER_MAX_W = 2.84    # rad/s


@dataclass
class FollowerConfig:
    cruise_v: float = 0.18
    max_v: float = BURGER_MAX_V
    max_w: float = 1.8               # below the 2.84 hardware limit: less odometry slip
    k_heading: float = 2.5
    turn_in_place: float = 0.6       # rad: rotate first if heading error is larger
    waypoint_tol: float = 0.10       # m: switch to the next intermediate waypoint
    goal_tol: float = 0.05           # m: final waypoint reached
    slow_radius: float = 0.35        # m: start slowing before the final waypoint
    min_v: float = 0.03
    max_acc_v: float = 0.5           # m/s^2
    max_acc_w: float = 4.0           # rad/s^2


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _slew(prev: float, target: float, max_step: float) -> float:
    return prev + _clip(target - prev, max_step)


class WaypointFollower:
    def __init__(self, cfg: FollowerConfig | None = None):
        self.cfg = cfg or FollowerConfig()
        self.waypoints: list[tuple[float, float]] = []
        self.index = 0
        self.v = 0.0
        self.w = 0.0

    def set_path(self, waypoints: list[tuple[float, float]]) -> None:
        self.waypoints = list(waypoints)
        self.index = 1 if len(self.waypoints) > 1 else 0   # [0] is the start point

    @property
    def active(self) -> bool:
        return self.index < len(self.waypoints)

    def stop(self) -> None:
        self.waypoints, self.index, self.v, self.w = [], 0, 0.0, 0.0

    def remaining(self, pose: Pose2D) -> float:
        if not self.active:
            return 0.0
        pts = [(pose.x, pose.y)] + self.waypoints[self.index:]
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    def step(self, pose: Pose2D, dt: float, max_forward: float | None = None) -> tuple[float, float, bool]:
        """Returns (v, w, done). `max_forward` caps forward speed (e.g. obstacle ahead)."""
        c = self.cfg
        if not self.active:
            self.v = self.w = 0.0
            return 0.0, 0.0, True
        # skip intermediate waypoints that are already reached
        while True:
            tx, ty = self.waypoints[self.index]
            dist = math.hypot(tx - pose.x, ty - pose.y)
            last = self.index == len(self.waypoints) - 1
            if dist < (c.goal_tol if last else c.waypoint_tol):
                if last:
                    self.stop()
                    return 0.0, 0.0, True
                self.index += 1
                continue
            break
        err = wrap_angle(math.atan2(ty - pose.y, tx - pose.x) - pose.yaw)
        w = _clip(c.k_heading * err, c.max_w)
        if abs(err) > c.turn_in_place:
            v = 0.0
        else:
            v = c.cruise_v * math.cos(err)
            if last:
                v = min(v, max(c.min_v, c.cruise_v * dist / c.slow_radius))
        v = min(v, c.max_v)
        if max_forward is not None:
            v = min(v, max(0.0, max_forward))
        dt = max(dt, 1e-3)
        self.v = _slew(self.v, v, c.max_acc_v * dt)
        self.w = _slew(self.w, w, c.max_acc_w * dt)
        return self.v, self.w, False


def front_clearance(ranges, angle_min: float, angle_increment: float, half_angle: float = 0.5,
                    range_min: float = 0.0) -> float:
    """Minimum valid lidar range within +-half_angle of the robot's heading."""
    best = math.inf
    for i, r in enumerate(ranges):
        a = wrap_angle(angle_min + i * angle_increment)
        if abs(a) <= half_angle and math.isfinite(r) and r > range_min:
            best = min(best, r)
    return best
