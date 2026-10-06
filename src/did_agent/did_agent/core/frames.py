"""Odom <-> world frame conversion. No ROS dependencies.

TASK.md: the robot spawns at (-2.0, -0.5) with zero yaw and the world is launched
with x_pose = y_pose = 0, so x_world = -2.0 + odom.x, y_world = -0.5 + odom.y.
"""
import math
from dataclasses import dataclass

START_X = -2.0
START_Y = -0.5
START_YAW = 0.0


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def odom_to_world(p: Pose2D, start: Pose2D = Pose2D(START_X, START_Y, START_YAW)) -> Pose2D:
    c, s = math.cos(start.yaw), math.sin(start.yaw)
    return Pose2D(start.x + c * p.x - s * p.y,
                  start.y + s * p.x + c * p.y,
                  wrap_angle(start.yaw + p.yaw))


def world_to_odom(p: Pose2D, start: Pose2D = Pose2D(START_X, START_Y, START_YAW)) -> Pose2D:
    c, s = math.cos(start.yaw), math.sin(start.yaw)
    dx, dy = p.x - start.x, p.y - start.y
    return Pose2D(c * dx + s * dy, -s * dx + c * dy, wrap_angle(p.yaw - start.yaw))
