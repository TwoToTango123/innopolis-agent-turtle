import math

import pytest

from did_agent.core.frames import Pose2D, odom_to_world, world_to_odom, wrap_angle, yaw_from_quaternion


def test_odom_origin_is_spawn_point():
    p = odom_to_world(Pose2D(0.0, 0.0, 0.0))
    assert (p.x, p.y, p.yaw) == pytest.approx((-2.0, -0.5, 0.0))


def test_task_formula():
    # TASK.md: x_world = -2.0 + odom.x, y_world = -0.5 + odom.y
    p = odom_to_world(Pose2D(1.3, -0.7, 0.4))
    assert (p.x, p.y) == pytest.approx((-0.7, -1.2))


def test_roundtrip():
    w = Pose2D(0.55, 1.6, 2.5)
    o = world_to_odom(w)
    back = odom_to_world(o)
    assert (back.x, back.y, back.yaw) == pytest.approx((w.x, w.y, w.yaw))


def test_wrap_angle():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)


def test_yaw_from_quaternion():
    a = 0.8
    assert yaw_from_quaternion(0.0, 0.0, math.sin(a / 2), math.cos(a / 2)) == pytest.approx(a)
