import math

import pytest

from did_agent.core.frames import Pose2D
from did_agent.core.frames import odom_to_world
from did_agent.core.localization import DeadReckoning, map_to_odom


def test_map_to_odom_composes_back_to_world_pose():
    world = Pose2D(0.4, 1.2, 2.0)
    odom = Pose2D(2.3, 1.9, 2.1)            # drifted odometry
    t = map_to_odom(world, odom)
    back = odom_to_world(odom, start=t)     # apply T to the odom pose
    assert (back.x, back.y, back.yaw) == pytest.approx((world.x, world.y, world.yaw))


def test_map_to_odom_without_drift_is_spawn_offset():
    t = map_to_odom(Pose2D(-1.0, -0.5, 0.0), Pose2D(1.0, 0.0, 0.0))
    assert (t.x, t.y, t.yaw) == pytest.approx((-2.0, -0.5, 0.0))


def _simulate(odom_yaw_drift_per_turn: float):
    """Drive a square-ish path; odometry heading drifts on every turn, IMU is exact.
    Returns (true pose, odom-only world pose, fused pose)."""
    start = Pose2D(-2.0, -0.5, 0.0)
    dr = DeadReckoning(start)
    tx, ty, tyaw = start.x, start.y, 0.0           # truth (world)
    ox, oy, oyaw = 0.0, 0.0, 0.0                   # wheel odometry (odom frame)
    dr.update_imu(0.3)                             # arbitrary IMU frame offset
    dr.update_odom(ox, oy, oyaw)
    for leg in range(6):
        for _ in range(100):                       # 1 m straight
            ds = 0.01
            tx += ds * math.cos(tyaw)
            ty += ds * math.sin(tyaw)
            ox += ds * math.cos(oyaw)
            oy += ds * math.sin(oyaw)
            dr.update_odom(ox, oy, oyaw)
        for _ in range(30):                        # turn 90 deg in place
            dyaw = math.pi / 2 / 30
            tyaw += dyaw
            oyaw += dyaw * (1 + odom_yaw_drift_per_turn)
            dr.update_imu(0.3 + tyaw)
            dr.update_odom(ox, oy, oyaw)
    odom_only = (start.x + ox, start.y + oy)
    return (tx, ty), odom_only, (dr.x, dr.y)


def test_fused_pose_ignores_odometry_heading_drift():
    truth, odom_only, fused = _simulate(0.04)
    assert math.dist(truth, odom_only) > 0.2       # odometry alone is off by > 20 cm
    assert math.dist(truth, fused) < 0.01


def test_no_drift_matches_odometry():
    truth, odom_only, fused = _simulate(0.0)
    assert math.dist(truth, odom_only) < 1e-6
    assert math.dist(truth, fused) == pytest.approx(0.0, abs=1e-6)


def test_backwards_motion_is_signed():
    dr = DeadReckoning(Pose2D(0.0, 0.0, 0.0))
    dr.update_imu(0.0)
    dr.update_odom(0.0, 0.0, 0.0)
    dr.update_odom(-0.5, 0.0, 0.0)                 # reversing
    assert (dr.x, dr.y) == pytest.approx((-0.5, 0.0))


def test_yaw_falls_back_to_odometry_without_imu():
    dr = DeadReckoning(Pose2D(-2.0, -0.5, 0.0))
    dr.update_odom(0.0, 0.0, 0.7)
    assert dr.yaw == pytest.approx(0.7)
    assert not dr.ready
