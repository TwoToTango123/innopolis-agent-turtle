"""Dead reckoning: distance from wheel odometry, heading from the IMU. No ROS dependencies.

Wheel odometry heading drifts on fast in-place turns (wheel slip): in the easy run
0.1 rad by the end, i.e. ~0.3 m position error 3 m away from the start - the same
size as the 0.30 m collect radius. The IMU heading does not drift, so we integrate
the odometry *distance* along the IMU *heading*.
"""
import math

from .frames import START_X, START_Y, START_YAW, Pose2D, wrap_angle


def map_to_odom(world: Pose2D, odom: Pose2D) -> Pose2D:
    """The map->odom correction that places the odometry pose `odom` at the estimated
    world pose `world` (what AMCL/SLAM publish). T = world * odom^-1."""
    yaw = wrap_angle(world.yaw - odom.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    return Pose2D(world.x - (c * odom.x - s * odom.y), world.y - (s * odom.x + c * odom.y), yaw)


class DeadReckoning:
    def __init__(self, start: Pose2D = Pose2D(START_X, START_Y, START_YAW)):
        self.start = start
        self.x, self.y = start.x, start.y
        self._imu0 = None          # IMU yaw at start (maps IMU frame -> world)
        self._imu_yaw = None
        self._odom = None          # last (x, y, yaw) in the odom frame

    @property
    def ready(self) -> bool:
        return self._odom is not None and self._imu_yaw is not None

    @property
    def yaw(self) -> float:
        if self._imu_yaw is None:
            return self.start.yaw if self._odom is None else wrap_angle(self.start.yaw + self._odom[2])
        return wrap_angle(self.start.yaw + self._imu_yaw - self._imu0)

    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    def update_imu(self, yaw: float) -> None:
        if self._imu0 is None:
            self._imu0 = yaw
        self._imu_yaw = yaw

    def update_odom(self, x: float, y: float, yaw: float) -> None:
        if self._odom is None:
            # first message: odom starts at ~0 at the spawn point
            self._odom = (x, y, yaw)
            c, s = math.cos(self.start.yaw), math.sin(self.start.yaw)
            self.x, self.y = self.start.x + c * x - s * y, self.start.y + s * x + c * y
            return
        px, py, pyaw = self._odom
        self._odom = (x, y, yaw)
        # signed distance travelled along the robot's own heading (odometry frame)
        mid = pyaw + wrap_angle(yaw - pyaw) / 2
        ds = (x - px) * math.cos(mid) + (y - py) * math.sin(mid)
        h = self.yaw
        self.x += ds * math.cos(h)
        self.y += ds * math.sin(h)
