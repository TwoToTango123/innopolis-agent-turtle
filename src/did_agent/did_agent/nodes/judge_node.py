"""DID judge: thin ROS wrapper around did_agent.core.judge.Judge.

Interface (TASK.md):
  /did/battery        std_msgs/Float32   battery left (start 60)
  /did/sample_sensor  std_msgs/Float32   proximity to nearest uncollected sample, 0..1, noisy
  /did/score          std_msgs/String    JSON state and score
  /did/events         std_msgs/String    JSON: collision | false_collect | hazard_hit | sample_collected
  /did/collect        std_srvs/Trigger   success if a sample is closer than 0.30 m
  /did/finish         std_srvs/Trigger   end of run (return to base)
"""
import json
import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from did_agent.core.frames import Pose2D, odom_to_world, yaw_from_quaternion
from did_agent.core.grid_map import GridMap
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.scenario import resolve_scenario
from did_agent.nodes import markers as mk


def _share(*p):
    return os.path.join(get_package_share_directory('did_agent'), *p)


class JudgeNode(Node):
    def __init__(self):
        super().__init__('did_judge')
        p = self.declare_parameter
        scenario = p('scenario', 'easy').value
        seed = p('seed', -1).value
        config = p('config', _share('config', 'judge.yaml')).value
        map_yaml = p('map', _share('maps', 'map.yaml')).value
        rate = p('rate_hz', 10.0).value
        self.pose_source = p('pose_source', 'ground_truth').value   # ground_truth | odom
        self.log_dir = p('log_dir', '').value

        grid = GridMap.from_yaml(map_yaml)
        self.scenario = resolve_scenario(scenario, seed, grid, _share('scenarios'))
        cfg = JudgeConfig.load(config)
        battery = p('battery', 0.0).value          # > 0 overrides the starting charge (demo of tight budgets)
        if battery > 0:
            cfg.battery.initial = float(battery)
        self.judge = Judge(self.scenario, cfg, grid, seed=None if seed < 0 else seed)
        self.t0 = None
        self._odom_pose = None

        self.gz = None
        if self.pose_source == 'ground_truth':
            try:
                from did_agent.nodes.gz_pose import GzModelPose
                self.gz = GzModelPose(p('model', 'burger').value)
            except Exception as e:  # noqa: BLE001 - fall back rather than die
                self.get_logger().error(f'ground truth unavailable ({e}); falling back to /odom')
                self.pose_source = 'odom'
        if self.pose_source == 'odom':
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_battery = self.create_publisher(Float32, '/did/battery', 10)
        self.pub_sensor = self.create_publisher(Float32, '/did/sample_sensor', 10)
        self.pub_score = self.create_publisher(String, '/did/score', 10)
        self.pub_events = self.create_publisher(String, '/did/events', reliable)
        self.create_service(Trigger, '/did/collect', self._on_collect)
        self.create_service(Trigger, '/did/finish', self._on_finish)
        self.create_timer(1.0 / rate, self._tick)
        # demo view of the hidden scenario (samples, zones) - the agent does not subscribe to it
        self.pub_markers = self.create_publisher(MarkerArray, '/did_judge/markers', 1)
        self.create_timer(1.0, self._publish_markers)

        sc = self.scenario
        self.get_logger().info(
            f'scenario {sc.name}: {len(sc.samples)} samples, {len(sc.terrain)} terrain zones, '
            f'{len(sc.hazards)} hazards, {len(sc.events)} hidden events; pose source: {self.pose_source}')

    # ---- inputs -------------------------------------------------------------
    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        w = odom_to_world(Pose2D(msg.pose.pose.position.x, msg.pose.pose.position.y,
                                 yaw_from_quaternion(q.x, q.y, q.z, q.w)))
        self._odom_pose = (w.x, w.y, w.yaw)

    def _pose(self):
        return self.gz.get() if self.gz is not None else self._odom_pose

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- loop ---------------------------------------------------------------
    def _tick(self):
        pose = self._pose()
        if pose is None:
            return
        now = self._now()
        if self.t0 is None:
            self.t0 = now
        for e in self.judge.update(now - self.t0, *pose):
            self._publish_event(e)
        self.pub_battery.publish(Float32(data=float(self.judge.battery.level)))
        self.pub_sensor.publish(Float32(data=float(self.judge.sensor_reading())))
        self.pub_score.publish(String(data=json.dumps(self.judge.score())))

    def _publish_markers(self):
        st = self.get_clock().now().to_msg()
        j, sc = self.judge, self.scenario
        ms = [mk.delete_all(st)]
        bx, by = sc.base
        ms.append(mk.disc('base', 0, bx, by, j.cfg.base_radius, (0.1, 0.7, 0.3, 0.35), st))
        ms.append(mk.text('base', 1, bx, by, 'BASE', (0.1, 0.9, 0.3, 1.0), st))
        for i, z in enumerate(j.terrain):
            if z.shape == 'circle':
                ms.append(mk.disc('terrain', i, z.cx, z.cy, z.r, (0.95, 0.66, 0.0, 0.30), st))
                ms.append(mk.text('terrain_label', i, z.cx, z.cy, f'{z.id} x{z.multiplier:g}', (1.0, 0.8, 0.2, 1.0), st))
        for i, h in enumerate(j.hazards):
            if h.shape == 'circle':
                ms.append(mk.disc('hazard', i, h.cx, h.cy, h.r, (0.82, 0.29, 0.36, 0.40), st))
                ms.append(mk.text('hazard_label', i, h.cx, h.cy, h.id, (1.0, 0.4, 0.4, 1.0), st))
        for i, s in enumerate(sc.samples):
            got = s.id in j.collected
            ms.append(mk.sphere('samples', i, s.x, s.y, 0.12, (0.5, 0.5, 0.5, 0.6) if got else (1.0, 0.85, 0.1, 1.0), st))
            ms.append(mk.text('sample_label', i, s.x, s.y, s.id + (' ✓' if got else ''), (1, 1, 1, 1), st, size=0.1))
        self.pub_markers.publish(MarkerArray(markers=ms))

    def _publish_event(self, e: dict):
        self.pub_events.publish(String(data=json.dumps(e)))
        self.get_logger().info(f'event: {e}')

    # ---- services -----------------------------------------------------------
    def _on_collect(self, req, resp):
        if self.t0 is None:
            resp.success, resp.message = False, 'judge has no robot pose yet'
            return resp
        resp.success, resp.message, events = self.judge.collect()
        for e in events:
            self._publish_event(e)
        self.get_logger().info(f'/did/collect -> {resp.success}: {resp.message}')
        return resp

    def _on_finish(self, req, resp):
        resp.success, resp.message = self.judge.finish()
        self.pub_score.publish(String(data=json.dumps(self.judge.score())))
        self.get_logger().info(f'/did/finish -> {resp.success}: {resp.message} | {json.dumps(self.judge.score())}')
        self.save_log()
        return resp

    def save_log(self):
        if not self.log_dir or getattr(self, '_saved_at_state', None) == self.judge.state:
            return
        self._saved_at_state = self.judge.state
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, time.strftime('%Y%m%d-%H%M%S') + f'_judge_{self.scenario.name}.json')
        with open(path, 'w') as f:
            json.dump({'scenario': self.scenario.to_dict(), 'score': self.judge.score(),
                       'log': self.judge.log, 'trajectory': self.judge.trajectory}, f, indent=1)
        self.get_logger().info(f'judge log saved to {path}')


def main():
    rclpy.init()
    node = JudgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
