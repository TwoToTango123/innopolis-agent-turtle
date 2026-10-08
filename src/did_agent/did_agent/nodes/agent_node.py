"""DID agent: thin ROS wrapper around MissionExecutor (planner -> subgoals -> navigator).

Inputs:  /odom (pose, converted to the world frame), /scan (front clearance),
         /did/battery, /did/sample_sensor, /did/score, /did/events
Outputs: /cmd_vel (TwistStamped), /did_agent/path (nav_msgs/Path), /did_agent/journal (JSON)
Calls:   /did/collect, /did/finish

Planners are chosen with the `planner` parameter; levels 2-4 add entries to make_planner().
"""
import json
import math
import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from did_agent.core.controller import front_clearance
from did_agent.core.costmap import CostMap
from did_agent.core.executor import MissionExecutor
from did_agent.core.frames import START_X, START_Y, Pose2D, odom_to_world, yaw_from_quaternion
from did_agent.core.grid_map import GridMap
from did_agent.core.localization import DeadReckoning
from did_agent.core.mission import AgentState, GoalQueuePlanner, ScriptedPlanner, parse_targets
from did_agent.core.navigator import Navigator
from did_agent.core.scenario import resolve_scenario
from did_agent.nodes import markers as mk


def _share(*p):
    return os.path.join(get_package_share_directory('did_agent'), *p)


class AgentNode(Node):
    def __init__(self):
        super().__init__('did_agent')
        p = self.declare_parameter
        self.base = (START_X, START_Y)
        grid = GridMap.from_yaml(p('map', _share('maps', 'map.yaml')).value)
        self.cm = CostMap(grid, inflation_radius=p('inflation_radius', 0.2).value)
        self.nav = Navigator(self.cm)
        self.mission = MissionExecutor(self.make_planner(grid), self.nav, self.base)
        self.rate = p('rate_hz', 20.0).value
        self.start_delay = p('start_delay', 3.0).value
        self.log_dir = p('log_dir', '').value

        self.pose: Pose2D | None = None
        self.front = math.inf
        self.battery = None
        self.sensor = 0.0
        self.score = {}
        self.t_ready = None
        self.pending_call = None
        self._path_id = None
        self._done_logged = False
        self._journal_len = 0

        # heading from the IMU, distance from wheel odometry (see core/localization.py)
        self.use_imu = p('use_imu_heading', True).value
        self.dr = DeadReckoning()
        if self.use_imu:
            self.create_subscription(Imu, '/imu', self._on_imu, 50)
        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 5)
        self.create_subscription(Float32, '/did/battery', lambda m: setattr(self, 'battery', m.data), 10)
        self.create_subscription(Float32, '/did/sample_sensor', lambda m: setattr(self, 'sensor', m.data), 10)
        self.create_subscription(String, '/did/score', lambda m: setattr(self, 'score', json.loads(m.data)), 10)
        self.create_subscription(String, '/did/events', self._on_event,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        self.pub_cmd = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/did_agent/path', 1)
        self.pub_journal = self.create_publisher(String, '/did_agent/journal', 10)
        self.cli = {'collect': self.create_client(Trigger, '/did/collect'),
                    'finish': self.create_client(Trigger, '/did/finish')}
        self.create_timer(1.0 / self.rate, self._tick)

        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal_pose, 10)
        self.create_subscription(PointStamped, '/clicked_point', self._on_clicked_point, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map_markers = self.create_publisher(MarkerArray, '/did_agent/map_markers', latched)
        self.pub_markers = self.create_publisher(MarkerArray, '/did_agent/markers', 1)
        self._publish_map_markers()
        self.create_timer(0.5, self._publish_markers)
        self.get_logger().info(f'planner: {type(self.mission.planner).__name__}')

    # ---- planner factory (levels 2-4 plug in here) --------------------------
    def make_planner(self, grid):
        p = self.declare_parameter
        kind = p('planner', 'scripted').value
        if kind == 'scripted':
            targets = parse_targets(p('targets', '').value)
            from_sc = p('targets_from_scenario', '').value
            if not targets and from_sc:
                # Level-1 demo crutch: sample coordinates given explicitly ("по заданным координатам").
                # Level 3 replaces this with search by /did/sample_sensor.
                sc = resolve_scenario(from_sc, p('seed', -1).value, grid, _share('scenarios'))
                targets = [(s.x, s.y) for s in sc.samples]
                self.get_logger().warn(f'targets taken from scenario {sc.name} (level-1 demo): {targets}')
            if not targets:
                raise ValueError('scripted planner needs `targets` or `targets_from_scenario`')
            return ScriptedPlanner(targets)
        if kind == 'manual':
            # operator mode: RViz "2D Goal Pose" = go there now, "Publish Point" = add to the route
            return GoalQueuePlanner(self.base, collect_at_goals=p('collect_at_goals', False).value)
        raise ValueError(f'unknown planner {kind!r}')

    # ---- operator goals from RViz ----------------------------------------------
    def _on_goal_pose(self, msg):
        x, y = msg.pose.position.x, msg.pose.position.y
        if not hasattr(self.mission.planner, 'set_goal'):
            self.get_logger().warn(f'goal ({x:.2f}, {y:.2f}) ignored: planner is not in manual mode')
            return
        if not self.cm.is_free_world(x, y):
            self.get_logger().warn(f'goal ({x:.2f}, {y:.2f}) is inside an obstacle margin; will stop at the nearest free cell')
        self.mission.planner.set_goal(x, y)
        self.get_logger().info(f'operator goal: ({x:.2f}, {y:.2f})')

    def _on_clicked_point(self, msg):
        x, y = msg.point.x, msg.point.y
        if not hasattr(self.mission.planner, 'add_point'):
            self.get_logger().warn(f'route point ({x:.2f}, {y:.2f}) ignored: planner is not in manual mode')
            return
        self.mission.planner.add_point(x, y)
        self.get_logger().info(f'route point added: ({x:.2f}, {y:.2f})')

    # ---- visualisation -----------------------------------------------------------
    def _publish_map_markers(self):
        """The static map as cubes (independent of RViz's Map display) + the forbidden margin."""
        st = self.get_clock().now().to_msg()
        g = self.cm.grid
        import numpy as np
        occ = [g.cell_to_world(ix, iy) for iy, ix in zip(*np.nonzero(g.occ == 100))]
        margin = [g.cell_to_world(ix, iy) for iy, ix in zip(*np.nonzero(self.cm.lethal & (g.occ == 0)))]
        self.pub_map_markers.publish(MarkerArray(markers=[
            mk.cubes('map_obstacles', 0, occ, g.resolution, (0.85, 0.85, 0.85, 1.0), st),
            mk.cubes('map_margin', 1, margin, g.resolution, (0.6, 0.2, 0.25, 0.35), st, z=-0.005)]))

    def _publish_markers(self):
        if self.pose is None:
            return
        st = self.get_clock().now().to_msg()
        ms = [mk.delete_all(st), mk.arrow('agent_pose', 0, self.pose.x, self.pose.y, self.pose.yaw, (0.0, 0.8, 1.0, 1.0), st)]
        planner = self.mission.planner
        route = []
        if self.nav.status == Navigator.ACTIVE and self.nav.goal is not None:
            route.append(self.nav.goal)
        if hasattr(planner, 'pending_targets'):
            route += planner.pending_targets()
        for i, (x, y) in enumerate(route):
            ms.append(mk.sphere('route', i, x, y, 0.1, (0.0, 0.8, 1.0, 0.9), st, z=0.05))
            ms.append(mk.text('route_label', i, x, y, str(i + 1), (1, 1, 1, 1), st, size=0.12))
        if len(route) > 1:
            ms.append(mk.line('route_line', 0, route, 0.015, (0.0, 0.8, 1.0, 0.5), st))
        self.pub_markers.publish(MarkerArray(markers=ms))

    # ---- inputs ---------------------------------------------------------------
    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        x, y, yaw = msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)
        if self.use_imu:
            self.dr.update_odom(x, y, yaw)
            self.pose = self.dr.pose()
        else:
            self.pose = odom_to_world(Pose2D(x, y, yaw))

    def _on_imu(self, msg):
        q = msg.orientation
        self.dr.update_imu(yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _on_scan(self, msg):
        self.front = front_clearance(msg.ranges, msg.angle_min, msg.angle_increment, 0.5, msg.range_min)

    def _on_event(self, msg):
        e = json.loads(msg.data)
        self.mission.add_events([e])
        self.get_logger().info(f'judge event: {e}')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- loop -------------------------------------------------------------------
    def _state(self, t) -> AgentState:
        return AgentState(t, self.pose.x, self.pose.y, self.pose.yaw, self.battery, self.sensor,
                          self.base, collected=self.score.get('collected', 0), score=self.score)

    def _tick(self):
        t = self._now()
        if self.pose is None or self.battery is None or t <= 0:
            return
        if self.t_ready is None:
            self.t_ready = t
            self.get_logger().info('inputs ready (odom, judge); starting in %.0f s' % self.start_delay)
        if t - self.t_ready < self.start_delay:
            return
        if self.pending_call is not None:
            self._send(0.0, 0.0)
            return
        s = self._state(t)
        cmd = self.mission.step(s, self.front)
        if cmd.call:
            self._call(cmd.call, s)
        self._send(cmd.v, cmd.w)
        self._publish_path()
        self._publish_journal()
        if self.mission.state == MissionExecutor.DONE and not self._done_logged:
            self._done_logged = True
            self.get_logger().info(f'mission done. score: {json.dumps(self.score)}')
            self._save()

    def _call(self, name, s):
        cli = self.cli[name]
        if not cli.wait_for_service(timeout_sec=1.0):
            self.mission.service_result(False, f'/did/{name} not available', s)
            return
        self.pending_call = name
        fut = cli.call_async(Trigger.Request())

        def done(f):
            self.pending_call = None
            r = f.result()
            self.get_logger().info(f'/did/{name} -> {r.success}: {r.message}')
            self.mission.service_result(r.success, r.message, self._state(self._now()))
        fut.add_done_callback(done)

    def _send(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def _publish_path(self):
        if self.nav.path is self._path_id or not self.nav.path:
            return
        self._path_id = self.nav.path
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.nav.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)
        self.get_logger().info(f'new path: {len(self.nav.path)} waypoints, {self.nav.planned_length:.2f} m '
                               f'-> {self.nav.goal}')

    def _publish_journal(self):
        j = self.mission.journal
        while self._journal_len < len(j):
            entry = j[self._journal_len]
            self._journal_len += 1
            self.pub_journal.publish(String(data=json.dumps(entry)))
            self.get_logger().info(f'subgoal: {entry}')

    def _save(self):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, time.strftime('%Y%m%d-%H%M%S') + '_agent.json')
        with open(path, 'w') as f:
            json.dump({'journal': self.mission.journal, 'planner_log': getattr(self.mission.planner, 'log', []),
                       'drain_per_meter': self.mission.drain_per_meter, 'score': self.score}, f, indent=1)
        self.get_logger().info(f'agent journal saved to {path}')

    def stop_robot(self):
        """Best effort on shutdown: the ROS context may already be going down."""
        try:
            self._send(0.0, 0.0)
        except Exception:  # noqa: BLE001 - rclpy raises RCLError once the context is invalid
            pass


def main():
    rclpy.init()
    node = AgentNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
