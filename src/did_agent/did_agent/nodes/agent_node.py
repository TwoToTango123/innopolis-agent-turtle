"""DID agent: thin ROS wrapper around MissionExecutor (planner -> subgoals -> navigator).

Inputs:  /odom (pose, converted to the world frame), /scan (front clearance),
         /did/battery, /did/sample_sensor, /did/score, /did/events
Outputs: /cmd_vel (TwistStamped), /did_agent/path (nav_msgs/Path), /did_agent/journal (JSON),
         /did_agent/lab (experiment journal entries, planner:=science)
Calls:   /did/collect, /did/finish

Planners are chosen with the `planner` parameter; levels 2-4 add entries to make_planner().
"""
import collections
import json
import math
import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Empty, Float32, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from did_agent.core.controller import front_clearance
from did_agent.core.costmap import CostMap
from did_agent.core.executor import MissionExecutor
from did_agent.core.frames import START_X, START_Y, Pose2D, odom_to_world, yaw_from_quaternion
from did_agent.core.grid_map import GridMap
from did_agent.core.llm_client import LLMClient, LLMError
from did_agent.core.llm_planner import DEFAULT_MISSION, LLMPlanner, Target
from did_agent.core.localization import DeadReckoning, map_to_odom
from did_agent.core.mission import AgentState, GoalQueuePlanner, ScriptedPlanner, parse_targets
from did_agent.core.navigator import Navigator
from did_agent.core.scenario import resolve_scenario
from did_agent.core.science import ScientificPlanner
from did_agent.nodes import markers as mk


def _jsonable(o):
    """numpy scalars (costs come from numpy arrays) -> plain Python for json.dumps."""
    return o.item() if hasattr(o, 'item') else str(o)


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, default=_jsonable)


def _share(*p):
    return os.path.join(get_package_share_directory('did_agent'), *p)


class AgentNode(Node):
    def __init__(self):
        super().__init__('did_agent')
        p = self.declare_parameter
        self.base = (START_X, START_Y)
        self.mode = p('planner', 'scripted').value
        self.auto_collect = p('auto_collect', True).value   # manual mode: collect when the sample sensor is high
        self._last_scan = None
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
        self.sensor_raw = None
        self.sensor_seq = 0
        self.score = {}
        self._lab_len = 0
        self._science_t = -1.0
        self.t_ready = None
        self.pending_call = None
        self._path_id = None
        self._done_logged = False
        self._journal_len = 0

        # heading from the IMU, distance from wheel odometry (see core/localization.py)
        self.use_imu = p('use_imu_heading', True).value
        self.dr = DeadReckoning()
        self.tf_broadcaster = TransformBroadcaster(self) if p('publish_map_odom', True).value else None
        if self.use_imu:
            self.create_subscription(Imu, '/imu', self._on_imu, 50)
        self.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 5)
        self.create_subscription(Float32, '/did/battery', lambda m: setattr(self, 'battery', m.data), 10)
        self._sensor_hist = collections.deque(maxlen=8)        # ~0.8 s at 10 Hz: smooths the sensor noise
        self.create_subscription(Float32, '/did/sample_sensor', self._on_sensor, 10)
        self.create_subscription(String, '/did/score', lambda m: setattr(self, 'score', json.loads(m.data)), 10)
        self.create_subscription(String, '/did/events', self._on_event,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
        self.pub_cmd = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/did_agent/path', 1)
        self.pub_journal = self.create_publisher(String, '/did_agent/journal', 10)
        self.pub_llm = self.create_publisher(String, '/did_agent/llm', 10)
        self.pub_lab = self.create_publisher(String, '/did_agent/lab', 50)
        self._llm_len = 0
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
        # web control panel: state as JSON + an operator "stop"
        self.pub_state = self.create_publisher(String, '/did_agent/state', 5)
        self.create_subscription(Empty, '/did_agent/cancel', self._on_cancel, 10)
        self.create_subscription(Empty, '/did_agent/collect_now', self._on_collect_now, 10)
        self.create_subscription(Bool, '/did_agent/auto_collect', self._on_auto_collect, 10)
        self.create_timer(0.2, self._publish_state)
        self.get_logger().info(f'planner: {type(self.mission.planner).__name__}')

    # ---- planner factory (levels 2-4 plug in here) --------------------------
    def make_planner(self, grid):
        p = self.declare_parameter
        kind = self.get_parameter('planner').value
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
        if kind == 'llm':
            # level 2: the LLM chooses which targets and in what order; geometry comes from A*
            from_sc = p('targets_from_scenario', '').value
            if not from_sc:
                raise ValueError('llm planner needs `targets_from_scenario` (level-2 demo: candidate targets)')
            sc = resolve_scenario(from_sc, p('seed', -1).value, grid, _share('scenarios'))
            targets = [Target(s.id, s.x, s.y) for s in sc.samples]
            model = p('llm_model', 'deepseek-v4.1-flash').value
            try:
                client = LLMClient(model, env_file=p('llm_env_file', os.path.join(os.getcwd(), '.env')).value)
            except LLMError as e:
                self.get_logger().error(f'LLM unavailable ({e}); the planner will use its greedy fallback')
                client = None
            self.get_logger().info(f'LLM planner: model {model}, {len(targets)} candidate targets from {sc.name}')
            return LLMPlanner(client, targets, self.base, self.nav.path_cost,
                              mission=p('mission', '').value or DEFAULT_MISSION)
        if kind == 'science':
            # levels 3-4: no sample coordinates; search by /did/sample_sensor, terrain from /did/battery
            learn = p('learn_terrain', True).value
            advisor = None
            if p('llm_advisor', False).value:
                # the LLM picks the strategy (sector to explore / go home) at key moments; the robot never waits for it
                from did_agent.core.llm_scientist import LLMScientist
                model = p('llm_model', 'deepseek-v4.1-flash').value
                try:
                    client = LLMClient(model, env_file=p('llm_env_file', os.path.join(os.getcwd(), '.env')).value)
                except LLMError as e:
                    self.get_logger().error(f'LLM unavailable ({e}); the algorithm decides alone')
                    client = None
                advisor = LLMScientist(client, mission=p('mission', '').value)
            knowledge = None
            kpath = p('knowledge', '').value
            if kpath:
                try:
                    with open(os.path.expanduser(kpath)) as f:
                        knowledge = json.load(f)
                except (OSError, ValueError) as e:
                    self.get_logger().warn(f'knowledge file {kpath} not loaded: {e}')
            self.get_logger().info(f'scientific planner: sensor search, terrain learning {"on" if learn else "OFF (H1 baseline)"}'
                                   f', LLM advisor {"on" if advisor else "off"}, knowledge {"loaded" if knowledge else "none"}')
            return ScientificPlanner(grid, self.cm, self.base, self.nav.path_cost, learn_terrain=learn, advisor=advisor,
                                     knowledge=knowledge)
        if kind == 'manual':
            # operator mode: RViz "2D Goal Pose" = go there now, "Publish Point" = add to the route
            return GoalQueuePlanner(self.base, collect_at_goals=p('collect_at_goals', False).value,
                                    auto_collect=self.auto_collect)
        raise ValueError(f'unknown planner {kind!r}')

    # ---- operator goals from RViz / the web panel ---------------------------------
    def _take_control(self, reason: str) -> None:
        """Operator takeover: an autonomous mission (llm/scripted) switches to manual mode."""
        if isinstance(self.mission.planner, GoalQueuePlanner):
            return
        old = type(self.mission.planner).__name__
        self.mission.replace_planner(GoalQueuePlanner(self.base, auto_collect=self.auto_collect), reason)
        self.mode = 'manual'
        self.get_logger().warn(f'operator took control ({reason}): {old} -> manual mode')

    def _on_goal_pose(self, msg):
        x, y = msg.pose.position.x, msg.pose.position.y
        self._take_control('цель оператора')
        if not self.cm.is_free_world(x, y):
            self.get_logger().warn(f'goal ({x:.2f}, {y:.2f}) is inside an obstacle margin; will stop at the nearest free cell')
        self.mission.planner.set_goal(x, y)
        self.get_logger().info(f'operator goal: ({x:.2f}, {y:.2f})')

    def _on_clicked_point(self, msg):
        x, y = msg.point.x, msg.point.y
        self._take_control('точка маршрута оператора')
        self.mission.planner.add_point(x, y)
        self.get_logger().info(f'route point added: ({x:.2f}, {y:.2f})')

    def _on_collect_now(self, _msg):
        self._take_control('сбор оператора')
        self.mission.planner.collect_now()
        self.get_logger().info('operator: collect here')

    def _on_auto_collect(self, msg):
        self.auto_collect = bool(msg.data)
        if isinstance(self.mission.planner, GoalQueuePlanner):
            self.mission.planner.auto_collect = self.auto_collect
        self.get_logger().info(f'auto-collect by sample sensor: {self.auto_collect}')

    def _on_cancel(self, _msg):
        """Stop: take control, drop the route and the current trip, stand still."""
        self._take_control('стоп оператора')
        self.mission.planner.queue = []
        self.mission.cancel_current('cancelled by operator')
        self.get_logger().info('operator: stop, route cleared')

    def _publish_state(self):
        """Everything the web panel needs, as one JSON message."""
        planner = self.mission.planner
        cur = self.mission.current
        st = {
            't': round(self._now(), 2),
            'mode': self.mode,
            'planner': type(planner).__name__,
            'executor': self.mission.state,
            'pose': None if self.pose is None else {'x': round(self.pose.x, 3), 'y': round(self.pose.y, 3), 'yaw': round(self.pose.yaw, 3)},
            'nav': {'status': self.nav.status, 'goal': self.nav.goal,
                    'path': [[round(x, 3), round(y, 3)] for x, y in self.nav.path] if self.nav.status == Navigator.ACTIVE else []},
            'route': [list(p) for p in planner.pending_targets()] if hasattr(planner, 'pending_targets') else
                     [list(s.target) for s in getattr(planner, 'queue', []) if s.kind == 'goto' and s.target],
            'current': None if cur is None else {'kind': cur.kind, 'target': cur.target, 'reason': cur.reason[:200]},
            'thinking': bool(getattr(planner, 'thinking', False)),
            'drain_per_meter': round(self.mission.drain_per_meter, 3),
            'sensor': round(self.sensor, 3),
            'auto_collect': self.auto_collect,
            'front': None if not math.isfinite(self.front) else round(self.front, 2),
            'scan': self._scan_world(),
        }
        if hasattr(planner, 'view') and st['t'] - self._science_t >= 1.0:      # heavier layers: once a second
            self._science_t = st['t']
            st['science'] = planner.view()
        self.pub_state.publish(String(data=dumps(st)))

    def _scan_world(self):
        """Lidar hits in the world frame (every 4th ray) for the web map."""
        if self.pose is None or self._last_scan is None:
            return []
        m = self._last_scan
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        pts = []
        for i in range(0, len(m.ranges), 4):
            r = m.ranges[i]
            if math.isfinite(r) and m.range_min < r < m.range_max:
                a = m.angle_min + i * m.angle_increment
                lx, ly = r * math.cos(a), r * math.sin(a)
                pts.append([round(self.pose.x + c * lx - s * ly, 2), round(self.pose.y + s * lx + c * ly, 2)])
        return pts

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
        if isinstance(planner, ScientificPlanner):
            v = planner.view()
            if v['unexplored']:
                ms.append(mk.cubes('science_unexplored', 0, v['unexplored'], 0.06, (1.0, 0.85, 0.2, 0.35), st, z=0.002))
            for i, z in enumerate(v['zones']):
                col = (0.5, 0.5, 0.5, 0.3) if z['status'] == 'опровергнута' else                       (0.85, 0.35, 0.1, 0.55) if z['status'] == 'изменилась' else (0.95, 0.55, 0.1, 0.45)
                c = v['cell']
                ms.append(mk.cubes('science_zone', i, [(x + c / 2, y + c / 2) for x, y in z['cells']], c, col, st, z=0.004))
            for i, (x, y) in enumerate(v['hazards']):
                ms.append(mk.disc('science_hazard', i, x, y, 0.55, (0.9, 0.1, 0.2, 0.35), st))
            if v['estimate']:
                ms.append(mk.sphere('science_estimate', 0, v['estimate'][0], v['estimate'][1], 0.14, (1.0, 0.2, 0.8, 0.9), st, z=0.07))
        if isinstance(planner, LLMPlanner):
            # RViz's marker font has no Cyrillic: show the plan in ASCII, the Russian thought goes to the log
            if planner.thinking:
                caption = 'LLM is thinking...'
            elif planner.journal:
                rec = planner.journal[-1]
                order = [s.split()[1] for s in rec.get('plan', []) if s.startswith('goto ')]
                caption = ('FALLBACK: ' if rec.get('fallback') else 'LLM: ') + '>'.join(order + ['base']) + \
                    f'\nforecast {rec.get("predicted_battery")}'
            else:
                caption = ''
            if caption:
                ms.append(mk.text('llm_plan', 0, 0.0, 2.35, caption, (1.0, 1.0, 0.6, 1.0), st, size=0.12, z=0.3))
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
        if self.tf_broadcaster is not None:
            # map->odom correction so RViz draws the robot and the scan where the agent believes it is
            t = map_to_odom(self.pose, Pose2D(x, y, yaw))
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id, tf.child_frame_id = 'map', 'odom'
            tf.transform.translation.x, tf.transform.translation.y = t.x, t.y
            tf.transform.rotation.z, tf.transform.rotation.w = math.sin(t.yaw / 2), math.cos(t.yaw / 2)
            self.tf_broadcaster.sendTransform(tf)

    def _on_sensor(self, msg):
        self.sensor_raw = msg.data
        self.sensor_seq += 1
        self._sensor_hist.append(msg.data)
        self.sensor = sum(self._sensor_hist) / len(self._sensor_hist)

    def _on_imu(self, msg):
        q = msg.orientation
        self.dr.update_imu(yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _on_scan(self, msg):
        self._last_scan = msg
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
                          self.base, collected=self.score.get('collected', 0), score=self.score,
                          sensor_raw=self.sensor_raw, sensor_seq=self.sensor_seq)

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
            self.pub_journal.publish(String(data=dumps(entry)))
            self.get_logger().info(f'subgoal: {entry}')
        lab = getattr(self.mission.planner, 'lab', None)
        while lab is not None and self._lab_len < len(lab.entries):
            e = lab.entries[self._lab_len]
            self._lab_len += 1
            self.pub_lab.publish(String(data=dumps(e)))
            self.get_logger().info(f'[журнал {e["t"]:.0f} с · {e["kind"]}] {e["text"]}')
        llm_journal = getattr(self.mission.planner, 'journal', None)
        while llm_journal is not None and self._llm_len < len(llm_journal):
            rec = llm_journal[self._llm_len]
            self._llm_len += 1
            self.pub_llm.publish(String(data=dumps(rec)))
            tries = len(rec.get('attempts', []))
            lat = sum(a.get('latency', 0) for a in rec.get('attempts', []))
            self.get_logger().info(
                f'LLM [{rec.get("trigger")}] {"FALLBACK " if rec.get("fallback") else ""}plan={rec.get("plan")} '
                f'(attempts {tries}, {lat:.1f}s, battery forecast {rec.get("predicted_battery")}) — {rec.get("thought")}')
            for a in rec.get('attempts', []):
                if a.get('error'):
                    self.get_logger().warn(f'LLM answer rejected: {a["error"]}')

    def _save(self):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        path = os.path.join(self.log_dir, stamp + '_agent.json')
        lab = getattr(self.mission.planner, 'lab', None)
        if lab is not None:
            with open(os.path.join(self.log_dir, stamp + '_lab.md'), 'w') as f:
                f.write(lab.to_markdown(f'Журнал эксперимента — {self.score.get("scenario", "")}'))
            kn = dict(self.mission.planner.export_knowledge(), scenario=self.score.get('scenario'), saved=stamp)
            for name in (stamp + '_knowledge.json', 'knowledge.json'):      # knowledge.json = the latest, for knowledge:=
                with open(os.path.join(self.log_dir, name), 'w') as f:
                    json.dump(kn, f, indent=1, ensure_ascii=False, default=_jsonable)
        with open(path, 'w') as f:
            json.dump({'journal': self.mission.journal, 'planner_log': getattr(self.mission.planner, 'log', []),
                       'lab': None if lab is None else {'entries': lab.entries,
                                                        'hypotheses': [h.__dict__ for h in lab.hypotheses.values()]},
                       'llm_journal': getattr(self.mission.planner, 'journal', []),
                       'drain_per_meter': self.mission.drain_per_meter, 'score': self.score},
                      f, indent=1, ensure_ascii=False, default=_jsonable)
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
        try:
            node.destroy_node()
        except BaseException:  # noqa: BLE001 - the context may be down, or a second Ctrl+C arrives
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
