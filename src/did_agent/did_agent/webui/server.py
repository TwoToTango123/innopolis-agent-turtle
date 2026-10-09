"""DID control panel: a localhost web UI to launch, watch and steer the robot.

    ./scripts/control_panel.sh            ->  open http://localhost:8080
    ros2 run did_agent control_panel --port 8080

One process = an HTTP server (standard library, Server-Sent Events for live data)
+ a ROS 2 node (subscribes to the judge/agent topics, publishes operator goals)
+ a launcher for `ros2 launch did_agent did.launch.py ...`. Binds to 127.0.0.1 only.
"""
import argparse
import base64
import collections
import glob
import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, Float32, String

from did_agent.core.costmap import CostMap
from did_agent.core.grid_map import OCCUPIED, GridMap
from did_agent.core.llm_planner import DEFAULT_MISSION

MODELS = ['deepseek-v4.1-flash', 'DeepSeek-V4-Flash', 'qwen3.8-flash-next', 'qwen3.6-35b-a3b', 'qwen3.8-27b', 'Qwen3.5-122B-A10B']
PLANNERS = ('science', 'llm', 'manual', 'scripted')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
LAUNCH_TS = re.compile(r'\[\d{10}\.\d+\] ')


def share(*p):
    return os.path.join(get_package_share_directory('did_agent'), *p)


# ---- ROS side --------------------------------------------------------------------
class Bridge(Node):
    def __init__(self):
        super().__init__('did_control_panel')
        self.lock = threading.Lock()
        self.reset()
        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Float32, '/did/battery', self._battery, 10)
        self.create_subscription(String, '/did/score', self._score, 10)
        self.create_subscription(String, '/did/events', self._event, reliable)
        self.create_subscription(String, '/did_agent/state', self._state, 5)
        self.create_subscription(String, '/did_judge/view', self._view, 2)
        self.create_subscription(String, '/did_agent/llm', self._llm, 10)
        self.create_subscription(String, '/did_agent/journal', self._journal, 50)
        self.create_subscription(String, '/did_agent/lab', self._lab, reliable)
        self.pub_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.pub_point = self.create_publisher(PointStamped, '/clicked_point', 10)
        self.pub_cancel = self.create_publisher(Empty, '/did_agent/cancel', 10)
        self.pub_collect = self.create_publisher(Empty, '/did_agent/collect_now', 10)
        self.pub_auto = self.create_publisher(Bool, '/did_agent/auto_collect', 10)

    def reset(self):
        with self.lock:
            self.battery = None
            self.score = {}
            self.agent = {}
            self.view = {}
            self.events = collections.deque(maxlen=60)
            self.journal = collections.deque(maxlen=60)
            self.llm = collections.deque(maxlen=20)
            self.lab = collections.deque(maxlen=300)
            self.science = None
            self.battery_hist = []
            self.last_msg = 0.0

    def _touch(self):
        self.last_msg = time.time()

    def _battery(self, m):
        with self.lock:
            self.battery = m.data
            self._touch()

    def _score(self, m):
        with self.lock:
            s = json.loads(m.data)
            t = s.get('t', 0.0)
            if self.battery_hist and t + 1.0 < self.battery_hist[-1][0]:
                self.battery_hist = []          # a new run started
                self.events.clear()
                self.journal.clear()
                self.llm.clear()
                self.lab.clear()
                self.science = None
            if not self.battery_hist or t - self.battery_hist[-1][0] >= 0.5:
                self.battery_hist.append([round(t, 1), round(s.get('battery', 0.0), 3)])
                if len(self.battery_hist) > 1200:      # keep the whole run: thin out instead of dropping the start
                    self.battery_hist = self.battery_hist[::2]
            self.score = s
            self._touch()

    def _event(self, m):
        with self.lock:
            self.events.append(json.loads(m.data))

    def _state(self, m):
        with self.lock:
            self.agent = json.loads(m.data)
            if 'science' in self.agent:              # sent once a second: keep the last one
                self.science = self.agent.pop('science')
            self._touch()

    def _lab(self, m):
        with self.lock:
            self.lab.append(json.loads(m.data))

    def _view(self, m):
        with self.lock:
            self.view = json.loads(m.data)

    def _llm(self, m):
        with self.lock:
            self.llm.append(json.loads(m.data))

    def _journal(self, m):
        with self.lock:
            self.journal.append(json.loads(m.data))

    # operator commands
    def goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)

    def point(self, x, y):
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y = float(x), float(y)
        self.pub_point.publish(msg)

    def cancel(self):
        self.pub_cancel.publish(Empty())

    def collect(self):
        self.pub_collect.publish(Empty())

    def auto_collect(self, on: bool):
        self.pub_auto.publish(Bool(data=bool(on)))

    def snapshot(self) -> dict:
        with self.lock:
            llm = [{k: r.get(k) for k in ('t', 'trigger', 'plan', 'thought', 'fallback', 'predicted_battery', 'model')}
                   | {'attempts': [{k: a.get(k) for k in ('latency', 'tokens', 'error')} for a in r.get('attempts', [])],
                      'options': r.get('options', [])}
                   for r in self.llm]
            return {'live': time.time() - self.last_msg < 3.0, 'battery': self.battery, 'score': self.score,
                    'agent': self.agent, 'view': self.view, 'events': list(self.events), 'journal': list(self.journal),
                    'llm': llm, 'battery_hist': self.battery_hist, 'lab': list(self.lab), 'science': self.science}

    def llm_full(self) -> list:
        with self.lock:
            return list(self.llm)


# ---- launcher ---------------------------------------------------------------------
class Sim:
    def __init__(self, root: str):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.params: dict = {}
        self.started = 0.0
        self.log = collections.deque(maxlen=1500)
        self.log_total = 0
        self.lock = threading.Lock()

    def external_running(self) -> bool:
        r = subprocess.run(['pgrep', '-f', 'gz sim -r -s'], capture_output=True, text=True)
        return bool(r.stdout.strip())

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict:
        own = self.running()
        return {'running': own or self.external_running(), 'own': own, 'params': self.params if own else {},
                'uptime': round(time.time() - self.started) if own else 0,
                'exit_code': None if self.proc is None or own else self.proc.returncode}

    def _add(self, line: str):
        line = LAUNCH_TS.sub('', ANSI.sub('', line.rstrip('\n')))
        if not line.strip() or 'gz_frame_id' in line or 'Stereo is NOT' in line:
            return
        with self.lock:
            self.log.append(line)
            self.log_total += 1

    def _reader(self, proc):
        for line in proc.stdout:
            self._add(line)
        self._add(f'[панель] процесс завершился, код {proc.wait()}')

    def start(self, p: dict) -> str:
        if self.running():
            raise ValueError('симуляция уже запущена из панели — сначала «Остановить»')
        if self.external_running():
            raise ValueError('симуляция уже запущена вне панели — нажмите «Остановить» (выполнит stop_sim.sh)')
        cmd = ['ros2', 'launch', 'did_agent', 'did.launch.py',
               f'scenario:={p["scenario"]}', f'planner:={p["planner"]}',
               f'battery:={p["battery"]}', f'seed:={p["seed"]}', f'llm_model:={p["model"]}',
               f'rviz:={"true" if p["rviz"] else "false"}', f'gui:={"true" if p["gui"] else "false"}',
               f'llm:={"true" if p["advisor"] else "false"}']
        if p['mission'] and p['mission'] != DEFAULT_MISSION:
            cmd.append(f'mission:={p["mission"]}')
        self._add('[панель] ' + ' '.join(c if ' ' not in c else repr(c) for c in cmd))
        self.proc = subprocess.Popen(cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
        self.params, self.started = p, time.time()
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        return 'запуск начат'

    def stop(self) -> str:
        if self.running():
            self._add('[панель] остановка (Ctrl+C)…')
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
        script = os.path.join(self.root, 'scripts', 'stop_sim.sh')
        if os.path.exists(script):
            out = subprocess.run([script], capture_output=True, text=True, timeout=60).stdout.strip()
            self._add('[панель] ' + (out or 'stop_sim.sh выполнен'))
        return 'остановлено'

    def log_since(self, since: int) -> dict:
        with self.lock:
            first = self.log_total - len(self.log)
            start = max(since, first)
            return {'lines': list(self.log)[start - first:], 'next': self.log_total}


# ---- static data --------------------------------------------------------------------
def map_payload() -> dict:
    grid = GridMap.from_yaml(share('maps', 'map.yaml'))
    cm = CostMap(grid, inflation_radius=0.2)
    occ = grid.occ == OCCUPIED
    ys, xs = np.nonzero(occ)
    pad = 6
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, grid.width - 1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, grid.height - 1)
    v = np.full(grid.occ.shape, 3, dtype=np.uint8)           # 3 = unknown
    v[grid.occ == 0] = 0                                       # 0 = free
    v[cm.lethal & (grid.occ == 0)] = 1                         # 1 = safety margin
    v[occ] = 2                                                 # 2 = obstacle
    crop = v[y0:y1 + 1, x0:x1 + 1]
    return {'resolution': grid.resolution, 'x0': grid.origin_x + x0 * grid.resolution,
            'y0': grid.origin_y + y0 * grid.resolution, 'w': int(crop.shape[1]), 'h': int(crop.shape[0]),
            'data': base64.b64encode(crop.tobytes()).decode()}


def runs_summary(root: str, n: int = 15) -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(root, 'runs', '*_judge_*.json')), reverse=True)[:n]:
        try:
            d = json.load(open(path))
        except (OSError, ValueError):
            continue
        s = d.get('score', {})
        out.append({'file': os.path.basename(path), 'time': os.path.basename(path)[:15], 'scenario': s.get('scenario'),
                    'state': s.get('state'), 'collected': s.get('collected'), 'total': s.get('samples_total'),
                    'returned': s.get('returned'), 'battery': s.get('battery'), 'distance': s.get('distance'),
                    'score': s.get('score'), 'penalties': s.get('penalties'), 't': s.get('t')})
    return out


def has_key(root: str) -> bool:
    if os.environ.get('MAI_API_KEY'):
        return True
    try:
        return any(l.startswith('MAI_API_KEY=') and len(l.strip()) > 14 for l in open(os.path.join(root, '.env')))
    except OSError:
        return False


# ---- HTTP -----------------------------------------------------------------------------
def make_handler(bridge: Bridge, sim: Sim, root: str, static_dir: str, mapdata: dict):
    def parse_start(b: dict) -> dict:
        scenario = str(b.get('scenario', 'medium'))
        if scenario not in ('easy', 'medium', 'hard') and not (scenario.endswith('.yaml') and os.path.exists(os.path.expanduser(scenario))):
            raise ValueError('неизвестный сценарий')
        planner = str(b.get('planner', 'science'))
        if planner not in PLANNERS:
            raise ValueError('неизвестный режим')
        battery = float(b.get('battery') or 0)
        if not 0 <= battery <= 500:
            raise ValueError('заряд должен быть от 0 до 500')
        seed = int(b.get('seed') if str(b.get('seed', '')).strip() not in ('', 'None') else -1)
        model = str(b.get('model', MODELS[0]))
        if model not in MODELS:
            raise ValueError('неизвестная модель')
        mission = ' '.join(str(b.get('mission') or '').split())[:600]
        return {'scenario': scenario, 'planner': planner, 'battery': battery, 'seed': seed, 'model': model,
                'mission': mission, 'rviz': bool(b.get('rviz')), 'gui': bool(b.get('gui')), 'advisor': bool(b.get('advisor'))}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ('/', '/index.html'):
                with open(os.path.join(static_dir, 'index.html'), 'rb') as f:
                    self._send(200, f.read(), 'text/html; charset=utf-8')
            elif u.path == '/api/config':
                self._json({'models': MODELS, 'planners': PLANNERS, 'scenarios': ['easy', 'medium', 'hard'],
                            'default_mission': DEFAULT_MISSION, 'has_key': has_key(root), 'root': root})
            elif u.path == '/api/map':
                self._json(mapdata)
            elif u.path == '/api/log':
                since = int(parse_qs(u.query).get('since', ['0'])[0])
                self._json(sim.log_since(since))
            elif u.path == '/api/llm':
                self._json(bridge.llm_full())
            elif u.path == '/api/runs':
                self._json(runs_summary(root))
            elif u.path == '/api/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                try:
                    while True:
                        payload = bridge.snapshot() | {'sim': sim.status()}
                        self.wfile.write(b'data: ' + json.dumps(payload, ensure_ascii=False).encode() + b'\n\n')
                        self.wfile.flush()
                        time.sleep(0.25)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._json({'error': 'not found'}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                return self._json({'error': 'bad json'}, 400)
            try:
                if u.path == '/api/start':
                    p = parse_start(body)
                    bridge.reset()
                    return self._json({'ok': True, 'message': sim.start(p)})
                if u.path == '/api/stop':
                    return self._json({'ok': True, 'message': sim.stop()})
                if u.path in ('/api/goal', '/api/point'):
                    x, y = float(body['x']), float(body['y'])
                    if not (-5 < x < 5 and -5 < y < 5):
                        raise ValueError('точка вне арены')
                    (bridge.goal if u.path == '/api/goal' else bridge.point)(x, y)
                    return self._json({'ok': True})
                if u.path == '/api/home':
                    base = (bridge.view or {}).get('base') or [-2.0, -0.5]
                    bridge.goal(*base)
                    return self._json({'ok': True})
                if u.path == '/api/cancel':
                    bridge.cancel()
                    return self._json({'ok': True})
                if u.path == '/api/collect':
                    bridge.collect()
                    return self._json({'ok': True})
                if u.path == '/api/auto_collect':
                    bridge.auto_collect(bool(body.get('on')))
                    return self._json({'ok': True})
            except (ValueError, KeyError, TypeError) as e:
                return self._json({'ok': False, 'error': str(e)}, 400)
            self._json({'error': 'not found'}, 404)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description='DID control panel (localhost)')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--root', default=os.getcwd(), help='project root (for .env, runs/, scripts/)')
    args, ros_args = ap.parse_known_args(argv)
    rclpy.init(args=ros_args)
    bridge = Bridge()
    sim = Sim(os.path.abspath(args.root))
    handler = make_handler(bridge, sim, sim.root, share('webui'), map_payload())
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        print(f'Порт {args.port} занят ({e.strerror}). Пульт уже запущен? Откройте http://localhost:{args.port} '
              f'или остановите старый: pkill -f did_agent/control_panel; другой порт: ./scripts/control_panel.sh 8090', flush=True)
        bridge.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)
    server.daemon_threads = True
    threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True).start()
    # Ctrl+C and kill (SIGTERM) both stop the server cleanly - and the simulation it started
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    signal.signal(signal.SIGINT, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f'DID control panel: http://localhost:{args.port}  (project root {sim.root})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if sim.running():
            print('stopping the simulation…', flush=True)
            sim.stop()
        server.server_close()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
