"""Mission executor: pulls subgoals from a Planner and runs them. No ROS dependencies.

goto / explore / return -> Navigator; collect / finish -> a service call that the
caller performs (ROS node: /did/collect, /did/finish; tests: Judge directly) and
reports back with `service_result`.
"""
import math
from dataclasses import dataclass

from .frames import Pose2D
from .mission import AgentState, Planner, Subgoal, SubgoalResult
from .navigator import Navigator


@dataclass
class Command:
    v: float = 0.0
    w: float = 0.0
    call: str | None = None       # 'collect' | 'finish': caller must call the service now


class MissionExecutor:
    IDLE, RUNNING, WAITING, DONE = 'idle', 'running', 'waiting_service', 'done'

    def __init__(self, planner: Planner, navigator: Navigator, base: tuple[float, float],
                 reserve: float = 1.3, margin: float = 1.0):
        self.planner = planner
        self.nav = navigator
        self.base = base
        self.reserve = reserve          # safety guard, independent of the planner:
        self.margin = margin            # a trip must leave enough battery to get home
        self.state = self.IDLE
        self.current: Subgoal | None = None
        self.journal: list[dict] = []
        self.drain_per_meter = 1.0      # prior; re-estimated from observed battery use
        self._total_used = 0.0
        self._total_dist = 0.0
        self._sg_start = None           # (t, battery, distance) when the subgoal started
        self._dist = 0.0
        self._last_xy = None
        self._last_battery = None
        self._pending_events: list[dict] = []
        self._planner_version = getattr(planner, 'version', None)

    # ---- bookkeeping --------------------------------------------------------
    def _track(self, s: AgentState) -> None:
        if self._last_xy is not None:
            d = math.hypot(s.x - self._last_xy[0], s.y - self._last_xy[1])
            self._dist += d
            self._total_dist += d
        if self._last_battery is not None:
            self._total_used += max(0.0, self._last_battery - s.battery)
        self._last_xy, self._last_battery = (s.x, s.y), s.battery
        if self._total_dist > 1.0:
            self.drain_per_meter = self._total_used / self._total_dist

    def return_cost(self, x: float, y: float) -> float | None:
        c = self.nav.path_cost((x, y), self.base)
        return None if c is None else c * self.drain_per_meter

    def trip_budget(self, x: float, y: float, target: tuple[float, float]) -> float | None:
        """Battery needed to reach `target` and then get home, with reserve."""
        there = self.nav.path_cost((x, y), target)
        home = self.nav.path_cost(target, self.base)
        if there is None or home is None:
            return None
        return (there + home) * self.drain_per_meter * self.reserve + self.margin

    def add_events(self, events: list[dict]) -> None:
        self._pending_events.extend(events)

    def _finish_subgoal(self, s: AgentState, success: bool, message: str) -> None:
        t0, b0, d0 = self._sg_start
        res = SubgoalResult(self.current, success, message, battery_used=b0 - s.battery,
                            distance=self._dist - d0, duration=s.t - t0)
        self.journal.append({'t': round(s.t, 2), 'kind': self.current.kind, 'target': self.current.target,
                             'reason': self.current.reason, 'success': success, 'message': message,
                             'battery_used': round(res.battery_used, 3), 'distance': round(res.distance, 3),
                             'duration': round(res.duration, 2), 'battery': round(s.battery, 3)})
        self.planner.on_result(res, s)
        self.current = None
        self.state = self.RUNNING

    # ---- main loop ------------------------------------------------------------
    def step(self, s: AgentState, front: float = math.inf) -> Command:
        self._track(s)
        if self.state == self.DONE:
            return Command()
        if self.state == self.WAITING:
            return Command()
        self.state = self.RUNNING
        # operator planners can replace the plan at any time: preempt the current trip
        version = getattr(self.planner, 'version', None)
        if version != self._planner_version:
            self._planner_version = version
            if self.current is not None and self.current.kind in ('goto', 'explore', 'return'):
                self.nav.cancel()
                self._finish_subgoal(s, False, 'preempted by a new goal')
        if self.current is None:
            s.events, self._pending_events = self._pending_events, []
            s.return_cost = self.return_cost(s.x, s.y)
            self.current = self.planner.next_subgoal(s)
            if self.current is None:
                if not getattr(self.planner, 'persistent', False):
                    self.state = self.DONE
                return Command()     # persistent planner: idle until a new goal arrives
            self._sg_start = (s.t, s.battery, self._dist)
            sg = self.current
            if sg.kind in ('goto', 'explore', 'return'):
                target = self.base if sg.kind == 'return' else sg.target
                if sg.kind != 'return':
                    need = self.trip_budget(s.x, s.y, target)
                    if need is not None and s.battery < need:
                        self._finish_subgoal(s, False, f'insufficient battery: need {need:.1f}, have {s.battery:.1f}')
                        return Command()
                if not self.nav.start(Pose2D(s.x, s.y, s.yaw), target, s.t):
                    self._finish_subgoal(s, False, self.nav.reason)
                    return Command()
            else:  # collect | finish
                self.state = self.WAITING
                return Command(call=sg.kind)
        v, w = self.nav.step(Pose2D(s.x, s.y, s.yaw), s.t, front)
        if self.nav.status == Navigator.ARRIVED:
            self._finish_subgoal(s, True, 'arrived')
        elif self.nav.status == Navigator.FAILED:
            self._finish_subgoal(s, False, self.nav.reason)
        return Command(v, w)

    def service_result(self, success: bool, message: str, s: AgentState) -> None:
        if self.current is None:      # preempted while the call was in flight
            return
        kind = self.current.kind
        self._finish_subgoal(s, success, message)
        if kind == 'finish':
            self.state = self.DONE
