"""Mission layer: planners emit subgoals, the executor (agent node) runs them.
No ROS dependencies.

Subgoal kinds (TASK.md level 2): explore | goto | collect | return | finish.
A planner only sees AgentState and SubgoalResult, so a scripted planner (level 1),
an LLM planner (level 2) or a scientific planner (levels 3-4) are interchangeable:
implement `Planner` and register it in PLANNERS.
"""
import math
from dataclasses import dataclass, field
from typing import Protocol

KINDS = ('explore', 'goto', 'collect', 'return', 'finish')


@dataclass
class Subgoal:
    kind: str
    target: tuple[float, float] | None = None
    params: dict = field(default_factory=dict)
    reason: str = ''          # human-readable why (goes to the experiment journal)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'unknown subgoal kind {self.kind!r}, expected one of {KINDS}')
        if self.kind in ('goto', 'explore') and self.target is None:
            raise ValueError(f'{self.kind} needs a target')


@dataclass
class SubgoalResult:
    subgoal: Subgoal
    success: bool
    message: str = ''
    battery_used: float = 0.0
    distance: float = 0.0
    duration: float = 0.0


@dataclass
class AgentState:
    t: float
    x: float
    y: float
    yaw: float
    battery: float
    sensor: float
    base: tuple[float, float]
    collected: int = 0
    score: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)      # judge events since the last call
    return_cost: float | None = None                      # planned battery to get home from here
    drain_per_meter: float = 1.0                          # measured by the executor


class Planner(Protocol):
    def next_subgoal(self, state: AgentState) -> Subgoal | None:
        """Next subgoal, or None when there is nothing left to do."""

    def on_result(self, result: SubgoalResult, state: AgentState) -> None:
        """Feedback after each subgoal (success, battery used, ...)."""


class ScriptedPlanner:
    """Level 1 baseline: visit known target points, collect at each, go home.
    Returns early if the battery would not cover the trip home with a reserve."""

    def __init__(self, targets: list[tuple[float, float]], reserve: float = 1.5, margin: float = 3.0):
        self.queue: list[Subgoal] = []
        for i, (x, y) in enumerate(targets):
            self.queue.append(Subgoal('goto', (x, y), reason=f'target {i + 1}/{len(targets)}'))
            self.queue.append(Subgoal('collect', reason=f'collect at target {i + 1}'))
        self.queue.append(Subgoal('return', reason='all targets visited'))
        self.queue.append(Subgoal('finish', reason='back at base'))
        self.reserve = reserve        # x planned return cost
        self.margin = margin          # + absolute battery margin
        self.log: list[str] = []

    def _must_go_home(self, s: AgentState) -> bool:
        if s.return_cost is None:
            return False
        return s.battery < s.return_cost * self.reserve + self.margin

    def next_subgoal(self, state: AgentState) -> Subgoal | None:
        if not self.queue:
            return None
        nxt = self.queue[0]
        if nxt.kind in ('goto', 'collect', 'explore') and self._must_go_home(state):
            self.log.append(f't={state.t:.1f}: battery {state.battery:.1f} < reserve for return '
                            f'({state.return_cost:.1f}), going home')
            self.queue = [Subgoal('return', reason='battery reserve'), Subgoal('finish', reason='back at base')]
        return self.queue.pop(0)

    def on_result(self, result: SubgoalResult, state: AgentState) -> None:
        self.log.append(f't={state.t:.1f}: {result.subgoal.kind} -> {"ok" if result.success else "FAIL"} '
                        f'{result.message} (battery -{result.battery_used:.2f}, {result.distance:.2f} m)')
        if result.subgoal.kind == 'goto' and not result.success and self.queue and self.queue[0].kind == 'collect':
            self.queue.pop(0)     # don't try to collect where we never arrived
        if not result.success and 'battery' in result.message:
            self.log.append(f't={state.t:.1f}: executor refused the trip, going home')
            self.queue = [Subgoal('return', reason='battery reserve'), Subgoal('finish', reason='back at base')]


class GoalQueuePlanner:
    """Operator mode: goals come from outside (RViz "2D Goal Pose" / "Publish Point").

    set_goal()  - go to this point now (replaces the queue, preempts the current trip)
    add_point() - append a point to the route
    A goal at the base means "return and finish the run".
    `persistent`: the executor waits for new goals instead of ending the mission.
    """
    persistent = True

    def __init__(self, base: tuple[float, float], collect_at_goals: bool = False, base_radius: float = 0.3):
        self.base = base
        self.collect_at_goals = collect_at_goals
        self.base_radius = base_radius
        self.queue: list[Subgoal] = []
        self.log: list[str] = []
        self.version = 0          # bumped on set_goal: the executor preempts the current subgoal
        self.route_points = 0

    def _goal_subgoals(self, x: float, y: float, reason: str) -> list[Subgoal]:
        if distance((x, y), self.base) <= self.base_radius:
            return [Subgoal('return', reason=f'{reason}: back to base'), Subgoal('finish', reason='operator sent the robot home')]
        out = [Subgoal('goto', (x, y), reason=reason)]
        if self.collect_at_goals:
            out.append(Subgoal('collect', reason=f'{reason}: try to collect'))
        return out

    def set_goal(self, x: float, y: float) -> None:
        self.queue = self._goal_subgoals(x, y, f'operator goal ({x:.2f}, {y:.2f})')
        self.version += 1
        self.log.append(f'new goal ({x:.2f}, {y:.2f})')

    def add_point(self, x: float, y: float) -> None:
        self.route_points += 1
        self.queue += self._goal_subgoals(x, y, f'route point {self.route_points} ({x:.2f}, {y:.2f})')
        self.log.append(f'route point ({x:.2f}, {y:.2f}), queue {len(self.queue)}')

    def pending_targets(self) -> list[tuple[float, float]]:
        return [s.target if s.kind == 'goto' else self.base for s in self.queue if s.kind in ('goto', 'return')]

    def next_subgoal(self, state: AgentState) -> Subgoal | None:
        return self.queue.pop(0) if self.queue else None

    def on_result(self, result: SubgoalResult, state: AgentState) -> None:
        self.log.append(f't={state.t:.1f}: {result.subgoal.kind} -> {"ok" if result.success else "FAIL"} {result.message}')
        if result.subgoal.kind == 'goto' and not result.success and self.queue and self.queue[0].kind == 'collect':
            self.queue.pop(0)


def parse_targets(text: str) -> list[tuple[float, float]]:
    """'x1,y1; x2,y2' -> [(x1, y1), (x2, y2)]"""
    out = []
    for chunk in text.replace('\n', ';').split(';'):
        chunk = chunk.strip()
        if chunk:
            x, y = (float(v) for v in chunk.split(','))
            out.append((x, y))
    return out


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
