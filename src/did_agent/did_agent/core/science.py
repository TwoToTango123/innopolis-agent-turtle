"""Levels 3-4: the scientific agent. No ROS dependencies.

The agent does NOT know where the samples or the expensive zones are. It knows the map,
the base, the battery and the sample sensor model s = exp(-d / sigma) + noise
(QUESTIONS.md#7), and runs the cycle hypothesis -> action -> data -> conclusion:

* SensorTracker  - averages raw /did/sample_sensor readings, estimates the noise level
                   (a jump = "sensor fault" hypothesis, level 4).
* SampleSearch   - the sensor is a range finder to the nearest uncollected sample:
                   d = -sigma * ln(s). A weak reading rules out a whole disc of the arena,
                   a strong one localises a sample by least squares over several readings.
* TerrainLearner - battery drain per metre on short straight segments; cells that cost
                   more than normal floor form zone hypotheses, which are tested by further
                   measurements, put into the cost map (A* avoids them) and re-checked:
                   a changed zone means the environment changed (level 4).
* LabJournal     - the experiment journal: hypotheses, tests, conclusions, decisions.
* ScientificPlanner - ties it together behind the Planner protocol (mission.py).
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .costmap import CostMap
from .grid_map import GridMap
from .mission import AgentState, Subgoal, SubgoalResult, distance
from .scenario import reachable_points

SIGMA = 0.7              # sensor model length scale, QUESTIONS.md#7
COLLECT_RADIUS = 0.30    # TASK.md


def fmt(v: float, nd: int = 1) -> str:
    return f'{v:.{nd}f}'.replace('.', ',')


# ---- experiment journal ------------------------------------------------------------------

@dataclass
class Hypothesis:
    id: str
    kind: str                  # baseline | terrain | sample | sensor | hazard
    statement: str
    status: str = 'проверяется'   # проверяется | подтверждена | опровергнута | изменилась
    created: float = 0.0
    updated: float = 0.0
    evidence: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


class LabJournal:
    def __init__(self):
        self.entries: list[dict] = []
        self.hypotheses: dict[str, Hypothesis] = {}
        self._n: dict[str, int] = {}

    def new_id(self, prefix: str) -> str:
        self._n[prefix] = self._n.get(prefix, 0) + 1
        return f'{prefix}{self._n[prefix]}'

    def log(self, t: float, kind: str, text: str, hid: str | None = None, **data) -> dict:
        e = {'t': round(t, 1), 'kind': kind, 'text': text}
        if hid:
            e['hypothesis'] = hid
        if data:
            e['data'] = data
        self.entries.append(e)
        return e

    def propose(self, t: float, prefix: str, kind: str, statement: str, test: str, **data) -> Hypothesis:
        h = Hypothesis(self.new_id(prefix), kind, statement, created=t, updated=t, data=data)
        self.hypotheses[h.id] = h
        self.log(t, 'гипотеза', f'{h.id}: {statement}. Проверка: {test}', h.id)
        return h

    def resolve(self, t: float, hid: str, status: str, evidence: str, action: str = '') -> None:
        h = self.hypotheses[hid]
        h.status, h.updated = status, t
        h.evidence.append(evidence)
        self.log(t, {'подтверждена': 'вывод', 'опровергнута': 'вывод', 'изменилась': 'адаптация'}.get(status, 'данные'),
                 f'{hid} {status}: {evidence}' + (f' → {action}' if action else ''), hid)

    def to_markdown(self, title: str = 'Журнал эксперимента') -> str:
        lines = [f'# {title}', '', '## Гипотезы', '', '| id | гипотеза | статус | доказательства |', '|---|---|---|---|']
        for h in self.hypotheses.values():
            lines.append(f'| {h.id} | {h.statement} | {h.status} | {"; ".join(h.evidence[-2:])} |')
        lines += ['', '## Ход эксперимента', '']
        for e in self.entries:
            lines.append(f'- **{fmt(e["t"])} с · {e["kind"]}** — {e["text"]}')
        return '\n'.join(lines) + '\n'


# ---- sensor -------------------------------------------------------------------------------

@dataclass
class Obs:
    x: float
    y: float
    r: float           # mean reading
    n: int
    sd: float          # standard error of the mean
    t: float


class SensorTracker:
    """Averages readings over short stretches (<= 5 readings or 0.1 m) and estimates the noise."""

    def __init__(self, prior_noise: float = 0.03, window: int = 5):
        self.window = window
        self.buf: list[tuple[float, float, float, float]] = []
        self.resid = deque(maxlen=60)
        self.prior = prior_noise
        self.baseline: float | None = None

    @property
    def noise(self) -> float:
        if len(self.resid) < 15:
            return self.prior
        # robust (MAD): a window that straddles a jump (a sample just collected) must not look like noise
        return float(1.4826 * np.median(np.abs(self.resid)))

    def add(self, x: float, y: float, r: float, t: float) -> Obs | None:
        self.buf.append((x, y, r, t))
        if len(self.buf) < self.window and math.hypot(x - self.buf[0][0], y - self.buf[0][1]) < 0.1:
            return None
        a = np.array(self.buf)
        self.buf = []
        m = float(a[:, 2].mean())
        if len(a) >= 3 and m < 0.3:     # far from samples the true value is ~flat over 0.1 m
            k = len(a)
            self.resid.extend(((a[:, 2] - m) * math.sqrt(k / (k - 1))).tolist())
        return Obs(float(a[:, 0].mean()), float(a[:, 1].mean()), m, len(a), self.noise / math.sqrt(len(a)), t)


class SampleSearch:
    """Where can an uncollected sample still be? Candidate points every 0.1 m on the free map."""

    def __init__(self, grid: GridMap, base, sigma: float = SIGMA, pose_err: float = 0.08):
        pts = []
        for x, y in reachable_points(grid, base, clearance=0.2):
            ix, iy = grid.world_to_cell(x, y)
            if ix % 2 == 0 and iy % 2 == 0:
                pts.append((x, y))
        self.P = np.array(pts)
        self.excluded = np.zeros(len(self.P), dtype=bool)
        self.abandoned = np.zeros(len(self.P), dtype=bool)
        self.sigma = sigma
        self.pose_err = pose_err
        self.obs: list[Obs] = []          # since the last collect: they all "see" the same nearest sample set

    def d_low(self, o: Obs) -> float:
        """The nearest uncollected sample is at least this far from (o.x, o.y)."""
        return -self.sigma * math.log(min(1.0, o.r + 3.5 * o.sd + 0.01)) - self.pose_err

    def d_high(self, o: Obs) -> float:
        return -self.sigma * math.log(max(o.r - 3.5 * o.sd - 0.01, 0.005)) + self.pose_err

    def _d(self, x: float, y: float) -> np.ndarray:
        return np.hypot(self.P[:, 0] - x, self.P[:, 1] - y)

    def add(self, o: Obs) -> None:
        self.obs.append(o)
        dl = self.d_low(o)
        if dl > 0.05:
            self.excluded |= self._d(o.x, o.y) < dl

    def exclude_disc(self, x: float, y: float, r: float) -> None:
        self.excluded |= self._d(x, y) < r

    def abandon_disc(self, x: float, y: float, r: float) -> None:
        self.abandoned |= self._d(x, y) < r

    def on_collected(self, x: float, y: float) -> None:
        self.obs = []
        self.exclude_disc(x, y, COLLECT_RADIUS)

    def strongest(self) -> Obs | None:
        return max(self.obs, key=lambda o: o.r) if self.obs else None

    def estimate(self, min_r: float = 0.2) -> tuple[tuple[float, float], dict] | None:
        """Least-squares position of the sample behind the strongest recent reading."""
        top = self.strongest()
        if top is None or top.r - 2 * top.sd < min_r:
            return None
        d = self._d(top.x, top.y)
        cand = (d <= self.d_high(top)) & ~self.abandoned
        if (cand & ~self.excluded).any():
            cand &= ~self.excluded
        if not cand.any():
            return None
        near = [o for o in self.obs if math.hypot(o.x - top.x, o.y - top.y) < 2.0]
        O = np.array([[o.x, o.y, o.r, max(o.sd, 0.01)] for o in near])
        C = self.P[cand]
        dist = np.hypot(C[:, None, 0] - O[None, :, 0], C[:, None, 1] - O[None, :, 1])
        res = np.clip((O[None, :, 2] - np.exp(-dist / self.sigma)) / O[None, :, 3], -4, 4)
        cost = np.square(res).sum(axis=1)
        k = int(np.argmin(cost))
        ok = cost <= cost[k] + 4.0            # ~2 sigma confidence region
        spread = float(np.hypot(C[ok, 0] - C[k, 0], C[ok, 1] - C[k, 1]).max())
        return (float(C[k, 0]), float(C[k, 1])), {'readings': len(near), 'max': round(top.r, 2), 'spread': round(spread, 2),
                                                  'anchor': (top.x, top.y)}

    def explore_target(self, x: float, y: float, ok=None) -> tuple[float, float] | None:
        """The unexplored point with the most unexplored area around it, discounted by distance."""
        U = ~self.excluded & ~self.abandoned
        if ok is not None:
            U &= ok
        if not U.any():
            return None
        Q = self.P[U]
        if len(Q) > 900:
            Q = Q[:: int(math.ceil(len(Q) / 900))]
        dens = (np.hypot(Q[:, None, 0] - Q[None, :, 0], Q[:, None, 1] - Q[None, :, 1]) < 0.7).sum(axis=1)
        score = dens / (np.hypot(Q[:, 0] - x, Q[:, 1] - y) + 0.8)
        k = int(np.argmax(score))
        return float(Q[k, 0]), float(Q[k, 1])

    def unexplored_fraction(self) -> float:
        return float((~self.excluded & ~self.abandoned).mean()) if len(self.P) else 0.0

    def is_excluded(self, x: float, y: float) -> bool:
        d = self._d(x, y)
        k = int(np.argmin(d))
        return bool(self.excluded[k] or self.abandoned[k])


# ---- terrain --------------------------------------------------------------------------------

@dataclass
class Zone:
    hid: str
    cells: set
    mu: float
    reported_mu: float
    nseg: int
    status: str = 'проверяется'


class TerrainLearner:
    """Battery drain per metre on straight 0.2 m segments -> cost multiplier per 0.25 m cell."""
    CELL = 0.25

    def __init__(self, cm: CostMap, seg_len: float = 0.2, expensive: float = 1.35, cheap: float = 1.15):
        self.cm = cm
        g = cm.grid
        xs = g.origin_x + (np.arange(g.width) + 0.5) * g.resolution
        ys = g.origin_y + (np.arange(g.height) + 0.5) * g.resolution
        X, Y = np.meshgrid(xs, ys)
        self._ix = np.floor(X / self.CELL).astype(int)            # coarse cell of every cost-map cell
        self._iy = np.floor(Y / self.CELL).astype(int)
        self._ox, self._oy = int(self._ix.min()), int(self._iy.min())
        self._shape = (int(self._iy.max()) - self._oy + 1, int(self._ix.max()) - self._ox + 1)
        self.prior = 1.0                       # expected multiplier of floor we have not driven on yet
        self.seg_len = seg_len
        self.expensive, self.cheap = expensive, cheap
        self.k: float | None = None            # battery per metre on normal floor
        self.rates: list[float] = []
        self.cells: dict[tuple[int, int], deque] = {}
        self.zones: list[Zone] = []
        self.segments: list[tuple[float, float, float]] = []     # (x, y, mu) for the map view
        self._start = None
        self._last = None
        self._dist = self._turn = 0.0

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / self.CELL)), int(math.floor(y / self.CELL))

    def add(self, s: AgentState) -> bool:
        """Feed one state; returns True when a segment was closed (cell stats changed)."""
        if self._last is None or s.battery is None:
            self._start = self._last = (s.x, s.y, s.yaw, s.battery, s.t)
            return False
        lx, ly, lyaw, _, _ = self._last
        self._dist += math.hypot(s.x - lx, s.y - ly)
        self._turn += abs(math.atan2(math.sin(s.yaw - lyaw), math.cos(s.yaw - lyaw)))
        self._last = (s.x, s.y, s.yaw, s.battery, s.t)
        if self._dist < self.seg_len:
            if s.t - self._start[4] > 3.0 and self._dist < 0.05:      # standing: restart the segment
                self._start, self._dist, self._turn = self._last, 0.0, 0.0
            return False
        x0, y0, _, b0, _ = self._start
        used, dist, turn = b0 - s.battery, self._dist, self._turn
        self._start, self._dist, self._turn = self._last, 0.0, 0.0
        if turn > 0.35 or used < 0:
            return False
        rate = used / dist
        self.rates.append(rate)
        if len(self.rates) >= 5:
            self.k = float(np.median(self.rates))
        mx, my = (x0 + s.x) / 2, (y0 + s.y) / 2
        self.segments.append((round(mx, 2), round(my, 2), rate))
        for px, py in ((x0, y0), (mx, my), (s.x, s.y)):
            self.cells.setdefault(self.cell(px, py), deque(maxlen=6)).append(rate)   # raw: divided by k later
        return self.k is not None

    def cell_mu(self, c) -> tuple[float, int]:
        v = self.cells.get(c)
        return (float(np.median(v)) / self.k, len(v)) if v and self.k else (1.0, 0)

    def cell_center(self, c) -> tuple[float, float]:
        return (c[0] + 0.5) * self.CELL, (c[1] + 0.5) * self.CELL

    def components(self) -> list[set]:
        hot = {c for c in self.cells if self.cell_mu(c)[0] >= self.expensive and self.cell_mu(c)[1] >= 2}
        comps, seen = [], set()
        for c in hot:
            if c in seen:
                continue
            comp, stack = set(), [c]
            seen.add(c)
            while stack:
                a = stack.pop()
                comp.add(a)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        b = (a[0] + dx, a[1] + dy)
                        if b in hot and b not in seen:
                            seen.add(b)
                            stack.append(b)
            comps.append(comp)
        return comps

    def zone_stats(self, cells) -> tuple[float, int, tuple[float, float]]:
        vals = [m / self.k for c in cells for m in self.cells.get(c, ())]
        xs = [self.cell_center(c) for c in cells]
        return float(np.median(vals)), len(vals), (float(np.mean([p[0] for p in xs])), float(np.mean([p[1] for p in xs])))

    def export(self) -> dict:
        return {'k': self.k, 'rates': self.rates[-200:],
                'cells': {f'{c[0]},{c[1]}': list(v) for c, v in self.cells.items()}}

    def load(self, d: dict) -> None:
        """Knowledge from earlier missions: raw drain per cell (re-checked by new measurements)."""
        self.rates = list(d.get('rates', []))
        self.k = d.get('k')
        for key, vals in d.get('cells', {}).items():
            i, j = (int(v) for v in key.split(','))
            self.cells[(i, j)] = deque(vals, maxlen=6)

    def update_prior(self) -> float:
        """Unmeasured floor is priced by what we have seen so far: share of expensive cells x their excess."""
        stats = [self.cell_mu(c) for c in self.cells]
        stats = [m for m, n in stats if n >= 2]
        if len(stats) >= 20:
            hot = [m for m in stats if m >= self.expensive]
            self.prior = float(min(1.6, 1.0 + len(hot) / len(stats) * (np.mean(hot) - 1.0))) if hot else 1.0
        return self.prior

    def apply_to_costmap(self) -> None:
        """Terrain layer = measured floor (x1 or the zone's multiplier) + the prior for unmeasured floor."""
        mu = np.full(self._shape, self.update_prior())
        def put(c, v):
            iy, ix = c[1] - self._oy, c[0] - self._ox
            if 0 <= iy < self._shape[0] and 0 <= ix < self._shape[1]:
                mu[iy, ix] = v
        for c in self.cells:
            m, n = self.cell_mu(c)
            put(c, 1.0 if m < self.expensive or n < 2 else m)
        for z in self.zones:
            if z.status != 'опровергнута':
                for c in z.cells:
                    for dx in (-1, 0, 1):              # one cell of margin: zone edges are blurry
                        for dy in (-1, 0, 1):
                            b = (c[0] + dx, c[1] + dy)
                            iy, ix = b[1] - self._oy, b[0] - self._ox
                            if 0 <= iy < self._shape[0] and 0 <= ix < self._shape[1]:
                                mu[iy, ix] = max(mu[iy, ix] if b not in self.cells else 1.0, z.mu if b in z.cells else (z.mu + 1) / 2)
        self.cm.terrain[:] = mu[self._iy - self._oy, self._ix - self._ox]


# ---- the planner ------------------------------------------------------------------------------

class ScientificPlanner:
    """Explore -> localise by the sensor -> collect; learn the terrain; return in time."""

    def __init__(self, grid: GridMap, cm: CostMap, base: tuple[float, float], path_cost,
                 sigma: float = SIGMA, reserve: float = 1.3, margin: float = 2.0, learn_terrain: bool = True,
                 advisor=None, knowledge: dict | None = None):
        self.cm = cm
        self.knowledge = knowledge                # what earlier missions learned (terrain, hazards)
        self.advisor = advisor                    # LLMScientist: chooses the strategy at key moments (optional)
        self.focus: int | None = None             # arena sector the advisor asked to explore
        self._consult_q: list[str] = []
        self._consult_t = -1e9
        self._battery0 = None
        self._battery_marks = [0.5, 0.3]
        self.base = base
        self.path_cost = path_cost
        self.sigma = sigma
        self.reserve, self.margin = reserve, margin
        self.learn_terrain = learn_terrain
        self.search = SampleSearch(grid, base, sigma)
        from .llm_scientist import sector_of
        self._sectors = sector_of(self.search.P)
        self.sensor = SensorTracker()
        self.terrain = TerrainLearner(cm)
        self.lab = LabJournal()
        self.log: list[str] = []
        self.version = 0
        self.queue: list[Subgoal] = []
        self.current: Subgoal | None = None
        self.mode = 'explore'                     # explore | localize | home
        self.samples_total: int | None = None
        self.collected = 0
        self._seq = None
        self._wait_readings: list[float] | None = None
        self._preempt_t = -1e9
        self._home_check_t = -1e9
        self._sample_h: Hypothesis | None = None
        self._loc_attempts = 0
        self._sensor_h: Hypothesis | None = None
        self._baseline_h: Hypothesis | None = None
        self._fault_since = None
        self._hazards: list[tuple[float, float]] = []
        self._prior_t = 0.0
        self._return_fails = 0
        self.last_estimate = None
        self._last_obs = None
        self._t = 0.0
        self.lab.log(0.0, 'план', 'Цель: найти образцы по датчику, оценить «цену» пола по расходу батареи, вернуться на базу с резервом.')
        self._knowledge_pending = bool(knowledge) and learn_terrain

    # ---- what the executor reads ----------------------------------------------------------
    @property
    def base_drain(self) -> float | None:
        return self.terrain.k if self.learn_terrain else None

    @property
    def journal_md(self) -> str:
        return self.lab.to_markdown()

    @property
    def journal(self) -> list[dict]:
        """LLM decisions (published on /did_agent/llm like the level-2 planner's)."""
        return self.advisor.journal if self.advisor is not None else []

    @property
    def thinking(self) -> bool:
        return self.advisor is not None and self.advisor.thinking

    # ---- the LLM advisor ------------------------------------------------------------------
    def _consult(self, trigger: str) -> None:
        if self.samples_total is not None and self.collected >= self.samples_total:
            return                                 # nothing to decide: the algorithm goes home
        if self.advisor is not None and self.mode != 'home':
            self._consult_q.append(trigger)

    def _advisor_tick(self, s: AgentState) -> None:
        if self.advisor is None:
            return
        rec = self.advisor.poll()
        if rec is not None:
            self._apply_advice(rec, s)
        if self._consult_q and self.mode != 'home' and not self.advisor.thinking and s.t - self._consult_t > 8.0:
            trigger = '; '.join(dict.fromkeys(self._consult_q))
            if self.advisor.ask(self, s, trigger):
                self._consult_q, self._consult_t = [], s.t
                rec = self.advisor.poll()              # synchronous mode (tests, offline)
                if rec is not None:
                    self._apply_advice(rec, s)

    def _apply_advice(self, rec: dict, s: AgentState) -> None:
        choice = rec.get('choice')
        tag = 'резерв' if rec.get('fallback') else 'LLM'
        self.lab.log(s.t, 'LLM', f'[{tag}] {choice}: {rec.get("thought", "")}', trigger=rec.get('trigger'))
        if rec.get('hypothesis'):
            self.lab.log(s.t, 'идея LLM', rec['hypothesis'])
        if choice == 'HOME':
            if self.mode != 'home':
                self.mode = 'home'
                self.lab.log(s.t, 'решение', 'Возвращаюсь на базу по решению LLM')
                self._preempt(s.t, 'LLM: home', 0.0)
        elif choice and choice.startswith('R'):
            self.focus = int(choice[1:]) - 1
            cur = self.current
            if cur is not None and cur.kind == 'goto' and cur.params.get('explore') and \
                    self._sectors[self._nearest_idx(*cur.target)] != self.focus:
                self._preempt(s.t, f'LLM: explore sector {choice}', 0.0)

    def _nearest_idx(self, x: float, y: float) -> int:
        P = self.search.P
        return int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))

    def pending_targets(self) -> list[tuple[float, float]]:
        return [s.target for s in self.queue if s.target is not None]

    # ---- continuous sensing -----------------------------------------------------------------
    def _preempt(self, t: float, why: str, min_gap: float = 1.5) -> None:
        if self.current is None or self.current.kind not in ('goto', 'explore', 'wait', 'return') or t - self._preempt_t < min_gap:
            return
        self._preempt_t = t
        self.version += 1
        self.log.append(f't={t:.1f}: preempt: {why}')

    def export_knowledge(self) -> dict:
        """The reusable part of what this mission learned (the 'knowledge base' for the next one)."""
        return {'terrain': self.terrain.export(), 'hazards': [list(h) for h in self._hazards],
                'hypotheses': [{'id': h.id, 'kind': h.kind, 'statement': h.statement, 'status': h.status}
                               for h in self.lab.hypotheses.values() if h.kind in ('terrain', 'hazard', 'baseline')]}

    def _load_knowledge(self, s: AgentState) -> None:
        self._knowledge_pending = False
        kn = self.knowledge
        self.terrain.load(kn.get('terrain', {}))
        for x, y in kn.get('hazards', []):
            self._hazards.append((x, y))
            self.cm.add_penalty_circle(x, y, 0.55)
        n_zones = sum(1 for h in kn.get('hypotheses', []) if h['kind'] == 'terrain' and h['status'] != 'опровергнута')
        self.lab.log(s.t, 'знания', f'Загружены знания прошлых миссий: расход k = {fmt(self.terrain.k or 0, 2)}, '
                                    f'{len(self.terrain.cells)} измеренных клеток, дорогих зон {n_zones}, опасных зон {len(self._hazards)}. '
                                    'Считаю их гипотезами: новые замеры их подтвердят или опровергнут')
        if self.terrain.k is not None:
            self._terrain_update(s)

    def observe(self, s: AgentState) -> None:
        self._t = s.t
        if self._knowledge_pending:
            self._load_knowledge(s)
        if self._battery0 is None and s.battery is not None:
            self._battery0 = s.battery
        if self._battery_marks and s.battery is not None and self._battery0 and s.battery < self._battery_marks[0] * self._battery0:
            self._consult(f'заряд ниже {int(100 * self._battery_marks.pop(0))} %')
        self._advisor_tick(s)
        for e in s.events:
            self._on_event(e, s)
        if s.sensor_raw is not None and s.sensor_seq != self._seq:
            self._seq = s.sensor_seq
            if self._wait_readings is not None:
                self._wait_readings.append(s.sensor_raw)
            o = self.sensor.add(s.x, s.y, s.sensor_raw, s.t)
            if o is not None:
                self._last_obs = o
                self.search.add(o)
                self._after_obs(o, s)
            self._check_sensor_noise(s.t)
        if self.learn_terrain and self.terrain.add(s):
            self._terrain_update(s)
        if self.mode != 'home' and s.t - self._home_check_t > 2.0 and self.current is not None and self.current.kind == 'goto':
            self._home_check_t = s.t
            if self._must_go_home(s):
                self.lab.log(s.t, 'решение', f'Заряд {fmt(s.battery)} — пора домой: возврат стоит ≈{fmt(self._home_cost(s) or 0)} с резервом')
                self.mode = 'home'
                self._preempt(s.t, 'battery reserve', 0.0)

    def _after_obs(self, o: Obs, s: AgentState) -> None:
        cur = self.current
        if self.mode == 'explore' and o.r - 2 * o.sd >= 0.3:
            self.mode = 'localize'
            self._preempt(s.t, f'strong sensor {o.r:.2f}', 0.0)
        elif self.mode == 'explore' and cur is not None and cur.kind == 'goto' and cur.params.get('explore') \
                and self.search.is_excluded(*cur.target) and distance((s.x, s.y), cur.target) > 0.5:
            self._preempt(s.t, 'explore target already ruled out', 2.0)
        elif self.mode == 'localize' and cur is not None and cur.kind == 'goto' and cur.params.get('localize'):
            est = self.search.estimate()
            if est is not None and distance(est[0], cur.target) > 0.25:
                self._preempt(s.t, 'sample estimate moved', 2.0)

    def _check_sensor_noise(self, t: float) -> None:
        nz = self.sensor.noise
        if self.sensor.baseline is None and len(self.sensor.resid) >= 40:
            self.sensor.baseline = nz
            self.lab.log(t, 'калибровка', f'Шум датчика образцов σ ≈ {fmt(nz, 3)}')
        base = self.sensor.baseline
        if base is None:
            return
        faulty = nz > max(3 * base, 0.08)
        if faulty and self._sensor_h is None:
            self._sensor_h = self.lab.propose(t, 'D', 'sensor', f'Датчик образцов неисправен или помеха: шум σ = {fmt(nz, 2)} вместо {fmt(base, 3)}',
                                              'держится ли повышенный шум; пока усредняю дольше и не доверяю слабым сигналам')
            self._fault_since = t
        elif faulty and self._sensor_h.status == 'проверяется' and t - self._fault_since > 5.0:
            self.lab.resolve(t, self._sensor_h.id, 'подтверждена', f'шум {fmt(nz, 2)} держится {fmt(t - self._fault_since, 0)} с',
                             'ожидание у точки 3 с вместо 1 с, широкие доверительные интервалы')
        elif not faulty and self._sensor_h is not None and nz < 2 * base:
            h = self._sensor_h
            if h.status == 'проверяется':
                self.lab.resolve(t, h.id, 'опровергнута', f'шум вернулся к {fmt(nz, 3)} — кратковременный выброс')
            else:
                self.lab.log(t, 'адаптация', f'{h.id}: датчик восстановился (σ = {fmt(nz, 3)}), обычный режим', h.id)
            self._sensor_h = None

    def _terrain_update(self, s: AgentState) -> None:
        tl = self.terrain
        if self._baseline_h is None and tl.k is not None:
            self._baseline_h = self.lab.propose(s.t, 'T', 'baseline', f'Обычный пол: расход k = {fmt(tl.k, 2)} ед./м',
                                                'медиана расхода на прямых отрезках по 0,2 м')
        elif self._baseline_h is not None and self._baseline_h.status == 'проверяется' and len(tl.rates) >= 15:
            self.lab.resolve(s.t, self._baseline_h.id, 'подтверждена', f'{len(tl.rates)} отрезков, k = {fmt(tl.k, 2)} ед./м',
                             'это единица цены пола')
        comps = tl.components()
        changed = False
        matched = set()
        for comp in comps:
            mu, nseg, (cx, cy) = tl.zone_stats(comp)
            z = max(tl.zones, key=lambda z: len(z.cells & comp), default=None)
            if z is None or not (z.cells & comp):
                h = self.lab.propose(s.t, 'T', 'terrain', f'Участок около ({fmt(cx)}; {fmt(cy)}) дороже обычного пола: расход ×{fmt(mu)}',
                                     'набрать ≥ 6 замеров; если подтвердится — в карту стоимостей, A* будет объезжать',
                                     x=round(cx, 2), y=round(cy, 2), mu=round(mu, 2))
                tl.zones.append(Zone(h.id, set(comp), mu, mu, nseg))
                changed = True
                matched.add(h.id)
                continue
            matched.add(z.hid)
            grew = comp - z.cells
            z.cells |= comp
            z.mu, z.nseg = mu, nseg
            h = self.lab.hypotheses[z.hid]
            h.data.update(x=round(cx, 2), y=round(cy, 2), mu=round(mu, 2))
            if z.status == 'проверяется' and nseg >= 6:
                z.status = 'подтверждена'
                z.reported_mu = mu
                self.lab.resolve(s.t, z.hid, 'подтверждена', f'{nseg} замеров, расход ×{fmt(mu)}', 'зона в карте стоимостей, маршруты её объезжают')
                self._consult(f'подтверждена дорогая зона {z.hid}')
                changed = True
            elif z.status in ('подтверждена', 'изменилась') and abs(mu - z.reported_mu) / z.reported_mu > 0.3:
                old = z.reported_mu
                z.status, z.reported_mu = 'изменилась', mu
                self.lab.resolve(s.t, z.hid, 'изменилась', f'расход ×{fmt(old)} → ×{fmt(mu)}: среда изменилась',
                                 'обновил карту стоимостей и перепланирую')
                self._consult(f'среда изменилась: зона {z.hid}')
                changed = True
            elif grew:
                changed = True
        for z in tl.zones:
            if z.hid in matched or z.status == 'опровергнута':
                continue
            mu, nseg, _ = tl.zone_stats(z.cells)
            if mu < tl.cheap:
                z.status = 'опровергнута'
                self.lab.resolve(s.t, z.hid, 'опровергнута', f'новые замеры: расход ×{fmt(mu)} — обычный пол', 'убрал из карты стоимостей')
                changed = True
        if changed:
            tl.apply_to_costmap()
            self._preempt(s.t, 'cost map changed', 3.0)
        elif s.t - self._prior_t > 5.0:
            self._prior_t = s.t
            tl.apply_to_costmap()        # newly measured cells + the prior for unmeasured floor

    def _on_event(self, e: dict, s: AgentState) -> None:
        typ = e.get('type')
        if typ == 'hazard_hit':
            cx, cy = s.x + 0.25 * math.cos(s.yaw), s.y + 0.25 * math.sin(s.yaw)
            if any(distance((cx, cy), h) < 0.4 for h in self._hazards):
                return
            self._hazards.append((cx, cy))
            h = self.lab.propose(s.t, 'H', 'hazard', f'Опасная зона около ({fmt(cx)}; {fmt(cy)})', 'штраф hazard_hit уже получен',
                                 x=round(cx, 2), y=round(cy, 2))
            self.cm.add_penalty_circle(cx, cy, 0.55)
            self.lab.resolve(s.t, h.id, 'подтверждена', 'штраф судьи −5', 'круг 0,55 м в карте рисков, A* его объезжает; перепланирую путь')
            self._preempt(s.t, 'hazard', 0.0)
            self._consult(f'штраф: опасная зона {h.id}')
        elif typ == 'collision':
            self.lab.log(s.t, 'данные', f'Столкновение у ({fmt(s.x)}; {fmt(s.y)}) — штраф')

    # ---- battery ----------------------------------------------------------------------------
    def _k(self) -> float:
        return self.terrain.k or 1.0

    def _home_cost(self, s: AgentState) -> float | None:
        c = self.path_cost((s.x, s.y), self.base)
        return None if c is None else c * self._k()

    def _must_go_home(self, s: AgentState) -> bool:
        c = self._home_cost(s)
        return c is not None and s.battery < c * self.reserve + self.margin

    def _affordable(self, s: AgentState, target) -> bool:
        there = self.path_cost((s.x, s.y), target)
        home = self.path_cost(target, self.base)
        if there is None or home is None:
            return False
        return s.battery >= (there + home) * self._k() * self.reserve + self.margin

    # ---- decisions ----------------------------------------------------------------------------
    def _go_home(self, s: AgentState, why: str) -> Subgoal:
        if self.mode != 'home':
            self.lab.log(s.t, 'решение', f'Возвращаюсь на базу: {why}')
        if self.cm.penalty.max() > 1.0:
            safe = self._home_cost(s)
            if safe is not None and s.battery < safe * 1.1 + 0.5:
                saved, self.cm.penalty = self.cm.penalty, np.ones_like(self.cm.penalty)
                direct = self._home_cost(s)
                if direct is not None and direct < safe * 0.9:
                    self.lab.log(s.t, 'решение', f'Объезд опасной зоны стоит ≈{fmt(safe)}, заряда {fmt(s.battery)} — '
                                                 f'иду напрямую (≈{fmt(direct)}): штраф −5 лучше, чем не вернуться')
                else:
                    self.cm.penalty = saved
        self.mode = 'home'
        self.queue = [Subgoal('finish', reason='на базе')]
        return Subgoal('return', reason=why)

    def next_subgoal(self, s: AgentState) -> Subgoal | None:
        self.current = self._next(s)
        return self.current

    def _next(self, s: AgentState) -> Subgoal | None:
        if s.score:
            self.samples_total = s.score.get('samples_total', self.samples_total)
            self.collected = s.score.get('collected', self.collected)
        if self.advisor is not None and not self.advisor.journal and not self._consult_q and not self.advisor.thinking:
            self._consult('начало миссии: куда вести разведку')
        if self.queue:
            return self.queue.pop(0)
        if self.mode == 'home':
            if distance((s.x, s.y), self.base) > 0.25:
                return self._go_home(s, 'заряд на исходе')
            return Subgoal('finish', reason='на базе')
        if self.samples_total is not None and self.collected >= self.samples_total:
            return self._go_home(s, f'собраны все {self.samples_total} образцов')
        if self._must_go_home(s):
            return self._go_home(s, f'заряд {fmt(s.battery)} — только на возврат с резервом')
        est = self.search.estimate()
        if est is not None:
            self.mode = 'localize'
            return self._localize(s, *est)
        self.mode = 'explore'
        return self._explore(s)

    def _localize(self, s: AgentState, q, info) -> Subgoal:
        self.last_estimate = q
        if self._sample_h is None:
            self._loc_attempts = 0
            self._sample_h = self.lab.propose(
                s.t, 'S', 'sample', f'Образец около ({fmt(q[0], 2)}; {fmt(q[1], 2)}): датчик до {fmt(info["max"], 2)}, замеров: {info["readings"]}',
                'подъехать, постоять, усреднить датчик; при ≥ 0,65 (ближе 0,3 м) — /did/collect', x=round(q[0], 2), y=round(q[1], 2))
        self._sample_h.data.update(x=round(q[0], 2), y=round(q[1], 2))
        if distance((s.x, s.y), q) > 0.15:
            if not self._affordable(s, q):
                self._abandon_sample(s, 'не хватит заряда доехать и вернуться')
                return self._go_home(s, 'образец рядом, но заряда на него не хватает')
            return Subgoal('goto', q, params={'localize': True},
                           reason=f'{self._sample_h.id}: к оценке положения образца (замеров: {info["readings"]}, разброс {fmt(info["spread"], 2)} м)')
        return self._measure()

    def _measure(self) -> Subgoal:
        self._wait_readings = []
        dur = 3.0 if self._sensor_h is not None else 1.0
        return Subgoal('wait', params={'duration': dur, 'measure': True}, reason=f'усредняю датчик {fmt(dur, 0)} с')

    def _explore(self, s: AgentState) -> Subgoal:
        if self.focus is not None:
            sector = self._sectors == self.focus
            if not (sector & ~self.search.excluded & ~self.search.abandoned).any():
                self.lab.log(s.t, 'данные', f'Сектор R{self.focus + 1} исследован')
                self.focus = None
                self._consult('выбранный сектор исследован')
        for _ in range(5):
            ok = (self._sectors == self.focus) if self.focus is not None else None
            q = self.search.explore_target(s.x, s.y, ok=ok)
            if q is None and ok is not None:
                self.focus = None
                q = self.search.explore_target(s.x, s.y)
            if q is None:
                break
            if not self.cm.is_free_world(*q):
                self.search.abandon_disc(*q, 0.2)
                continue
            if self._affordable(s, q):
                where = f'сектор R{self.focus + 1} (решение LLM), ' if self.focus is not None else ''
                return Subgoal('goto', q, params={'explore': True},
                               reason=f'разведка: {where}не исследовано {fmt(100 * self.search.unexplored_fraction(), 0)} % арены')
            self.search.abandon_disc(*q, 0.5)
        left = None if self.samples_total is None else self.samples_total - self.collected
        if self.search.unexplored_fraction() == 0.0 and left:
            self.lab.log(s.t, 'вывод', f'Арена исследована, но {left} образц. не найдено — датчик мог ошибиться; заряда больше не трачу')
        return self._go_home(s, 'дальше разведка не окупается или арена исследована')

    def _abandon_sample(self, s: AgentState, why: str) -> None:
        if self._sample_h is not None:
            self.lab.resolve(s.t, self._sample_h.id, 'опровергнута', why)
            x, y = self._sample_h.data['x'], self._sample_h.data['y']
            self.search.abandon_disc(x, y, 0.45)
            self._sample_h = None
        self.search.obs = []

    # ---- feedback ----------------------------------------------------------------------------
    def on_result(self, r: SubgoalResult, s: AgentState) -> None:
        sg = r.subgoal
        self.current = None
        self.log.append(f't={s.t:.1f}: {sg.kind} -> {"ok" if r.success else "FAIL"} {r.message}')
        if sg.kind == 'wait' and sg.params.get('measure'):
            vals = self._wait_readings or []
            self._wait_readings = None
            if not r.success or not vals:
                return
            m = float(np.mean(vals))
            sd = self.sensor.noise / math.sqrt(len(vals))
            thr = math.exp(-COLLECT_RADIUS / self.sigma)
            self._loc_attempts += 1
            if m >= thr - sd:
                self.lab.log(s.t, 'проверка', f'Датчик {fmt(m, 2)} ± {fmt(sd, 2)} ≥ {fmt(thr, 2)}: образец ближе 0,3 м — собираю',
                             self._sample_h.id if self._sample_h else None)
                self.queue.insert(0, Subgoal('collect', reason=f'датчик {m:.2f} ≥ {thr:.2f}'))
            elif self._loc_attempts >= 6:
                self._abandon_sample(s, f'6 попыток, датчик не выше {fmt(m, 2)}')
            else:
                self.lab.log(s.t, 'данные', f'Датчик {fmt(m, 2)} < {fmt(thr, 2)}: образец не здесь, уточняю положение',
                             self._sample_h.id if self._sample_h else None)
        elif sg.kind == 'collect':
            if r.success:
                h = self._sample_h
                if h is not None:
                    self.lab.resolve(s.t, h.id, 'подтверждена', f'/did/collect: {r.message}')
                    self._sample_h = None
                self.collected += 1
                self._consult(f'собран образец ({self.collected} из {self.samples_total})')
                self.search.on_collected(s.x, s.y)
                self.sensor.buf = []          # readings before the collect saw the old sample
                self.mode = 'explore'
            else:
                self.search.exclude_disc(s.x, s.y, COLLECT_RADIUS - 0.03)
                self.lab.log(s.t, 'данные', f'Ложный сбор ({r.message}) — исключаю круг 0,27 м', self._sample_h.id if self._sample_h else None)
        elif sg.kind == 'return' and not r.success:
            self._return_fails += 0 if 'preempt' in r.message else 1
            if self._return_fails < 3 and 'battery depleted' not in r.message:
                self.queue = []           # plan the way home again (the cost map changed), then finish
        elif sg.kind == 'goto' and not r.success:
            if 'preempt' in r.message:
                return
            if 'battery' in r.message:
                self.mode = 'home'
                self.lab.log(s.t, 'решение', f'Исполнитель отказал в поездке ({r.message}) — домой')
            elif sg.params.get('localize'):
                self._abandon_sample(s, f'не доехать: {r.message}')
            else:
                self.search.abandon_disc(*sg.target, 0.35)
        elif sg.kind == 'goto' and r.success and sg.params.get('localize'):
            self.queue.insert(0, self._measure())

    # ---- for the map view / logs ---------------------------------------------------------------
    def view(self) -> dict:
        P = self.search.P
        U = ~self.search.excluded & ~self.search.abandoned
        return {
            'mode': self.mode,
            'unexplored': [[round(float(x), 2), round(float(y), 2)] for x, y in P[U][::3]],
            'estimate': None if self.last_estimate is None or self._sample_h is None else [round(v, 2) for v in self.last_estimate],
            'zones': [{'id': z.hid, 'status': z.status, 'mu': round(z.mu, 2),
                       'cells': [[c[0] * TerrainLearner.CELL, c[1] * TerrainLearner.CELL] for c in z.cells]} for z in self.terrain.zones],
            'cell': TerrainLearner.CELL,
            'hazards': [[round(x, 2), round(y, 2)] for x, y in self._hazards],
            'k': None if self.terrain.k is None else round(self.terrain.k, 3),
            'noise': round(self.sensor.noise, 3),
            'hypotheses': [{'id': h.id, 'kind': h.kind, 'statement': h.statement, 'status': h.status} for h in self.lab.hypotheses.values()],
            'lab': self.lab.entries[-30:],
        }
