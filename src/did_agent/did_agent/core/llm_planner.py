"""Level 2: LLM mission planner. No ROS dependencies.

Division of labour (TASK.md: "LLM works only at the top level, not in the velocity loop"):
  * the algorithm computes geometry: A* path costs between the robot, every target and the base,
    converted to battery with the measured drain per metre;
  * the LLM gets the mission text + state + that cost table and decides WHAT to do:
    which targets, in what order, when to go home - referring to targets by id, never by coordinates;
  * every answer is validated (schema, known ids, collect after goto, battery budget). On errors the
    model gets the error text and tries again; after `max_retries` a deterministic greedy plan is used.
  * replanning: plan finished, a subgoal failed, a judge penalty, or a leg cost far above the forecast.

Every call is journaled (trigger, prompt, raw answer, model reasoning, latency, validation) for the demo.
"""
import json
import math
import re
import threading
from dataclasses import dataclass

from .mission import AgentState, Subgoal, SubgoalResult

SYSTEM_PROMPT = """Ты — планировщик верхнего уровня автономного робота-исследователя TurtleBot3.
Прокладку маршрутов, объезд препятствий и управление скоростью делает навигационный алгоритм.
Ты решаешь только, ЧТО делать дальше, и объясняешь почему.

Действия:
- {"action": "goto", "target": "<id>"} — поехать к цели из таблицы целей;
- {"action": "collect"} — попытаться собрать образец (только сразу после goto к образцу);
- {"action": "return"} — вернуться на базу;
- {"action": "finish"} — завершить миссию (только сразу после return).

Правила:
1. Используй только id из таблицы целей со статусом pending, каждую цель — не больше одного раза.
2. После каждого goto к образцу ставь collect.
3. Расход всего плана вместе с возвратом на базу должен быть не больше «доступно для плана». Расходы и варианты маршрута уже посчитаны алгоритмом — не перебирай комбинации и не вычисляй расстояния сам.
4. Обычно выбирай один из ВАРИАНТОВ: больше образцов — лучше, но учитывай запас и наблюдения (если переезды оказываются дороже прогноза, запас нужен больше).
5. Если переезд оказался дороже прогноза, участок пола дорогой — избегай похожих переездов.
6. План всегда заканчивается действиями return, finish.
Думай кратко: решение — это выбор стратегии, а не вычисления.

Ответь ОДНИМ JSON-объектом без текста вокруг:
{"thought": "1–2 предложения по-русски: почему такой план", "plan": [{"action": "goto", "target": "s1"}, {"action": "collect"}, {"action": "return"}, {"action": "finish"}]}"""

DEFAULT_MISSION = 'Собери как можно больше образцов, избегай дорогих участков пола и вернись на базу до разрядки батареи.'
PENALTY_EVENTS = ('collision', 'false_collect', 'hazard_hit')


@dataclass
class Target:
    id: str
    x: float
    y: float
    kind: str = 'sample'
    status: str = 'pending'          # pending | collected | failed | skipped


class PlanError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """The outermost JSON object in a model answer (tolerates ```json fences and chatter)."""
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise PlanError('в ответе нет JSON-объекта')
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise PlanError(f'JSON не разбирается: {e.msg} (позиция {e.pos})') from None
    if not isinstance(obj, dict):
        raise PlanError('ответ должен быть JSON-объектом')
    return obj


class LLMPlanner:
    persistent = True                 # the executor idles while the model thinks

    def __init__(self, client, targets: list[Target], base: tuple[float, float], cost_fn,
                 mission: str = DEFAULT_MISSION, reserve: float = 3.0, max_retries: int = 2,
                 async_mode: bool = True, deviation: float = 1.4, max_llm_calls: int = 12):
        self.client = client
        self.targets = {t.id: t for t in targets}
        self.base = base
        self.cost_fn = cost_fn        # (a_xy, b_xy) -> path cost in metres * terrain (None if unreachable)
        self.mission = mission
        self.reserve = reserve
        self.max_retries = max_retries
        self.async_mode = async_mode
        self.deviation = deviation
        self.max_llm_calls = max_llm_calls

        self.queue: list[Subgoal] = []
        self.journal: list[dict] = []     # one record per planning call
        self.observations: list[str] = []
        self.log: list[str] = []
        self.last_thought = ''
        self.finished = False
        self._trigger = 'старт миссии'
        self._pending_thread: threading.Thread | None = None
        self._result: tuple[list[Subgoal], dict] | None = None
        self._last_target: str | None = None
        self._failures: list[str] = []

    # ---- costs ------------------------------------------------------------------
    def _xy(self, node: str, state: AgentState) -> tuple[float, float]:
        if node == 'robot':
            return state.x, state.y
        if node == 'base':
            return self.base
        t = self.targets[node]
        return t.x, t.y

    def _cost_table(self, state: AgentState) -> dict:
        """Battery cost between robot, pending targets and base (path cost x measured drain)."""
        drain = state.drain_per_meter
        nodes = ['robot'] + [i for i, t in self.targets.items() if t.status == 'pending'] + ['base']
        table = {}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                if a == 'robot' or b != 'robot':
                    c = self.cost_fn(self._xy(a, state), self._xy(b, state))
                    table[(a, b)] = table[(b, a)] = None if c is None else round(c * drain, 2)
        return table

    # ---- options: the combinatorics is solved by the algorithm, not by the model ------------
    def route_options(self, state: AgentState, costs: dict, max_options: int = 4) -> list[dict]:
        """For every number of samples k: the cheapest visiting order that fits the budget
        (exact DP over subsets, n <= ~10). Returns options sorted by k descending."""
        pending = [i for i, t in self.targets.items() if t.status == 'pending']
        budget = state.battery - self.reserve
        n = len(pending)
        best = {}          # (mask, last) -> (cost, order)
        for i, tid in enumerate(pending):
            c = costs.get(('robot', tid))
            if c is not None:
                best[(1 << i, i)] = (c, (tid,))
        for mask in range(1, 1 << n):
            for last in range(n):
                cur = best.get((mask, last))
                if cur is None:
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    c = costs.get((pending[last], pending[j]))
                    if c is None:
                        continue
                    key, val = (mask | (1 << j), j), (cur[0] + c, cur[1] + (pending[j],))
                    if key not in best or val[0] < best[key][0]:
                        best[key] = val
        per_k = {}
        for (mask, last), (c, order) in best.items():
            home = costs.get((pending[last], 'base'))
            if home is None or c + home > budget + 1e-6:
                continue
            total = c + home
            k = len(order)
            if k not in per_k or total < per_k[k][0]:
                per_k[k] = (total, order)
        opts = [{'samples': k, 'order': list(o), 'cost': round(c, 2), 'margin': round(budget - c, 2)}
                for k, (c, o) in sorted(per_k.items(), reverse=True)][:max_options]
        for letter, o in zip('ABCD', opts):
            o['id'] = letter
        return opts

    # ---- prompt -------------------------------------------------------------------
    def build_prompt(self, state: AgentState, costs: dict, options: list[dict] | None = None) -> str:
        pending = [i for i, t in self.targets.items() if t.status == 'pending']
        budget = state.battery - self.reserve
        lines = [f'МИССИЯ: {self.mission}', '',
                 f'ПРИЧИНА ПЛАНИРОВАНИЯ: {self._trigger}', '',
                 'СОСТОЯНИЕ:',
                 f'- время {state.t:.0f} с; заряд {state.battery:.1f}; резерв {self.reserve:.1f}; '
                 f'доступно для плана {budget:.1f}',
                 f'- измеренный расход {state.drain_per_meter:.2f} на метр; собрано '
                 f'{max(state.collected, sum(t.status == "collected" for t in self.targets.values()))}',
                 f'- робот в ({state.x:.2f}, {state.y:.2f}); база в ({self.base[0]:.2f}, {self.base[1]:.2f}); '
                 f'возврат отсюда стоит {costs.get(("robot", "base"))}', '',
                 'ЦЕЛИ (расход батареи посчитан планировщиком пути):',
                 '| id | статус | от робота | до базы |']
        for i, t in self.targets.items():
            if t.status == 'pending':
                lines.append(f'| {i} | pending | {costs.get(("robot", i))} | {costs.get((i, "base"))} |')
            else:
                lines.append(f'| {i} | {t.status} | — | — |')
        if len(pending) > 1:
            lines += ['', 'РАСХОД МЕЖДУ ЦЕЛЯМИ:']
            for a_i, a in enumerate(pending):
                for b in pending[a_i + 1:]:
                    lines.append(f'- {a}–{b}: {costs.get((a, b))}')
        if options:
            lines += ['', 'ВАРИАНТЫ (алгоритм нашёл самый дешёвый порядок для каждого числа образцов; все укладываются в бюджет):']
            for o in options:
                lines.append(f'- {o["id"]}: {" → ".join(o["order"])} → база; образцов {o["samples"]}; '
                             f'расход {o["cost"]}; останется сверх резерва {o["margin"]}')
        elif any(t.status == 'pending' for t in self.targets.values()):
            lines += ['', 'ВАРИАНТЫ: ни одна цель не укладывается в бюджет с возвратом — нужно возвращаться на базу.']
        if self.observations:
            lines += ['', 'НАБЛЮДЕНИЯ:'] + [f'- {o}' for o in self.observations[-8:]]
        if self._failures:
            lines += ['', 'СБОИ:'] + [f'- {f}' for f in self._failures[-5:]]
        return '\n'.join(lines)

    # ---- validation ---------------------------------------------------------------
    def validate(self, obj: dict, state: AgentState, costs: dict) -> tuple[list[Subgoal], list[str], float]:
        """Returns (subgoals, warnings, predicted battery). Raises PlanError with a message for the model."""
        plan = obj.get('plan')
        if not isinstance(plan, list) or not plan:
            raise PlanError('поле "plan" должно быть непустым списком действий')
        thought = str(obj.get('thought', '')).strip()
        warnings, steps, seen = [], [], set()
        for k, step in enumerate(plan):
            if not isinstance(step, dict) or 'action' not in step:
                raise PlanError(f'шаг {k + 1}: нужен объект с полем "action"')
            a = str(step['action']).strip().lower()
            if a not in ('goto', 'collect', 'return', 'finish'):
                raise PlanError(f'шаг {k + 1}: неизвестное действие "{a}"')
            if a == 'goto':
                tid = str(step.get('target', '')).strip()
                if tid not in self.targets:
                    raise PlanError(f'шаг {k + 1}: цели "{tid}" нет в таблице')
                if self.targets[tid].status != 'pending':
                    raise PlanError(f'шаг {k + 1}: цель {tid} уже {self.targets[tid].status}')
                if tid in seen:
                    raise PlanError(f'шаг {k + 1}: цель {tid} в плане дважды')
                seen.add(tid)
                steps.append(('goto', tid))
            elif a == 'collect':
                if not steps or steps[-1][0] != 'goto':
                    raise PlanError(f'шаг {k + 1}: collect допустим только сразу после goto')
                steps.append(('collect', steps[-1][1]))
            else:
                steps.append((a, None))
            if a == 'finish':
                if k < len(plan) - 1:
                    warnings.append('действия после finish отброшены')
                break
        # repairs that do not change the model's intent
        fixed = []
        for i, (a, tid) in enumerate(steps):
            fixed.append((a, tid))
            if a == 'goto' and self.targets[tid].kind == 'sample' and (i + 1 >= len(steps) or steps[i + 1][0] != 'collect'):
                fixed.append(('collect', tid))
                warnings.append(f'добавлен collect после goto {tid}')
        steps = [s for s in fixed if s[0] not in ('return', 'finish')]
        if [a for a, _ in fixed][-2:] != ['return', 'finish']:
            warnings.append('план дополнен return, finish')
        steps += [('return', None), ('finish', None)]
        # battery budget
        pos, need = 'robot', 0.0
        for a, tid in steps:
            if a == 'goto':
                c = costs.get((pos, tid))
                if c is None:
                    raise PlanError(f'цель {tid} недостижима')
                need += c
                pos = tid
        back = costs.get((pos, 'base'))
        need += back or 0.0
        budget = state.battery - self.reserve
        # going straight home is always allowed: it is the safest thing to do on a low battery
        if any(a == 'goto' for a, _ in steps) and need > budget + 1e-6:
            raise PlanError(f'план требует {need:.1f} заряда вместе с возвратом, а доступно {budget:.1f} — убери часть целей')
        # to subgoals
        out = []
        pos = 'robot'
        for a, tid in steps:
            if a == 'goto':
                t = self.targets[tid]
                out.append(Subgoal('goto', (t.x, t.y), {'target': tid, 'predicted': costs.get((pos, tid))},
                                   reason=f'LLM: {tid}. {thought}'[:300]))
                pos = tid
            elif a == 'collect':
                out.append(Subgoal('collect', params={'target': tid}, reason=f'LLM: collect {tid}'))
            elif a == 'return':
                out.append(Subgoal('return', params={'predicted': costs.get((pos, 'base'))}, reason=f'LLM: return. {thought}'[:300]))
            else:
                out.append(Subgoal('finish', reason='LLM: finish'))
        return out, warnings, need

    # ---- fallback -------------------------------------------------------------------
    def fallback_plan(self, state: AgentState, costs: dict, options: list[dict]) -> tuple[list[Subgoal], float, str]:
        """Deterministic plan when the model fails: option A (most samples, cheapest order)."""
        order = options[0]['order'] if options else []
        obj = {'thought': 'резервный план: вариант с наибольшим числом образцов',
               'plan': [x for tid in order for x in ({'action': 'goto', 'target': tid}, {'action': 'collect'})]
               + [{'action': 'return'}, {'action': 'finish'}]}
        subgoals, _, need = self.validate(obj, state, costs)
        return subgoals, need, obj['thought']

    def greedy_plan(self, state: AgentState, costs: dict) -> tuple[list[Subgoal], float]:
        """Nearest feasible target first, as long as the trip home stays in budget."""
        budget = state.battery - self.reserve
        pos, spent, order = 'robot', 0.0, []
        left = {i for i, t in self.targets.items() if t.status == 'pending'}
        while left:
            best = None
            for tid in left:
                c, home = costs.get((pos, tid)), costs.get((tid, 'base'))
                if c is None or home is None or spent + c + home > budget:
                    continue
                if best is None or c < best[1]:
                    best = (tid, c)
            if best is None:
                break
            order.append(best[0])
            spent += best[1]
            pos = best[0]
            left.discard(best[0])
        obj = {'thought': 'резервный жадный план: ближайшая выполнимая цель', 'plan':
               [x for tid in order for x in ({'action': 'goto', 'target': tid}, {'action': 'collect'})]
               + [{'action': 'return'}, {'action': 'finish'}]}
        subgoals, _, need = self.validate(obj, state, costs)
        return subgoals, need

    # ---- the planning call ------------------------------------------------------------
    def plan_now(self, state: AgentState) -> tuple[list[Subgoal], dict]:
        costs = self._cost_table(state)
        if not any(t.status == 'pending' for t in self.targets.values()):
            # nothing left to decide: no need to ask the model
            subgoals = [Subgoal('return', params={'predicted': costs.get(('robot', 'base'))}, reason='все цели обработаны'),
                        Subgoal('finish', reason='все цели обработаны')]
            return subgoals, {'t': round(state.t, 1), 'trigger': self._trigger, 'attempts': [], 'fallback': False,
                              'plan': ['return', 'finish'], 'thought': 'целей не осталось — возвращаюсь на базу без вызова LLM',
                              'predicted_battery': costs.get(('robot', 'base'))}
        options = self.route_options(state, costs)
        user = self.build_prompt(state, costs, options)
        rec = {'t': round(state.t, 1), 'trigger': self._trigger, 'model': getattr(self.client, 'model', '?'),
               'prompt': user, 'options': options, 'attempts': [], 'fallback': False}
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
        llm_calls = sum(len(r['attempts']) for r in self.journal)
        for attempt in range(self.max_retries + 1):
            if self.client is None or llm_calls >= self.max_llm_calls:
                rec['attempts'].append({'error': 'LLM недоступна или исчерпан лимит вызовов'})
                break
            entry = {}
            rec['attempts'].append(entry)
            llm_calls += 1
            try:
                reply = self.client.chat(messages)
                entry.update(latency=round(reply.latency, 2), tokens=reply.completion_tokens,
                             content=reply.content[:2000], reasoning=reply.reasoning[:3000], finish_reason=reply.finish_reason)
                if not reply.content.strip() and reply.finish_reason == 'length':
                    raise PlanError('ответ обрезан: рассуждение заняло весь лимит токенов. Не перебирай варианты — '
                                    'выбери один из ВАРИАНТОВ и сразу ответь коротким JSON')
                obj = extract_json(reply.content)
                subgoals, warnings, need = self.validate(obj, state, costs)
                entry['warnings'] = warnings
                rec.update(plan=[self._describe(s) for s in subgoals], thought=str(obj.get('thought', '')),
                           predicted_battery=round(need, 2))
                return subgoals, rec
            except PlanError as e:
                entry['error'] = str(e)
                messages += [{'role': 'assistant', 'content': entry.get('content', '')},
                             {'role': 'user', 'content': f'Ошибка в плане: {e}. Исправь и пришли JSON заново по тем же правилам.'}]
            except Exception as e:  # noqa: BLE001 - network/model errors: fall back, never crash the robot
                entry['error'] = f'{type(e).__name__}: {e}'
                break
        subgoals, need, thought = self.fallback_plan(state, costs, options)
        rec.update(fallback=True, plan=[self._describe(s) for s in subgoals],
                   thought=thought + ' (LLM не дала корректный план)', predicted_battery=round(need, 2))
        return subgoals, rec

    @staticmethod
    def _describe(s: Subgoal) -> str:
        return s.kind + (f' {s.params["target"]}' if s.params.get('target') and s.kind == 'goto' else '')

    def _run(self, state: AgentState) -> None:
        try:
            self._result = self.plan_now(state)
        except Exception as e:  # noqa: BLE001
            self._result = ([Subgoal('return', reason=f'planner crashed: {e}'), Subgoal('finish', reason='planner crashed')],
                            {'t': round(state.t, 1), 'trigger': self._trigger, 'error': str(e), 'fallback': True, 'attempts': []})

    # ---- Planner protocol --------------------------------------------------------------
    def _replan(self, why: str) -> None:
        self.queue = []
        self._trigger = why
        self.log.append(f'replan: {why}')

    def next_subgoal(self, state: AgentState) -> Subgoal | None:
        for e in state.events:
            if e.get('type') in PENALTY_EVENTS:
                self.observations.append(f't={e.get("t")}: штраф {e["type"]} {e.get("penalty", "")}')
                if self.queue and self.queue[0].kind != 'finish':
                    self._replan(f'штраф судьи: {e["type"]}')
        if self._result is not None:                       # a plan has arrived
            subgoals, rec = self._result
            self._result, self._pending_thread = None, None
            self.journal.append(rec)
            self.last_thought = rec.get('thought', '')
            self.queue = subgoals
            self.log.append(f't={state.t:.1f}: plan {rec.get("plan")} fallback={rec.get("fallback")}')
        if self.queue:
            sg = self.queue.pop(0)
            if sg.kind == 'goto':
                self._last_target = sg.params.get('target')
            if sg.kind == 'finish':
                self.finished = True
            return sg
        if self.finished or self._pending_thread is not None:
            return None
        if self.async_mode:
            self._pending_thread = threading.Thread(target=self._run, args=(state,), daemon=True)
            self._pending_thread.start()
            return None
        self._run(state)
        return self.next_subgoal(state)

    @property
    def thinking(self) -> bool:
        return self._pending_thread is not None

    def on_result(self, result: SubgoalResult, state: AgentState) -> None:
        sg = result.subgoal
        tid = sg.params.get('target')
        self.log.append(f't={state.t:.1f}: {sg.kind} {tid or ""} -> {"ok" if result.success else "FAIL"} {result.message}')
        if sg.kind == 'collect' and tid in self.targets:
            self.targets[tid].status = 'collected' if result.success else 'failed'
            if not result.success:
                self._failures.append(f'collect {tid}: {result.message}')
                self._replan(f'не удалось собрать {tid}')
        elif sg.kind in ('goto', 'return'):
            name = tid or 'база'
            if result.distance > 0.05:
                self.observations.append(f'переезд до {name}: {result.distance:.2f} м, расход {result.battery_used:.2f} '
                                         f'({result.battery_used / result.distance:.2f}/м)')
            if not result.success:
                if tid in self.targets:
                    # the executor refuses trips it cannot afford: that target is skipped, not broken
                    self.targets[tid].status = 'skipped' if 'battery' in result.message else 'failed'
                self._failures.append(f'{sg.kind} {name}: {result.message}')
                if sg.kind == 'goto' and self.queue and self.queue[0].kind == 'collect':
                    self.queue.pop(0)
                self._replan(f'сбой подцели {sg.kind} {name}: {result.message}')
                return
            pred = sg.params.get('predicted')
            if pred and result.battery_used > self.deviation * pred and result.battery_used - pred > 0.5:
                self.observations.append(f'переезд до {name} дороже прогноза: {result.battery_used:.2f} против {pred:.2f} — '
                                         f'похоже, по пути дорогой грунт')
                if self.queue and self.queue[0].kind not in ('return', 'finish') and sg.kind == 'goto':
                    # keep the collect at this target, replan the rest
                    keep = [self.queue.pop(0)] if self.queue[0].kind == 'collect' else []
                    self._replan(f'расход на переезде до {name} вышел за прогноз')
                    self.queue = keep
