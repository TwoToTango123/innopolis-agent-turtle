"""LLM as the scientific lead of the scientific agent (levels 2 + 3-4). No ROS dependencies.

The algorithm keeps driving the robot (exploration, sample localisation, terrain learning).
At key moments - a sample collected, a hypothesis confirmed or changed, a penalty, the battery
below 50 % / 30 % - the model gets the experiment journal and options computed by the algorithm
(arena sectors to explore, with unexplored share, expected samples and battery cost on the learned
cost map; or HOME) and chooses the strategy. The call is asynchronous: the robot does not stop.
Every answer is validated (choice id, budget, JSON); errors go back to the model once, then the
algorithm's own choice is used. The model never sees coordinates it could get wrong: only ids.
"""
import json
import math
import threading

import numpy as np

from .llm_planner import extract_json

SYSTEM_PROMPT = """Ты — научный руководитель автономного робота-исследователя (TurtleBot3 на арене около 5×5 м).
Робот сам ищет скрытые образцы по датчику близости и сам измеряет «цену» пола по расходу батареи.
Геометрию, пути и расход считает алгоритм. Ты принимаешь стратегическое решение: какой сектор арены
разведывать дальше или пора возвращаться на базу — и объясняешь его.

Правила:
1. Выбирай ровно один вариант из ВАРИАНТЫ по его id (R1…R6 — разведать сектор, HOME — вернуться на базу).
2. Вариант с пометкой «не хватит заряда» выбирать нельзя.
3. Цель — собрать как можно больше образцов и обязательно вернуться на базу до разрядки.
   Полезный ориентир — ожидаемые образцы на единицу заряда; дальний сектор оправдан, только если там больше шансов.
4. Если все образцы собраны или оставшийся заряд разумнее сохранить — HOME.
5. Подтверждённые дорогие зоны и опасности уже учтены в цене вариантов. «Среда изменилась» в журнале — повод
   пересмотреть приоритеты, а не прекращать миссию.
6. Ответ — только JSON без пояснений вокруг:
{"thought": "1–2 предложения по-русски: почему этот вариант", "choice": "R3", "hypothesis": "необязательно: одна проверяемая гипотеза о среде"}"""

SECTORS = 6
NAMES = ['восток', 'северо-восток', 'северо-запад', 'запад', 'юго-запад', 'юго-восток']


class AdviceError(ValueError):
    pass


def sector_of(P: np.ndarray) -> np.ndarray:
    """Sector index 0..5 of points around the arena centre (0, 0); sector 0 is centred on +x (east)."""
    a = np.arctan2(P[:, 1], P[:, 0])
    return (np.floor(((a + math.pi / SECTORS) % (2 * math.pi)) / (2 * math.pi / SECTORS))).astype(int) % SECTORS


class LLMScientist:
    def __init__(self, client, mission: str = '', max_calls: int = 20, max_retries: int = 1, async_mode: bool = True):
        self.client = client
        self.mission = mission or 'Собери как можно больше образцов, избегай дорогих и опасных зон, вернись на базу до разрядки.'
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.async_mode = async_mode
        self.journal: list[dict] = []          # same shape as LLMPlanner.journal (panel / topic)
        self._thread: threading.Thread | None = None
        self._result: dict | None = None
        self.calls = 0

    @property
    def thinking(self) -> bool:
        return self._thread is not None

    # ---- options: computed by the algorithm ---------------------------------------------
    def options(self, planner, s) -> list[dict]:
        search = planner.search
        U = ~search.excluded & ~search.abandoned
        P = search.P
        sec = sector_of(P)
        left = None if planner.samples_total is None else max(0, planner.samples_total - planner.collected)
        total_u = max(1, int(U.sum()))
        k = planner._k()
        out = []
        for i in range(SECTORS):
            m = U & (sec == i)
            n = int(m.sum())
            if n == 0:
                continue
            Q = P[m]
            c = Q.mean(axis=0)
            q = Q[int(np.argmin(np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1])))]
            there = planner.path_cost((s.x, s.y), (float(q[0]), float(q[1])))
            home = planner.path_cost((float(q[0]), float(q[1])), planner.base)
            if there is None or home is None:
                continue
            there, home = float(there), float(home)
            need = (there + home) * k * planner.reserve + planner.margin
            exp = None if left is None else left * n / total_u
            out.append({'id': f'R{i + 1}', 'sector': i, 'name': NAMES[i], 'point': (round(float(q[0]), 2), round(float(q[1]), 2)),
                        'unexplored': round(n / len(P), 3), 'expected': None if exp is None else round(exp, 2),
                        'cost_there': round(there * k, 1), 'cost_total': round(need, 1), 'affordable': bool(need <= s.battery),
                        'value': None if exp is None else round(exp / max(need, 1.0), 3)})
        home = planner._home_cost(s)
        out.append({'id': 'HOME', 'name': 'вернуться на базу', 'cost_total': None if home is None else round(float(home), 1), 'affordable': True})
        return out

    @staticmethod
    def default_choice(options: list[dict], left) -> str:
        """The algorithm's own pick (fallback): best expected samples per battery among affordable sectors."""
        if left == 0:
            return 'HOME'
        best = [o for o in options if o['id'] != 'HOME' and o['affordable']]
        if not best:
            return 'HOME'
        return max(best, key=lambda o: (o['value'] or 0.0, o['unexplored']))['id']

    # ---- prompt -----------------------------------------------------------------------------
    def build_prompt(self, planner, s, options: list[dict], trigger: str) -> str:
        left = None if planner.samples_total is None else planner.samples_total - planner.collected
        hyps = [f'- {h.id} [{h.status}]: {h.statement}' + (f' ({h.evidence[-1]})' if h.evidence else '')
                for h in planner.lab.hypotheses.values() if h.kind != 'sample' or h.status == 'проверяется']
        recent = [f'- {e["t"]:.0f} с, {e["kind"]}: {e["text"]}' for e in planner.lab.entries[-8:]]
        rows = []
        for o in options:
            if o['id'] == 'HOME':
                rows.append(f'- HOME: вернуться на базу, стоит ≈{o["cost_total"]}')
            else:
                rows.append(f'- {o["id"]} ({o["name"]}): не исследовано {100 * o["unexplored"]:.0f} % арены, '
                            f'ожидаемо образцов {o["expected"]}, туда ≈{o["cost_there"]}, туда и домой с резервом ≈{o["cost_total"]}'
                            + ('' if o['affordable'] else ' — не хватит заряда'))
        return '\n'.join([
            f'МИССИЯ: {self.mission}',
            f'ПОВОД ДЛЯ РЕШЕНИЯ: {trigger}',
            'СОСТОЯНИЕ:',
            f'- время {s.t:.0f} с; заряд {s.battery:.1f}; собрано {planner.collected} из {planner.samples_total}'
            + (f' (осталось найти {left})' if left is not None else ''),
            f'- расход на обычном полу {planner._k():.2f} ед./м; шум датчика {planner.sensor.noise:.3f}',
            'ГИПОТЕЗЫ:', *(hyps or ['- пока нет']),
            'ПОСЛЕДНИЕ ЗАПИСИ ЖУРНАЛА:', *(recent or ['- нет']),
            'ВАРИАНТЫ (посчитаны алгоритмом по выученной карте стоимостей):', *rows,
            'Выбери вариант и ответь JSON.'])

    def validate(self, obj: dict, options: list[dict], left) -> tuple[str, str, str]:
        choice = str(obj.get('choice', '')).strip().upper()
        ids = {o['id']: o for o in options}
        if choice not in ids:
            raise AdviceError(f'вариант {choice!r} не из списка: {", ".join(ids)}')
        if not ids[choice]['affordable']:
            raise AdviceError(f'на вариант {choice} не хватит заряда — выбери другой или HOME')
        if left == 0 and choice != 'HOME':
            raise AdviceError('все образцы собраны — правильный ответ HOME')
        return choice, str(obj.get('thought', '')).strip(), str(obj.get('hypothesis', '') or '').strip()

    # ---- the call ------------------------------------------------------------------------------
    def decide(self, planner, s, trigger: str) -> dict:
        """Synchronous: options and prompt from the planner's current state, then the model."""
        left = None if planner.samples_total is None else planner.samples_total - planner.collected
        options = self.options(planner, s)
        return self._call(options, self.build_prompt(planner, s, options, trigger), left, trigger, s.t)

    def _call(self, options: list[dict], user: str, left, trigger: str, t: float) -> dict:
        rec = {'t': round(t, 1), 'trigger': trigger, 'model': getattr(self.client, 'model', '?'), 'prompt': user,
               'options': options, 'attempts': [], 'fallback': False}
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
        for _ in range(self.max_retries + 1):
            if self.client is None or self.calls >= self.max_calls:
                rec['attempts'].append({'error': 'LLM недоступна или исчерпан лимит вызовов'})
                break
            entry = {}
            rec['attempts'].append(entry)
            self.calls += 1
            try:
                reply = self.client.chat(messages)
                entry.update(latency=round(reply.latency, 2), tokens=reply.completion_tokens, content=reply.content[:1500],
                             reasoning=reply.reasoning[:2500], finish_reason=reply.finish_reason)
                if not reply.content.strip():
                    raise AdviceError('пустой ответ: сразу выбери вариант и ответь коротким JSON')
                choice, thought, hyp = self.validate(extract_json(reply.content), options, left)
                rec.update(choice=choice, thought=thought, hypothesis=hyp)
                break
            except AdviceError as e:
                entry['error'] = str(e)
                messages += [{'role': 'assistant', 'content': entry.get('content', '')},
                             {'role': 'user', 'content': f'Ошибка: {e}. Пришли JSON заново по тем же правилам.'}]
            except Exception as e:  # noqa: BLE001 - network / parsing: fall back, never stop the robot
                entry['error'] = f'{type(e).__name__}: {e}'
                if isinstance(e, (json.JSONDecodeError, ValueError)):
                    messages += [{'role': 'assistant', 'content': entry.get('content', '')},
                                 {'role': 'user', 'content': 'Ответ не JSON. Пришли только JSON по формату.'}]
                    continue
                break
        if 'choice' not in rec:
            rec.update(choice=self.default_choice(options, left), fallback=True,
                       thought='выбор алгоритма: больше всего ожидаемых образцов на единицу заряда (LLM не дала корректный ответ)',
                       hypothesis='')
        rec['plan'] = ['return'] if rec['choice'] == 'HOME' else [f'goto {rec["choice"]}']
        return rec

    def ask(self, planner, s, trigger: str) -> bool:
        """Options and prompt are computed here (main thread, consistent with the cost map);
        only the network call runs in the background. False if a call is already in flight."""
        if self._thread is not None:
            return False
        left = None if planner.samples_total is None else planner.samples_total - planner.collected
        options = self.options(planner, s)
        user = self.build_prompt(planner, s, options, trigger)
        if not self.async_mode:
            self._result = self._call(options, user, left, trigger, s.t)
            return True

        def run():
            try:
                self._result = self._call(options, user, left, trigger, s.t)
            except Exception as e:  # noqa: BLE001
                self._result = {'t': round(s.t, 1), 'trigger': trigger, 'choice': None, 'fallback': True,
                                'thought': f'ошибка советника: {e}', 'attempts': [], 'plan': []}
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True

    def poll(self) -> dict | None:
        if self._result is None:
            return None
        rec, self._result, self._thread = self._result, None, None
        self.journal.append(rec)
        return rec
