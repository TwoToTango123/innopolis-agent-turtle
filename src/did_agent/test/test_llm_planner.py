"""LLM planner without the network: a scripted fake model exercises parsing, validation,
retries with error feedback, the greedy fallback and replanning."""
import json
import os

import pytest

from did_agent.core.costmap import CostMap
from did_agent.core.judge import Judge, JudgeConfig
from did_agent.core.llm_client import LLMClient, LLMError, LLMReply, load_env_file
from did_agent.core.llm_planner import LLMPlanner, PlanError, Target, extract_json
from did_agent.core.mission import AgentState
from did_agent.core.navigator import Navigator
from did_agent.core.offline_sim import run_mission
from did_agent.core.scenario import Scenario

PKG = os.path.join(os.path.dirname(__file__), '..')
CONFIG = os.path.join(PKG, 'config', 'judge.yaml')
BASE = (-2.0, -0.5)


class FakeLLM:
    model = 'fake'

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        a = self.answers.pop(0) if self.answers else '{}'
        if isinstance(a, Exception):
            raise a
        return LLMReply(content=a if isinstance(a, str) else json.dumps(a, ensure_ascii=False), reasoning='...', latency=0.01)


def plan(*steps, thought='тест'):
    out = []
    for s in steps:
        if s in ('collect', 'return', 'finish'):
            out.append({'action': s})
        else:
            out.append({'action': 'goto', 'target': s})
    return {'thought': thought, 'plan': out}


@pytest.fixture
def easy():
    return Scenario.load(os.path.join(PKG, 'scenarios', 'easy.yaml'))


def make(world_map, sc, answers, **kw):
    nav = Navigator(CostMap(world_map, inflation_radius=0.2))
    client = FakeLLM(answers)
    p = LLMPlanner(client, [Target(s.id, s.x, s.y) for s in sc.samples], BASE, nav.path_cost, async_mode=False, **kw)
    return p, client


def state(battery=60.0):
    return AgentState(0.0, -2.0, -0.5, 0.0, battery, 0.0, BASE)


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('Вот план:\n```json\n{"plan": [], "thought": "x"}\n```') == {'plan': [], 'thought': 'x'}
    with pytest.raises(PlanError):
        extract_json('нет json')
    with pytest.raises(PlanError):
        extract_json('{"plan": [}')


def test_valid_plan_becomes_subgoals(world_map, easy):
    p, client = make(world_map, easy, [plan('s1', 'collect', 's3', 'collect', 's2', 'collect', 'return', 'finish')])
    sg = p.next_subgoal(state())
    assert (sg.kind, sg.params['target']) == ('goto', 's1')
    kinds = [sg.kind] + [s.kind for s in p.queue]
    assert kinds == ['goto', 'collect'] * 3 + ['return', 'finish']
    rec = p.journal[0]
    assert not rec['fallback'] and rec['plan'][0] == 'goto s1' and rec['predicted_battery'] > 0
    prompt = client.calls[0][1]['content']
    assert 'ЦЕЛИ' in prompt and '| s1 | pending |' in prompt and 'доступно для плана' in prompt


def test_repairs_missing_collect_and_return(world_map, easy):
    p, _ = make(world_map, easy, [plan('s1')])
    p.next_subgoal(state())
    assert [s.kind for s in p.queue] == ['collect', 'return', 'finish']
    assert 'добавлен collect после goto s1' in p.journal[0]['attempts'][0]['warnings']


def test_bad_answers_are_retried_with_error_feedback(world_map, easy):
    p, client = make(world_map, easy, ['я думаю, что...', plan('zz', 'return', 'finish'), plan('s3', 'collect', 'return', 'finish')])
    sg = p.next_subgoal(state())
    assert sg.params['target'] == 's3'
    errs = [a.get('error') for a in p.journal[0]['attempts']]
    assert errs[0] == 'в ответе нет JSON-объекта' and 'zz' in errs[1] and errs[2] is None
    # the model saw its own mistake
    assert 'Ошибка в плане' in client.calls[2][-1]['content']


def test_over_budget_plan_is_rejected(world_map, easy):
    p, _ = make(world_map, easy, [plan('s1', 'collect', 's2', 'collect', 's3', 'collect', 'return', 'finish'),
                                  plan('s1', 'collect', 'return', 'finish')])
    sg = p.next_subgoal(state(battery=8.0))
    assert 'требует' in p.journal[0]['attempts'][0]['error']
    assert sg.params['target'] == 's1'


def test_route_options_are_optimal_and_in_budget(world_map, easy):
    import itertools
    p, _ = make(world_map, easy, [])
    s = state(battery=60.0)
    costs = p._cost_table(s)
    opts = p.route_options(s, costs)
    assert [o['samples'] for o in opts] == [3, 2, 1] and opts[0]['id'] == 'A'

    def tour(order):
        stops = ['robot', *order, 'base']
        return sum(costs[(a, b)] for a, b in zip(stops, stops[1:]))
    brute = min(tour(o) for o in itertools.permutations(['s1', 's2', 's3']))
    assert opts[0]['cost'] == pytest.approx(brute, abs=0.02)
    tight = p.route_options(state(battery=3.0 + opts[1]['cost'] + 0.01), costs)
    assert tight[0]['samples'] == 2                  # three no longer fit
    assert all(o['margin'] >= 0 for o in tight)


def test_truncated_answer_gets_specific_feedback_and_fallback_uses_option_a(world_map, easy):
    class Truncated(FakeLLM):
        def chat(self, messages):
            self.calls.append(messages)
            return LLMReply(content='', reasoning='x' * 100, latency=0.01, finish_reason='length')
    nav = Navigator(CostMap(world_map, inflation_radius=0.2))
    client = Truncated([])
    p = LLMPlanner(client, [Target(s.id, s.x, s.y) for s in easy.samples], BASE, nav.path_cost, async_mode=False)
    p.next_subgoal(state())
    rec = p.journal[0]
    assert 'обрезан' in rec['attempts'][0]['error'] and rec['fallback']
    assert [x for x in rec['plan'] if x.startswith('goto')] == ['goto ' + t for t in rec['options'][0]['order']]


def test_going_home_is_allowed_even_below_reserve(world_map, easy):
    p, client = make(world_map, easy, [plan('return', 'finish')])
    sg = p.next_subgoal(state(battery=2.0))        # battery below the 3.0 reserve
    assert sg.kind == 'return' and not p.journal[0]['fallback']


def test_no_llm_call_when_nothing_is_left(world_map, easy):
    p, client = make(world_map, easy, [])
    for t in p.targets.values():
        t.status = 'collected'
    sg = p.next_subgoal(state())
    assert sg.kind == 'return' and client.calls == []


def test_fallback_after_retries_or_network_error(world_map, easy):
    p, _ = make(world_map, easy, ['мусор'] * 3)
    sg = p.next_subgoal(state())
    assert p.journal[0]['fallback'] and sg.kind == 'goto'
    p2, _ = make(world_map, easy, [LLMError('network error: timeout')])
    p2.next_subgoal(state())
    assert p2.journal[0]['fallback'] and 'network' in p2.journal[0]['attempts'][0]['error']


def test_full_offline_mission_with_llm_planner(world_map, easy):
    answers = [plan('s1', 'collect', 's3', 'collect', 's2', 'collect', 'return', 'finish', thought='s1, s3, затем s2')]
    p, client = make(world_map, easy, answers)
    res = run_mission(world_map, Judge(easy, JudgeConfig.load(CONFIG), world_map, seed=0), p)
    assert res.score['collected'] == 3 and res.score['returned'] and res.score['penalties']['collision'] == 0
    assert [j['kind'] for j in res.journal][:2] == ['goto', 'collect']
    # the s1 -> s3 -> s2 leg order came from the "model"
    assert [j['target'] for j in res.journal if j['kind'] == 'goto'] == [(-1.38, -1.42), (-0.38, -0.47), (-0.17, 1.58)]


def test_collect_failure_triggers_replan(world_map, easy):
    moved = Scenario.from_dict({**easy.to_dict(), 'samples': [{'id': 's1', 'x': -1.38, 'y': -1.42},
                                                              {'id': 's2', 'x': -0.17, 'y': 1.58},
                                                              {'id': 's3', 'x': -0.38, 'y': -0.47}]})
    # the judge's s1 is elsewhere -> collect at the planner's s1 fails
    judge_sc = Scenario.from_dict({**moved.to_dict(), 'samples': [{'id': 's1', 'x': 1.6, 'y': -1.6},
                                                                  {'id': 's2', 'x': -0.17, 'y': 1.58},
                                                                  {'id': 's3', 'x': -0.38, 'y': -0.47}]})
    answers = [plan('s1', 'collect', 'return', 'finish'), plan('s3', 'collect', 'return', 'finish')]
    p, client = make(world_map, moved, answers)
    res = run_mission(world_map, Judge(judge_sc, JudgeConfig.load(CONFIG), world_map, seed=0), p)
    assert len(client.calls) == 2
    assert 'не удалось собрать s1' in p.journal[1]['trigger']
    assert 'collect s1' in p.journal[1]['prompt'] and p.targets['s1'].status == 'failed'
    assert res.score['collected'] == 1 and res.score['returned']


def test_client_needs_key_and_hides_it(tmp_path, monkeypatch):
    monkeypatch.delenv('MAI_API_KEY', raising=False)
    with pytest.raises(LLMError):
        LLMClient('m', env_file=str(tmp_path / 'missing.env'))
    f = tmp_path / '.env'
    f.write_text('MAI_API_KEY=sk-secret123\nMAI_BASE_URL=https://example.org/v1\n')
    assert load_env_file(str(f))['MAI_API_KEY'] == 'sk-secret123'
    c = LLMClient('m', env_file=str(f))
    assert 'sk-secret' not in repr(c) and c.base_url == 'https://example.org/v1'


@pytest.mark.skipif(not os.environ.get('RUN_LLM_LIVE'), reason='set RUN_LLM_LIVE=1 to call the real model')
def test_live_model_plans_easy(world_map, easy):
    nav = Navigator(CostMap(world_map, inflation_radius=0.2))
    client = LLMClient(os.environ.get('LLM_MODEL', 'deepseek-v4.1-flash'), env_file=os.path.expanduser('~/innopolis_proj/.env'))
    p = LLMPlanner(client, [Target(s.id, s.x, s.y) for s in easy.samples], BASE, nav.path_cost, async_mode=False)
    sg = p.next_subgoal(state())
    print(json.dumps(p.journal[0], ensure_ascii=False, indent=1)[:3000])
    assert sg.kind == 'goto' and not p.journal[0]['fallback']
