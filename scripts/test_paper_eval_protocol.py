#!/usr/bin/env python3
"""Offline protocol regression tests; no model/GPU/JVM required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().with_name('run_audited_paper_eval.py')
spec = importlib.util.spec_from_file_location('evaluator', path)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class FakeAgent:
    def __init__(self):
        self.messages = []

    def count(self, messages):
        return 1000

    def generate(self, messages, task_id, step):
        self.messages.append(messages)
        return 'Thought: inspect the lab.\n\nAction: look around', {}

    def trace(self, *args):
        pass


class ProtocolTests(unittest.TestCase):
    def test_native_alf_initial_template(self):
        actual = ev.native_prompt('alfworld', 'goal', 'Your task is to: goal', ['look', 'help'], [])
        namespace = ev.runpy.run_path(str(ev.VERL / 'environments/prompts/alfworld.py'))
        expected = namespace['ALFWORLD_TEMPLATE_NO_HIS'].format(
            current_observation='Your task is to: goal', admissible_actions="'look'")
        self.assertEqual(actual, [{'role': 'user', 'content': expected}])

    def test_history_records_pre_action_observation_and_step_number(self):
        steps = [{'observation': str(i), 'action': 'look'} for i in range(4)]
        actual = ev.history_text(steps)
        self.assertEqual(actual, "[Observation 3: '2', Action 3: 'look']\n[Observation 4: '3', Action 4: 'look']")

    def test_projection_is_upstream_not_fuzzy_recovery(self):
        act, valid = ev.project('alfworld', '<think>x</think><action>LOOK</action>', ['look'])
        self.assertEqual((act, valid), ('look', True))
        act, valid = ev.project('alfworld', '[action] look', ['look'])
        self.assertEqual(act, '[action] look')
        self.assertFalse(valid)

    def test_web_projection_takes_first_action(self):
        act, valid = ev.project('webshop', '<think>x</think><action>search[a]</action><action>search[b]</action>', [])
        self.assertEqual((act, valid), ('search[a]', True))

    def test_web_observation_format(self):
        self.assertEqual(ev.web_observation('WebShop [SEP] Instruction: [SEP] goal [SEP] Search', 'goal'), "'Search'")

    def test_overflow_is_rejected_before_model_request(self):
        agent = ev.Agent(SimpleNamespace(disable_thinking=False), Path('/tmp/unused'))
        agent.count = lambda messages: 4097
        with patch.object(ev, 'http') as api:
            with self.assertRaisesRegex(RuntimeError, '4096-token'):
                agent.generate([], 'task', 1)
            api.assert_not_called()

    def web(self, reward):
        backend = ev.Web.__new__(ev.Web)
        backend.args = SimpleNamespace(max_steps=1)
        backend.data = {'items': 'a', 'attributes': 'b'}
        goal = {'instruction_text': 'goal'}
        backend.env = SimpleNamespace(server=SimpleNamespace(goals=[goal]), instruction_text='goal',
            reset=lambda **kwargs: ('WebShop [SEP] Instruction: [SEP] goal [SEP] buy now', None),
            get_available_actions=lambda: {'has_search_bar': False, 'clickables': ['buy now']},
            step=lambda action: ('terminal purchase state', reward, True, {}))
        agent = FakeAgent()
        agent.bounded_native = lambda *args: ([{'role':'user','content':'prompt'}], 0)
        agent.generate = lambda *args: ('<think>x</think><action>click[buy now]</action>', {})
        row = dict(task_id='webshop_0', index=0, data_sha256=backend.data,
            goal_sha256=ev.hashlib.sha256(ev.json.dumps(goal, sort_keys=True).encode()).hexdigest())
        return backend.run(row, agent)

    def test_web_partial_reward_is_not_success(self):
        result = self.web(.4)
        self.assertFalse(result['success'])
        self.assertEqual(result['score'], 40.0)
        self.assertEqual(result['steps'][0]['next_observation'], 'terminal purchase state')

    def test_web_perfect_purchase_is_success(self):
        self.assertTrue(self.web(1.0)['success'])

    def science(self, score, done):
        args = SimpleNamespace(science_url='http://fake', max_steps=2)
        backend = ev.Science(args)
        state = dict(task_name='boil', var_num=1, task_description='PERMANENT GOAL',
                     observation='first lab observation', done=False, score=0)
        def api(method, url, **kwargs):
            if url.endswith('/create'):
                return {'id': 1}
            if url.endswith('/reset'):
                return state.copy()
            if url.endswith('/admissible_actions'):
                return {'admissible_actions': ['look around', 'connect battery to wire']}
            if url.endswith('/step'):
                # Deliberately no task_description on step, matching AgentGym.
                return dict(observation='later lab observation', done=done, score=score)
            return True
        agent = FakeAgent()
        with patch.object(ev, 'http', api):
            result = backend.run(dict(task_id='sciworld_0', index=0, taskName='boil', variationIdx=1), agent)
        return result, agent

    def test_science_partial_score_is_not_success(self):
        result, _ = self.science(1.0, True)
        self.assertFalse(result['success'])
        self.assertEqual(result['score'], 1.0)

    def test_science_full_score_is_success(self):
        result, _ = self.science(100.0, True)
        self.assertTrue(result['success'])

    def test_science_goal_preserved_on_later_steps_and_circuit_actions_allowed(self):
        result, agent = self.science(10.0, False)
        self.assertIn('PERMANENT GOAL', agent.messages[1][-1]['content'])
        self.assertIn('connect', agent.messages[1][0]['content'])
        self.assertIn('connect battery to wire', result['steps'][1]['admissible_actions'])


if __name__ == '__main__':
    unittest.main()
