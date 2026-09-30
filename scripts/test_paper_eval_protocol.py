#!/usr/bin/env python3
"""Offline protocol regression tests; no model/GPU/JVM required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

path = Path(__file__).resolve().with_name('run_audited_paper_eval.py')
spec = importlib.util.spec_from_file_location('evaluator', path)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
super_spec = importlib.util.spec_from_file_location('supervisor', path.with_name('supervise_paper_eval.py'))
supervisor = importlib.util.module_from_spec(super_spec)
super_spec.loader.exec_module(supervisor)


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
    def test_supervisor_web_manifest_is_isolated(self):
        self.assertEqual(supervisor.manifest_path(Path('/data'), 'webshop'),
                         Path('/data/webshop_index_v2.json'))
        self.assertEqual(supervisor.manifest_path(Path('/data'), 'alfworld'),
                         Path('/data/alfworld.json'))
        self.assertEqual(supervisor.manifest_path(Path('/data'), 'scienceworld'),
                         Path('/data/scienceworld_context_v2.json'))

    def test_web_only_summary_and_read_only_combined_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            reference = Path(tmp) / 'old'
            out.mkdir()
            reference.mkdir()
            jobs = {'webshop_' + m: {'expected': 500} for m in ('qwen25', 'qwen38', 'ee_il')}
            supervisor.update(out, jobs, {})
            self.assertIn('| Model | WebShop success (index-v2) |', (out / 'RESULTS.md').read_text())
            old_stats = {b + '_' + m: dict(status='running', completed=1, total=200,
                successes=0, errors=0, success_rate=0.0, average_native_score=0.0)
                for b in ('alfworld', 'scienceworld') for m in ('qwen25', 'qwen38', 'ee_il')}
            original = ev.json.dumps({'results': old_stats})
            (reference / 'summary.json').write_text(original)
            supervisor.update(out, jobs, {}, reference)
            self.assertIn('ALFWorld success', (out / 'RESULTS.md').read_text())
            self.assertEqual((reference / 'summary.json').read_text(), original)
            self.assertEqual(set(ev.json.loads((out / 'summary.json').read_text())['results']), set(jobs))

    def index(self, ids):
        return SimpleNamespace(num_docs=len(ids), doc=lambda i:
            SimpleNamespace(raw=lambda: ev.json.dumps({'id': ids[i]})))

    def test_web_index_accepts_exact_catalog(self):
        self.assertEqual(ev.validate_web_index(self.index(['b', 'a']), ['a', 'b']),
                         ev.hashlib.sha256(b'a\nb').hexdigest())

    def test_web_index_rejects_wrong_size(self):
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
            ev.validate_web_index(self.index(['a', 'b', 'c']), ['a', 'b'])

    def test_web_index_rejects_wrong_ids(self):
        with self.assertRaisesRegex(RuntimeError, 'ASIN mismatch'):
            ev.validate_web_index(self.index(['a', 'c']), ['a', 'b'])

    def test_web_index_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(RuntimeError, 'ASIN mismatch'):
            ev.validate_web_index(self.index(['a', 'a']), ['a', 'b'])

    def test_web_protocol_is_separate_from_old_results(self):
        self.assertEqual(ev.protocol_for_env('webshop'), 'webshop-index-v2')
        self.assertEqual(ev.protocol_for_env('scienceworld'), 'scienceworld-context-v2')

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
        actual = agent.messages[1]
        self.assertEqual(sum('PERMANENT GOAL' in m['content'] for m in actual), 1)
        self.assertEqual(actual[-1]['content'], 'later lab observation')
        self.assertIn('connect', agent.messages[1][0]['content'])
        self.assertIn('connect battery to wire', result['steps'][1]['admissible_actions'])

    def test_science_demo_is_full_released_training_dialogue(self):
        backend = ev.Science(SimpleNamespace(science_url='http://fake'))
        self.assertGreater(len(backend.prelude), 4)
        self.assertIn('activate stove', '\n'.join(m['content'] for m in backend.prelude))

    def test_science_retains_all_history_when_it_fits(self):
        steps = [dict(observation='obs%d' % i, output='Thought: reason\nAction: action%d' % i) for i in range(8)]
        messages, keep = ev.bounded_science(FakeAgent(), [], 'GOAL', 'current', steps)
        self.assertEqual(keep, 8)
        self.assertEqual(len(messages), 17)
        self.assertEqual(messages[0]['content'], 'GOAL\nobs0')
        self.assertEqual(messages[-1]['content'], 'current')
        self.assertEqual(sum('GOAL' in m['content'] for m in messages), 1)
        self.assertEqual([m['role'] for m in messages], ['user','assistant'] * 8 + ['user'])

    def test_science_overflow_drops_whole_oldest_turns_only(self):
        steps = [dict(observation='obs%d' % i, output='action%d' % i) for i in range(8)]
        agent = FakeAgent()
        agent.count = lambda messages: len(messages) * 500
        messages, keep = ev.bounded_science(agent, [], 'GOAL', 'current', steps)
        self.assertEqual(keep, 3)
        self.assertEqual(messages[0]['content'], 'GOAL\nobs5')
        self.assertNotIn('obs4', str(messages))
        self.assertEqual(messages[-1]['content'], 'current')

    def test_science_oversized_current_state_is_infrastructure_error(self):
        agent = FakeAgent(); agent.count = lambda messages: 4097
        with self.assertRaisesRegex(RuntimeError, '4096'):
            ev.bounded_science(agent, [], 'GOAL', 'current', [])

    def test_science_official_task_validation(self):
        rows = [dict(item_id='sciworld_%d' % i) for i in range(200)]
        games = [dict(taskName='boil', variationIdx=i) for i in range(200)]
        frozen = ev.validate_science_tasks(rows, games)
        self.assertEqual(len(frozen), 200)
        self.assertEqual(ev.validate_science_tasks(rows, games, frozen), frozen)
        with self.assertRaisesRegex(RuntimeError, 'unique'):
            ev.validate_science_tasks(rows[:199] + rows[:1], games)
        with self.assertRaisesRegex(RuntimeError, 'out of range'):
            ev.validate_science_tasks(rows, games[:199])
        frozen[0]['variationIdx'] = 1
        with self.assertRaisesRegex(RuntimeError, 'mapping changed'):
            ev.validate_science_tasks(rows, games, frozen)

    def test_science_checks_goal_hash_before_inference(self):
        backend = ev.Science(SimpleNamespace(science_url='http://fake', max_steps=2))
        def api(method, url, **kwargs):
            if url.endswith('/create'): return {'id': 1}
            if url.endswith('/reset'): return dict(task_name='boil', var_num=1, task_description='changed')
            return True
        agent = FakeAgent()
        with patch.object(ev, 'http', api), self.assertRaisesRegex(AssertionError, 'goal changed'):
            backend.run(dict(task_id='sciworld_0', index=0, taskName='boil', variationIdx=1,
                             task_description_sha256='wrong'), agent)
        self.assertEqual(agent.messages, [])

    def test_multiple_reference_summaries_do_not_replace_current_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'science'; old = Path(tmp) / 'old'; web = Path(tmp) / 'web'
            for p in (out,old,web): p.mkdir()
            def stat(total):
                return dict(status='completed',completed=total,total=total,successes=0,errors=0,
                            success_rate=0.,average_native_score=0.)
            ev.atomic(old/'summary.json',dict(results={b+'_'+m:stat(140 if b=='alfworld' else 200)
                for b in ('alfworld','scienceworld') for m in ('qwen25','qwen38','ee_il')}))
            ev.atomic(web/'summary.json',dict(results={'webshop_'+m:stat(500) for m in ('qwen25','qwen38','ee_il')}))
            jobs = {'scienceworld_'+m:dict(expected=200) for m in ('qwen25','qwen38','ee_il')}
            stats = supervisor.update(out,jobs,{},[old,web])
            self.assertEqual(stats['scienceworld_ee_il']['completed'],0)
            table = (out/'RESULTS.md').read_text()
            self.assertIn('ALFWorld success',table); self.assertIn('WebShop success',table)
            self.assertIn('waiting; 0/200',table)


if __name__ == '__main__':
    unittest.main()
