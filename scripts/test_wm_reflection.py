#!/usr/bin/env python3
"""Behavioral checks for bounded reflection and isolated prediction."""
import copy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflection.controller import ReflectionController, parse_review, valid_replacement
from worldmodel.backends import (OracleWorldModel, Word2WorldModel, PredictionError,
                                 RandomIsolation, same_transition, science_observation_key)
from worldmodel.contexts import alf_action, checkpoint_observation, initial_context, ScienceActionView
from worldmodel import runtime


def transition(n):
    return dict(observation='state %d' % n, raw_observation='state %d' % n,
                score=n, done=False, success=False, wm_observation='state %d' % n)


class Agent:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.messages = []

    def count(self, messages):
        return 100

    def generate(self, messages, *args):
        self.messages.append(messages)
        return json.dumps(next(self.decisions)), {'usage': {'total_tokens': 10}}


def decision(kind, action):
    return dict(decision=kind, action=action, reason='reviewed the consequence')


class MemoryWM:
    backend = 'oracle'
    def __init__(self):
        self.actions = []
    def predict(self, action):
        self.actions.append(action)
        return dict(observation='hypothetical ' + action, seconds=0)


class ReflectionTests(unittest.TestCase):
    def controller(self, decisions, wm=None):
        agent = Agent(decisions)
        wm = wm or MemoryWM()
        return ReflectionController(wm, agent, 'alfworld', 'GOAL', 'test'), wm, agent

    def test_keep_early_stops(self):
        c, wm, agent = self.controller([decision('keep', 'look')])
        chosen, log = c.select('look', 'current', [], ['look', 'inventory'], 1)
        self.assertEqual(chosen, 'look')
        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(wm.actions, ['look'])
        self.assertFalse(log['hit_round_limit'])

    def test_three_replacements_four_predictions_no_fourth_review(self):
        c, wm, agent = self.controller([decision('replace', a) for a in ('inventory', 'go to cabinet 1', 'look')])
        chosen, log = c.select('go to fridge 1', 'current', [],
                               ['inventory', 'go to cabinet 1', 'look', 'go to fridge 1'], 2)
        self.assertEqual(chosen, 'look')
        self.assertEqual(len(agent.messages), 3)
        self.assertEqual(wm.actions, ['go to fridge 1', 'inventory', 'go to cabinet 1', 'look'])
        self.assertTrue(log['hit_round_limit'])
        for messages in agent.messages:
            data = json.loads(messages[0]['content'].split('INPUT DATA:\n')[1])
            self.assertEqual(data['current_observation'], 'current')
            self.assertEqual(data['executed_history'], [])

    def test_replacement_then_keep(self):
        c, wm, agent = self.controller([decision('replace', 'inventory'), decision('keep', 'inventory')])
        chosen, log = c.select('look', 'current', [], ['look', 'inventory'], 1)
        self.assertEqual(chosen, 'inventory')
        self.assertEqual(log['replacements'], 1)
        self.assertEqual(len(wm.actions), 2)

    def test_oracle_privileged_fields_not_given_to_agent(self):
        wm = MemoryWM()
        wm.predict = lambda _: dict(observation='ordinary feedback', score='SECRET SCORE',
                                  hidden_state='SECRET ROOM', seconds=0, success=True)
        c, _, agent = self.controller([decision('keep', 'look')], wm)
        c.select('look', 'current', [], ['look'], 1)
        content = agent.messages[0][0]['content']
        self.assertNotIn('SECRET', content)
        self.assertNotIn('"success":', content)

    def test_invalid_replacements_consume_rounds_without_fabricating_an_action(self):
        c, wm, agent = self.controller([decision('replace', 'teleport moon')] * 3)
        chosen, log = c.select('look', 'current', [], ['look'], 1)
        self.assertEqual(chosen, 'look')
        self.assertEqual(len(agent.messages), 3)
        self.assertEqual(log['invalid_reviews'], 3)
        self.assertEqual(log['replacements'], 0)
        self.assertIn('invalid', agent.messages[-1][0]['content'])

    def test_empty_checkpoint_is_unknown_evidence_and_can_be_reflected(self):
        wm = MemoryWM()
        wm.predict = lambda _: dict(observation='', prediction_available=False,
            observation_complete=False, empty_output=True, seconds=0)
        c, _, agent = self.controller([decision('replace', 'inventory'), decision('keep', 'inventory')], wm)
        chosen, log = c.select('bad tagged action', 'CURRENT REAL STATE', [], ['inventory'], 1)
        self.assertEqual(chosen, 'inventory')
        self.assertEqual(log['rounds'], 2)
        content = agent.messages[0][0]['content']
        self.assertIn('UNKNOWN', content)
        self.assertIn('CURRENT REAL STATE', content)
        self.assertNotIn('prediction hit its token limit', content)
        self.assertNotIn('hypothetical', content)

    def test_required_fields_overflow_is_error(self):
        c, wm, agent = self.controller([])
        agent.count = lambda _: 4097
        with self.assertRaises(PredictionError):
            c.select('look', 'current', [], ['look'], 1)
        self.assertEqual(agent.messages, [])

    def test_drop_old_history_preserves_candidate_prediction_and_goal(self):
        c, wm, agent = self.controller([decision('keep', 'look')])
        agent.count = lambda messages: 5000 if 'OLD HISTORY' in messages[0]['content'] else 100
        old = dict(observation='OLD HISTORY', action='inventory', next_observation='old result')
        _, log = c.select('look', 'current', [old], ['look'], 1)
        self.assertEqual(log['reviews'][0]['history_length'], 0)
        content = agent.messages[0][0]['content']
        self.assertIn('GOAL', content)
        self.assertIn('hypothetical look', content)

    def test_strict_json_and_keep_consistency(self):
        for bad in ('```json\n{}\n```', '{}', '[]', json.dumps(decision('keep', 'inventory')),
                    json.dumps(decision('replace', 'look'))):
            with self.subTest(bad=bad), self.assertRaises(PredictionError):
                parse_review(bad, 'look')

    def test_round_limits(self):
        for n in (0, 4, -1):
            with self.assertRaises(ValueError):
                ReflectionController(MemoryWM(), Agent([]), 'alfworld', 'goal', 'test', n)

    def test_environment_action_grammar(self):
        self.assertTrue(valid_replacement('webshop', 'search[red shoes]', ['search[<your query>]']))
        self.assertFalse(valid_replacement('webshop', 'click[unknown]', ['click[buy now]']))
        self.assertTrue(valid_replacement('webshop', 'click[next >]', ['click[next >]']))
        self.assertTrue(valid_replacement('scienceworld', 'connect wire to battery', []))
        self.assertFalse(valid_replacement('scienceworld', 'Action: teleport moon', []))


class ScienceExitTests(unittest.TestCase):
    HEAD = 'Inside the hallway is: \n\tthe agent\n\ta substance called air\n\ta painting\n'
    DOORS = ['A door to the bedroom (that is open)',
             'A door to the kitchen (that is closed)',
             'A door to the workshop (that is closed)']

    def room(self, doors=None):
        return self.HEAD + '\n You also see: ' + ', '.join(self.DOORS if doors is None else doors) + '.'

    def state(self, text):
        return dict(observation=text, raw_observation=text, native_observation=text,
                    score=0, reward=0, done=False, success=False, wm_observation=text)

    def test_all_exit_permutations_are_equivalent(self):
        from itertools import permutations
        expected = self.state(self.room())
        for doors in permutations(self.DOORS):
            self.assertTrue(same_transition(expected, self.state(self.room(doors))))

    def test_real_door_changes_and_multiplicity_still_fail(self):
        expected = self.state(self.room())
        variants = [self.DOORS[:-1], self.DOORS + [self.DOORS[0]],
                    [d.replace('open', 'closed') for d in self.DOORS],
                    [d.replace('kitchen', 'art studio') for d in self.DOORS]]
        for doors in variants:
            with self.subTest(doors=doors):
                self.assertFalse(same_transition(expected, self.state(self.room(doors))))

    def test_room_contents_and_numeric_feedback_still_fail(self):
        expected = self.state(self.room())
        self.assertFalse(same_transition(expected, self.state(self.room().replace('painting', 'book'))))
        for key, value in [('score', 1), ('reward', 1), ('done', True), ('success', True)]:
            actual = self.state(self.room(list(reversed(self.DOORS))))
            actual[key] = value
            self.assertFalse(same_transition(expected, actual))

    def test_unsupported_layouts_and_other_environments_stay_strict(self):
        a = self.state(self.room())
        b = self.state(self.room(list(reversed(self.DOORS))))
        a.pop('native_observation')
        b.pop('native_observation')
        self.assertFalse(same_transition(a, b))
        for text in ['You also see: ' + self.room(),
                     self.room().replace('A door to the kitchen', 'An exit to the kitchen'),
                     self.room() + '\n', self.room()[:-1]]:
            self.assertEqual(science_observation_key(text), text)

    def test_oracle_commit_and_replay_preserve_native_text(self):
        native = self.room()
        reordered = self.room(list(reversed(self.DOORS)))
        factory = lambda: (lambda action: self.state(reordered), lambda: None, self.state(reordered))
        wm = OracleWorldModel(factory, self.state(native))
        prediction = wm.predict('look around')
        self.assertEqual(prediction['observation'], reordered)
        wm.commit('look around', self.state(native))
        self.assertEqual(wm.history[0][1]['observation'], native)
        self.assertEqual(wm.predict('look around')['verified_prefix_steps'], 1)


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.closed = 0
        self.live = 0
        def factory():
            current = [0]
            def step(action):
                current[0] += int(action)
                return transition(current[0])
            def close():
                self.closed += 1
            return step, close, transition(0)
        self.factory = factory
        self.wm = OracleWorldModel(factory, transition(0))

    def test_rejected_candidates_leave_no_history(self):
        self.wm.predict('1')
        self.wm.predict('2')
        self.assertEqual(self.wm.history, [])
        self.assertEqual(self.live, 0)
        self.wm.commit('2', transition(2))
        prediction = self.wm.predict('3')
        self.assertEqual(prediction['observation'], 'state 5')
        self.assertEqual(prediction['verified_prefix_steps'], 1)
        self.assertEqual(self.closed, 3)

    def test_commit_divergence_fails(self):
        self.wm.predict('1')
        with self.assertRaises(PredictionError):
            self.wm.commit('1', transition(9))
        self.assertEqual(self.wm.history, [])

    def test_replay_divergence_fails_and_closes(self):
        self.wm.history = [('1', transition(3))]
        with self.assertRaises(PredictionError):
            self.wm.predict('1')
        self.assertEqual(self.closed, 1)

    def test_initial_divergence_fails_and_closes(self):
        self.wm.initial = transition(4)
        with self.assertRaises(PredictionError):
            self.wm.predict('1')
        self.assertEqual(self.closed, 1)

    def test_rng_restoration_including_on_exception(self):
        state = random.getstate()
        with self.assertRaises(RuntimeError):
            with RandomIsolation():
                random.seed(12345)
                random.random()
                raise RuntimeError()
        self.assertEqual(random.getstate(), state)


class RuntimeTests(unittest.TestCase):
    def test_science_menu_permutations_preserve_numbered_action_meaning(self):
        header = 'Ambiguous request: Please enter the number for the action you intended (or blank to cancel):\n'
        first, second = ScienceActionView(), ScienceActionView()
        a = first.observe(dict(observation=header + '0:\tfocus on zebra\n1:\tfocus on apple\n'))
        b = second.observe(dict(observation=header + '0:\tfocus on apple\n1:\tfocus on zebra\n'))
        self.assertEqual(a['observation'], b['observation'])
        self.assertEqual(first.action('0'), '1')
        self.assertEqual(second.action('0'), '0')
        self.assertNotEqual(a['native_observation'], b['native_observation'])
        self.assertTrue(valid_replacement('scienceworld', '0', ['0', '1']))
        self.assertFalse(valid_replacement('scienceworld', '0', ['look around']))
        first.observe(dict(observation='You focus on apple.'))
        self.assertEqual(first.action('0'), '0')

    def test_science_menu_preserves_indistinguishable_choices_and_native_numbers(self):
        header = 'Ambiguous request: Please enter the number for the action you intended (or blank to cancel):\n'
        native=header+'7:\tfocus on book\n3:\tfocus on book\n9:\tfocus on wood\n'
        view=ScienceActionView()
        observed=view.observe(dict(observation=native))
        self.assertEqual(observed['observation'],native)
        self.assertEqual(observed['native_observation'],native)
        self.assertEqual(view.mapping,{'7':'7','3':'3','9':'9'})
        self.assertEqual(view.action('7'),'7')
        self.assertEqual(view.action('3'),'3')

    def test_science_menu_still_rejects_duplicate_native_numbers(self):
        header = 'Ambiguous request: Please enter the number for the action you intended (or blank to cancel):\n'
        with self.assertRaises(PredictionError):
            ScienceActionView().observe(dict(observation=header+'0:\tfocus on book\n0:\tfocus on book\n'))

    def test_science_duplicate_menu_mapping_resets_on_next_observation(self):
        header = 'Ambiguous request: Please enter the number for the action you intended (or blank to cancel):\n'
        view=ScienceActionView()
        view.observe(dict(observation=header+'0:\tfocus on book\n1:\tfocus on book\n'))
        view.observe(dict(observation='You focus on the book.'))
        self.assertEqual(view.mapping,{})

    def test_committed_action_reason_survives_later_invalid_reviews(self):
        from types import SimpleNamespace
        reviews = [dict(decision=dict(decision='replace', action='inventory', reason='valid reason')),
                   dict(decision=dict(decision='invalid', action='inventory', reason='validation error'))]
        controller = SimpleNamespace(env='scienceworld', select=lambda *args: ('inventory', dict(reviews=reviews)))
        chosen, output, metadata = runtime.select(controller, 'look around', 'original', 'room', [], [], 1)
        self.assertEqual(chosen, 'inventory')
        self.assertEqual(output, 'Thought: valid reason\nAction: inventory')
        self.assertEqual(metadata['initial_output'], 'original')

    def test_wrong_service_context_window_fails_before_prediction(self):
        from types import SimpleNamespace
        config = dict(url='http://fake', served_model_name='wm', checkpoint='/tmp/wm', max_model_len=32768)
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(data=[
            dict(id='wm', root='/tmp/wm', max_model_len=4096)]))
        args = SimpleNamespace(wm_backend='word2world', wm_config='fake.json')
        with patch('worldmodel.runtime.load_config', return_value=config), \
             patch('worldmodel.runtime.requests.get', return_value=response), \
             self.assertRaisesRegex(PredictionError, 'context window'):
            runtime.make_controller(args, 'alfworld', {}, 'task', {}, None, None, ROOT)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.wm = Word2WorldModel(dict(max_tokens=512, max_model_len=12288,
            served_model_name='wm', url='http://fake', provenance={}), 'INITIAL STATE', lambda _: 100)
        self.payloads = []
        class Response:
            def raise_for_status(self): pass
            def json(self):
                return dict(choices=[dict(message=dict(content='future [SUCCESS]'), finish_reason='stop')])
        def post(url, json, timeout):
            self.payloads.append(copy.deepcopy(json))
            return Response()
        self.post = post

    def test_predictions_are_branches_and_real_commit_is_teacher_forced(self):
        with patch('worldmodel.backends.requests.post', self.post):
            a = self.wm.predict('A')
            self.wm.predict('B')
            self.assertTrue(a['predicted_success'])
            self.assertEqual(a['observation'], 'future')
            self.assertEqual(len(self.wm.history), 1)
            self.wm.commit('B', dict(wm_observation='REAL FEEDBACK'))
            self.wm.predict('C')
        self.assertEqual(self.payloads[-1]['messages'][-2]['content'], 'REAL FEEDBACK')
        self.assertNotIn('future', str(self.payloads[-1]['messages']))
        self.assertEqual(self.payloads[0]['temperature'], 0)
        self.assertEqual(self.payloads[0]['max_tokens'], 512)

    def test_context_overflow_never_silently_truncates(self):
        self.wm.count = lambda _: 12000
        with self.assertRaises(PredictionError):
            self.wm.predict('A')

    def test_published_token_limit_returns_explicitly_partial_observation(self):
        class Response:
            def raise_for_status(self): pass
            def json(self):
                return dict(choices=[dict(message=dict(content='partial'), finish_reason='length')])
        with patch('worldmodel.backends.requests.post', return_value=Response()):
            prediction = self.wm.predict('A')
        self.assertEqual(prediction['observation'], 'partial')
        self.assertFalse(prediction['observation_complete'])
        self.assertEqual(prediction['finish_reason'], 'length')

    def test_empty_prediction_is_explicit_missing_evidence(self):
        class Response:
            def raise_for_status(self): pass
            def json(self):
                return dict(choices=[dict(message=dict(content=''), finish_reason='stop')])
        with patch('worldmodel.backends.requests.post', return_value=Response()):
            prediction = self.wm.predict('A')
        self.assertEqual(prediction['observation'], '')
        self.assertFalse(prediction['prediction_available'])
        self.assertTrue(prediction['empty_output'])
        self.assertEqual(len(self.wm.history), 1)

    def test_alf_action_adapter_only_changes_placement_prefix(self):
        self.assertEqual(alf_action('put mug 1 in/on table 1'), 'move mug 1 in/on table 1')
        self.assertEqual(alf_action('go to cupboard 1'), 'go to cupboard 1')
        self.assertEqual(checkpoint_observation('alfworld', 'room\nAVAILABLE ACTIONS: look', ''), 'room')

    def test_science_initial_context_has_the_actual_task_once(self):
        from types import SimpleNamespace
        config = dict(env='scienceworld')
        fake = SimpleNamespace(get_sciworld_description=lambda _: 'PRIVATE ROOM DATA')
        with patch('worldmodel.contexts.collector', return_value=fake):
            result = initial_context(config, dict(index=123), 'ACTUAL GOAL',
                                     dict(wm_observation='INITIAL OBSERVATION'))
        self.assertEqual(result.count('ACTUAL GOAL'), 1)
        self.assertIn('PRIVATE ROOM DATA', result)
        self.assertIn('INITIAL OBSERVATION', result)


if __name__ == '__main__':
    unittest.main()
