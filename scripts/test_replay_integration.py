#!/usr/bin/env python3
"""CPU contract tests. Synthetic inputs here are NOT scientific outcomes."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflection.replay_review import parse_review, review_messages, PROTOCOL
from gates.core import RuleGate, GateInput, ExternalInput
from worldmodel.ground_truth_environment import differences


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/gates/dev_distribution_phase5_v1.json').read_text())
        self.config.update(review_protocol=PROTOCOL, max_forecasts_per_invocation=1)

    def test_single_forecast_budget(self):
        gate = RuleGate(self.config, 'always')
        self.assertTrue(gate.decide(GateInput('x', 1, remaining_forecast_budget=1))['call_wm'])
        self.assertFalse(gate.decide(GateInput('x', 1, remaining_forecast_budget=0))['call_wm'])

    def test_legacy_protocol_still_requires_four(self):
        self.config['review_protocol'] = 'json'
        with self.assertRaises(ValueError):
            RuleGate(self.config)

    def test_never_and_exhaustion(self):
        self.assertFalse(RuleGate(self.config, 'never').decide(GateInput('x', 20))['call_wm'])
        self.assertFalse(RuleGate(self.config, 'always').decide(GateInput('x', 0))['call_wm'])

    def test_unknown_and_missing_conservative(self):
        result = RuleGate(self.config, 'full').decide(GateInput('x', 1, external=ExternalInput()))
        self.assertTrue(result['call_wm'])
        self.assertEqual(result['experience_conflict'], 'unknown')

    def test_reconstruction_failure_never_reaches_reviewer(self):
        for value in [dict(status='reconstruction_failed',prediction_available=False),
                      dict(status='ok', prediction_available=False)]:
            with self.assertRaises(ValueError):
                review_messages('g','o',[],'look around',value)

    def test_hidden_audit_fields_never_reach_prompt(self):
        value = dict(status='ok', prediction_available=True, predicted_observation='door opens',
                     predicted_reward=0, predicted_done=False, predicted_score=0,
                     selected_rng='PRIVATE_SENTINEL', object_tree_sha256='PRIVATE_SENTINEL',
                     mismatch=dict(secret='PRIVATE_SENTINEL'))
        prompt = json.dumps(review_messages('g','o',[],'open door',value))
        self.assertNotIn('PRIVATE_SENTINEL', prompt)

    def test_keep_replace_and_format(self):
        self.assertEqual(parse_review('Decision: KEEP\nReason: observed evidence\nFinal Action: wait','wait')['final_action'],'wait')
        self.assertEqual(parse_review('Decision: REPLACE\nReason: current visible door\nFinal Action: open door','wait')['final_action'],'open door')
        self.assertEqual(parse_review('Decision: KEEP\nReason: x\nFinal Action: different','wait')['decision'],'INVALID')
        for text in ['bad', 'Decision: REPLACE\nReason: x\nFinal Action: a\nAction: b']:
            self.assertEqual(parse_review(text,'wait')['final_action'],'wait')

    def test_real_state_fields_not_ignored(self):
        state = dict(transition=dict(observation='o', raw_observation='o',score=0,done=False,moves=0,reward=0),
                     selected_rng={'seed':1},object_tree_sha256='a',inventory='empty')
        for field, value in [('selected_rng',{'seed':2}),('object_tree_sha256','b'),('inventory','item')]:
            other = copy.deepcopy(state)
            other[field] = value
            self.assertIn(field, differences(state,other))
        for key in ['score','done','moves','reward','observation']:
            other = copy.deepcopy(state)
            other['transition'][key] = 'different'
            self.assertIn('transition', differences(state,other))


if __name__ == '__main__':
    unittest.main()
