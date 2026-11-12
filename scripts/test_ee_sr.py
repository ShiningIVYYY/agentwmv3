#!/usr/bin/env python3
"""CPU contract fixtures; no synthetic results enter trial metrics."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from internal_signals.ee_sr_prompt import native_messages,prompt,parse_action
from internal_signals.decision import messages
from internal_signals.ee_sr_actor import SRTokenNLLProvider,BACKEND,REVISION


class Contracts(unittest.TestCase):
    source=dict(instruction='GENERIC_INSTRUCTIONS',acknowledgement='GENERIC_ACK')
    def test_native_action(self):self.assertEqual(parse_action('Thought:\nI can open it.\n\nAction:\nopen door to kitchen')['action'],'open door to kitchen')
    def test_structured_action(self):self.assertEqual(parse_action('Premise: p\nExpected Consequence: c\nAction: look around')['action'],'look around')
    def test_ambiguous_number_preserved(self):self.assertEqual(parse_action('Action:\n2')['action'],'2')
    def test_no_action(self):self.assertIsNone(parse_action('Only thoughts')['action'])
    def test_multiple_actions(self):self.assertIsNone(parse_action('Action: a\nAction: b')['action'])
    def test_empty_action(self):self.assertIsNone(parse_action('Action:\n')['action'])
    def test_trailing_prose_not_silently_dropped(self):self.assertIsNone(parse_action('Action: a\nmore prose')['action'])
    def test_structured_prompt_identical_to_baseline(self):self.assertEqual(prompt('structured','g','o',[],self.source),messages('g','o',[],keep=2))
    def test_native_history_bound_no_expert_action(self):
        h=[dict(observation='DROP',action='oracle_not_visible'),dict(observation='o1',action='open door'),dict(observation='o2',action='go to hallway')]
        p=native_messages('g','now',h,self.source);s=json.dumps(p)
        self.assertNotIn('DROP',s);self.assertNotIn('oracle_not_visible',s);self.assertEqual(p[-1]['content'],'now');self.assertEqual(p[2]['content'],'g\no1')
    def test_native_single_initial_observation(self):self.assertEqual(native_messages('g','obs',[],self.source)[2]['content'],'g\nobs')
    def test_backend_identity(self):
        c=dict(backend=BACKEND,dtype='bfloat16',model_revision=REVISION,missing_policy='null');p=SRTokenNLLProvider(object(),c);self.assertEqual(p.backend,BACKEND)
    def test_old_backend_rejected(self):
        with self.assertRaises(ValueError):SRTokenNLLProvider(object(),dict(backend='hf_qwen35_generation_raw_bf16_v1',dtype='bfloat16',model_revision=REVISION,missing_policy='null'))
    def test_wrong_revision_rejected(self):
        with self.assertRaises(ValueError):SRTokenNLLProvider(object(),dict(backend=BACKEND,dtype='bfloat16',model_revision='main',missing_policy='null'))
    def test_trial_has_no_gate_or_old_thresholds(self):
        p=json.loads((ROOT/'configs/evaluation/ee_sr_actor_trial_v1.json').read_text());self.assertTrue(p['no_gate']);self.assertTrue(p['no_old_nll_threshold_reuse']);self.assertEqual(len(p['tasks']),3)
    def test_overlap_not_reported_as_exact_training_membership(self):
        d=json.loads((ROOT/'docs/ee_sr_trial/data_overlap.json').read_text());self.assertFalse(d['checkpoint_actual_training_row_membership_verified']);self.assertTrue(d['partitions']['test']['normalized_goal_overlap_ids'])


if __name__=='__main__':unittest.main()
