#!/usr/bin/env python3
"""CPU contract fixtures. Synthetic inputs here are never research measurements."""
import inspect,json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflection.replay_review_v2 import ReviewerProtocol,VERSION,parse_review,review_messages
from reflection.replay_review import parse_review as old_parse
from evaluation.independent_dev import run_episode


class Contracts(unittest.TestCase):
    def output(self,**kw):return json.dumps(dict(decision='KEEP',final_action='look around',reason_code='trial_supported',**kw))
    def valid(self,obj):return parse_review(json.dumps(obj),'look around')
    def test_keep(self):self.assertEqual(parse_review(self.output(),'look around')['decision'],'KEEP')
    def test_replace(self):self.assertEqual(self.valid(dict(decision='REPLACE',final_action='open door to hallway',reason_code='trial_failed'))['changed'],True)
    def test_json_whitespace_permitted(self):self.assertEqual(parse_review(' \n'+self.output()+'\n','look around')['decision'],'KEEP')
    def test_extra_key_rejected(self):self.assertEqual(self.valid(dict(decision='KEEP',final_action='look around',reason_code='trial_supported',reason='prose'))['decision'],'INVALID')
    def test_missing_key_rejected(self):self.assertEqual(self.valid(dict(decision='KEEP',final_action='look around'))['error'],'json_schema_keys')
    def test_duplicate_key_rejected(self):self.assertEqual(parse_review('{"decision":"KEEP","decision":"REPLACE"}','a')['error'],'invalid_or_incomplete_json')
    def test_markdown_not_salvaged(self):self.assertEqual(parse_review('```json\n'+self.output()+'\n```','look around')['decision'],'INVALID')
    def test_truncated_not_relabelled_keep(self):
        x=parse_review('{"decision":"REPLACE",','x');self.assertEqual(x['decision'],'INVALID');self.assertEqual(x['fallback'],'keep_draft');self.assertEqual(x['final_action'],'x')
    def test_reason_consistency(self):self.assertEqual(self.valid(dict(decision='KEEP',final_action='look around',reason_code='trial_harmful'))['decision'],'INVALID')
    def test_keep_cannot_change(self):self.assertEqual(self.valid(dict(decision='KEEP',final_action='inventory',reason_code='trial_supported'))['error'],'keep_changed_draft')
    def test_multiline_action_rejected(self):self.assertEqual(self.valid(dict(decision='REPLACE',final_action='open door\ngo hallway',reason_code='trial_failed'))['error'],'non_single_action_string')
    def test_action_empty_long_or_control(self):
        for action in ['','a'*257,'abc\x00']:
            self.assertEqual(self.valid(dict(decision='REPLACE',final_action=action,reason_code='trial_failed'))['decision'],'INVALID')
    def test_nonstring_rejected(self):self.assertEqual(self.valid(dict(decision='KEEP',final_action=True,reason_code='trial_supported'))['error'],'json_schema_types')
    def test_unicode_quotes_roundtrip(self):
        a='look at book ("Origin of Species")';x=parse_review(json.dumps(dict(decision='KEEP',final_action=a,reason_code='trial_supported')),a);self.assertEqual(x['final_action'],a)
    def test_no_private_feedback_leak(self):
        sim=dict(status='ok',prediction_available=True,predicted_observation='public feedback',predicted_reward=0,predicted_done=False,predicted_score=0,object_tree='SECRET_PRIVATE',future='UNSEEN_FUTURE')
        result=review_messages('goal','current',[], 'look around',sim)[0]['content'];self.assertNotIn('SECRET_PRIVATE',result);self.assertNotIn('UNSEEN_FUTURE',result)
    def test_history_bound(self):
        sim=dict(status='ok',prediction_available=True,predicted_observation='ok',predicted_reward=0,predicted_done=False,predicted_score=0)
        text=review_messages('g','o',[dict(observation='DROP_OLD',action='a'),dict(observation='keep1',action='b'),dict(observation='keep2',action='c')],'a',sim)[0]['content'];self.assertNotIn('DROP_OLD',text)
    def test_unverified_feedback_refused(self):
        with self.assertRaises(ValueError):review_messages('g','o',[],'a',dict(status='failed',prediction_available=False))
    def test_original_preserved(self):
        raw='Decision: KEEP\nReason: ok\nFinal Action: look around'
        self.assertEqual(ReviewerProtocol('original').parse(raw,'look around'),old_parse(raw,'look around'))
        self.assertEqual(old_parse(raw.replace('\nFinal','\n\nFinal'),'look around')['decision'],'INVALID')
    def test_generation_proxy_never_mutates_actor(self):
        actor=SimpleNamespace(config=dict(max_new_tokens=512,model_id='id',model_revision='rev'),model=object(),tokenizer=object(),device=object(),render=lambda m:m,sync=lambda:None,fingerprint='fp')
        c=dict(actor.config)
        def generate(proxy,messages,role):self.assertIs(proxy.model,actor.model);self.assertEqual(proxy.config['max_new_tokens'],192);return 'text',{}
        with patch('gates.generation.unscored_generate',side_effect=generate):ReviewerProtocol(VERSION).generate(actor,[],42)
        self.assertEqual(actor.config,c)
    def test_optional_episode_hook_default_unchanged(self):self.assertIsNone(inspect.signature(run_episode).parameters['reviewer'].default)
    def test_historical_pre_hash_uses_full_capture(self):
        from scripts.run_reviewer_repair import historical_pre_hash
        from worldmodel.ground_truth_environment import digest,state_fingerprint
        state=dict(transition=dict(observation='canonical',raw_observation='raw order'))
        self.assertEqual(historical_pre_hash(state),digest(state))
        self.assertNotEqual(historical_pre_hash(state),state_fingerprint(state))
    def test_historical_pre_hash_does_not_drop_raw(self):
        from scripts.run_reviewer_repair import historical_pre_hash
        a=dict(transition=dict(observation='same',raw_observation='a'))
        b=dict(transition=dict(observation='same',raw_observation='b'))
        self.assertNotEqual(historical_pre_hash(a),historical_pre_hash(b))
    def test_all_real_source_schemas_preflight_before_gpu(self):
        from scripts.run_reviewer_repair import preflight_cases
        p=json.loads((ROOT/'configs/evaluation/phase510_reviewer_repair_v1.json').read_text());preflight_cases(p)
    def test_missing_phase59_hash_is_not_silently_ignored(self):
        from scripts.run_reviewer_repair import historical_post_hash
        with self.assertRaises(KeyError):historical_post_hash({},dict(source='outputs/phase59_independent_v2/episodes/x/steps.jsonl'))
        self.assertIsNone(historical_post_hash({},dict(source='outputs/replay_fix_130_e2e_v1/evaluation_steps.jsonl')))
    def test_frozen_task_isolation_and_caps(self):
        p=json.loads((ROOT/'configs/evaluation/phase510_reviewer_repair_v1.json').read_text());old=json.loads((ROOT/'configs/evaluation/phase59_independent_dev_v1.json').read_text())
        self.assertFalse({r['task_id'] for r in p['tasks']}&{r['task_id'] for r in old['tasks']});self.assertEqual(p['unused_dev_goal_groups'],0)
        self.assertEqual(p['wm_invocation_budgets'],old['wm_invocation_budgets']);self.assertEqual(len(p['tasks']),3)
        self.assertEqual(ReviewerProtocol(VERSION).max_new_tokens,192);self.assertEqual(ReviewerProtocol('original').max_new_tokens,512)


if __name__=='__main__':unittest.main()
