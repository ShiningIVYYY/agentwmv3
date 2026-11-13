"""CPU contracts; synthetic inputs here are never experimental evidence."""
import json,sys,unittest,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.alfworld_memory import match_rule,synthesize,AlfExperienceMemory
from worldmodel.alfworld_environment import fingerprint
from internal_signals.alfworld_prompts import draft_messages,review_messages,parse_review
from gates.core import RuleGate,GateInput,ExternalInput


class Contracts(unittest.TestCase):
    def test_three_way_goal_isolation(self):
        m=json.loads((ROOT/'docs/alfworld_pilot/split_manifest.json').read_text());sets=[{r['goal_group'] for r in m[k]} for k in ['memory_tasks','calibration_tasks','evaluation_tasks','final_valid_seen']]
        for i,a in enumerate(sets):
            for b in sets[i+1:]:self.assertFalse(a&b)
        self.assertEqual(len(m['final_valid_seen']),140)
    def test_no_test_game_selected(self):
        m=json.loads((ROOT/'docs/alfworld_pilot/split_manifest.json').read_text())
        for key in ['memory_tasks','calibration_tasks','evaluation_tasks']:
            self.assertTrue(all(r['game'].startswith('json_2.1.1/train/') for r in m[key]))
    def test_aligned_alf_limits(self):
        p=json.loads((ROOT/'configs/evaluation/alfworld_pilot_v1.json').read_text());self.assertEqual((p['maximum_interaction_turns'],p['history_length'],p['prompt_limit'],p['response_limit']),(50,2,2048,512))
        self.assertEqual(p['wm_invocation_budgets'],dict(never=0,always=50,internal=5,experience=5,full=5))
    def test_draft_has_native_commands_and_fields(self):
        text=draft_messages('goal','observation',[],['go to fridge 1'])[0]['content'];self.assertIn('Premise:',text);self.assertIn('Expected Consequence:',text);self.assertIn('go to fridge 1',text)
    def test_history_two_only(self):
        h=[dict(observation='old'+str(i),action='look') for i in range(4)];s=draft_messages('g','o',h,[])[0]['content'];self.assertNotIn('old0',s);self.assertNotIn('old1',s);self.assertIn('old2',s)
    def test_reviewer_does_not_receive_private_digest(self):
        sim=dict(status='ok',prediction_available=True,predicted_observation='public',predicted_reward=0,predicted_done=False,predicted_score=0,private_state_sha256='PRIVATE_CANARY')
        s=review_messages('g','o',[],['look'],'look',sim)[0]['content'];self.assertNotIn('PRIVATE_CANARY',s);self.assertIn('public',s)
    def test_unverified_trial_not_reviewed(self):
        with self.assertRaises(ValueError):review_messages('g','o',[],[],'look',dict(prediction_available=False))
    def test_original_reviewer_parser(self):
        self.assertEqual(parse_review('Decision: KEEP\nReason: grounded\nFinal Action: look','look')['decision'],'KEEP')
    def test_invalid_reviewer_explicit(self):self.assertEqual(parse_review('oops','look')['decision'],'INVALID')
    def test_private_facts_affect_identity(self):self.assertNotEqual(fingerprint(dict(observation='same',private_state_sha256='a')),fingerprint(dict(observation='same',private_state_sha256='b')))
    def test_open_conflict_requires_explicit_state(self):
        self.assertIsNotNone(match_rule('repeat_open','You open the fridge 1.','open fridge 1',[]));self.assertIsNone(match_rule('repeat_open','Nothing happens.','open fridge 1',[]))
    def test_no_conflict_for_current_admissible_command(self):self.assertIsNone(match_rule('repeat_open','You open the fridge 1.','open fridge 1',['open fridge 1']))
    def test_closed_take_condition(self):self.assertIsNotNone(match_rule('closed_take','The fridge 1 is closed.','take apple 1 from fridge 1',[]))
    def test_one_episode_cannot_make_supported_card(self):
        c=dict(task_id='a',source_split='train',wm_called=False,decision_at_ns=1,outcome_available_at_ns=2,probe_kind='repeat_open',environment_feedback='Nothing happens.',episode_id='e',case_id='e/1')
        self.assertEqual(synthesize([c,c],[dict(task_id='a')])['cards'],[])
    def test_illegal_case_source_rejected(self):
        with self.assertRaises(ValueError):synthesize([dict(task_id='test',source_split='test',wm_called=False)],[])
    def test_unknown_signal_abstains(self):
        snapshot=dict(schema='alfworld-experience-snapshot-v1',cross_domain_memory=False,cases=[],cards=[],built_at_ns=1)
        m=AlfExperienceMemory(snapshot,dict(memory_tasks=[]));q=dict(task_id='d',episode_id='ep',goal='g',observation='o',draft_action='look',commands=['look'],available_at_ns=2)
        signal,resolved=m.evaluate(q);self.assertEqual(signal['conflict'],'unknown');self.assertFalse(resolved.evidence_verified)
        with self.assertRaises(ValueError):m.evaluate(dict(q,environment_feedback='FUTURE'))


if __name__=='__main__':unittest.main()
