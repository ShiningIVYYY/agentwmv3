#!/usr/bin/env python3
"""Synthetic unit contracts only; never interpret fixtures as scientific results."""
import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.reviewer import PredictionOnly,ValidityChecked,official_validity
from analysis.reviewer_metrics import termination_causes,feedback_kind,verified_pair_utility
from diagnostics.initialization import observed_room,validated_room_view,teleport_command
from evaluation.profile import canonicalize_observation


class Model:
    backend='word2world'
    def __init__(self):self.calls=[]
    def predict(self,action):self.calls.append(action);return dict(observation='unit fixture',seconds=0,usage={'total_tokens':1})


class Original:
    def __init__(self,chosen='open door'):
        self.wm=Model();self.chosen=chosen;self.review_protocol='json';self.max_rounds=3;self.events=[]
    def recorder(self,*args):self.events.append(args)
    def select(self,candidate,*args):
        predictions=[dict(action=candidate,**self.wm.predict(candidate))]
        if self.chosen!=candidate:predictions.append(dict(action=self.chosen,**self.wm.predict(self.chosen)))
        return self.chosen,dict(initial_action=candidate,final_action=self.chosen,predictions=predictions,reviews=[{'output':'original raw JSON'}],rounds=1)
    def commit(self,action,actual,log):log['committed']=action;return log


class DiagnosticTests(unittest.TestCase):
    def test_membership_is_enumerated_not_semantic_proof(self):self.assertEqual(official_validity('open door',['open door']),'api_enumerated')
    def test_absence_remains_unknown(self):self.assertEqual(official_validity('go kitchen',[]),'unknown')
    def test_prediction_only_keeps_draft(self):
        original=Original();c=PredictionOnly(original);chosen,log=c.select('go kitchen','state',[],[],1)
        self.assertEqual(chosen,'go kitchen');self.assertEqual(log['reviews'],[]);self.assertEqual(original.wm.calls,['go kitchen'])
    def test_prediction_only_does_not_run_reviewer(self):
        original=Original();original.select=lambda *args: self.fail('Reviewer must not run')
        PredictionOnly(original).select('look around','state',[],[],1)
    def test_positive_listed_replacement_kept(self):
        chosen,log=ValidityChecked(Original()).select('go kitchen','state',[],['open door'],1)
        self.assertEqual(chosen,'open door');self.assertFalse(log['guard_blocked_unknown_replacement'])
    def test_unknown_replacement_reverts_without_fake_generation(self):
        original=Original();chosen,log=ValidityChecked(original).select('go kitchen','state',[],[],1)
        self.assertEqual(chosen,'go kitchen');self.assertEqual(log['reviews'][0]['output'],'original raw JSON')
        self.assertEqual(log['official_action_validity'],'unknown');self.assertTrue(log['predictions'][-1]['cached'])
    def test_cached_revert_has_no_new_model_call(self):
        original=Original();_,log=ValidityChecked(original).select('go kitchen','state',[],[],1)
        self.assertEqual(original.wm.calls,['go kitchen','open door'])
        self.assertEqual(sum(not p.get('cached',False) for p in log['predictions']),2)
    def test_original_unknown_not_claimed_safe(self):
        _,log=ValidityChecked(Original()).select('invented action','state',[],[],1)
        self.assertEqual(log['guard_original_draft_validity'],'unknown');self.assertTrue(log['semantic_utility_not_certified'])
    def test_kept_original_does_not_add_prediction(self):
        original=Original('look around');_,log=ValidityChecked(original).select('look around','state',[],[],1)
        self.assertFalse(log['guard_blocked_unknown_replacement']);self.assertEqual(len(log['predictions']),1)
    def test_commit_delegates_real_feedback(self):
        wrapper=ValidityChecked(Original());self.assertEqual(wrapper.commit('wait',{'score':0},{})['committed'],'wait')
    def test_preserves_protocol_attributes(self):
        wrapper=PredictionOnly(Original());self.assertEqual(wrapper.max_rounds,3);self.assertEqual(wrapper.review_protocol,'json')


class NativeAndPairTests(unittest.TestCase):
    def native(self,**kw):return dict(negative_score_termination=False,simulator_completed=False,sdk_clock_limit_exceeded=False,horizon_truncated=False,**kw)
    def pair(self):
        a=dict(task_id='unit-fixture',source_step=3,remaining_actor_budget=18,initial_score=0,initial_simulator_moves=2,
            restore_provenance=dict(heap_matches=True,snapshot_id='unit-0'),final_score=0)
        return a,copy.deepcopy(a)
    def test_clock_is_not_task_completed(self):
        n=self.native();n['sdk_clock_limit_exceeded']=True;self.assertEqual(termination_causes(n),['SDK_simulator_clock_limit'])
    def test_negative_score_separate_from_clock(self):
        n=self.native();n['negative_score_termination']=True;self.assertEqual(termination_causes(n),['negative_native_score'])
    def test_horizon_is_not_success(self):
        n=self.native();n['horizon_truncated']=True;self.assertEqual(termination_causes(n),['actor20_turn_cap'])
    def test_unknown_cause_not_invented(self):self.assertEqual(termination_causes(self.native()),[])
    def test_unknown_object_not_assumed_bad_syntax(self):self.assertEqual(feedback_kind('No known action matches that input.',0),'parser_or_object_resolution_failure')
    def test_successful_focus_text_can_be_task_failure(self):self.assertEqual(feedback_kind('You focus on the object.',-100),'observed_task_score_decrease')
    def test_verified_pair_arithmetic(self):
        a,b=self.pair();b['final_score']=25;self.assertEqual(verified_pair_utility(a,b),25)
    def test_different_snapshot_rejected(self):
        a,b=self.pair();b['restore_provenance']['snapshot_id']='other'
        with self.assertRaises(ValueError):verified_pair_utility(a,b)
    def test_missing_heap_rejected(self):
        a,b=self.pair();b['restore_provenance']={}
        with self.assertRaises(ValueError):verified_pair_utility(a,b)
    def test_different_budget_rejected(self):
        a,b=self.pair();b['remaining_actor_budget']=17
        with self.assertRaises(ValueError):verified_pair_utility(a,b)
    def test_different_clock_rejected(self):
        a,b=self.pair();b['initial_simulator_moves']=3
        with self.assertRaises(ValueError):verified_pair_utility(a,b)
    def test_different_initial_score_rejected(self):
        a,b=self.pair();b['initial_score']=10
        with self.assertRaises(ValueError):verified_pair_utility(a,b)


class InitializationTests(unittest.TestCase):
    def test_only_official_teleport_used(self):self.assertEqual(teleport_command('kitchen',['teleport to kitchen']),'teleport to kitchen')
    def test_terminal_not_assumed_room(self):self.assertIsNone(teleport_command('terminal 1',['teleport to kitchen']))
    def test_terminal_cannot_label_kitchen(self):self.assertFalse(validated_room_view('terminal 1','This room is called the kitchen. In it, you see:'))
    def test_real_room_verified(self):self.assertTrue(validated_room_view('kitchen','This room is called the kitchen. In it, you see:'))
    def test_outside_verified(self):self.assertTrue(validated_room_view('outside','This outside location is called the outside. Here you see:'))
    def test_unknown_look_does_not_become_valid(self):self.assertIsNone(observed_room('No known action matches that input.'))
    def test_flat_order_not_factual_prediction_error(self):
        self.assertEqual(canonicalize_observation('room\n\ta zebra\n\tan apple'),canonicalize_observation('room\n\tan apple\n\ta zebra'))
    def test_changed_door_state_not_normalized_away(self):
        self.assertNotEqual(canonicalize_observation('A door (closed)'),canonicalize_observation('A door (open)'))


if __name__=='__main__':unittest.main(verbosity=2)
