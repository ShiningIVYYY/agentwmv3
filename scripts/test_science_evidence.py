"""Regression cases for known bad overrides, metadata isolation and branching."""
import copy
import json
from types import SimpleNamespace
import unittest

from reflection.controller import ReflectionController
from reflection import science_evidence as se


def prediction(text, score=0, done=False, success=False):
    return dict(observation=text,score=score,done=done,success=success,reward=0,seconds=0)


class WM:
    def __init__(self, mapping, backend='oracle'):
        self.mapping,self.backend,self.calls=mapping,backend,[]
    def predict(self, action):
        self.calls.append(action)
        return copy.deepcopy(self.mapping[action])


class Agent:
    def __init__(self, outputs=()):
        self.outputs=list(outputs);self.calls=[]
    def count(self,messages):
        return len(json.dumps(messages))//4
    def generate(self,messages,*args,**kwargs):
        self.calls.append((copy.deepcopy(messages),kwargs))
        if not self.outputs:
            raise AssertionError('Unexpected model call')
        value=self.outputs.pop(0)
        return value if isinstance(value,tuple) else (value,dict(usage=dict(total_tokens=10),finish_reason='stop'))


def keep():
    return json.dumps(dict(decision='keep',issue='none',evidence_source='none',evidence='',reason='Useful setup'))


def replan(action):
    return '<think>Fix the verified problem</think>\n<action>'+action+'</action>'


def pair(a,b,choice='alternative'):
    return json.dumps(dict(choice=choice,fixes_original_issue=True,introduces_risk=False,
        original_evidence=a,alternative_evidence=b,reason='Alternative fixes the input error'))


class EvidenceTests(unittest.TestCase):
    def run_select(self,mapping,outputs=(),protocol='science-evidence-v1',backend='oracle',steps=(),original='move metal pot to red box'):
        wm=WM(mapping,backend);agent=Agent(outputs)
        c=ReflectionController(wm,agent,'scienceworld','REAL GOAL','dev_task',3,lambda *args:None,protocol)
        result,log=c.select(original,'CURRENT REAL STATE',list(steps),[],len(steps)+1)
        return result,log,agent,wm

    def test_demonstrated_success_cannot_be_vetoed(self):
        action='move metal pot to red box'
        result,log,agent,wm=self.run_select({action:prediction('You move the metal pot to the red box.',100,True,True)})
        self.assertEqual(result,action)
        self.assertEqual(agent.calls,[])
        self.assertIn('oracle_success_preserved',log['policy_events'])

    def test_positive_partial_or_zero_reward_is_not_a_defect(self):
        action='pick up metal pot'
        result,log,_,_=self.run_select({action:prediction('You move the metal pot to the inventory.',74)},[keep()],original=action)
        self.assertEqual(result,action)
        self.assertIsNone(log['original_defect'])

    def test_unsupported_criticism_does_not_interrupt_setup(self):
        action='pick up metal pot'
        critique=json.dumps(dict(decision='replan',issue='terminal_failure',evidence_source='prediction',
            evidence='You move the metal pot to the inventory.',reason='I would prefer looking around'))
        result,log,agent,_=self.run_select({action:prediction('You move the metal pot to the inventory.',74)},[critique],original=action)
        self.assertEqual(result,action)
        self.assertEqual(len(agent.calls),1)
        self.assertEqual(log['reviews'][0]['guard'],'unsupported_or_inconsistent_review')

    def test_fabricated_quote_is_rejected(self):
        d=dict(decision='replan',issue='harmful_change',evidence_source='prediction',
               evidence='The apparatus was destroyed.',reason='Dangerous')
        issue,guard=se.judge_critique(d,'activate stove',prediction('The stove is now activated.'),'oracle',[],True)
        self.assertIsNone(issue)
        self.assertEqual(guard,'ungrounded_review_quote')

    def test_invalid_input_can_be_repaired(self):
        a,b='focus on egg parrot','look around'
        mapping={a:prediction('No known action matches that input.'),b:prediction('This room is called the outside.')}
        result,log,_,_=self.run_select(mapping,[replan(b)],original=a)
        self.assertEqual(result,b)
        self.assertEqual(log['original_defect'],'invalid_input')

    def test_stage_two_compares_both_predictions(self):
        a,b='focus on egg parrot','look around'
        p1,p2='No known action matches that input.','This room is called the outside.'
        result,log,_,wm=self.run_select({a:prediction(p1),b:prediction(p2)},[replan(b),pair(p1,p2)],protocol='science-evidence-v2',original=a)
        self.assertEqual(result,b)
        self.assertEqual(wm.calls,[a,b])
        self.assertIn('comparison',log['reviews'][0])

    def test_uncertain_comparison_retains_original(self):
        a,b='focus on egg parrot','look around'
        p1,p2='No known action matches that input.','This room is called the outside.'
        result,log,_,wm=self.run_select({a:prediction(p1),b:prediction(p2)},
            [replan(b),pair(p1,p2,'uncertain'),replan(b),replan(b)],protocol='science-evidence-v2',original=a)
        self.assertEqual(result,a)
        self.assertEqual(log['rounds'],3)
        self.assertEqual(wm.calls,[a,b])
        self.assertEqual(log['predictions'][-1]['action'],a)
        self.assertTrue(log['predictions'][-1]['cached'])

    def test_terminal_failure_guard_is_not_limited_to_focus(self):
        a,b='mix bowl','look around'
        result,log,_,_=self.run_select({a:prediction('The mixture reacts.',-100,True,False),
            b:prediction('This room is called the kitchen.')},[replan(b)],original=a)
        self.assertEqual(result,b)
        self.assertEqual(log['original_defect'],'terminal_failure')

    def test_fallback_does_not_execute_known_terminal_failure(self):
        a,b='focus on chameleon','look around'
        p1,p2='You focus on the chameleon.','This room is called the outside.'
        result,log,_,_=self.run_select({a:prediction(p1,-100,True,False),b:prediction(p2)},
            [replan(b),pair(p1,p2,'uncertain'),replan(b),replan(b)],protocol='science-evidence-v2',original=a)
        self.assertEqual(result,b)
        self.assertIn('safe_fallback_from_verified_terminal_failure',log['policy_events'])

    def test_terminal_failure_alternative_never_replaces_original(self):
        a,b='focus on egg parrot','focus on chameleon'
        result,log,_,_=self.run_select({a:prediction('No known action matches that input.'),
            b:prediction('You focus on the chameleon.',-100,True,False)},[replan(b)]*3,original=a)
        self.assertEqual(result,a)
        self.assertEqual(log['predictions'][-1]['action'],a)

    def test_all_bad_replans_use_verified_nonfatal_observation_fallback(self):
        a,b='focus on mouse','focus on chameleon'
        result,log,_,_=self.run_select({a:prediction('You focus on the mouse.',-100,True,False),
            b:prediction('You focus on the chameleon.',-100,True,False),
            'look around':prediction('This room is called the outside.')},[replan(b)]*3,original=a)
        self.assertEqual(result,'look around')
        self.assertEqual(log['rounds'],3)
        self.assertIn('safe_fallback_from_verified_terminal_failure',log['policy_events'])

    def test_observation_only_decisions_are_invariant_to_hidden_labels(self):
        a='focus on chameleon';outputs=[keep()]
        results=[];prompts=[]
        for p in [prediction('You focus on the chameleon.',100,True,True),prediction('You focus on the chameleon.',-100,True,False)]:
            result,log,agent,_=self.run_select({a:p},outputs,protocol='science-evidence-v3-obs',original=a)
            results.append(result);prompts.append(agent.calls[0][0])
            self.assertFalse(log['transition_metadata_visible'])
            self.assertEqual(log['policy_events'],[])
        self.assertEqual(results,[a,a]);self.assertEqual(prompts[0],prompts[1])

    def test_word2world_never_treats_numeric_fields_as_ground_truth(self):
        a='focus on chameleon'
        result,log,_,_=self.run_select({a:prediction('You focus on the chameleon.',-100,True,False)},
            [keep()],backend='word2world',original=a)
        self.assertEqual(result,a);self.assertIsNone(log['original_defect'])
        self.assertNotIn('score',se.visible(prediction('Text',100,True,True),'word2world'))

    def test_missing_or_partial_forecast_does_not_justify_intervention(self):
        a='pick up metal pot'
        for key in ['prediction_available','observation_complete']:
            p=prediction('You move the pot.');p[key]=False
            result,log,agent,_=self.run_select({a:p},original=a)
            self.assertEqual(result,a);self.assertEqual(agent.calls,[])

    def test_waits_are_not_misclassified_as_no_progress_loops(self):
        for a in ['wait1','wait','activate stove','connect red wire to battery']:
            steps=[dict(action=a,next_observation='Nothing changes visibly.')]*3
            self.assertFalse(se.loop_evidence(a,steps))
        steps=[dict(action='look around',next_observation='An unchanged room.')]*3
        self.assertTrue(se.loop_evidence('look around',steps))

    def test_memory_is_only_committed_source_labelled_evidence(self):
        s=dict(action='use thermometer on tin',next_observation='The temperature is 232 degrees celsius.',observation='Before')
        rows=se.facts([s]+[dict(action='wait1',next_observation='Nothing visible.',observation='Before')]*12)
        self.assertEqual(rows[0]['step'],1)
        self.assertEqual(rows[0]['observed_excerpt'],s['next_observation'])
        self.assertNotIn('success',json.dumps(rows))
        self.assertNotIn('predicted',json.dumps(rows))

    def test_forecasts_branch_from_same_real_state_and_final_is_last(self):
        a,b='focus on egg parrot','look around'
        result,log,_,wm=self.run_select({a:prediction('No known action matches that input.'),b:prediction('This room is called the outside.')},[replan(b)],original=a)
        self.assertEqual(wm.calls,[a,b])
        self.assertEqual(log['predictions'][-1]['action'],result)
        self.assertLessEqual(log['rounds'],3)

    def test_protocol_cannot_be_used_in_other_benchmarks(self):
        with self.assertRaises(ValueError):
            ReflectionController(WM({}),Agent(),'webshop','goal','task',review_protocol='science-evidence-v1')


if __name__=='__main__':
    unittest.main()
