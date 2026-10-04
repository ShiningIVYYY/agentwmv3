#!/usr/bin/env python3
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflection.controller import ReflectionController
from reflection.agentwm import (PROTOCOL, POLICY, extract_object, parse_direct,
    parse_critique, parse_replan, feedback, reject_replan, REVIEW_SCHEMAS)
from worldmodel.backends import PredictionError
from worldmodel.runtime import metrics, select as runtime_select, review_protocol
from test_native_reflection import Agent, xml
from types import SimpleNamespace
from unittest.mock import patch
import tempfile


class ReviewAgent(Agent):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.schemas=[]
    def generate(self,messages,*args,response_schema=None):
        self.schemas.append(response_schema)
        return super().generate(messages,*args)


class WM:
    def __init__(self, observations, backend='oracle'):
        self.observations, self.backend, self.calls = observations, backend, []
    def predict(self, action):
        self.calls.append(action)
        value = self.observations[action]
        if isinstance(value, str):
            value = dict(observation=value)
        return dict(seconds=1, verified_prefix_steps=0, **value)


def direct(action, change=True):
    return json.dumps(dict(assessment='reviewed', action=action, change=change, reason='model reason'))


def critique(replan=True, suggestion='go to kitchen'):
    return json.dumps(dict(assessment='observed defect', execute_proposed=not replan,
                          needs_replan=replan, suggested_action=suggestion, reason='model review'))


class AgentWMTests(unittest.TestCase):
    def controller(self, forecasts, outputs, env='alfworld', backend='oracle'):
        wm, agent = WM(forecasts, backend), ReviewAgent(outputs)
        return ReflectionController(wm, agent, env, 'GOAL', 'task', review_protocol=PROTOCOL), wm, agent

    def test_exact_copies_match_frozen_origin_hashes(self):
        folder=Path(__file__).resolve().parents[1]/'reflection'
        provenance=json.loads((folder/'agentwm_prompt_provenance.json').read_text())
        self.assertEqual(len(provenance['files']),3)
        for row in provenance['files']:
            self.assertEqual(hashlib.sha256((folder/row['local_file']).read_bytes()).hexdigest(),row['sha256'])

    def test_alfworld_json_change_is_honored_without_v8_defect_gate(self):
        c,wm,agent=self.controller({'look':'valid useful view','inventory':'valid inventory'},
                                 [direct('inventory'),direct('inventory',False)])
        action,log=c.select('look','CURRENT',[],['look','inventory'],1)
        self.assertEqual(action,'inventory')
        self.assertEqual(wm.calls,['look','inventory'])
        self.assertEqual(log['selection_policy'],POLICY)
        self.assertIn('Is there another admissible action',agent.messages[0][-1]['content'])

    def test_alfworld_keep_uses_declared_change_false(self):
        self.assertEqual(parse_direct(direct('inventory',False),'look')[0]['action'],'look')

    def test_extracts_explicit_json_from_model_thought_but_never_native_action(self):
        self.assertEqual(extract_object('Reason</think>\n'+direct('look',False))['action'],'look')
        for text in (xml('look'),direct('look')+direct('inventory'),'{bad json}'):
            with self.assertRaises(PredictionError):extract_object(text)

    def test_oracle_metadata_visible_and_hidden_fields_excluded(self):
        c,wm,agent=self.controller({'look':dict(observation='preview',reward=1,done=True,success=True,score=100,hidden='PRIVATE')},[direct('look',False)])
        c.select('look','CURRENT',[],['look'],1)
        content=agent.messages[0][-1]['content']
        self.assertIn('- reward: 1',content)
        self.assertIn('- success: True',content)
        self.assertNotIn('PRIVATE',content)

    def test_learned_feedback_does_not_leak_real_labels(self):
        prediction=dict(observation='preview',reward=12345,score=12345,success=True,done=True,predicted_success=False)
        self.assertEqual(feedback(prediction,'word2world'),dict(reward=None,score=None,done=None,success=False))
        c,wm,agent=self.controller({'look':prediction},[direct('look',False)],backend='word2world')
        c.select('look','CURRENT',[],['look'],1)
        content=agent.messages[0][-1]['content']
        self.assertNotIn('12345',content)
        self.assertIn('None means UNKNOWN',content)

    def test_truncated_success_marker_is_not_assumed_false(self):
        self.assertIsNone(feedback(dict(predicted_success=False,observation_complete=False),'word2world')['success'])

    def test_science_keep_does_not_make_replan_call(self):
        c,wm,agent=self.controller({'go to hallway':'You move into hallway.'},[critique(False)],'scienceworld')
        action,log=c.select('go to hallway','CURRENT WORKSHOP',[],['go to hallway'],1)
        self.assertEqual(action,'go to hallway')
        self.assertEqual(len(agent.messages),1)
        self.assertNotIn('replan',log['reviews'][0])

    def test_science_replan_is_separate_call_and_suggestion_not_executed(self):
        c,wm,agent=self.controller({'open door to kitchen':'No known action matches that input.',
                                    'go to hallway':'You move into hallway.'},
            [critique(True,'go to kitchen'),xml('go to hallway')],'scienceworld')
        chosen,text,meta=runtime_select(c,'open door to kitchen','ORIGINAL','CURRENT WORKSHOP',[],['go to hallway'],1)
        self.assertEqual(chosen,'go to hallway')
        self.assertEqual(wm.calls,['open door to kitchen','go to hallway'])
        self.assertEqual(len(agent.messages),2)
        self.assertIn('Do not output the final replacement action',agent.messages[0][-1]['content'])
        self.assertIn('CURRENT WORKSHOP',agent.messages[1][-1]['content'])
        self.assertIn('Generate a replacement action for the current real environment state',agent.messages[1][-1]['content'])
        self.assertEqual(text,'Thought: The preview supports this choice.\nAction: go to hallway')
        self.assertEqual(meta['reflection']['reviews'][0]['replan']['output'],xml('go to hallway'))

    def test_science_native_checkpoint_replan_output_is_supported(self):
        d=parse_replan('Thought: move through current door\nAction: go to hallway','look around')
        self.assertEqual(d['action'],'go to hallway')
        with self.assertRaises(PredictionError):parse_replan(xml('go to hallway')+xml('look around'),'look around')

    def test_science_failed_replans_leave_original_cached_for_commit(self):
        c,wm,agent=self.controller({'open door to kitchen':'No known action matches that input.',
            'go to kitchen':'No known action matches that input.'},
            [critique(),xml('go to kitchen'),critique(False)],'scienceworld')
        chosen,log=c.select('open door to kitchen','CURRENT WORKSHOP',[],['go to kitchen'],1)
        self.assertEqual(chosen,'open door to kitchen')
        self.assertTrue(log['predictions'][-1]['cached'])
        self.assertEqual(log['predictions'][-1]['action'],chosen)
        self.assertEqual(len(wm.calls),2)

    def test_science_three_bad_replans_have_only_three_reviews_four_predictions(self):
        alternatives=['go to kitchen','go to bedroom','go to outside']
        c,wm,agent=self.controller({a:'No known action matches that input.' for a in ['open door to kitchen']+alternatives},
            [x for a in alternatives for x in (critique(),xml(a))],'scienceworld')
        chosen,log=c.select('open door to kitchen','CURRENT',[],alternatives,1)
        self.assertEqual(log['rounds'],3)
        self.assertEqual(len(agent.messages),6)
        self.assertEqual(len(wm.calls),4)
        self.assertEqual(chosen,'open door to kitchen')
        self.assertTrue(log['predictions'][-1]['cached'])
        log['prediction_exact_match']=True
        m=metrics([dict(reflection=log)])['reflection_metrics']
        self.assertEqual((m['reflection_calls'],m['replan_calls'],m['wm_calls']),(3,3,4))

    def test_terminal_wrong_focus_and_missing_partial_replans_are_rejected(self):
        self.assertEqual(reject_replan('focus on wrong',dict(observation='done',done=True,success=False,score=-100),'oracle'),'predicted_terminal_bad_focus')
        for value in (dict(observation='',prediction_available=False),dict(observation='seems okay',observation_complete=False)):
            self.assertIsNotNone(reject_replan('go to hallway',value,'word2world'))

    def test_webshop_copied_rules_and_native_action(self):
        c,wm,agent=self.controller({'search[x]':'Page 1','search[y]':'Page 1'},[xml('search[y]'),'<action>search[y]</action>'],'webshop')
        chosen,log=c.select('search[x]','CURRENT',[],['search[<your query>]'],1)
        self.assertEqual(chosen,'search[y]')
        self.assertIn('Your job is not to solve the task from scratch',agent.messages[0][-1]['content'])
        self.assertIn('Default decision is KEEP',agent.messages[0][-1]['content'])

    def test_invalid_current_replacement_does_not_get_executed(self):
        c,wm,agent=self.controller({'look':'view'},[direct('go to moon 1')]*3)
        chosen,log=c.select('look','CURRENT',[],['look'],1)
        self.assertEqual(chosen,'look')
        self.assertEqual(log['rounds'],3)
        self.assertTrue(all(r['error_kind']=='invalid_action' for r in log['reviews']))

    def test_direct_oscillation_stops_without_a_fourth_review(self):
        c,wm,agent=self.controller({'look':'view','inventory':'items'},[direct('inventory'),direct('look')])
        chosen,log=c.select('look','CURRENT',[],['look','inventory'],1)
        self.assertEqual(chosen,'inventory')
        self.assertEqual(log['reviews'][-1]['guard'],'oscillation_detected')
        self.assertEqual(len(agent.messages),2)

    def test_final_direct_replacement_is_predicted(self):
        actions=['look','inventory','go to desk 1','go to shelf 1']
        c,wm,agent=self.controller({a:'view' for a in actions},[direct(a) for a in actions[1:]])
        chosen,log=c.select('look','CURRENT',[],actions,1)
        self.assertEqual(chosen,actions[-1])
        self.assertEqual(len(wm.calls),4)
        self.assertEqual(log['predictions'][-1]['action'],chosen)

    def test_history_is_real_and_limited_to_five_steps(self):
        c,wm,agent=self.controller({'look':'FUTURE'},[direct('look',False)])
        steps=[dict(action='look',observation='OLD'+str(i),next_observation='REAL'+str(i)) for i in range(7)]
        c.select('look','CURRENT',steps,['look'],8)
        content=agent.messages[0][-1]['content']
        self.assertNotIn('REAL0',content)
        self.assertIn('REAL2',content)
        self.assertNotIn('Step 1 Observation: FUTURE',content)

    def test_prompt_overflow_keeps_original_with_honest_zero_reviews(self):
        c,wm,agent=self.controller({'look':'preview'},[])
        agent.count=lambda messages:5000
        chosen,log=c.select('look','CURRENT',[],['look'],1)
        self.assertEqual((chosen,log['rounds'],log['budget_skip']),('look',0,True))
        self.assertEqual(agent.messages,[])

    def test_v8_and_qwen_defaults_are_unchanged(self):
        self.assertEqual(review_protocol(SimpleNamespace(model='ee-alfworld')),'ee-conservative-v2')
        self.assertEqual(review_protocol(SimpleNamespace(model='qwen25')),'json')
        self.assertEqual(review_protocol(SimpleNamespace(model='ee-alfworld',reflection_review_protocol=PROTOCOL)),PROTOCOL)

    def test_structured_decoding_is_for_review_only_not_replanning(self):
        c,wm,agent=self.controller({'go to kitchen':'No known action matches that input.',
                                    'go to hallway':'hallway'},[critique(),xml('go to hallway')],'scienceworld')
        c.select('go to kitchen','CURRENT',[],['go to hallway'],1)
        self.assertEqual(agent.schemas,[REVIEW_SCHEMAS['scienceworld'],None])

    def test_real_client_schema_payload_does_not_change_sampling(self):
        import run_audited_paper_eval as ev
        args=SimpleNamespace(model='ee-alfworld-il',max_tokens=1024,disable_thinking=False,
                             generation_lock=None,url='http://test')
        with tempfile.TemporaryDirectory() as d:
            agent=ev.Agent(args,Path(d));agent.count=lambda messages:100
            with patch.object(ev,'http',return_value=dict(choices=[dict(message=dict(content=direct('look',False)),finish_reason='stop')],usage={})) as client:
                raw,meta=agent.generate([dict(role='user',content='UNCHANGED')],'task',1,response_schema=REVIEW_SCHEMAS['alfworld'])
                payload=client.call_args.kwargs['json']
                self.assertEqual((payload['temperature'],payload['top_p'],payload['seed'],payload['max_tokens']),(.4,1.,0,1024))
                self.assertEqual(payload['response_format']['json_schema']['schema'],REVIEW_SCHEMAS['alfworld'])
                self.assertEqual(payload['messages'][0]['content'],'UNCHANGED')
                self.assertEqual(meta['response_format'],payload['response_format'])


if __name__=='__main__':
    unittest.main()
