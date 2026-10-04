import copy
from pathlib import Path
import sys
import unittest
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflection.controller import ReflectionController, valid_replacement
from reflection.conservative import failure_feedback, original_defect, POLICY
from worldmodel.runtime import metrics, select as runtime_select
from test_native_reflection import Agent, xml


class WM:
    backend='oracle'
    def __init__(self, observations):
        self.observations=observations
        self.calls=[]
        self.history=[]
    def predict(self, action):
        self.calls.append(action)
        result=self.observations[action]
        if isinstance(result,str):result=dict(observation=result)
        return dict(seconds=0,verified_prefix_steps=len(self.history),**result)


class ConservativeTests(unittest.TestCase):
    def controller(self, forecasts, outputs, env='alfworld'):
        wm,agent=WM(forecasts),Agent(outputs)
        return ReflectionController(wm,agent,env,'GOAL','task',review_protocol='ee-conservative-v2'),wm,agent

    def test_correct_book_pickup_cannot_be_overwritten_by_next_step(self):
        c,wm,agent=self.controller({'take book 2 from bed 1':'You pick up the book 2 from the bed 1.'},[xml('go to sidetable 1')])
        action,log=c.select('take book 2 from bed 1','At bed 1, book visible.',[],['take book 2 from bed 1','go to sidetable 1'],2)
        self.assertEqual(action,'take book 2 from bed 1')
        self.assertEqual(log['original_defect'],None)
        self.assertEqual(log['replacements'],0)
        self.assertEqual(log['reviews'][0]['guard'],'no_observable_original_defect')
        self.assertEqual(log['reviews'][0]['model_proposal']['action'],'go to sidetable 1')
        self.assertEqual(wm.calls,['take book 2 from bed 1'])

    def test_actual_state_is_final_turn_and_future_not_in_history(self):
        c,wm,agent=self.controller({'go to hallway':'You move to the hallway.'},['Thought: inspect kitchen.\nAction: open door to kitchen'],'scienceworld')
        action,log=c.select('go to hallway','CURRENT BEDROOM',[],['go to hallway'],1)
        self.assertEqual(action,'go to hallway')
        content=agent.messages[0][-1]['content']
        self.assertLess(content.index('UNEXECUTED PREVIEW'),content.index('ACTUAL CURRENT STATE AND EXECUTED HISTORY'))
        self.assertTrue(content.endswith("GOAL\nCURRENT BEDROOM\n['go to hallway']\n[]"))
        self.assertIn('assume you have entered a room',content)

    def test_explicit_failure_gets_one_verified_repair_and_stops(self):
        c,wm,agent=self.controller({'open desk 1':'Nothing happens.','go to shelf 1':'You arrive at shelf 1.'},[xml('go to shelf 1')])
        action,log=c.select('open desk 1','CURRENT',[],['open desk 1','go to shelf 1'],1)
        self.assertEqual(action,'go to shelf 1')
        self.assertEqual(log['replacements'],1)
        self.assertEqual(log['rounds'],1)
        self.assertEqual(wm.calls,['open desk 1','go to shelf 1'])
        self.assertEqual(log['reviews'][-1]['guard'],'verified_repair')

    def test_future_room_repair_is_checked_against_same_prefix(self):
        c,wm,agent=self.controller({'open door to kitchen':'No known action matches that input.',
            'go to kitchen':'No known action matches that input.', 'go to hallway':'You move to the hallway.'},
            ['Thought: move to kitchen.\nAction: go to kitchen','Thought: move through hallway first.\nAction: go to hallway'],'scienceworld')
        action,log=c.select('open door to kitchen','CURRENT WORKSHOP',[],['go to hallway'],1)
        self.assertEqual(action,'go to hallway')
        self.assertEqual(len(agent.messages),2)
        self.assertTrue(all(p['verified_prefix_steps']==0 for p in log['predictions']))
        self.assertNotIn('You move to the hallway.',agent.messages[0][-1]['content'])

    def test_repeated_rejected_proposals_do_not_oscillate(self):
        c,wm,agent=self.controller({'open desk 1':'Nothing happens.','open shelf 1':'Nothing happens.'},[xml('open shelf 1'),xml('open shelf 1')])
        action,log=c.select('open desk 1','CURRENT',[],['open desk 1','open shelf 1'],1)
        self.assertEqual(action,'open desk 1')
        self.assertEqual(log['rounds'],2)
        self.assertEqual(wm.calls,['open desk 1','open shelf 1'])
        self.assertTrue(log['predictions'][-1]['cached'])
        self.assertEqual(log['predictions'][-1]['action'],action)

    def test_three_failed_proposals_four_calls_cached_original_fallback(self):
        choices=['open cabinet 1','open cabinet 2','open cabinet 3']
        forecasts={a:'Nothing happens.' for a in choices+['open desk 1']}
        c,wm,agent=self.controller(forecasts,[xml(a) for a in choices])
        action,log=c.select('open desk 1','CURRENT',[],list(forecasts),1)
        self.assertEqual(action,'open desk 1')
        self.assertEqual(len(agent.messages),3)
        self.assertEqual(len(wm.calls),4)
        self.assertEqual(len(log['predictions']),5)
        self.assertTrue(log['predictions'][-1]['cached'])
        log['prediction_exact_match']=True
        self.assertEqual(metrics([dict(reflection=log)])['reflection_metrics']['wm_calls'],4)

    def test_final_round_success_has_no_fourth_review(self):
        c,wm,agent=self.controller({'open desk 1':'Nothing happens.','go to shelf 1':'You arrive at shelf 1.'},
            ['bad output','bad output',xml('go to shelf 1')])
        action,log=c.select('open desk 1','CURRENT',[],['open desk 1','go to shelf 1'],1)
        self.assertEqual(action,'go to shelf 1')
        self.assertEqual(len(agent.messages),3)
        self.assertEqual(len(wm.calls),2)

    def test_failed_missing_partial_or_identical_repair_cannot_commit(self):
        variants=[dict(observation='',prediction_available=False),dict(observation='looks good',observation_complete=False),
                  dict(observation='Nothing happens.')]
        for alt in variants:
            c,wm,agent=self.controller({'open desk 1':'Nothing happens.','go to shelf 1':alt},[xml('go to shelf 1'),xml('open desk 1')])
            action,log=c.select('open desk 1','CURRENT',[],['open desk 1','go to shelf 1'],1)
            self.assertEqual(action,'open desk 1')
            self.assertEqual(log['replacements'],0)

    def test_missing_original_evidence_does_not_become_failure(self):
        c,wm,agent=self.controller({'look':dict(observation='',prediction_available=False)},[xml('inventory')])
        action,log=c.select('look','CURRENT',[],['look','inventory'],1)
        self.assertEqual(action,'look')
        self.assertIsNone(log['original_defect'])

    def test_invalid_current_original_can_be_repaired(self):
        c,wm,agent=self.controller({'take book 1 from bed 1':'Nothing happens.','go to bed 1':'You arrive at bed 1.'},[xml('go to bed 1')])
        action,log=c.select('take book 1 from bed 1','Far from bed.',[],['go to bed 1'],1)
        self.assertEqual(action,'go to bed 1')
        self.assertEqual(log['original_defect'],'not_current_admissible')

    def test_future_admissible_action_does_not_override_current_actions(self):
        c,wm,agent=self.controller({'look':'Nothing happens.'},[xml('take book 1 from bed 1')]*3)
        action,log=c.select('look','CURRENT',[],['look','go to bed 1'],1)
        self.assertEqual(action,'look')
        self.assertEqual(log['invalid_reviews'],3)
        self.assertEqual(len(wm.calls),1)

    def test_observation_repetition_does_not_reject_wait_or_look(self):
        for action in ('wait','wait1','look','look around'):
            c,wm,agent=self.controller({action:'SAME STATE'},[xml(action)])
            steps=[dict(action=action,observation='SAME STATE',next_observation='SAME STATE')]*4
            self.assertIsNone(original_defect(c,action,wm.predict(action),steps,[action]))

    def test_actual_three_repeated_futile_actions_can_trigger_repair(self):
        c,wm,agent=self.controller({'focus on pot':'You focus on the metal pot.','pick up metal pot':'You move the metal pot to the inventory.'},
            ['Thought: progress beyond repeated focus.\nAction: pick up metal pot'],'scienceworld')
        steps=[dict(action='focus on pot',observation='You focus on the metal pot.',next_observation='You focus on the metal pot.')]*3
        action,log=c.select('focus on pot','CURRENT',steps,['focus on pot','pick up metal pot'],4)
        self.assertEqual(action,'pick up metal pot')

    def test_no_privileged_score_or_hidden_state_enters_review(self):
        c,wm,agent=self.controller({'look':dict(observation='visible',score='SECRET',done=True,success=True,hidden='SECRET')},[xml('look')])
        c.select('look','CURRENT',[],['look'],1)
        self.assertNotIn('SECRET',str(agent.messages))
        self.assertNotIn('success',agent.messages[0][-1]['content'])

    def test_training_context_is_preserved_and_real_history_trimmed(self):
        c,wm,agent=self.controller({'look':'preview'},[xml('look')])
        prelude=[dict(role='user',content='TRAIN USER'),dict(role='assistant',content='TRAIN ACTION')]*27
        agent.reflection_context=lambda env,task,obs,steps,actions,keep: copy.deepcopy(prelude)+[dict(role='user',content='GOAL CURRENT'+(' OLD' if keep else ''))]
        agent.count=lambda messages:5000 if 'OLD' in messages[-1]['content'] else 100
        c.select('look','CURRENT',[dict(action='look',observation='OLD',next_observation='CURRENT')],['look'],1)
        self.assertEqual(agent.messages[0][:-1],prelude)
        self.assertNotIn('OLD',str(agent.messages))

    def test_only_hypothetical_preview_is_clipped_to_token_budget(self):
        c,wm,agent=self.controller({'look':'X'*9000},[xml('look')])
        agent.count=lambda messages:len(messages[-1]['content'])
        action,log=c.select('look','CURRENT',[],['look'],1)
        self.assertEqual(action,'look')
        self.assertTrue(log['reviews'][0]['preview_clipped'])
        self.assertLessEqual(len(agent.messages[0][-1]['content']),4096)
        self.assertIn('PARTIAL',agent.messages[0][-1]['content'])
        self.assertIn('GOAL',agent.messages[0][-1]['content'])
        self.assertIn('CURRENT',agent.messages[0][-1]['content'])

    def test_required_context_overflow_is_honest_fallback_not_fake_review(self):
        c,wm,agent=self.controller({'look':'preview'},[])
        agent.count=lambda messages:5000
        action,log=c.select('look','CURRENT',[],['look'],1)
        self.assertEqual(action,'look')
        self.assertTrue(log['budget_skip'])
        self.assertEqual(log['rounds'],0)
        self.assertEqual(agent.messages,[])

    def test_keep_preserves_original_actor_output_exactly(self):
        c,wm,agent=self.controller({'look':'visible'},[xml('inventory')])
        chosen,text,metadata=runtime_select(c,'look','ORIGINAL ACTOR TEXT','CURRENT',[],['look','inventory'],1)
        self.assertEqual(text,'ORIGINAL ACTOR TEXT')

    def test_accepted_repair_keeps_actual_native_model_output(self):
        raw='Thought: The door is already open; move through it.\nAction: go to hallway'
        c,wm,agent=self.controller({'open door to hallway':'The door is already open.','go to hallway':'You move to the hallway.'},[raw],'scienceworld')
        chosen,text,metadata=runtime_select(c,'open door to hallway','OLD','CURRENT',[],['go to hallway'],1)
        self.assertEqual(text,raw)

    def test_sdk_aliases_and_partial_failure_evidence(self):
        self.assertTrue(valid_replacement('scienceworld','drop metal pot',[]))
        self.assertTrue(valid_replacement('scienceworld','native alias', ['native alias']))
        self.assertIsNone(failure_feedback(dict(observation='Nothing happens.',observation_complete=False),'alfworld'))
        self.assertEqual(failure_feedback(dict(observation='Page 1 (Total results: 0)'),'webshop'),'empty_search_results')

    def test_real_agent_reflection_context_uses_complete_science_demo(self):
        import run_audited_paper_eval as evaluator
        prelude=[dict(role='user',content='TRAIN'),dict(role='assistant',content='Thought: train\nAction: wait1')]*27
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'data/paper_eval/sources/science_expert_sample.jsonl'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(dict(messages=prelude))+'\n')
            with patch.object(evaluator,'ROOT',root):
                agent=evaluator.Agent(SimpleNamespace(disable_thinking=False),root)
                messages=agent.reflection_context('scienceworld','GOAL','CURRENT',[],[],0)
            self.assertEqual(messages[:-1],prelude)
            self.assertEqual(messages[-1]['content'],'GOAL\nCURRENT')


if __name__=='__main__':
    unittest.main()
