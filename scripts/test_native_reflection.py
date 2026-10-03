import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflection.controller import ReflectionController
from reflection.native import parse_native_review
from worldmodel.backends import PredictionError
from worldmodel.runtime import metrics


class WM:
    backend = 'oracle'
    def __init__(self):
        self.actions = []
    def predict(self, action):
        self.actions.append(action)
        return dict(observation='preview '+action, score='SECRET', hidden_state='SECRET', seconds=0)


class Agent:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []
    def reflection_context(self, env, task, observation, steps, actions, keep):
        return [dict(role='user',content=task+'\n'+observation+'\n'+str(actions)+'\n'+
                     str(steps[-keep:] if keep else []))]
    def count(self, messages):
        return 100
    def generate(self, messages, *args):
        self.messages.append(copy.deepcopy(messages))
        return next(self.outputs), dict(usage={})


def xml(action):
    return '<think>The preview supports this choice.</think><action>'+action+'</action>'


class NativeTests(unittest.TestCase):
    def controller(self, outputs, env='alfworld'):
        wm, agent = WM(), Agent(outputs)
        return ReflectionController(wm,agent,env,'GOAL','task',review_protocol='ee-native-action-v1'), wm, agent

    def test_keep_and_optional_prefilled_thought(self):
        for output in (xml('look'), 'Reviewed.</think><action>LOOK</action>'):
            self.assertEqual(parse_native_review(output,'look','alfworld')['decision'],'keep')

    def test_replacement_then_keep_and_no_privileged_feedback(self):
        c, wm, agent = self.controller([xml('inventory'),xml('inventory')])
        chosen, log = c.select('look','CURRENT',[],['look','inventory'],1)
        self.assertEqual(chosen,'inventory')
        self.assertEqual(wm.actions,['look','inventory'])
        self.assertEqual(log['replacements'],1)
        for messages in agent.messages:
            content=messages[-1]['content']
            self.assertIn('CURRENT',content)
            self.assertIn('GOAL',content)
            self.assertIn('hypothetical',content)
            self.assertNotIn('SECRET',content)

    def test_three_replacements_four_predictions(self):
        c, wm, agent = self.controller([xml(a) for a in ('inventory','go to desk 1','look')])
        chosen, log = c.select('go to shelf 1','CURRENT',[],['look','inventory','go to desk 1'],1)
        self.assertEqual(chosen,'look')
        self.assertEqual(len(agent.messages),3)
        self.assertEqual(len(wm.actions),4)
        self.assertIn('no fourth review',agent.messages[-1][-1]['content'])

    def test_science_native_and_numbered_choice(self):
        c, wm, agent = self.controller(['Thought: This resolves the current menu.\nAction: 0','Thought: Keep.\nAction: 0'],'scienceworld')
        chosen, log=c.select('look around','CURRENT',[],['0','1'],1)
        self.assertEqual(chosen,'0')
        self.assertEqual(log['replacements'],1)

    def test_web_search(self):
        c, wm, agent=self.controller(['<action>search[blue shoes]</action>',xml('search[blue shoes]')],'webshop')
        chosen, log=c.select('search[red shoes]','CURRENT',[],['search[<your query>]'],1)
        self.assertEqual(chosen,'search[blue shoes]')
        content=agent.messages[0][-1]['content']
        self.assertLess(content.index('preview search'),content.index('ACTUAL CURRENT STATE'))
        self.assertTrue(content.endswith("['search[<your query>]']\n[]"))

    def test_web_action_only_must_be_one_complete_native_block(self):
        self.assertEqual(parse_native_review('<action>click[B01ABC]</action>','click[b01abc]','webshop'),
                         dict(decision='keep',action='click[b01abc]',reason=''))
        for output in ('{"action":"click[B01ABC]</action>"}',
                       '<action>click[B01ABC]</action> trailing',
                       '<action>click[B01ABC]</action><action>search[x]</action>'):
            with self.assertRaises(PredictionError):
                parse_native_review(output,'click[b01abc]','webshop')

    def test_format_errors_and_inapplicable_actions_are_separate(self):
        c, wm, agent=self.controller(['{"action":"look</action>"}',xml('go to moon 1'),xml('look')])
        chosen, log=c.select('look','CURRENT',[],['look'],1)
        self.assertEqual(chosen,'look')
        self.assertEqual([r['error_kind'] for r in log['reviews']],['invalid_format','invalid_action',None])
        log['prediction_exact_match']=True
        values=metrics([dict(reflection=log)])['reflection_metrics']
        self.assertEqual(values['valid_review_calls'],1)
        self.assertEqual(values['review_format_errors'],1)
        self.assertEqual(values['invalid_replacement_reviews'],1)

    def test_strict_single_action_no_malformed_json_salvage(self):
        for output in ('{"action":"look</action>"}',xml('look')+xml('inventory'),xml('look')+' trailing',
                       '<action>look</action>','```'+xml('look')+'```',xml('')):
            with self.subTest(output=output),self.assertRaises(PredictionError):
                parse_native_review(output,'look','alfworld')

    def test_science_rejects_multiple_actions_or_wrong_format(self):
        for output in ('Thought: x\nAction: look around\nAction: wait',xml('look around'),
                       '{"action":"look around"}'):
            with self.assertRaises(PredictionError):
                parse_native_review(output,'look around','scienceworld')

    def test_missing_prediction_is_explicit_unknown(self):
        c, wm, agent=self.controller([xml('look')])
        wm.predict=lambda action: dict(observation='',prediction_available=False)
        c.select('look','CURRENT',[],['look'],1)
        self.assertIn('UNKNOWN',agent.messages[0][-1]['content'])

    def test_history_trimming_keeps_current_goal_and_prediction(self):
        c, wm, agent=self.controller([xml('look')])
        agent.count=lambda messages: 5000 if 'OLD' in messages[-1]['content'] else 100
        steps=[dict(observation='OLD',action='inventory',next_observation='CURRENT')]
        c.select('look','CURRENT',steps,['look'],1)
        self.assertNotIn('OLD',agent.messages[0][-1]['content'])
        self.assertIn('GOAL',agent.messages[0][-1]['content'])
        self.assertIn('preview look',agent.messages[0][-1]['content'])


if __name__ == '__main__':
    unittest.main()
