import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import run_audited_paper_eval as ev
import run_scienceworld_sr_context as repro


class ContextTests(unittest.TestCase):
    def test_goal_and_episode_boundary_survive_history_eviction(self):
        prelude = [dict(role='user', content='DEMO OTHER GOAL'),
                   dict(role='assistant', content='Action: demo')]
        previous = [dict(observation='old state '+str(i), output='Action: move')
                    for i in range(50)]
        original = copy.deepcopy(prelude)
        messages = ev.anchored_science_messages(prelude, 'REAL GOAL', 'NOW', previous, 2)
        self.assertIn('REAL GOAL', messages[-1]['content'])
        self.assertIn('NOW', messages[-1]['content'])
        self.assertIn('separate task', messages[2]['content'])
        self.assertIn('old state 48', messages[2]['content'])
        self.assertNotIn('old state 47', json.dumps(messages))
        self.assertEqual(prelude, original)
        self.assertEqual([r['role'] for r in messages],
                         ['user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'user'])

    def test_explicit_demo_loads_and_no_history_still_has_goal(self):
        path = ev.ROOT/'data/paper_eval/sources/science_short_verified_demo.json'
        science = ev.Science(SimpleNamespace(science_url='http://unused', science_demo=str(path)))
        self.assertEqual(len(science.prelude), 10)
        messages = ev.anchored_science_messages(science.prelude, 'REAL GOAL', 'NOW', [], 0)
        self.assertIn('separate task', messages[-1]['content'])
        self.assertIn('REAL GOAL', messages[-1]['content'])

    def test_cohort_is_pure_sr_with_original_generation_limits(self):
        config = dict(manifest='/fixed/tasks', model='ee-scienceworld-sr',
            tokenizer_path='/fixed/checkpoint', tokenizer_url='http://tokenizer',
            science_url='http://science', max_steps=100, max_tokens=1024)
        experiment = dict(original_config=config, output='/new-cohort', workers=8,
            model_url='http://router', science_context='task-anchored-v1', science_demo='/train/demo')
        args, path = repro.command(experiment, False)
        for flag, value in [('--wm-backend', 'none'), ('--max-steps', '100'),
                            ('--max-tokens', '1024'), ('--workers', '8'),
                            ('--science-context', 'task-anchored-v1')]:
            self.assertEqual(args[args.index(flag)+1], value)
        self.assertEqual(path, Path('/new-cohort/none/scienceworld_ee_sr'))
        self.assertNotIn('--reflection-max-rounds', args)


if __name__ == '__main__':
    unittest.main()
