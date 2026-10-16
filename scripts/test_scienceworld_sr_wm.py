"""Check baseline gating, unchanged v2 controls and corrected-baseline pairing."""
import copy
from pathlib import Path
import tempfile
import unittest

import run_scienceworld_sr_wm as wm
from experiments.matrix import save


class WMTests(unittest.TestCase):
    def fixture(self, root):
        config = dict(env='scienceworld', manifest='/fixed/tasks', url='http://router',
            model='ee-scienceworld-sr', tokenizer_path='/checkpoint',
            tokenizer_url='http://tokenizer', science_url='http://science',
            max_steps=100, max_tokens=1024, generation_lock=None,
            science_context='task-anchored-v1', science_demo='/train/verified-demo.json')
        profile = dict(env='scienceworld', baseline=str(root/'none'),
                       task_ids=['a','b'], wm_config='/fixed/science-wm.json')
        jobs = {backend+'/'+wm.PROFILE: dict(backend=backend, profile=wm.PROFILE,
            config=config, workers=16, output=str(root/backend), preflight=str(root/'smoke'/backend))
            for backend in wm.BACKENDS}
        return dict(root=str(root), jobs=jobs, profiles={wm.PROFILE:profile},
                    policy=dict(workers=4, smoke_limit=4, smoke_steps=3))

    def test_commands_preserve_context_and_v2_method(self):
        matrix = self.fixture(Path('/tmp/context-wm-test'))
        for backend in wm.BACKENDS:
            key = backend+'/'+wm.PROFILE
            args = wm.command(matrix, key)
            for flag, value in [('--workers','16'), ('--max-steps','100'),
                ('--max-tokens','1024'), ('--reflection-max-rounds','3'),
                ('--reflection-review-protocol','agentwm-copied-v1'),
                ('--science-context','task-anchored-v1'),
                ('--science-demo','/train/verified-demo.json')]:
                self.assertEqual(args[args.index(flag)+1], value)
            self.assertEqual('--wm-config' in args, backend=='word2world')
            smoke = wm.command(matrix, key, True)
            self.assertEqual(smoke[smoke.index('--workers')+1], '4')
            self.assertEqual(smoke[smoke.index('--max-steps')+1], '3')

    def test_pairing_references_new_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = self.fixture(root)
            for backend, outcomes in [('none',[True,False]), ('oracle',[False,True])]:
                for task, success in zip(['a','b'], outcomes):
                    save(root/backend/'tasks'/(task+'.json'), dict(task_id=task,
                        status='completed', success=success, score=100*success))
            paired = wm.sr.paired(matrix, 'oracle/'+wm.PROFILE)
            self.assertEqual(paired['improved_task_ids'], ['b'])
            self.assertEqual(paired['degraded_task_ids'], ['a'])

    def test_baseline_gate_rejects_hidden_wm_or_wrong_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'none'/wm.PROFILE
            tasks = [dict(task_id='task_'+str(i)) for i in range(200)]
            save(root/'manifest.json', dict(tasks=tasks))
            save(directory/'config.json', dict(manifest=str(root/'manifest.json'),
                science_context='task-anchored-v1'))
            row = dict(status='completed', done=True, score=100, success=True,
                task_description='real goal', steps=[dict(valid_format=True,
                finish_reason='stop', prompt_tokens=2048,
                messages=[dict(role='user',content='real goal current state')])])
            for task in tasks:
                save(directory/'tasks'/(task['task_id']+'.json'), row)
            self.assertEqual(wm.audit_baseline(root)['successes'], 200)
            path = directory/'tasks/task_0.json'
            changed = copy.deepcopy(row)
            changed['steps'][0]['reflection'] = {}
            save(path, changed)
            with self.assertRaises(RuntimeError):
                wm.audit_baseline(root)
            changed = copy.deepcopy(row)
            changed['success'] = False
            save(path, changed)
            with self.assertRaises(RuntimeError):
                wm.audit_baseline(root)


if __name__ == '__main__':
    unittest.main()
