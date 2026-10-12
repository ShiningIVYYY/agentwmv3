#!/usr/bin/env python3
"""Check SR controls, baseline dependencies, and matched task accounting."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('sr_runner', ROOT/'scripts/run_sr_experiments.py')
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


class SRTests(unittest.TestCase):
    def matrix(self, root):
        profiles, jobs = {}, {}
        for env in sr.ENVS:
            key = env+'_ee_sr'
            config = dict(env=env, manifest=str(root/(env+'.json')), url='http://primary',
                model='ee-'+env+'-sr', tokenizer_path='/checkpoints/'+env+'/sr', tokenizer_url='http://tokenizer',
                science_url='http://science', max_steps=sr.SPEC[env][2], max_tokens=sr.SPEC[env][3],
                disable_thinking=False, generation_lock=None)
            profiles[key] = dict(env=env, config=config, task_ids=['a','b','c','d'], baseline=str(root/'none'/key), wm_config='/wm/'+env)
            for backend in sr.GROUPS:
                c = dict(config)
                if backend == 'word2world':
                    c['url'] = 'http://replica'
                jobs[backend+'/'+key] = dict(profile=key, backend=backend, config=c,
                    output=str(root/backend/key), preflight=str(root/'preflight'/backend/key))
        return dict(root=str(ROOT), profiles=profiles, jobs=jobs,
                    policy=dict(workers=4, smoke_limit=4, smoke_steps=3))

    def test_nine_commands_preserve_agent_controls(self):
        matrix = self.matrix(Path('/tmp/test-sr'))
        for key, job in matrix['jobs'].items():
            args = sr.command(matrix, key)
            for name in ('model','tokenizer_path','max_steps','max_tokens','url'):
                self.assertEqual(args[args.index('--'+name.replace('_','-'))+1], str(job['config'][name]))
            self.assertEqual(args[args.index('--workers')+1], '4')
            self.assertEqual(args[args.index('--wm-backend')+1], job['backend'])
            self.assertEqual('--reflection-review-protocol' in args, job['backend'] != 'none')
            self.assertEqual('--wm-config' in args, job['backend'] == 'word2world')
            self.assertNotIn('--limit', args)
            smoke = sr.command(matrix, key, True)
            self.assertEqual(smoke[smoke.index('--max-steps')+1], '3')
            self.assertEqual(smoke[smoke.index('--limit')+1], '4')

    def test_WM_waits_for_clean_baseline_completion(self):
        state = dict(jobs={})
        self.assertEqual(len(sr.eligible_jobs(state)), 3)
        for status in ('running','failed'):
            state['jobs']['full/none/alfworld_ee_sr'] = dict(status=status)
            self.assertEqual(len(sr.eligible_jobs(state)), 3)
        state['jobs']['full/none/alfworld_ee_sr']['status'] = 'completed'
        self.assertEqual(len(sr.eligible_jobs(state)), 5)
        self.assertIn('word2world/alfworld_ee_sr', sr.eligible_jobs(state))
        self.assertNotIn('oracle/scienceworld_ee_sr', sr.eligible_jobs(state))

    def test_pairing_uses_only_same_completed_ids_and_new_SR_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = self.matrix(root)
            for backend, values in [('none', [True,False,True,False]), ('oracle',[False,True,True,True])]:
                for key, success in zip(['a','b','c','d'], values):
                    sr.save(root/backend/'alfworld_ee_sr'/'tasks'/(key+'.json'),
                        dict(task_id=key, status='error' if backend == 'oracle' and key == 'd' else 'completed',
                             success=success, score=100 if success else 0))
            result = sr.paired(matrix, 'oracle/alfworld_ee_sr')
            self.assertEqual(result['matched_tasks'], 3)
            self.assertEqual(result['improved_task_ids'], ['b'])
            self.assertEqual(result['degraded_task_ids'], ['a'])
            self.assertEqual(result['net_successes'], 0)


if __name__ == '__main__':
    unittest.main()
