#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.matrix import ENVS, MODELS, BACKENDS, SPEC, ARGUMENTS, command, save, sha, statistics, verify, prepare
from experiments.scheduler import MatrixScheduler, process_identity, matches_process


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.matrix = dict(root=str(self.root), output=str(self.root / 'out'), models=list(MODELS), profiles={}, jobs={},
                           reflection_max_rounds=3, hashes={}, policy=dict(smoke_steps=2, smoke_limit=1, max_runner_launches=3))
        for env in ENVS:
            for model in MODELS:
                profile_key = env + '_' + model
                config = dict(env=env, manifest='frozen-' + env + '.json', url='http://agent-' + model,
                    model=model, tokenizer_path='/checkpoint/' + profile_key, tokenizer_url='http://tokens',
                    science_url='http://science', max_steps=SPEC[env][2], max_tokens=SPEC[env][3],
                    generation_lock='/shared.lock' if model == 'qwen38' else None,
                    disable_thinking=model == 'qwen38', protocol='baseline', limit=None, manifest_sha256='fixed')
                self.matrix['profiles'][profile_key] = dict(config=config, model=model, env=env,
                    expected=SPEC[env][1], task_ids=['task-0'], baseline=str(self.root / 'baseline' / profile_key), wm_config='wm-' + env + '.json')
                for backend in BACKENDS:
                    key = backend + '/' + profile_key
                    self.matrix['jobs'][key] = dict(profile=profile_key, backend=backend,
                        output=str(self.root / 'out' / key), preflight=str(self.root / 'out/preflight' / key))
        save(self.root / 'out/experiment.json', self.matrix)
        self.scheduler = MatrixScheduler(self.matrix)

    def actual_config(self, key, smoke):
        job = self.matrix['jobs'][key]
        config = copy.deepcopy(self.matrix['profiles'][job['profile']]['config'])
        config.update(wm_backend=job['backend'], reflection_max_rounds=3)
        if smoke:
            config.update(max_steps=2, limit=1)
        save(Path(job['preflight'] if smoke else job['output']) / 'config.json', config)

    def test_eighteen_full_commands_inherit_all_original_agent_arguments(self):
        self.assertEqual(len(self.matrix['jobs']), 18)
        self.assertEqual(sum(SPEC[env][1] for env in ENVS) * 6, 5040)
        for key, job in self.matrix['jobs'].items():
            config = self.matrix['profiles'][job['profile']]['config']
            argv = command(self.matrix, key)
            for field in ARGUMENTS:
                flag = '--' + field.replace('_', '-')
                if config[field] is None:
                    self.assertNotIn(flag, argv)
                else:
                    self.assertEqual(argv[argv.index(flag)+1], str(config[field]))
            self.assertEqual('--disable-thinking' in argv, config['disable_thinking'])
            self.assertNotIn('--limit', argv)
            self.assertEqual(argv[argv.index('--reflection-max-rounds')+1], '3')

    def test_smoke_limits_do_not_modify_frozen_full_configuration(self):
        before = copy.deepcopy(self.matrix)
        for key in self.matrix['jobs']:
            argv = command(self.matrix, key, True)
            self.assertEqual(argv[argv.index('--max-steps')+1], '2')
            self.assertEqual(argv[argv.index('--limit')+1], '1')
        self.assertEqual(self.matrix, before)

    def test_actual_config_cannot_silently_change_tokens_or_task_count(self):
        key = 'word2world/alfworld_qwen25'
        self.actual_config(key, False)
        self.scheduler.validate_config(key, False)
        path = Path(self.matrix['jobs'][key]['output']) / 'config.json'
        for field, value in (('max_tokens', 512), ('limit', 1)):
            self.actual_config(key, False)
            config = json.loads(path.read_text())
            config[field] = value
            save(path, config)
            with self.assertRaises(RuntimeError):
                self.scheduler.validate_config(key, False)

    def test_changed_frozen_source_rejects_resume(self):
        path = self.root / 'source.py'
        path.write_text('initial')
        self.matrix['hashes'] = {str(path):sha(path)}
        verify(self.matrix)
        path.write_text('changed')
        with self.assertRaises(RuntimeError):
            verify(self.matrix)

    def test_infrastructure_errors_cannot_be_counted_as_complete_or_model_failures(self):
        directory = self.root / 'results'
        save(directory / 'tasks/task-0.json', dict(task_id='task-0', status='completed', success=False, score=0))
        save(directory / 'tasks/task-1.json', dict(task_id='task-1', status='error', error='service unavailable'))
        stats = statistics(directory, ['task-0','task-1'])
        self.assertEqual((stats['completed'], stats['errors'], stats['complete']), (1,1,False))
        self.assertEqual(stats['success_rate'], 0)
        save(directory / 'tasks/task-1.json', dict(task_id='task-1', status='completed', success=True, score=100))
        stats = statistics(directory, ['task-0','task-1'])
        self.assertTrue(stats['complete'])
        self.assertEqual(stats['success_rate'], 50)

    def test_unknown_task_result_cannot_pollute_denominator(self):
        save(self.root / 'tasks/foreign.json', dict(task_id='foreign', status='completed', success=True, score=100))
        with self.assertRaises(RuntimeError):
            statistics(self.root, ['task-0'])

    def test_inherited_metrics_without_empty_count_merge_with_new_results(self):
        save(self.root / 'tasks/task-0.json', dict(task_id='task-0', status='completed', success=False,
            score=0, reflection_metrics=dict(steps=2, reflection_calls=3, mean_rounds=1.5)))
        save(self.root / 'tasks/task-1.json', dict(task_id='task-1', status='completed', success=False,
            score=0, reflection_metrics=dict(steps=1, reflection_calls=2, mean_rounds=2,
                wm_empty_predictions=1)))
        metrics = statistics(self.root, ['task-0','task-1'])['reflection']
        self.assertEqual(metrics['wm_empty_predictions'], 1)
        self.assertEqual(metrics['mean_rounds'], 5 / 3)

    def test_preflight_failure_blocks_entire_model_wave_full_runs(self):
        key = 'oracle/alfworld_qwen25'
        self.scheduler.state['jobs'][self.scheduler.key(key, True)] = dict(status='failed')
        with patch.object(self.scheduler, 'ensure_front_agent') as services, patch.object(self.scheduler, 'drive_job') as launch:
            self.assertTrue(self.scheduler.drive_unit('qwen25', None, False))
            services.assert_not_called()
            launch.assert_not_called()
        for key in self.scheduler.jobs_for('qwen25'):
            self.assertEqual(self.scheduler.state['jobs'][self.scheduler.key(key, False)]['status'], 'blocked_preflight')

    def test_exit_zero_with_missing_tasks_retries_only_to_fixed_limit(self):
        key = 'oracle/alfworld_qwen25'
        self.scheduler.state['jobs'][self.scheduler.key(key, False)] = dict(status='running', launches=3)
        with patch('experiments.scheduler.active_runners', return_value=[]), patch('experiments.scheduler.matches_process', return_value=False):
            self.assertEqual(self.scheduler.drive_job(key, False), 'failed')

    def test_smoke_prediction_must_match_committed_action(self):
        key = 'oracle/alfworld_qwen25'
        self.actual_config(key, True)
        path = Path(self.matrix['jobs'][key]['preflight']) / 'tasks/task-0.json'
        reflection = dict(rounds=3, final_action='look', predictions=[dict(action='inventory', observation='room')],
                          prediction_exact_match=True, actual=dict(observation='room'))
        save(path, dict(task_id='task-0', steps=[dict(action='look', reflection=reflection)]))
        with self.assertRaises(RuntimeError):
            self.scheduler.validate_smoke(key)
        reflection['predictions'] = [dict(action='look', observation='room')] * 4
        save(path, dict(task_id='task-0', steps=[dict(action='look', reflection=reflection)]))
        self.scheduler.validate_smoke(key)
        reflection['rounds'] = 4
        save(path, dict(task_id='task-0', steps=[dict(action='look', reflection=reflection)]))
        with self.assertRaises(RuntimeError):
            self.scheduler.validate_smoke(key)

    def test_science_oracle_smoke_accepts_exit_order_but_rejects_door_state(self):
        key = 'oracle/scienceworld_qwen25'
        path = Path(self.matrix['jobs'][key]['preflight']) / 'tasks/task-0.json'
        head = 'Inside the hallway is: \n\tthe agent\n\n You also see: '
        a, b = 'A door to the bedroom (that is open)', 'A door to the kitchen (that is closed)'
        def state(text):
            return dict(observation=text, raw_observation=text, native_observation=text,
                        score=0, reward=0, done=False, success=False)
        prediction = dict(state(head + b + ', ' + a + '.'), action='look around')
        reflection = dict(rounds=1, final_action='look around', predictions=[prediction],
                          prediction_exact_match=False, actual=state(head + a + ', ' + b + '.'))
        row = dict(task_id='task-0', steps=[dict(action='look around', reflection=reflection)])
        save(path, row)
        self.scheduler.validate_smoke(key)
        reflection['actual'] = state(head + a + ', ' + b.replace('closed', 'open') + '.')
        save(path, row)
        with self.assertRaises(RuntimeError):
            self.scheduler.validate_smoke(key)

    def test_baseline_completion_also_waits_for_original_runners_to_exit(self):
        for env in ENVS:
            profile = self.matrix['profiles'][env + '_qwen38']
            save(Path(profile['baseline']) / 'tasks/task-0.json', dict(task_id='task-0', status='completed', success=True, score=100))
        directory = self.matrix['profiles']['webshop_qwen38']['baseline']
        with patch('experiments.scheduler.active_runners', return_value=[dict(pid=123, command=['python','--output',directory])]):
            self.assertFalse(self.scheduler.qwen38_dependencies())
        with patch('experiments.scheduler.active_runners', return_value=[]):
            self.assertTrue(self.scheduler.qwen38_dependencies())

    def test_pid_reuse_cannot_adopt_wrong_worker(self):
        with patch('experiments.scheduler.process_identity', return_value=dict(start_ticks='new', command=['python','worker'])):
            self.assertFalse(matches_process(dict(pid=123, start_ticks='old', command=['python','worker'])))

    def test_both_available_model_families_are_validated_before_long_full_runs(self):
        self.assertEqual(self.scheduler.units[:4], [('qwen25',None,True),
            ('ee_il','alfworld',True), ('ee_il','scienceworld',True), ('ee_il','webshop',True)])
        self.assertTrue(all(not unit[2] for unit in self.scheduler.units[4:]))

    def test_two_agent_scope_has_no_qwen38_jobs_or_dependency(self):
        matrix = copy.deepcopy(self.matrix)
        matrix['models'] = ['qwen25','ee_il']
        matrix['profiles'] = {k:v for k,v in matrix['profiles'].items() if v['model'] != 'qwen38'}
        matrix['jobs'] = {k:v for k,v in matrix['jobs'].items() if v['profile'] in matrix['profiles']}
        scheduler = MatrixScheduler(matrix)
        self.assertEqual(len(matrix['jobs']), 12)
        self.assertEqual(sum(SPEC[env][1] for env in ENVS) * 4, 3360)
        self.assertEqual(scheduler.state['qwen38_stage'], 'not_selected')
        self.assertFalse(scheduler.jobs_for('qwen38'))

    def test_changed_model_selection_cannot_resume_a_different_matrix(self):
        with self.assertRaisesRegex(RuntimeError, 'Selected models changed'):
            prepare(self.root, self.root / 'out', ['qwen25','ee_il'])


if __name__ == '__main__':
    unittest.main()
