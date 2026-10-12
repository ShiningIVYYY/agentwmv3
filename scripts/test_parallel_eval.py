#!/usr/bin/env python3
"""Exercise real OS locks, worker crashes, task recovery and process supervision."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.matrix import read, save, statistics
from experiments.parallel import run_parallel, worker_command
from experiments.task_queue import TaskQueue


def execute(output, manifest, rows, run, events):
    queue = TaskQueue(output, manifest, rows, run, 4)
    for row in queue.claims():
        events.put(('start', row['task_id'], os.getpid(), time.time()))
        time.sleep(.02 * (1 + int(row['task_id'].split('-')[1]) % 4))
        save(queue.result_file(row['task_id']), dict(task_id=row['task_id'], status='completed', success=True, score=100))
        events.put(('end', row['task_id'], os.getpid(), time.time()))


def die_during_claim(output, manifest, rows, published):
    queue = TaskQueue(output, manifest, rows, 'same-run', 4)
    for row in queue.claims():
        if published:
            save(queue.result_file(row['task_id']), dict(task_id=row['task_id'], status='completed', success=True, score=100))
        os._exit(17)


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        self.rows = [dict(task_id='task-%02d' % i) for i in range(20)]
        self.manifest = self.root / 'manifest.json'
        save(self.manifest, dict(tasks=self.rows))
        self.ctx = mp.get_context('fork')

    def workers(self, run='same-run'):
        events = self.ctx.Queue()
        children = [self.ctx.Process(target=execute, args=(self.output, self.manifest, self.rows, run, events)) for _ in range(4)]
        for process in children:
            process.start()
        for process in children:
            process.join(15)
            if process.is_alive():
                process.kill()
                process.join()
                self.fail('Queue worker did not exit')
            self.assertEqual(process.exitcode, 0)
        rows = []
        while not events.empty():
            rows.append(events.get())
        return rows

    def test_four_processes_claim_each_task_once_and_overlap(self):
        events = self.workers()
        starts = [r for r in events if r[0] == 'start']
        self.assertEqual(sorted(r[1] for r in starts), sorted(r['task_id'] for r in self.rows))
        self.assertEqual(len(set(r[2] for r in starts)), 4)
        concurrent = peak = 0
        for event in sorted(events, key=lambda r:r[3]):
            concurrent += 1 if event[0] == 'start' else -1
            peak = max(peak, concurrent)
        self.assertEqual(peak, 4)
        self.assertTrue(statistics(self.output, [r['task_id'] for r in self.rows])['complete'])

    def test_dead_process_releases_claim_without_losing_task(self):
        child = self.ctx.Process(target=die_during_claim, args=(self.output, self.manifest, self.rows, False))
        child.start(); child.join(5)
        self.assertEqual(child.exitcode, 17)
        starts = [r for r in self.workers() if r[0] == 'start']
        self.assertEqual(len(starts), 20)
        self.assertEqual(sum(r[1] == 'task-00' for r in starts), 1)

    def test_completed_result_before_crash_is_not_executed_twice(self):
        child = self.ctx.Process(target=die_during_claim, args=(self.output, self.manifest, self.rows, True))
        child.start(); child.join(5)
        self.assertEqual(child.exitcode, 17)
        starts = [r for r in self.workers() if r[0] == 'start']
        self.assertEqual(len(starts), 19)
        self.assertNotIn('task-00', [r[1] for r in starts])

    def test_failed_task_is_not_reclaimed_until_next_cohort(self):
        queue = TaskQueue(self.output, self.manifest, self.rows[:1], 'first', 4)
        for row in queue.claims():
            save(queue.result_file(row['task_id']), dict(task_id=row['task_id'], status='error', error='test'))
        self.assertEqual(list(queue.claims()), [])
        next_queue = TaskQueue(self.output, self.manifest, self.rows[:1], 'second', 4)
        claims = next_queue.claims()
        self.assertEqual(next(claims)['task_id'], 'task-00')
        claims.close()

    def test_queue_rejects_changed_manifest_and_duplicate_ids(self):
        TaskQueue(self.output, self.manifest, self.rows, 'first', 4)
        save(self.manifest, dict(tasks=list(reversed(self.rows))))
        with self.assertRaises(RuntimeError):
            TaskQueue(self.output, self.manifest, self.rows, 'first', 4)
        with self.assertRaises(RuntimeError):
            TaskQueue(self.output, self.manifest, [self.rows[0]] * 2, 'second', 4)

    def test_worker_configuration_preserves_model_and_sampling_arguments(self):
        command = worker_command(['eval.py', '--output', str(self.output), '--workers', '4',
            '--url', 'http://agent', '--max-steps', '100', '--max-tokens', '1024',
            '--reflection-max-rounds', '3'], self.output, 'run', 1, 4)
        for flag, value in (('--workers', '1'), ('--queue-workers', '4'), ('--url', 'http://agent'),
                            ('--max-steps', '100'), ('--max-tokens', '1024'), ('--reflection-max-rounds', '3')):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotEqual(command[command.index('--output') + 1], str(self.output))

    def test_worker_configs_cannot_differ_or_replace_the_manifest(self):
        queue = TaskQueue(self.output, self.manifest, self.rows, 'run', 4)
        config = dict(output='/worker-0', model='unchanged', manifest=str(self.manifest), max_tokens=1024)
        queue.publish_config(config)
        queue.publish_config(dict(config, output='/worker-1'))
        self.assertEqual(read(self.output / 'config.json')['max_tokens'], 1024)
        with self.assertRaises(RuntimeError):
            queue.publish_config(dict(config, max_tokens=512))

    def fake_runner(self, failure):
        path = self.root / 'fake_eval.py'
        path.write_text('''import argparse,os,sys,time
from pathlib import Path
sys.path.insert(0, ROOT)
from experiments.matrix import read,save
from experiments.task_queue import TaskQueue
p=argparse.ArgumentParser()
for flag in ('output','workers','task-queue','queue-run','queue-workers','manifest'): p.add_argument('--'+flag)
a=p.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
q=TaskQueue(a.task_queue,a.manifest,read(a.manifest)['tasks'],a.queue_run,int(a.queue_workers))
q.publish_config(dict(output=str(out),model='fake',max_steps=100,max_tokens=1024))
for row in q.claims():
 save(out/'progress.json',dict(task_id=row['task_id'],step=1,state='fake'))
 time.sleep(.01)
 save(q.result_file(row['task_id']),dict(task_id=row['task_id'],status='completed',success=False,score=0))
if FAILURE:
 marker=out/'crashed_once'
 if FAILURE=='always' or not marker.exists():
  marker.touch(); os._exit(17)
save(out/'progress.json',dict(state='finished'))
'''.replace('ROOT', repr(str(ROOT))).replace('FAILURE', repr(failure)))
        return path

    def supervise(self, failure, backend='oracle'):
        args = SimpleNamespace(task_queue=None,queue_run=None,queue_workers=None,wm_backend=backend,
            generation_lock=None,output=str(self.output),manifest=str(self.manifest),limit=None,workers=2)
        script = self.fake_runner(failure)
        argv = [str(script),'--output',str(self.output),'--workers','2','--manifest',str(self.manifest)]
        return run_parallel(args, self.root, argv)

    def test_supervisor_waits_for_all_clean_process_exits(self):
        self.assertEqual(self.supervise(None), 0)
        state = read(self.output / 'parallel_state.json')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['worker_failures'], 0)
        self.assertTrue(all(r['exits'][-1]['returncode'] == 0 for r in state['children'].values()))
        self.assertEqual(read(self.output / 'progress.json')['active_workers'], 0)

    def test_pure_agent_uses_same_task_queue_and_clean_exit_checks(self):
        self.assertEqual(self.supervise(None, backend='none'), 0)
        self.assertTrue(statistics(self.output, [r['task_id'] for r in self.rows])['complete'])
        self.assertEqual(read(self.output / 'parallel_state.json')['worker_failures'], 0)

    def test_crash_after_results_is_observed_and_retried(self):
        self.assertEqual(self.supervise('once'), 0)
        state = read(self.output / 'parallel_state.json')
        self.assertEqual(state['worker_failures'], 2)
        for record in state['children'].values():
            self.assertEqual([r['returncode'] for r in record['exits']], [17,0])

    def test_persistent_teardown_crash_cannot_pass_with_complete_json(self):
        with self.assertRaisesRegex(RuntimeError, 'retry budget'):
            self.supervise('always')
        state = read(self.output / 'parallel_state.json')
        self.assertGreaterEqual(state['worker_failures'], 3)
        self.assertTrue(all(r['launches'] <= 3 for r in state['children'].values()))
        self.assertTrue(statistics(self.output, [r['task_id'] for r in self.rows])['complete'])
        self.assertNotEqual(state.get('status'), 'completed')


if __name__ == '__main__':
    unittest.main()
