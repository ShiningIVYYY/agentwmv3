#!/usr/bin/env python3
"""Add the v2 Oracle/Word2World method to the completed context-v3 baseline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.matrix import read, save, sha, source_files, statistics, verify
from experiments.scheduler import process_identity, matches_process
import run_sr_experiments as sr

PROFILE = 'scienceworld_ee_sr'
BACKENDS = ('oracle', 'word2world')
CONTEXT_FIELDS = ('science_context', 'science_demo')


def audit_baseline(out):
    directory = out/'none'/PROFILE
    config = read(directory/'config.json')
    tasks = read(config['manifest'])['tasks']
    ids = [r['task_id'] for r in tasks]
    rows = [read(directory/'tasks'/(task+'.json')) for task in ids]
    if len(ids) != 200 or len(set(ids)) != 200 or any(r['status'] != 'completed' for r in rows):
        raise RuntimeError('Requires the complete fixed 200-task baseline')
    if config.get('science_context') != 'task-anchored-v1':
        raise RuntimeError('Requires the corrected ScienceWorld context')
    steps = [step for row in rows for step in row['steps']]
    if not all(row['success'] == bool(row['done'] and row['score'] == 100) for row in rows):
        raise RuntimeError('Baseline success accounting differs from the environment')
    if (any('reflection' in step for step in steps) or
        any(not step['valid_format'] or step['finish_reason'] != 'stop' for step in steps) or
        any(step['prompt_tokens'] > 4096 for step in steps) or
        any(row['task_description'] not in step['messages'][-1]['content'] for row in rows for step in row['steps'])):
        raise RuntimeError('Baseline prompt, generation or pure-agent audit failed')
    return dict(passed=True, count=200, successes=sum(r['success'] for r in rows),
        success_rate=100*sum(r['success'] for r in rows)/200,
        average_score=sum(r['score'] for r in rows)/200, steps=len(steps),
        max_prompt_tokens=max(s['prompt_tokens'] for s in steps),
        valid_format_steps=len(steps), stop_completions=len(steps),
        baseline_result_hashes={str(directory/'tasks'/(task+'.json')):
            sha(directory/'tasks'/(task+'.json')) for task in ids})


def prepare(out, workers):
    path = out/'wm_experiment.json'
    if path.exists():
        matrix = read(path)
        if matrix['wm_workers'] != workers or matrix['output'] != str(out):
            raise RuntimeError('WM scheduling changed; use a recorded transition')
        verify(matrix)
        return matrix
    baseline_experiment = read(out/'experiment.json')
    verify(baseline_experiment)
    audit = audit_baseline(out)
    save(out/'wm_baseline_audit.json', audit)
    reference_dir = ROOT/'outputs/wm_reflection/ee_sr_agentwm_prompts_v2'
    reference = read(reference_dir/'experiment.json')
    original = reference['profiles'][PROFILE]
    config = read(out/'none'/PROFILE/'config.json')
    ids = [r['task_id'] for r in read(config['manifest'])['tasks']]
    if ids != original['task_ids']:
        raise RuntimeError('WM comparison task IDs differ from v2')
    config = dict(config)
    config.pop('execution', None)
    profiles = {PROFILE: dict(original, config=config, baseline=str(out/'none'/PROFILE), task_ids=ids)}
    jobs = {}
    for backend in ('none',)+BACKENDS:
        c = dict(config)
        c['output'] = str(out/backend/PROFILE)
        jobs[backend+'/'+PROFILE] = dict(profile=PROFILE, backend=backend,
            config=c, expected=200, workers=workers,
            output=c['output'], preflight=str(out/'wm_preflight'/backend/PROFILE))
    services = [s for s in reference['services'] if s['env'] == 'scienceworld']
    if {s['gpu'] for s in services} != {3, 4, 5, 6}:
        raise RuntimeError('Expected the four retained ScienceWorld replicas')
    relevant = source_files(ROOT) + [Path(__file__), ROOT/'scripts/run_sr_experiments.py',
        Path(config['manifest']), Path(config['science_demo']), out/'experiment.json',
        out/'none'/PROFILE/'config.json', out/'wm_baseline_audit.json',
        ROOT/'checkpoints/early-experience-scienceworld/sr/sr_provenance.json']
    # Verify that the locally copied prompt, critique/replan policy and WM code
    # really are the v2 implementation rather than a silently changed method.
    method_hashes = {p: digest for p, digest in reference['hashes'].items()
                     if '/reflection/' in p or '/worldmodel/' in p or p == original['wm_config']}
    for filename, digest in method_hashes.items():
        if sha(filename) != digest:
            raise RuntimeError('v2 WM method source changed: '+filename)
    hashes = {str(p): sha(p) for p in relevant}
    hashes.update(audit['baseline_result_hashes'])
    hashes.update(method_hashes)
    matrix = dict(root=str(ROOT), output=str(out), created=time.time(),
        protocol='scienceworld-context-v3-agentwm-wm-v1', profiles=profiles, jobs=jobs,
        services=services, inference_router=reference['inference_router'], hashes=hashes,
        wm_workers=workers, baseline_success_rate=audit['success_rate'],
        method_reference=str(reference_dir), method_hashes=method_hashes,
        reflection_max_rounds=3, sampling=reference['sampling'],
        policy=dict(workers=4, smoke_limit=4, smoke_steps=3, max_runner_launches=3,
            reflection_protocol='agentwm-copied-v1', structured_outputs=sr.DECODE,
            schedule='both-WM-preflights-before-both-full-WM-arms',
            science_workers_per_group=workers, wm_gpus=[2],
            gpu_layout=[(s['env'],s['gpu'],s['port']) for s in services],
            released_gpus=[0,1,7]))
    save(path, matrix)
    services_dir = out/'services'
    services_dir.mkdir(exist_ok=True)
    router = read(reference_dir/'services/science_router.json')
    if not matches_process(router):
        raise RuntimeError('Existing inference router is not running')
    save(services_dir/'science_router.json', router)
    old_state = read(reference_dir/'runstate.json')
    save(out/'wm_runstate.json', dict(matrix_sha256=sha(path),
        jobs={'full/none/'+PROFILE: dict(status='completed', launches=0)},
        services={str(s['port']):old_state['services'][str(s['port'])] for s in services},
        phase='prepared'))
    return matrix


def command(matrix, key, smoke=False):
    args = sr.command(matrix, key, smoke)
    for field in CONTEXT_FIELDS:
        args += ['--'+field.replace('_', '-'), str(matrix['jobs'][key]['config'][field])]
    return args


class Scheduler(sr.Scheduler):
    def __init__(self, matrix):
        self.matrix = matrix
        self.root, self.out = Path(matrix['root']), Path(matrix['output'])
        self.state = read(self.out/'wm_runstate.json')
        if self.state['matrix_sha256'] != sha(self.out/'wm_experiment.json'):
            raise RuntimeError('Frozen WM experiment changed')
        self.children = {}

    def persist(self):
        self.state['updated'] = time.time()
        save(self.out/'wm_runstate.json', self.state)

    def event(self, event, **fields):
        row = dict(event=event, updated=time.time(), **fields)
        print(json.dumps(row), flush=True)
        with (self.out/'wm_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def validate(self, key, smoke):
        super().validate(key, smoke)
        job = self.matrix['jobs'][key]
        actual = read(Path(job['preflight'] if smoke else job['output'])/'config.json')
        baseline = read(self.out/'none'/PROFILE/'config.json')
        for name in CONTEXT_FIELDS+('protocol','science_prompt','science_demo_sha256','science_demo_messages'):
            if actual.get(name) != baseline.get(name):
                raise RuntimeError('WM and pure-agent context differ: '+name)

    def drive(self, key, smoke):
        phase = ('preflight/' if smoke else 'full/')+key
        record = self.state['jobs'].setdefault(phase, dict(status='pending', launches=0))
        if record['status'] == 'completed':
            return True
        job = self.matrix['jobs'][key]
        directory = Path(job['preflight'] if smoke else job['output'])
        ids = self.matrix['profiles'][PROFILE]['task_ids']
        ids = ids[:4] if smoke else ids
        if record.get('pid'):
            child = self.children.get(phase)
            code = child.poll() if child else None
            if code is None and matches_process(record):
                return False
            native = read(directory/'parallel_state.json') if (directory/'parallel_state.json').exists() else {}
            code = code if code is not None else (0 if native.get('status') == 'completed' else -1)
            record.update(pid=None, exit_code=code)
            if code == 0 and statistics(directory, ids)['complete']:
                self.validate(key, smoke)
                record['status'] = 'completed'
                self.persist()
                self.event('job_completed', job=key, smoke=smoke)
                return True
            record['status'] = 'failed'
            self.event('job_failed', job=key, smoke=smoke, exit_code=code)
        if record['launches'] >= (1 if smoke else 3):
            raise RuntimeError('Runner exhausted retries: '+phase)
        verify(self.matrix)
        directory.mkdir(parents=True, exist_ok=True)
        argv = command(self.matrix, key, smoke)
        with (directory/'runner.log').open('a') as stream:
            child = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1'))
        self.children[phase] = child
        record.update(pid=child.pid, **process_identity(child.pid),
                      status='running', launches=record['launches']+1)
        self.persist()
        self.event('job_started', job=key, smoke=smoke, pid=child.pid)
        return False

    def refresh(self):
        results, comparisons = {}, {}
        for key, job in self.matrix['jobs'].items():
            result = statistics(job['output'], self.matrix['profiles'][PROFILE]['task_ids'])
            result['status'] = self.state['jobs'].get('full/'+key, {}).get('status', 'pending')
            result['preflight'] = self.state['jobs'].get('preflight/'+key, {}).get('status', 'pending')
            results[key] = result
            if job['backend'] != 'none':
                comparisons[key] = sr.paired(self.matrix, key)
        complete = all(r['complete'] and r['status']=='completed' for r in results.values())
        save(self.out/'wm_summary.json', dict(updated=time.time(), complete=complete,
            phase=self.state.get('phase'), error=self.state.get('error'),
            manager_pid=os.getpid(), results=results))
        save(self.out/'wm_comparison.json', comparisons)
        lines = ['# ScienceWorld SR context-v3 + WM', '',
                 '| Group | Completed | Success rate | Mean score | Errors | Status |',
                 '|---|---:|---:|---:|---:|---|']
        for key, row in results.items():
            rate = '—' if row['success_rate'] is None else '%.2f%%'%row['success_rate']
            score = '—' if row['average_score'] is None else '%.2f'%row['average_score']
            lines.append('| %s | %d/200 | %s | %s | %d | %s |'%
                (key,row['completed'],rate,score,row['errors'],row['status']))
        lines += ['', '| Group vs corrected pure SR | Matched | Failure → success | Success → failure | Net |',
                  '|---|---:|---:|---:|---:|']
        for key, row in comparisons.items():
            lines.append('| %s | %d | %d | %d | %+d |'%
                (key,row['matched_tasks'],row['improved'],row['degraded'],row['net_successes']))
        (self.out/'WM_RESULTS.md').write_text('\n'.join(lines)+'\n')
        return complete

    def run(self):
        with (self.out/'.wm_manager.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            save(self.out/'wm_manager.json', dict(pid=os.getpid(), **process_identity(os.getpid())))
            try:
                self.state.pop('error', None)
                self.state['phase'] = 'services'
                self.persist()
                self.refresh()
                self.services()
                self.ensure_router()
                keys = [backend+'/'+PROFILE for backend in BACKENDS]
                self.state['phase'] = 'preflight'
                while True:
                    done = [self.drive(key, True) for key in keys]
                    self.refresh()
                    if all(done):
                        break
                    time.sleep(5)
                self.event('both_WM_preflights_passed')
                self.state['phase'] = 'full'
                while True:
                    self.ensure_router()
                    for key in keys:
                        self.drive(key, False)
                    self.persist()
                    if self.refresh():
                        self.state['phase'] = 'completed'
                        self.persist()
                        self.refresh()
                        self.event('both_WM_arms_completed')
                        return
                    time.sleep(10)
            except Exception as error:
                self.state.update(phase='blocked', error=repr(error))
                self.persist()
                self.refresh()
                self.event('manager_error', error=repr(error))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 33), default=16)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.status:
        print((out/'WM_RESULTS.md').read_text())
    else:
        matrix = prepare(out, args.workers)
        if not args.prepare_only:
            Scheduler(matrix).run()
