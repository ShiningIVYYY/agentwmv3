#!/usr/bin/env python3
"""Run the pinned SR baseline/oracle/Word2World comparison on eight GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.matrix import ENVS, ARGUMENTS, SPEC, read, save, sha, source_files, statistics, task_rows, verify
from experiments.scheduler import MatrixScheduler, active_runners, process_identity, matches_process, model_ready, decode_config_matches

GROUPS = ('none', 'oracle', 'word2world')
LAYOUT = [('alfworld', 0, 8201), ('scienceworld', 4, 8202), ('webshop', 5, 8203),
          ('alfworld', 6, 8211), ('scienceworld', 7, 8212)]
DECODE = dict(backend='xgrammar', disable_any_whitespace=True)
VLLM = '/home/shn/miniconda3/envs/verl-agent/bin/vllm'


def prepare(root, out, compact=False):
    layout = [('alfworld', 0, 8201), ('scienceworld', 6, 8202), ('webshop', 5, 8203)] if compact else LAYOUT
    target = out / 'experiment.json'
    if target.exists():
        matrix = read(target)
        if matrix['root'] != str(root) or matrix['output'] != str(out):
            raise RuntimeError('SR experiment root changed')
        verify(matrix)
        return matrix
    old = read(root / 'outputs/wm_reflection/ee_il_agentwm_prompts_v10_no_json_whitespace/experiment.json')
    profiles, jobs, services = {}, {}, []
    hashes = {str(p): sha(p) for p in source_files(root)}
    hashes[str(root / 'scripts/run_sr_experiments.py')] = sha(root / 'scripts/run_sr_experiments.py')
    checkpoints = {}
    for env in ENVS:
        checkpoint = root / 'checkpoints' / ('early-experience-' + env) / 'sr'
        provenance = checkpoint / 'sr_provenance.json'
        record = read(provenance)
        if record['repo_id'] != 'osunlp/early-experience-' + env or record['subfolder'] != 'sr':
            raise RuntimeError('Wrong SR checkpoint provenance')
        checkpoints[env] = record
        for name in ('config.json', 'generation_config.json', 'tokenizer_config.json',
                     'tokenizer.json', 'chat_template.jinja', 'sr_provenance.json'):
            hashes[str(checkpoint / name)] = sha(checkpoint / name)
        source = old['profiles'][env + '_ee_il']
        manifest = read(source['config']['manifest'])
        if (sha(source['config']['manifest']) != source['config']['manifest_sha256'] or
            [r['task_id'] for r in manifest['tasks']] != source['task_ids'] or
            len(source['task_ids']) != SPEC[env][1]):
            raise RuntimeError('Audited task manifest changed: ' + env)
        config = dict(source['config'])
        for key in ('protocol', 'manifest_sha256', 'science_prompt', 'science_demo_sha256', 'science_demo_messages'):
            config.pop(key, None)
        primary = next(port for e, gpu, port in layout if e == env)
        config.update(model='ee-' + env + '-sr', tokenizer_path=str(checkpoint),
                      url='http://127.0.0.1:' + str(primary), generation_lock=None,
                      output=str(out / 'none' / (env + '_ee_sr')))
        key = env + '_ee_sr'
        profiles[key] = dict(config=config, task_ids=source['task_ids'], expected=source['expected'],
            env=env, model='ee_sr', baseline=config['output'], wm=source['wm'], wm_config=source['wm_config'])
        hashes[config['manifest']] = sha(config['manifest'])
        hashes[source['wm_config']] = sha(source['wm_config'])
        for p in (Path(source['wm']['checkpoint']) / 'provenance.json',
                  Path(source['wm']['source_dir']) / 'REVISION', Path(source['wm']['source_dir']) / 'SHA256.json'):
            hashes[str(p)] = sha(p)
        for backend in GROUPS:
            jobkey = backend + '/' + key
            c = dict(config)
            if backend == 'word2world' and sum(e == env for e, gpu, port in layout) > 1:
                port = next(port for e, gpu, port in layout if e == env and gpu in (6, 7))
                c['url'] = 'http://127.0.0.1:' + str(port)
            jobs[jobkey] = dict(profile=key, backend=backend, config=c, expected=source['expected'],
                output=str(out / jobkey), preflight=str(out / 'preflight' / jobkey))
    for env, gpu, port in layout:
        profile = profiles[env + '_ee_sr']
        services.append(dict(env=env, gpu=gpu, port=port, model=profile['config']['model'],
            checkpoint=profile['config']['tokenizer_path'], url='http://127.0.0.1:' + str(port)))
    matrix = dict(protocol='sr-three-arm-v1', root=str(root), output=str(out), created=time.time(),
        commit_at_prepare=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        benchmarks=list(ENVS), models=['ee_sr'], backends=list(GROUPS), experiment_count=9,
        task_instances=sum(j['expected'] for j in jobs.values()), profiles=profiles, jobs=jobs,
        sampling=old['sampling'], reflection_max_rounds=3, sr_checkpoints=checkpoints,
        services=services, hashes=hashes, policy=dict(workers=4, smoke_limit=4, smoke_steps=3,
            max_runner_launches=3, reflection_protocol='agentwm-copied-v1', structured_outputs=DECODE,
            schedule='all-nine-preflights-then-three-environments-in-parallel/baseline-before-WM-per-environment',
            gpu_layout=layout, wm_gpus=[1, 2, 3], history='commit-real-feedback-only'))
    save(target, matrix)
    return matrix


def command(matrix, jobkey, smoke=False):
    job = matrix['jobs'][jobkey]
    profile = matrix['profiles'][job['profile']]
    c = job['config']
    python = '/home/shn/miniconda3/envs/' + SPEC[profile['env']][0] + '/bin/python'
    args = [python, str(Path(matrix['root']) / 'scripts/run_audited_paper_eval.py')]
    for key in ARGUMENTS:
        value = matrix['policy']['smoke_steps'] if smoke and key == 'max_steps' else c.get(key)
        if value is not None:
            args += ['--' + key.replace('_', '-'), str(value)]
    args += ['--output', job['preflight'] if smoke else job['output'], '--wm-backend', job['backend'],
             '--workers', str(matrix['policy']['workers'])]
    if job['backend'] != 'none':
        args += ['--reflection-max-rounds', '3', '--reflection-review-protocol', 'agentwm-copied-v1']
    if job['backend'] == 'word2world':
        args += ['--wm-config', profile['wm_config']]
    if smoke:
        args += ['--limit', str(matrix['policy']['smoke_limit'])]
    return args


def paired(matrix, key):
    job = matrix['jobs'][key]
    p = matrix['profiles'][job['profile']]
    base = task_rows(p['baseline'], p['task_ids'])
    treatment = task_rows(job['output'], p['task_ids'])
    common = sorted(k for k in base.keys() & treatment.keys()
                    if base[k]['status'] == treatment[k]['status'] == 'completed')
    improved = [k for k in common if treatment[k]['success'] and not base[k]['success']]
    degraded = [k for k in common if base[k]['success'] and not treatment[k]['success']]
    return dict(matched_tasks=len(common), improved=len(improved), degraded=len(degraded),
                net_successes=len(improved)-len(degraded), improved_task_ids=improved, degraded_task_ids=degraded)


def eligible_jobs(state):
    jobs = []
    for env in ENVS:
        base = 'none/' + env + '_ee_sr'
        jobs.append(base)
        if state['jobs'].get('full/' + base, {}).get('status') == 'completed':
            jobs.extend(backend + '/' + env + '_ee_sr' for backend in ('oracle', 'word2world'))
    return jobs


class Scheduler:
    def __init__(self, matrix):
        self.matrix = matrix
        self.root, self.out = Path(matrix['root']), Path(matrix['output'])
        path = self.out / 'runstate.json'
        self.state = read(path) if path.exists() else dict(matrix_sha256=sha(self.out/'experiment.json'), jobs={}, services={})
        if self.state['matrix_sha256'] != sha(self.out/'experiment.json'):
            raise RuntimeError('Frozen SR experiment changed')
        self.children = {}

    def persist(self):
        self.state['updated'] = time.time()
        save(self.out / 'runstate.json', self.state)

    def event(self, event, **fields):
        row = dict(event=event, updated=time.time(), **fields)
        print(json.dumps(row), flush=True)
        with (self.out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def stop_old_gpu0(self):
        allowed = (self.root/'checkpoints/Qwen2.5-7B-Instruct').resolve()
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            pid = int(proc.name)
            ident = process_identity(pid)
            if not ident:
                continue
            argv = ident['command']
            if 'serve' not in argv or '--served-model-name' not in argv:
                continue
            try:
                if ((proc/'cwd').resolve() != self.root or
                    b'CUDA_VISIBLE_DEVICES=0' not in (proc/'environ').read_bytes().split(b'\0') or
                    Path(argv[argv.index('serve')+1]).resolve() != allowed):
                    continue
                if argv[argv.index('--served-model-name')+1] != 'qwen25' or argv[argv.index('--port')+1] != '8100':
                    raise RuntimeError('Unexpected GPU0 service identity')
                for runner in active_runners():
                    if 'http://127.0.0.1:8100' in runner['command']:
                        raise RuntimeError('Qwen25 service still has a running evaluator')
                if process_identity(pid) != ident:
                    raise RuntimeError('Process changed before stop')
                self.event('stop_idle_owned_service', pid=pid, identity=ident)
                os.kill(pid, signal.SIGTERM)
                deadline = time.time()+180
                while process_identity(pid):
                    if time.time()>deadline:
                        raise RuntimeError('Owned service did not stop')
                    time.sleep(2)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue

    def services(self):
        import requests
        for url in ('http://127.0.0.1:8105/', 'http://127.0.0.1:36001/docs'):
            requests.get(url, timeout=10).raise_for_status()
        for p in self.matrix['profiles'].values():
            wm = p['wm']
            if not model_ready(wm['url'], wm['served_model_name'], wm['checkpoint'], wm['max_model_len'], self.root):
                raise RuntimeError('Existing Word2World service unavailable: '+wm['url'])
        self.stop_old_gpu0()
        for service in self.matrix['services']:
            key = str(service['port'])
            ready = model_ready(service['url'], service['model'], service['checkpoint'], 5120, self.root)
            record = self.state['services'].get(key)
            service_record = self.out/'services'/('gpu%d.json' % service['gpu'])
            if record is None and service_record.exists():
                record = read(service_record)
                if record.get('gpu') != service['gpu'] or not matches_process(record):
                    record = None
                elif (record['command'][record['command'].index('--port')+1] != key or
                      record['command'][record['command'].index('serve')+1] != service['checkpoint']):
                    raise RuntimeError('Stored SR service configuration differs')
                if record:
                    self.state['services'][key] = record
                    self.persist()
            if ready and not (record and matches_process(record) and decode_config_matches(record['command'], DECODE)):
                raise RuntimeError('SR endpoint has no matching owned service record: '+key)
            if ready or (record and matches_process(record)):
                continue
            used, total = [int(x.strip()) for x in subprocess.check_output(['nvidia-smi', '-i', str(service['gpu']),
                '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).split(',')]
            if total-used < int(total*.9):
                raise RuntimeError('GPU is occupied: '+str(service['gpu']))
            argv = [VLLM, 'serve', service['checkpoint'], '--served-model-name', service['model'],
                '--tensor-parallel-size', '1', '--max-model-len', '5120', '--max-num-seqs', '8',
                '--gpu-memory-utilization', '.90', '--host', '127.0.0.1', '--port', key,
                '--structured-outputs-config', json.dumps(DECODE, sort_keys=True)]
            log = self.out/'services'/('gpu%d.log' % service['gpu'])
            log.parent.mkdir(exist_ok=True)
            with log.open('a') as stream:
                child = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(service['gpu']), PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1'))
            ident = process_identity(child.pid)
            self.state['services'][key] = dict(pid=child.pid, **ident, gpu=service['gpu'])
            save(service_record, self.state['services'][key])
            self.persist()
            self.event('service_started', pid=child.pid, **service)
        deadline = time.time()+900
        while True:
            waiting = []
            for service in self.matrix['services']:
                if not model_ready(service['url'], service['model'], service['checkpoint'], 5120, self.root):
                    record = self.state['services'][str(service['port'])]
                    if not matches_process(record):
                        raise RuntimeError('SR service exited: GPU '+str(service['gpu']))
                    waiting.append(service['port'])
            if not waiting:
                self.event('all_model_services_ready', sr_services=len(self.matrix['services']), wm_services=3)
                return
            if time.time()>deadline:
                raise RuntimeError('SR service startup timed out: '+str(waiting))
            time.sleep(5)

    def validate(self, key, smoke):
        job = self.matrix['jobs'][key]
        profile = self.matrix['profiles'][job['profile']]
        path = Path(job['preflight'] if smoke else job['output'])
        actual = read(path/'config.json')
        for name in ARGUMENTS + ('disable_thinking',):
            expected = self.matrix['policy']['smoke_steps'] if smoke and name == 'max_steps' else job['config'].get(name)
            if actual.get(name) != expected:
                raise RuntimeError('Job configuration differs: '+key+'/'+name)
        if actual['manifest_sha256'] != sha(job['config']['manifest']):
            raise RuntimeError('Manifest changed')
        if actual.get('wm_backend', 'none') != job['backend']:
            raise RuntimeError('Wrong WM backend')
        if actual.get('execution', {}).get('workers') != self.matrix['policy']['workers']:
            raise RuntimeError('Wrong worker count')
        if job['backend'] != 'none':
            if actual['reflection_max_rounds'] != 3 or actual['reflection_review_protocol'] != 'agentwm-copied-v1':
                raise RuntimeError('Wrong reflection protocol')
        if actual.get('limit') != (self.matrix['policy']['smoke_limit'] if smoke else None):
            raise RuntimeError('Wrong task limit')
        if not smoke:
            return
        validator = object.__new__(MatrixScheduler)
        validator.matrix = self.matrix
        reviews, steps, valid_actor, truncated = [], 0, 0, 0
        for task in profile['task_ids'][:self.matrix['policy']['smoke_limit']]:
            row = read(path/'tasks'/(task+'.json'))
            if not row.get('steps'):
                raise RuntimeError('Preflight executed no environment steps')
            if job['backend'] != 'none':
                validator.validate_smoke_task(key, task)
            for step in row['steps']:
                steps += 1
                valid_actor += bool(step.get('valid_format') or step.get('admissible'))
                if job['backend'] == 'none' and step.get('reflection'):
                    raise RuntimeError('Pure SR unexpectedly used reflection')
                if job['backend'] != 'none':
                    reviews.extend(step['reflection']['reviews'])
        valid = sum(r['decision']['decision'] in ('keep', 'replace') for r in reviews)
        format_errors = sum(r.get('error_kind') == 'invalid_format' for r in reviews)
        truncated = sum(r.get('generation',{}).get('finish_reason') == 'length' for r in reviews)
        if not valid_actor or (job['backend'] != 'none' and not valid):
            raise RuntimeError('Preflight produced no valid actor/reviewer decisions')
        # Errors from model judgments remain measurable outcomes. Reject systemic
        # interface failures before any full task is allowed to start.
        if reviews and (format_errors/len(reviews) > .25 or truncated/len(reviews) > .25):
            raise RuntimeError('Preflight has excessive review format errors/truncation')
        save(path/'validation.json', dict(passed=True, steps=steps, valid_actor_steps=valid_actor,
            reviews=len(reviews), valid_reviews=valid, format_errors=format_errors, truncated_reviews=truncated))

    def drive(self, key, smoke):
        phase = ('preflight/' if smoke else 'full/')+key
        record = self.state['jobs'].setdefault(phase, dict(status='pending', launches=0))
        if record['status'] == 'completed':
            return True
        job = self.matrix['jobs'][key]
        directory = Path(job['preflight'] if smoke else job['output'])
        ids = self.matrix['profiles'][job['profile']]['task_ids']
        ids = ids[:self.matrix['policy']['smoke_limit']] if smoke else ids
        if not record.get('pid') and record.get('exit_code') == 0 and statistics(directory, ids)['complete']:
            self.validate(key, smoke)
            record['status'] = 'completed'
            self.persist()
            return True
        if record.get('pid'):
            child = self.children.get(phase)
            code = child.poll() if child else None
            if code is None and matches_process(record):
                return False
            status = read(directory/'parallel_state.json') if (directory/'parallel_state.json').exists() else {}
            code = code if code is not None else (0 if status.get('status') == 'completed' else -1)
            record.update(pid=None, exit_code=code)
            if code == 0 and statistics(directory, ids)['complete']:
                self.validate(key, smoke)
                record['status'] = 'completed'
                self.persist()
                self.event('job_completed', job=key, smoke=smoke)
                return True
            record['status'] = 'failed'
            self.event('job_failed', job=key, smoke=smoke, exit_code=code)
        limit = 1 if smoke else self.matrix['policy']['max_runner_launches']
        if record['launches'] >= limit:
            raise RuntimeError('Runner exhausted retries: '+phase)
        verify(self.matrix)
        directory.mkdir(parents=True, exist_ok=True)
        argv = command(self.matrix, key, smoke)
        with (directory/'runner.log').open('a') as stream:
            child = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1'))
        self.children[phase] = child
        record.update(pid=child.pid, **process_identity(child.pid), status='running', launches=record['launches']+1)
        self.persist()
        self.event('job_started', job=key, smoke=smoke, pid=child.pid)
        return False

    def refresh(self):
        results, comparisons = {}, {}
        for key, job in self.matrix['jobs'].items():
            p = self.matrix['profiles'][job['profile']]
            result = statistics(job['output'], p['task_ids'])
            result['status'] = self.state['jobs'].get('full/'+key, {}).get('status', 'pending')
            result['preflight'] = self.state['jobs'].get('preflight/'+key, {}).get('status', 'pending')
            results[key] = result
            if job['backend'] != 'none':
                comparisons[key] = paired(self.matrix, key)
        complete = all(r['complete'] and r['status'] == 'completed' for r in results.values())
        summary = dict(updated=time.time(), complete=complete, results=results, manager_pid=os.getpid(),
                       phase=self.state.get('phase'), error=self.state.get('error'))
        save(self.out/'summary.json', summary)
        save(self.out/'comparison.json', comparisons)
        lines = ['# SR + WM evaluation', '', '| Group | Completed | Success rate | Mean score | Errors | Status |',
                 '|---|---:|---:|---:|---:|---|']
        for key, r in results.items():
            rate = '—' if r['success_rate'] is None else '%.2f%%' % r['success_rate']
            score = '—' if r['average_score'] is None else '%.2f' % r['average_score']
            lines.append('| %s | %d/%d | %s | %s | %d | %s |' %
                (key, r['completed'], r['expected'], rate, score, r['errors'], r['status']))
        lines += ['', '| WM group vs pure SR | Matched | Failure → success | Success → failure | Net |',
                  '|---|---:|---:|---:|---:|']
        for key, c in comparisons.items():
            lines.append('| %s | %d | %d | %d | %+d |' %
                (key,c['matched_tasks'],c['improved'],c['degraded'],c['net_successes']))
        (self.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        return complete

    def run(self):
        with (self.out/'.manager.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state.pop('error', None)
            save(self.out/'manager.json', dict(pid=os.getpid(), **process_identity(os.getpid())))
            try:
                self.state['phase'] = 'services'
                self.persist()
                self.refresh()
                self.services()
                self.state['phase'] = 'preflight'
                while True:
                    done = [self.drive(key, True) for key in self.matrix['jobs']]
                    self.refresh()
                    if all(done):
                        break
                    time.sleep(5)
                self.state['phase'] = 'full'
                self.event('all_nine_preflights_passed')
                while True:
                    for key in eligible_jobs(self.state):
                        self.drive(key, False)
                    self.persist()
                    if self.refresh():
                        self.state['phase'] = 'completed'
                        self.persist()
                        self.refresh()
                        self.event('all_nine_experiments_completed')
                        return
                    time.sleep(10)
            except Exception as error:
                self.state.update(phase='blocked', error=repr(error))
                self.persist()
                self.refresh()
                self.event('manager_error', error=repr(error))
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-for-checkpoints', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--compact-gpus', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.status:
        if (out/'RESULTS.md').exists():
            print((out/'RESULTS.md').read_text())
        elif (out/'setup.json').exists():
            print(json.dumps(read(out/'setup.json'), indent=2))
        else:
            print('SR experiment has not started.')
        return
    if args.wait_for_checkpoints:
        with (out/'.setup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                ready, pending = [], []
                for env in ENVS:
                    checkpoint = ROOT/'checkpoints'/('early-experience-'+env)/'sr'
                    (ready if (checkpoint/'sr_provenance.json').exists() else pending).append(env)
                save(out/'setup.json', dict(phase='waiting_for_checkpoints' if pending else 'checkpoints_ready',
                    updated=time.time(), pid=os.getpid(), ready=ready, pending=pending))
                if not pending:
                    break
                downloader = read(ROOT/'outputs/wm_reflection_setup/sr_checkpoints/download_process.json')
                identity = process_identity(downloader['pid'])
                if not identity or not any(Path(a).name == 'download_sr_checkpoints.py' for a in identity['command']):
                    raise RuntimeError('SR downloader stopped before verification completed: '+str(pending))
                time.sleep(10)
    matrix = prepare(ROOT, out, args.compact_gpus)
    if not args.prepare_only:
        Scheduler(matrix).run()


if __name__ == '__main__':
    main()
