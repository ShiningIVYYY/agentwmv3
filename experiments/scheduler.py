import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import requests

from .matrix import ENVS, BACKENDS, ARGUMENTS, command, read, save, sha, statistics, task_rows, verify, worker_count, smoke_limit
from worldmodel.backends import same_transition


def native_reflection_required(matrix, job_key):
    profile = matrix['profiles'][matrix['jobs'][job_key]['profile']]
    return (profile['model'] == 'ee_il' and
            matrix['policy'].get('ee_reflection_protocol') in ('ee-native-action-v1', 'ee-conservative-v2', 'agentwm-copied-v1'))


def process_identity(pid, proc=Path('/proc')):
    try:
        directory = proc / str(pid)
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(start_ticks=fields[19], command=argv)
    except (OSError, ValueError, IndexError):
        return None


def matches_process(record):
    current = process_identity(record.get('pid', -1))
    return bool(current and current['start_ticks'] == record.get('start_ticks') and
                current['command'][1:] == record.get('command', [])[1:])


def active_runners():
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        value = process_identity(int(entry.name))
        if value and any(Path(a).name == 'run_audited_paper_eval.py' for a in value['command']):
            rows.append(dict(pid=int(entry.name), **value))
    return rows


def decode_config_matches(argv, expected):
    if expected is None:
        return True
    try:
        actual = json.loads(argv[argv.index('--structured-outputs-config') + 1])
    except (ValueError, IndexError, TypeError):
        return False
    if not isinstance(actual, dict):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def front_decode_ready(root, config, expected, proc=Path('/proc')):
    if expected is None:
        return True
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        identity = process_identity(int(entry.name), proc)
        if not identity:
            continue
        argv = identity['command']
        try:
            if ('serve' in argv and (entry / 'cwd').resolve() == Path(root).resolve()
                    and b'CUDA_VISIBLE_DEVICES=0' in (entry / 'environ').read_bytes().split(b'\0')
                    and argv[argv.index('--served-model-name') + 1] == config['model']
                    and config['url'] == 'http://127.0.0.1:' + argv[argv.index('--port') + 1]
                    and decode_config_matches(argv, expected)):
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def model_ready(url, name, weights=None, context=None, root=None):
    try:
        response = requests.get(url.rstrip('/') + '/v1/models', timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return False
    rows = [m for m in response.json()['data'] if m['id'] == name]
    if not rows:
        raise RuntimeError('Service occupied by a different model: ' + url)
    if weights is not None:
        served = Path(rows[0].get('root', ''))
        served = served if served.is_absolute() else Path(root) / served
        if served.resolve() != Path(weights).resolve() or rows[0].get('max_model_len') != context:
            raise RuntimeError('Service checkpoint/context differs: ' + url)
    return True


class MatrixScheduler:
    def __init__(self, matrix):
        self.matrix = matrix
        self.root = Path(matrix['root'])
        self.out = Path(matrix['output'])
        self.state_path = self.out / 'runstate.json'
        self.state = read(self.state_path) if self.state_path.exists() else dict(
            matrix_sha256=sha(self.out / 'experiment.json'), jobs={}, front_index=0,
            qwen38_stage='dependency', restored=False, service_events=[])
        if self.state['matrix_sha256'] != sha(self.out / 'experiment.json'):
            raise RuntimeError('Experiment matrix changed after execution started')
        self.children = {}
        self.models = matrix.get('models', ['qwen25', 'qwen38', 'ee_il'])
        self.units = [('qwen25', None, True)] if 'qwen25' in self.models else []
        if 'ee_il' in self.models:
            self.units += [('ee_il', env, True) for env in ENVS]
        if 'qwen25' in self.models:
            self.units += [('qwen25', None, False)]
        if 'ee_il' in self.models:
            self.units += [('ee_il', env, False) for env in ENVS]
        if 'qwen38' not in self.models:
            self.state['qwen38_stage'] = 'not_selected'
        self.services_checked = False

    def persist(self):
        self.state['updated'] = time.time()
        save(self.state_path, self.state)

    def jobs_for(self, model, env=None):
        return [key for key, job in self.matrix['jobs'].items()
                if self.matrix['profiles'][job['profile']]['model'] == model and
                (env is None or self.matrix['profiles'][job['profile']]['env'] == env)]

    def key(self, job, smoke):
        return ('preflight/' if smoke else 'full/') + job

    def event(self, event, **data):
        item = dict(event=event, updated=time.time(), **data)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        with open(self.out / 'events.jsonl', 'a') as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')

    def wait_ready(self, url, name, weights=None, context=None):
        deadline = time.time() + 600
        while not model_ready(url, name, weights, context, self.root):
            if time.time() >= deadline:
                raise RuntimeError('Model did not become ready: ' + url)
            time.sleep(5)

    def ensure_shared_services(self):
        response = requests.get('http://127.0.0.1:8105/', timeout=5)
        response.raise_for_status()
        response = requests.get('http://127.0.0.1:36001/docs', timeout=5)
        response.raise_for_status()
        for env in ENVS:
            wm = next(p['wm'] for p in self.matrix['profiles'].values() if p['env'] == env)
            if not model_ready(wm['url'], wm['served_model_name'], wm['checkpoint'], wm['max_model_len'], self.root):
                subprocess.run(['/home/shn/miniconda3/envs/verl-agent/bin/python',
                    str(self.root / 'scripts/start_worldmodel_service.py'), '--env', env,
                    '--gpu', str(wm['gpu'])], cwd=self.root, check=True)
                self.wait_ready(wm['url'], wm['served_model_name'], wm['checkpoint'], wm['max_model_len'])
        self.services_checked = True

    def ensure_front_agent(self, model, env=None):
        profile = self.matrix['profiles'].get((env or 'alfworld') + '_' + model)
        config = profile['config'] if profile else self.matrix['front_restore_profile']
        decode = self.matrix['policy'].get('ee_structured_outputs_config') if model == 'ee_il' else None
        if (model_ready(config['url'], config['model'], config['tokenizer_path'], 5120, self.root)
                and front_decode_ready(self.root, config, decode)):
            return
        # Only an agent service in GPU 0 owned by this repository may be switched.
        approved = {Path(p['config']['tokenizer_path']).resolve():p['config']
                    for p in self.matrix['profiles'].values() if p['model'] != 'qwen38'}
        restore = self.matrix.get('front_restore_profile')
        if restore:
            approved[Path(restore['tokenizer_path']).resolve()] = restore
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            identity = process_identity(pid)
            if not identity:
                continue
            argv = identity['command']
            if 'serve' not in argv or '--served-model-name' not in argv or '--port' not in argv:
                continue
            try:
                cwd = (entry / 'cwd').resolve()
                environ = (entry / 'environ').read_bytes().split(b'\0')
                if cwd != self.root or b'CUDA_VISIBLE_DEVICES=0' not in environ:
                    continue
                weight = Path(argv[argv.index('serve') + 1])
                weight = (cwd / weight).resolve() if not weight.is_absolute() else weight.resolve()
                if weight not in approved:
                    continue
                old = approved[weight]
                name = argv[argv.index('--served-model-name') + 1]
                port = argv[argv.index('--port') + 1]
                if name != old['model'] or old['url'] != 'http://127.0.0.1:' + port:
                    raise RuntimeError('GPU0 service metadata mismatch; refusing to switch')
                for runner in active_runners():
                    args = runner['command']
                    if '--url' in args and args[args.index('--url') + 1] == old['url']:
                        raise RuntimeError('Agent service is still used by runner %d' % runner['pid'])
                if process_identity(pid) != identity:
                    raise RuntimeError('Service identity changed during switch')
                os.kill(pid, signal.SIGTERM)
                record = dict(event='stop_idle_gpu0_agent', pid=pid, model=name, command=argv, gpu=0)
                self.state['service_events'].append(record)
                self.persist()
                self.event(**record)
                deadline = time.time() + 180
                while process_identity(pid):
                    if time.time() >= deadline:
                        raise RuntimeError('Idle agent failed to stop gracefully')
                    time.sleep(2)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        deadline = time.time() + 180
        while True:
            used, total = [int(s.strip()) for s in subprocess.check_output([
                'nvidia-smi','-i','0','--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'], text=True).split(',')]
            if total - used >= int(total * .90):
                break
            if time.time() >= deadline:
                raise RuntimeError('GPU0 memory unavailable; no unrelated processes were stopped')
            time.sleep(3)
        port = config['url'].rsplit(':', 1)[1]
        argv = ['/home/shn/miniconda3/envs/verl-agent/bin/vllm', 'serve', config['tokenizer_path'],
            '--served-model-name', config['model'], '--tensor-parallel-size', '1', '--max-model-len', '5120',
            '--max-num-seqs', '8', '--gpu-memory-utilization', '.90', '--host', '127.0.0.1', '--port', port]
        if decode is not None:
            argv += ['--structured-outputs-config', json.dumps(decode, sort_keys=True)]
        log = self.out / 'services' / (config['model'] + '.log')
        log.parent.mkdir(exist_ok=True)
        with open(log, 'a') as stream:
            process = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1'))
        record = dict(event='start_gpu0_agent', pid=process.pid, model=config['model'], command=argv, gpu=0)
        self.state['service_events'].append(record)
        self.persist()
        self.event(**record)
        self.wait_ready(config['url'], config['model'], config['tokenizer_path'], 5120)
        if not front_decode_ready(self.root, config, decode):
            raise RuntimeError('Agent JSON whitespace constraints were not applied')

    def drive_job(self, job_key, smoke):
        key = self.key(job_key, smoke)
        job = self.matrix['jobs'][job_key]
        profile = self.matrix['profiles'][job['profile']]
        directory = Path(job['preflight'] if smoke else job['output'])
        ids = profile['task_ids'][:smoke_limit(self.matrix, job_key)] if smoke else profile['task_ids']
        record = self.state['jobs'].setdefault(key, dict(launches=0, status='queued'))
        if record['status'] in ('completed', 'failed', 'blocked_preflight'):
            return record['status']
        process = self.children.get(key)
        if process is not None:
            code = process.poll()  # Reap finished children before reading /proc.
            if code is None:
                # A result can be written just before native teardown. Even if
                # /proc changes between reads, wait for an observed exit code.
                record['status'] = 'running'
                return 'running'
            if code is not None:
                self.children.pop(key)
                exit_record = dict(pid=process.pid, returncode=code, updated=time.time())
                record.setdefault('runner_exits', []).append(exit_record)
                record['last_exit'] = exit_record
                record['exit_failed'] = code != 0
                if code != 0:
                    record['runner_failures'] = record.get('runner_failures', 0) + 1
                    if smoke and (directory / 'tasks').exists():
                        archive = directory / 'failed_attempts' / ('launch_%d' % record['launches'])
                        archive.mkdir(parents=True, exist_ok=True)
                        (directory / 'tasks').rename(archive / 'tasks')
                        exit_record['failed_tasks_archive'] = str(archive / 'tasks')
                self.event('runner_exited', job=job_key, smoke=smoke, pid=process.pid, returncode=code)
        if matches_process(record):
            record['status'] = 'running'
            return 'running'
        stats = statistics(directory, ids)
        if stats['complete'] and not record.get('exit_failed', False):
            try:
                self.validate_config(job_key, smoke)
                if smoke:
                    self.validate_smoke(job_key)
            except Exception as error:
                record.update(status='failed', error=repr(error))
                self.event('validation_failed', job=job_key, smoke=smoke, error=repr(error))
                return 'failed'
            record.update(status='completed', completed_at=time.time())
            self.event('job_completed', job=job_key, smoke=smoke)
            return 'completed'
        argv = command(self.matrix, job_key, smoke)
        existing = [r for r in active_runners() if r['command'][1:] == argv[1:]]
        if len(existing) > 1:
            raise RuntimeError('Duplicate experiment runners: ' + key)
        if existing:
            record.update(existing[0], status='running', launches=max(1, record['launches']))
            self.persist()
            return 'running'
        if record['launches'] >= self.matrix['policy']['max_runner_launches']:
            record.update(status='failed', residual_errors=stats['errors'])
            self.event('job_failed', job=job_key, smoke=smoke, completed=stats['completed'], errors=stats['errors'])
            return 'failed'
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / 'runner.log', 'a') as stream:
            process = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
                         TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1'))
        identity = process_identity(process.pid)
        if identity is None:
            identity = dict(start_ticks=None, command=argv)
        # Store the expected command even before exec() has updated /proc.
        record.update(pid=process.pid, start_ticks=identity['start_ticks'], command=argv,
                      status='running', launches=record['launches'] + 1, started=time.time())
        self.children[key] = process
        self.persist()
        self.event('runner_started', job=job_key, smoke=smoke, pid=process.pid, launch=record['launches'])
        return 'running'

    def validate_config(self, job_key, smoke):
        job = self.matrix['jobs'][job_key]
        baseline = self.matrix['profiles'][job['profile']]['config']
        actual = read(Path(job['preflight'] if smoke else job['output']) / 'config.json')
        for key in ARGUMENTS + ('disable_thinking', 'protocol', 'manifest_sha256',
                                'science_prompt', 'science_demo_sha256', 'science_demo_messages'):
            expected = self.matrix['policy']['smoke_steps'] if smoke and key == 'max_steps' else baseline.get(key)
            if actual.get(key) != expected:
                raise RuntimeError('Derived agent config differs from baseline: ' + key)
        if actual['wm_backend'] != job['backend'] or actual['reflection_max_rounds'] != 3:
            raise RuntimeError('Derived WM/reflection configuration differs')
        if actual.get('limit') != (smoke_limit(self.matrix, job_key) if smoke else baseline.get('limit')):
            raise RuntimeError('Derived task limit differs')
        if native_reflection_required(self.matrix, job_key):
            if actual.get('wm_reflection', {}).get('reflection_review_protocol') != self.matrix['policy']['ee_reflection_protocol']:
                raise RuntimeError('EE-IL reflection protocol differs')
            if self.matrix['policy'].get('ee_selection_policy') and actual['wm_reflection'].get('selection_policy') != self.matrix['policy']['ee_selection_policy']:
                raise RuntimeError('EE-IL selection policy differs')
        if worker_count(self.matrix, job_key) > 1:
            if actual.get('execution', {}).get('workers') != worker_count(self.matrix, job_key):
                raise RuntimeError('Parallel worker configuration differs')

    def validate_smoke(self, job_key):
        job = self.matrix['jobs'][job_key]
        profile = self.matrix['profiles'][job['profile']]
        for task_id in profile['task_ids'][:smoke_limit(self.matrix, job_key)]:
            self.validate_smoke_task(job_key, task_id)
        if native_reflection_required(self.matrix, job_key):
            reviews = []
            for task_id in profile['task_ids'][:smoke_limit(self.matrix, job_key)]:
                row = read(Path(job['preflight']) / 'tasks' / (task_id + '.json'))
                for step in row['steps']:
                    reflection = step['reflection']
                    if reflection.get('review_protocol') != self.matrix['policy']['ee_reflection_protocol']:
                        raise RuntimeError('EE-IL preflight used the wrong reflection protocol')
                    reviews.extend(reflection['reviews'])
            valid = sum(r['decision']['decision'] in ('keep', 'replace') for r in reviews)
            if not valid:
                raise RuntimeError('EE-IL preflight produced no valid reflection decisions')
            save(Path(job['preflight']) / 'reflection_validation.json', dict(
                protocol=self.matrix['policy']['ee_reflection_protocol'], total_reviews=len(reviews), valid_reviews=valid, passed=True))

    def validate_smoke_task(self, job_key, task_id):
        job = self.matrix['jobs'][job_key]
        row = read(Path(job['preflight']) / 'tasks' / (task_id + '.json'))
        if not row['steps']:
            raise RuntimeError('Preflight did not execute a real environment step')
        for step in row['steps']:
            reflection = step['reflection']
            conservative = reflection.get('review_protocol') == 'ee-conservative-v2'
            copied = reflection.get('review_protocol') == 'agentwm-copied-v1'
            rounds_ok = 1 <= reflection['rounds'] <= 3 or ((conservative or copied) and reflection['rounds'] == 0 and reflection.get('budget_skip'))
            physical_calls = sum(not p.get('cached', False) for p in reflection['predictions'])
            if not (rounds_ok and 1 <= physical_calls <= 4 and len(reflection['predictions']) <= (5 if conservative or copied else 4)):
                raise RuntimeError('Preflight exceeded reflection/prediction budgets')
            if copied:
                from reflection.agentwm import POLICY, reject_replan
                if reflection.get('selection_policy') != POLICY:
                    raise RuntimeError('Preflight copied AgentWM policy missing')
                if self.matrix['profiles'][job['profile']]['env'] == 'scienceworld':
                    if sum('replan' in r for r in reflection['reviews']) > 3:
                        raise RuntimeError('Preflight exceeded ScienceWorld replanning budget')
                    from reflection.agentwm import SCIENCE_REPLAN_REGEX
                    import re
                    for review in reflection['reviews']:
                        if 'replan' not in review:
                            continue
                        replan = review['replan']
                        if (replan.get('generation', {}).get('structured_outputs') != dict(regex=SCIENCE_REPLAN_REGEX)
                                or re.fullmatch(SCIENCE_REPLAN_REGEX, replan['output']) is None):
                            raise RuntimeError('ScienceWorld preflight replan did not produce the requested action format')
                    if reflection['final_action'] != reflection['initial_action']:
                        accepted = [r for r in reflection['reviews'] if r['decision']['decision'] == 'replace']
                        if not accepted or not accepted[-1].get('replan') or not accepted[-1]['critique']['needs_replan']:
                            raise RuntimeError('ScienceWorld changed action without a separate replan')
                        if reject_replan(reflection['final_action'], reflection['predictions'][-1], job['backend']):
                            raise RuntimeError('ScienceWorld committed a rejected replan')
            if conservative:
                from reflection.conservative import POLICY, usable, failure_feedback
                if reflection.get('selection_policy') != POLICY:
                    raise RuntimeError('Preflight conservative policy missing')
                if reflection['final_action'] != reflection['initial_action']:
                    if not (reflection.get('original_defect') and reflection['reviews'][-1].get('guard') == 'verified_repair'):
                        raise RuntimeError('Preflight changed action without evidence for repair')
                    last = reflection['predictions'][-1]
                    if not usable(last) or failure_feedback(last, self.matrix['profiles'][job['profile']]['env']):
                        raise RuntimeError('Preflight committed a failed or unknown repair')
            if reflection['predictions'][-1]['action'] != step['action'] or reflection['final_action'] != step['action']:
                raise RuntimeError('Preflight committed a different action than predicted')
            if job['backend'] == 'oracle' and ('actual' not in reflection or not same_transition(
                    reflection['predictions'][-1], reflection['actual'])):
                raise RuntimeError('Preflight oracle transition did not match commit')

    def drive_unit(self, model, env, smoke):
        jobs = self.jobs_for(model, env)
        if not smoke:
            prerequisites = [self.state['jobs'].get(self.key(key, True), {}).get('status')
                             for key in self.jobs_for(model)]
            if any(s in ('failed', 'blocked_preflight') for s in prerequisites):
                for key in jobs:
                    self.state['jobs'].setdefault(self.key(key, False), dict(launches=0)).update(status='blocked_preflight')
                return True
            if not all(s == 'completed' for s in prerequisites):
                return False
        if all(self.state['jobs'].get(self.key(key, smoke), {}).get('status') == 'completed' for key in jobs):
            return True
        if model != 'qwen38':
            self.ensure_front_agent(model, env)
        else:
            config = self.matrix['profiles']['alfworld_qwen38']['config']
            if not model_ready(config['url'], config['model']):
                raise RuntimeError('Original Qwen3.8 service unavailable')
        statuses = [self.drive_job(key, smoke) for key in jobs]
        return all(s in ('completed', 'failed', 'blocked_preflight') for s in statuses)

    def qwen38_dependencies(self):
        dependencies = {}
        for env in ENVS:
            profile = self.matrix['profiles'][env + '_qwen38']
            dependencies[env] = statistics(profile['baseline'], profile['task_ids'])
        paths = {p['baseline'] for p in self.matrix['profiles'].values() if p['model'] == 'qwen38'}
        active = [r['pid'] for r in active_runners() if '--output' in r['command'] and
                  r['command'][r['command'].index('--output') + 1] in paths]
        self.state['dependencies'] = dict(qwen38_baselines=dependencies, active_baseline_pids=active)
        return all(s['complete'] for s in dependencies.values()) and not active

    def refresh(self):
        results, comparison = {}, {}
        continuation_path = self.matrix['policy'].get('continuation_provenance')
        continuation = read(continuation_path) if continuation_path else {}
        for key, job in self.matrix['jobs'].items():
            profile = self.matrix['profiles'][job['profile']]
            stats = statistics(job['output'], profile['task_ids'])
            record = self.state['jobs'].get(self.key(key, False), {})
            preflight = self.state['jobs'].get(self.key(key, True), {})
            stats.update(status=record.get('status', 'queued'), preflight=preflight.get('status', 'queued'),
                pid=record.get('pid') if matches_process(record) else None, launches=record.get('launches', 0),
                runner_failures=record.get('runner_failures', 0), last_runner_exit=record.get('last_exit'))
            progress = Path(job['output']) / 'progress.json'
            stats['progress'] = read(progress) if progress.exists() else {}
            stats['worker_count'] = worker_count(self.matrix, key)
            parallel = Path(job['output']) / 'parallel_state.json'
            parallel_state = read(parallel) if parallel.exists() else {}
            stats['worker_failures'] = parallel_state.get('worker_failures', 0)
            reflection = stats.get('reflection', {})
            total = reflection.get('reflection_calls', 0)
            valid = total - reflection.get('invalid_reviews', 0)
            stats['reflection_health'] = dict(total_reviews=total, valid_reviews=valid,
                format_errors=reflection.get('review_format_errors'),
                status='pending' if not total else ('has_valid_reviews' if valid else 'no_valid_reviews'))
            if continuation:
                retained = set(continuation['groups'].get(key, {}).get('tasks', {}))
                stats['decoder_cohorts'] = {
                    'before_whitespace_fix': statistics(job['output'], profile['task_ids'], retained),
                    'after_whitespace_fix': statistics(job['output'], profile['task_ids'], set(profile['task_ids']) - retained),
                }
            results[key] = stats
        for key, profile in self.matrix['profiles'].items():
            ids = profile['task_ids']
            base_rows = task_rows(profile['baseline'], ids)
            item = dict(baseline=statistics(profile['baseline'], ids), treatments={})
            for backend in BACKENDS:
                treatment = task_rows(self.matrix['jobs'][backend + '/' + key]['output'], ids)
                paired = [i for i in ids if base_rows.get(i, {}).get('status') == 'completed' and
                          treatment.get(i, {}).get('status') == 'completed']
                item['treatments'][backend] = dict(results[backend + '/' + key], matched_tasks=len(paired),
                    paired_success_change_pp=100 * sum(treatment[i]['success']-base_rows[i]['success'] for i in paired)/len(paired) if paired else None,
                    paired_score_change=sum(treatment[i]['score']-base_rows[i]['score'] for i in paired)/len(paired) if paired else None)
                if continuation:
                    retained = set(continuation['groups'].get(backend + '/' + key, {}).get('tasks', {}))
                    for name, selected in (('before_whitespace_fix', retained),
                                           ('after_whitespace_fix', set(ids) - retained)):
                        matched = [i for i in paired if i in selected]
                        cohort = item['treatments'][backend]['decoder_cohorts'][name]
                        cohort.update(matched_tasks=len(matched),
                            improved=sum(treatment[i]['success'] and not base_rows[i]['success'] for i in matched),
                            degraded=sum(not treatment[i]['success'] and base_rows[i]['success'] for i in matched),
                            paired_success_change_pp=100 * sum(treatment[i]['success']-base_rows[i]['success'] for i in matched)/len(matched) if matched else None)
            comparison[key] = item
        complete = all(v['complete'] and v['status'] == 'completed' for v in results.values())
        save(self.out / 'summary.json', dict(updated=time.time(), complete=complete, results=results,
            dependencies=self.state.get('dependencies'), front_index=self.state['front_index'],
            qwen38_stage=self.state['qwen38_stage'], manager_pid=os.getpid()))
        save(self.out / 'comparison.json', dict(updated=time.time(), complete=complete, rows=comparison))
        lines = ['# %d agents × three benchmarks × two world models' % len(self.models), '',
            'Partial results are provisional. Infrastructure errors are excluded and shown separately. '
            'ScienceWorld reports mean SDK score and full success; ALFWorld/WebShop report success.', '',
            '| Benchmark | Agent | Pure agent | Oracle + reflection | Word2World + reflection |',
            '|---|---|---|---|---|']
        if continuation:
            lines[2:2] = ['Decoder-repair continuation: aggregate results combine retained pre-fix tasks '
                'and tasks generated after the JSON whitespace repair. These are separate cohorts; '
                'see `decoder_cohorts` in `comparison.json` and the immutable copies recorded in '
                '`resume_provenance.json`.', '']
        def cell(stats, env):
            value = '%.2f%%' % stats['success_rate'] if stats['success_rate'] is not None else '—'
            if env == 'scienceworld' and stats['average_score'] is not None:
                value = 'score %.2f; success %s' % (stats['average_score'], value)
            return '%s (%d/%d, %s, errors=%d, runner_failures=%d, worker_failures=%d)' % (value, stats['completed'], stats['expected'],
                stats.get('status', 'complete' if stats['complete'] else 'pending'), stats['errors'],
                stats.get('runner_failures', 0), stats.get('worker_failures', 0))
        for env in ENVS:
            for model in self.models:
                item = comparison[env + '_' + model]
                lines.append('| %s | %s | %s | %s | %s |' % (env, model, cell(item['baseline'], env),
                    cell(item['treatments']['oracle'], env), cell(item['treatments']['word2world'], env)))
        lines += ['', 'Frozen original configs, sampling, task IDs, and source hashes: `experiment.json`.',
            'EE-IL uses %d independent workers per WM group; Qwen2.5 remains serial. '
            'Concurrent batches preserve sampling settings but need not reproduce serial text exactly.' % self.matrix['policy'].get('ee_workers', 1),
            'Preflight results: `preflight/`. Live stage: `summary.json`. Paired comparisons: `comparison.json`.',
            'ScienceWorld ambiguity menus use a lossless stable label order with native-number translation. '
            'Oracle room exit lists compare unordered full door descriptions; native Agent text is preserved. '
            'Checkpoint predictions reaching 512 tokens are marked partial. Oracle divergence is an infrastructure error.']
        path = self.out / 'RESULTS.md'
        tmp = path.with_suffix('.md.tmp')
        tmp.write_text('\n'.join(lines) + '\n')
        tmp.replace(path)
        return complete

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / '.matrix.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state.pop('manager_error', None)
            save(self.out / 'manager.json', dict(pid=os.getpid(), **process_identity(os.getpid())))
            self.ensure_shared_services()
            self.event('matrix_started', experiments=self.matrix['experiment_count'], task_instances=self.matrix['task_instances'])
            while True:
                verify(self.matrix)
                if self.state['front_index'] < len(self.units):
                    unit = self.units[self.state['front_index']]
                    if self.drive_unit(*unit):
                        self.event('front_unit_finished', model=unit[0], env=unit[1], smoke=unit[2])
                        self.state['front_index'] += 1
                elif not self.state['restored']:
                    if 'qwen25' in self.models or self.matrix.get('front_restore_profile'):
                        self.ensure_front_agent('qwen25')
                    self.state['restored'] = True
                stage = self.state['qwen38_stage']
                if stage == 'dependency' and self.qwen38_dependencies():
                    self.state['qwen38_stage'] = 'preflight'
                    self.event('qwen38_baseline_dependency_satisfied')
                elif stage == 'preflight' and self.drive_unit('qwen38', None, True):
                    self.state['qwen38_stage'] = 'full'
                elif stage == 'full' and self.drive_unit('qwen38', None, False):
                    self.state['qwen38_stage'] = 'done'
                self.persist()
                complete = self.refresh()
                if self.state['front_index'] == len(self.units) and self.state['qwen38_stage'] in ('done', 'not_selected') and self.state['restored']:
                    self.event('matrix_finished', complete=complete)
                    return 0 if complete else 2
                time.sleep(15)
