import hashlib
import json
from pathlib import Path
import subprocess
import time

ENVS = ('alfworld', 'scienceworld', 'webshop')
MODELS = ('qwen25', 'qwen38', 'ee_il')
BACKENDS = ('word2world', 'oracle')
SPEC = dict(alfworld=('ee-alfworld', 140, 50, 1024),
            scienceworld=('ee-scienceworld', 200, 100, 1024),
            webshop=('ee-webshop', 500, 15, 512))
FOLDERS = dict(alfworld='audited', scienceworld='scienceworld_context_v2', webshop='webshop_index_v2')
ARGUMENTS = ('env', 'manifest', 'url', 'model', 'tokenizer_path', 'tokenizer_url',
             'science_url', 'max_steps', 'max_tokens', 'generation_lock')
RESULT_CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def baseline_directory(root, env, model):
    return root / 'outputs/paper_eval' / FOLDERS[env] / (env + '_' + model)


def source_files(root):
    files = [root / 'scripts/run_audited_paper_eval.py', root / 'scripts/paper_eval_tokenizers.py',
             root / 'scripts/run_wm_experiment_matrix.py',
             root / 'scripts/validate_scienceworld_protocol.py', root / 'scripts/validate_webshop_index.py']
    for folder in ('worldmodel', 'reflection', 'experiments', 'configs/worldmodel'):
        files.extend(p for p in (root / folder).glob('*') if p.is_file())
    files.extend([root / 'data/paper_eval/sources/science_expert_sample.jsonl',
                  root / 'outputs/paper_eval/scienceworld_context_v2/preflight.json'])
    snapshot = root / 'outputs/wm_reflection_setup/science_snapshot'
    files.append(snapshot / 'provenance.json')
    files.extend((snapshot / 'classes').rglob('*.class'))
    return sorted(set(files))


def prepare(root, out, models=MODELS, ee_workers=1, ee_reflection_protocol='ee-conservative-v2'):
    root, out = Path(root).resolve(), Path(out).resolve()
    if not models or not set(models) <= set(MODELS):
        raise ValueError('Select known Agent models')
    models = tuple(model for model in MODELS if model in models)
    if ee_reflection_protocol not in ('ee-conservative-v2', 'agentwm-copied-v1'):
        raise ValueError('Unknown EE reflection protocol')
    target = out / 'experiment.json'
    if target.exists():
        matrix = read(target)
        if matrix['root'] != str(root) or matrix['output'] != str(out):
            raise RuntimeError('Experiment root/output changed')
        if matrix['models'] != list(models):
            raise RuntimeError('Selected models changed; use a new experiment directory')
        if matrix['policy'].get('ee_workers', 1) != ee_workers:
            raise RuntimeError('Worker count changed; use a new experiment directory')
        if matrix['policy'].get('ee_reflection_protocol', 'ee-conservative-v2') != ee_reflection_protocol:
            raise RuntimeError('Reflection protocol changed; use a new experiment directory')
        verify(matrix)
        return matrix
    protected = root / 'outputs/paper_eval'
    if out == protected or protected in out.parents:
        raise RuntimeError('Matrix outputs must be separate from baseline results')
    hashes = {str(p): sha(p) for p in source_files(root)}
    continuation = out / 'resume_provenance.json'
    if continuation.exists():
        provenance = read(continuation)
        if provenance.get('protocol') != 'decoder-repair-continuation-v1':
            raise RuntimeError('Unknown continuation provenance')
        hashes[str(continuation)] = sha(continuation)
        for key, group in provenance['groups'].items():
            for task_id, record in group['tasks'].items():
                path = out / key / 'tasks' / (task_id + '.json')
                if sha(path) != record['sha256'] or read(path).get('status') != 'completed':
                    raise RuntimeError('Inherited completed result changed: ' + str(path))
                hashes[str(path)] = record['sha256']
    profiles, jobs = {}, {}
    sampling = dict(temperature=.4, top_p=1.0, repetition_penalty=1.0,
                    generation_seed=0, environment_seed=1000, prompt_limit=4096,
                    top_k=dict(vllm=-1, qwen38=0))
    for env in ENVS:
        _, count, horizon, tokens = SPEC[env]
        audit_path = root / 'outputs/paper_eval' / FOLDERS[env] / 'audit.json'
        audit = read(audit_path)
        for key, value in sampling.items():
            if audit.get(key) != value:
                raise RuntimeError('Baseline sampling differs: %s/%s' % (env, key))
        hashes[str(audit_path)] = sha(audit_path)
        wm_config = root / 'configs/worldmodel' / (env + '.json')
        from worldmodel.backends import load_config
        wm = load_config(wm_config, root, env)
        for p in (Path(wm['checkpoint']) / 'provenance.json',
                  Path(wm['source_dir']) / 'REVISION', Path(wm['source_dir']) / 'SHA256.json'):
            hashes[str(p)] = sha(p)
        for model in models:
            key = env + '_' + model
            directory = baseline_directory(root, env, model)
            path = directory / 'config.json'
            config = read(path)
            if (config['env'], config['max_steps'], config['max_tokens'], config.get('limit')) != (env, horizon, tokens, None):
                raise RuntimeError('Baseline horizon/task configuration differs: ' + key)
            if config.get('wm_reflection') or config.get('wm_backend', 'none') != 'none':
                raise RuntimeError('Expected a pure-agent baseline: ' + key)
            if config['disable_thinking'] != (model == 'qwen38'):
                raise RuntimeError('Baseline thinking configuration differs: ' + key)
            manifest = read(config['manifest'])
            ids = [r['task_id'] for r in manifest['tasks']]
            if len(ids) != count or len(set(ids)) != count or sha(config['manifest']) != config['manifest_sha256']:
                raise RuntimeError('Baseline task manifest changed: ' + key)
            hashes[str(path)] = sha(path)
            hashes[config['manifest']] = sha(config['manifest'])
            checkpoint = Path(config['tokenizer_path'])
            for name in ('config.json', 'tokenizer_config.json', 'generation_config.json'):
                if (checkpoint / name).exists():
                    hashes[str(checkpoint / name)] = sha(checkpoint / name)
            profiles[key] = dict(config=config, baseline=str(directory), task_ids=ids,
                expected=count, model=model, env=env, wm=wm, wm_config=str(wm_config))
            for backend in BACKENDS:
                job_key = backend + '/' + key
                jobs[job_key] = dict(profile=key, backend=backend, output=str(out / job_key),
                    preflight=str(out / 'preflight' / job_key), expected=count)
    matrix = dict(protocol='wm-full-matrix-v1', root=str(root), output=str(out),
        created=time.time(), commit_at_prepare=subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        models=list(models), benchmarks=list(ENVS), backends=list(BACKENDS),
        experiment_count=len(jobs), task_instances=sum(j['expected'] for j in jobs.values()),
        sampling=sampling, reflection_max_rounds=3, profiles=profiles, jobs=jobs, hashes=hashes,
        policy=dict(smoke_limit=1, smoke_steps=2, max_runner_launches=3,
            ee_workers=ee_workers, parallel_smoke_limit=ee_workers,
            ee_reflection_protocol=ee_reflection_protocol,
            ee_selection_policy='agentwm-direct/science-critique-guarded-replan-v1' if ee_reflection_protocol == 'agentwm-copied-v1' else 'evidence-gated-repair-v1',
            front_gpu=0, wm_gpus=[1,2,3], qwen38_gpus=[4,5,6,7],
            preflight='all available-model combinations must pass before full runs',
            qwen38_dependency='all three original qwen38 baselines complete and their runners exited' if 'qwen38' in models else 'not selected',
            science_ambiguity='stable-label-order-with-native-number-translation',
            science_oracle_exits='unordered-full-door-descriptions/native-agent-text-preserved',
            science_oracle_state='live-jvm-heap-checkpoint/rollback-rejected/adopt-final-state-once',
            alfworld_native_resources='two-exclusively-owned-native-engines/cleanup-and-reuse/no-early-dlclose',
            temporary_directory='owned-worker-directory-on-experiment-volume',
            empty_prediction='explicit-missing-evidence/reflect-with-current-state',
            history='commit-real-feedback-only'))
    if 'ee_il' in models and ee_reflection_protocol == 'agentwm-copied-v1':
        matrix['policy']['ee_structured_outputs_config'] = dict(
            backend='xgrammar', disable_any_whitespace=True)
    if continuation.exists():
        matrix['policy']['continuation_provenance'] = str(continuation)
        matrix['policy']['result_cohorts'] = 'retained-before-repair-and-generated-after-repair/reported-separately'
    if 'qwen25' not in models:
        # A restore profile is service metadata only; no Qwen tasks are selected.
        restore_path = baseline_directory(root, 'alfworld', 'qwen25') / 'config.json'
        matrix['front_restore_profile'] = read(restore_path)
        matrix['hashes'][str(restore_path)] = sha(restore_path)
    save(target, matrix)
    return matrix


def verify(matrix):
    for path, expected in matrix['hashes'].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('Frozen experiment input changed: ' + path)


def worker_count(matrix, job_key):
    profile = matrix['profiles'][matrix['jobs'][job_key]['profile']]
    return matrix['policy'].get('ee_workers', 1) if profile['model'] == 'ee_il' else 1


def smoke_limit(matrix, job_key):
    workers = worker_count(matrix, job_key)
    return matrix['policy'].get('parallel_smoke_limit', workers) if workers > 1 else matrix['policy']['smoke_limit']


def command(matrix, job_key, smoke=False):
    job = matrix['jobs'][job_key]
    profile = matrix['profiles'][job['profile']]
    config = profile['config']
    root = Path(matrix['root'])
    python = '/home/shn/miniconda3/envs/' + SPEC[profile['env']][0] + '/bin/python'
    argv = [python, str(root / 'scripts/run_audited_paper_eval.py')]
    for key in ARGUMENTS:
        value = config.get(key)
        if smoke and key == 'max_steps':
            value = matrix['policy']['smoke_steps']
        if value is not None:
            argv += ['--' + key.replace('_', '-'), str(value)]
    if config['disable_thinking']:
        argv.append('--disable-thinking')
    argv += ['--output', job['preflight'] if smoke else job['output'],
             '--wm-backend', job['backend'], '--reflection-max-rounds', str(matrix['reflection_max_rounds'])]
    if profile['model'] == 'ee_il' and matrix['policy'].get('ee_reflection_protocol') == 'agentwm-copied-v1':
        argv += ['--reflection-review-protocol', 'agentwm-copied-v1']
    if job['backend'] == 'word2world':
        argv += ['--wm-config', profile['wm_config']]
    if smoke:
        argv += ['--limit', str(smoke_limit(matrix, job_key))]
    workers = worker_count(matrix, job_key)
    if workers > 1:
        argv += ['--workers', str(workers)]
    return argv


def task_rows(directory, ids):
    allowed = set(ids)
    rows = {}
    for path in (Path(directory) / 'tasks').glob('*.json'):
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        cached = RESULT_CACHE.get(str(path))
        if cached and cached[0] == signature:
            row = cached[1]
        else:
            full = read(path)
            row = {k:full[k] for k in ('task_id', 'status', 'success', 'score', 'reflection_metrics', 'error') if k in full}
            RESULT_CACHE[str(path)] = (signature, row)
        key = row['task_id']
        if key not in allowed or key in rows or path.stem != key:
            raise RuntimeError('Unexpected or duplicate task result: ' + str(path))
        rows[key] = row
    return rows


def statistics(directory, ids, cohort_ids=None):
    rows = task_rows(directory, ids)
    expected = len(ids)
    if cohort_ids is not None:
        selected = set(cohort_ids)
        if not selected <= set(ids):
            raise RuntimeError('Cohort contains unknown task IDs')
        rows = {key:row for key,row in rows.items() if key in selected}
        expected = len(selected)
    completed = {k:r for k,r in rows.items() if r['status'] == 'completed'}
    errors = {k:r for k,r in rows.items() if r['status'] == 'error'}
    count = len(completed)
    result = dict(completed=count, expected=expected, errors=len(errors),
        complete=count == expected, success_rate=100 * sum(r['success'] for r in completed.values()) / count if count else None,
        average_score=sum(r['score'] for r in completed.values()) / count if count else None)
    metrics = [r['reflection_metrics'] for r in completed.values() if r.get('reflection_metrics')]
    if metrics:
        keys = set().union(*(r.keys() for r in metrics)) - {'mean_rounds'}
        totals = {k:sum(r.get(k, 0) for r in metrics) for k in keys}
        totals['mean_rounds'] = totals['reflection_calls'] / totals['steps'] if totals['steps'] else 0
        result['reflection'] = totals
    return result
