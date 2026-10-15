#!/usr/bin/env python3
"""Run a separate, pinned pure-SR cohort with a verified compact one-shot."""
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

PYTHON = '/home/shn/miniconda3/envs/ee-scienceworld/bin/python'


def prepare(out, workers):
    path = out/'experiment.json'
    if path.exists():
        experiment = read(path)
        assert experiment['workers'] == workers
        verify(experiment)
        return experiment
    old = ROOT/'outputs/wm_reflection/ee_sr_agentwm_prompts_v2'
    config = read(old/'none/scienceworld_ee_sr/config.json')
    manifest = read(config['manifest'])
    demo = ROOT/'data/paper_eval/sources/science_short_verified_demo.json'
    provenance = ROOT/'outputs/wm_reflection_setup/science_sr_reproduction/prompt_provenance.json'
    assert read(demo)['provenance']['done'] and read(demo)['provenance']['score'] == 100
    assert read(demo)['provenance']['task_id'] not in {r['task_id'] for r in manifest['tasks']}
    inputs = source_files(ROOT) + [Path(__file__), demo, provenance,
        Path(config['manifest']), Path(config['tokenizer_path'])/'sr_provenance.json']
    experiment = dict(root=str(ROOT), output=str(out), created=time.time(), workers=workers,
        protocol='scienceworld-task-anchored-v1', task_ids=[r['task_id'] for r in manifest['tasks']],
        original_config=config, original_baseline=str(old/'none/scienceworld_ee_sr'),
        model_url='http://127.0.0.1:8292', science_demo=str(demo),
        science_context='task-anchored-v1', hashes={str(p): sha(p) for p in inputs},
        checkpoint_provenance=read(Path(config['tokenizer_path'])/'sr_provenance.json'),
        author_score=66.0, author_exact_eval_prompt_available=False,
        purpose='Repair demonstrated context contamination; measure rather than assume a reproduced score.')
    save(path, experiment)
    return experiment


def command(experiment, smoke):
    c = experiment['original_config']
    out = Path(experiment['output'])
    directory = out/('preflight' if smoke else 'none')/'scienceworld_ee_sr'
    args = [PYTHON, str(ROOT/'scripts/run_audited_paper_eval.py')]
    settings = dict(env='scienceworld', manifest=c['manifest'], url=experiment['model_url'],
        model=c['model'], tokenizer_path=c['tokenizer_path'], tokenizer_url=c['tokenizer_url'],
        science_url=c['science_url'], max_steps=3 if smoke else c['max_steps'],
        max_tokens=c['max_tokens'], output=str(directory), wm_backend='none',
        workers=4 if smoke else experiment['workers'],
        science_context=experiment['science_context'], science_demo=experiment['science_demo'])
    for key, value in settings.items():
        args += ['--'+key.replace('_', '-'), str(value)]
    if smoke:
        args += ['--limit', '4']
    return args, directory


def run(experiment):
    out = Path(experiment['output'])
    state_path = out/'runstate.json'
    state = read(state_path) if state_path.exists() else dict(jobs={})
    children = {}
    with (out/'.manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(out/'manager.json', dict(pid=os.getpid(), **process_identity(os.getpid())))
        for smoke in (True, False):
            stage = 'preflight' if smoke else 'full'
            ids = experiment['task_ids'][:4] if smoke else experiment['task_ids']
            argv, directory = command(experiment, smoke)
            job = state['jobs'].setdefault(stage, dict(launches=0))
            while True:
                result = statistics(directory, ids)
                child = children.get(stage)
                alive = bool(job.get('pid') and matches_process(job))
                native = read(directory/'parallel_state.json') if (directory/'parallel_state.json').exists() else {}
                clean_exit = native.get('status') == 'completed'
                state.update(stage=stage, updated=time.time())
                save(state_path, state)
                save(out/'summary.json', dict(updated=time.time(), stage=stage,
                    complete=not smoke and result['complete'] and clean_exit,
                    results=result, original_pure_success_rate=37.0, reported_author_rate=66.0))
                if not alive and result['complete'] and clean_exit:
                    if smoke:
                        rows = [read(directory/'tasks'/(task+'.json')) for task in ids]
                        assert all(r['steps'] for r in rows)
                        assert all(s['valid_format'] for r in rows for s in r['steps'])
                        assert all(not s.get('reflection') for r in rows for s in r['steps'])
                        save(directory/'validation.json', dict(passed=True, steps=sum(len(r['steps']) for r in rows)))
                    job['status'] = 'completed'
                    save(state_path, state)
                    break
                if not alive:
                    if child:
                        job['last_exit_code'] = child.poll()
                    if job['launches'] >= 3:
                        raise RuntimeError('Evaluation retries exhausted: '+stage)
                    verify(experiment)
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory/'runner.log').open('a') as stream:
                        child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1'))
                    children[stage] = child
                    job.update(pid=child.pid, **process_identity(child.pid),
                               status='running', launches=job['launches']+1)
                    save(state_path, state)
                    print(json.dumps(dict(event='started', stage=stage, pid=child.pid)), flush=True)
                time.sleep(10)
        print('Corrected pure-SR ScienceWorld cohort completed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 33))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run(prepare(out, args.workers))
