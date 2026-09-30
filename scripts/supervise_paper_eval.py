#!/usr/bin/env python3
"""Launch and monitor nine fixed-config jobs, without losing work on logout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
CONDA = Path('/home/shn/miniconda3/envs')
SPEC = {
    'alfworld': ('ee-alfworld', 50, 1024, 140),
    # The paper omits this horizon. Retain ScienceWorld SDK's default 100;
    # explicitly disclose this operational choice, never call it paper-specified.
    'scienceworld': ('ee-scienceworld', 100, 1024, 200),
    'webshop': ('ee-webshop', 15, 512, 500),
}
IL_PORTS = {'alfworld': 8101, 'scienceworld': 8103, 'webshop': 8104}


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def child_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
                HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')


def ready(port, model=None):
    try:
        if port == 8105:
            return requests.get('http://127.0.0.1:8105/', timeout=3).status_code == 200
        response = requests.get('http://127.0.0.1:%d/v1/models' % port, timeout=3)
        return response.ok and (model is None or model in [m['id'] for m in response.json()['data']])
    except Exception:
        return False


def update(out, jobs, children):
    stats = {}
    for key, job in jobs.items():
        files = list((out / key / 'tasks').glob('*.json'))
        rows = [json.loads(f.read_text()) for f in files]
        good = [r for r in rows if r['status'] == 'completed']
        errors = [r for r in rows if r['status'] == 'error']
        proc = children.get(key)
        status = 'completed' if len(good) == job['expected'] else (
            'running' if proc and proc.poll() is None else ('incomplete' if files else 'waiting'))
        progress_file = out / key / 'progress.json'
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        stats[key] = dict(status=status, completed=len(good), total=job['expected'],
            successes=sum(r['success'] for r in good), errors=len(errors),
            success_rate=100 * sum(r['success'] for r in good) / len(good) if good else None,
            average_native_score=sum(r['score'] for r in good)/len(good) if good else None,
            pid=proc.pid if proc and proc.poll() is None else None, progress=progress)
    save(out / 'summary.json', dict(updated=time.time(), results=stats))
    lines = ['# Audited evaluation — live results', '',
        'Partial rows are not final benchmark scores. ScienceWorld = mean raw SDK score (failed tasks may return -100); full success is reported separately.', '',
        '| Model | ALFWorld success | ScienceWorld score / full success | WebShop success |',
        '|---|---|---|---|']
    for model in ('qwen25', 'qwen38', 'ee_il'):
        cells = []
        for benchmark in SPEC:
            s = stats[benchmark + '_' + model]
            if s['success_rate'] is None:
                cell = '%s; 0/%d' % (s['status'], s['total'])
            else:
                cell = '%.2f%%; %d/%d; %s' % (s['success_rate'], s['completed'], s['total'], s['status'])
                if benchmark == 'scienceworld':
                    cell = '%.2f / %s' % (s['average_native_score'], cell)
            if s['errors']:
                cell += '; errors=%d (excluded; not final)' % s['errors']
            cells.append(cell)
        lines.append('| %s | %s |' % (model, ' | '.join(cells)))
    lines += ['', 'Protocol limitations, source hashes, and fixed settings: `audit.json`. Qwen3.8 is an additional baseline, not a paper-reported model.']
    tmp = out / 'RESULTS.md.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(out / 'RESULTS.md')
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'outputs/paper_eval/audited'))
    parser.add_argument('--limit', type=int, help='Smoke only, in a separate output directory')
    parser.add_argument('--smoke-steps', type=int, help='Shortened horizon for smoke only')
    parser.add_argument('--refresh-audit-only', action='store_true', help='Refresh metadata without launching duplicate jobs')
    args = parser.parse_args()
    if args.smoke_steps and not args.limit:
        parser.error('--smoke-steps requires --limit and a separate output directory')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    lock = open(out / '.supervisor.lock', 'a')
    if not args.refresh_audit_only:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / 'supervisor.pid').write_text(str(os.getpid()))
    tokens = None
    if not ready(8105):
        tokens = subprocess.Popen([str(CONDA / 'ee-qwen38-direct/bin/python'),
            str(ROOT / 'scripts/paper_eval_tokenizers.py')], cwd=ROOT, env=child_env(),
            stdout=open(logs / 'tokenizers.log', 'a'), stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    for port, model in [(8105, None), (8100, 'qwen25'), (8101, 'ee-alfworld-il'),
                        (8103, 'ee-scienceworld-il'), (8104, 'ee-webshop-il'), (8102, None)]:
        while not ready(port, model):
            if time.time() > deadline:
                raise RuntimeError('Model/tokenizer service not ready on port %d' % port)
            time.sleep(5)
        print('Ready:', port, model, flush=True)
    data = ROOT / 'data/paper_eval/audited'
    data.mkdir(exist_ok=True)
    for benchmark, (environment, horizon, max_tokens, _) in SPEC.items():
        manifest = data / (benchmark + '.json')
        if not manifest.exists():
            subprocess.run([str(CONDA / environment / 'bin/python'),
                str(ROOT / 'scripts/run_audited_paper_eval.py'), '--env', benchmark,
                '--manifest', str(manifest), '--prepare', '--max-steps', str(horizon),
                '--max-tokens', str(max_tokens)], cwd=ROOT, env=child_env(), check=True,
                stdout=open(logs / ('prepare_' + benchmark + '.log'), 'a'), stderr=subprocess.STDOUT)
    jobs = {}
    for benchmark, (environment, horizon, max_tokens, expected) in SPEC.items():
        if args.smoke_steps:
            horizon = args.smoke_steps
        for model in ('qwen25', 'qwen38', 'ee_il'):
            if model == 'qwen25':
                port, served, weights = 8100, 'qwen25', ROOT / 'checkpoints/Qwen2.5-7B-Instruct'
            elif model == 'qwen38':
                port, served, weights = 8102, str(ROOT / 'checkpoints/Qwen3.8-27B'), ROOT / 'checkpoints/Qwen3.8-27B'
            else:
                port, served = IL_PORTS[benchmark], 'ee-' + benchmark + '-il'
                weights = ROOT / ('checkpoints/early-experience-' + benchmark + '/il')
            key = benchmark + '_' + model
            command = [str(CONDA / environment / 'bin/python'),
                str(ROOT / 'scripts/run_audited_paper_eval.py'), '--env', benchmark,
                '--manifest', str(data / (benchmark + '.json')), '--url', 'http://127.0.0.1:%d' % port,
                '--model', served, '--tokenizer-path', str(weights), '--output', str(out / key),
                '--max-steps', str(horizon), '--max-tokens', str(max_tokens)]
            if model == 'qwen38':
                command += ['--disable-thinking', '--generation-lock', str(ROOT / 'outputs/paper_eval/qwen38_generation.lock')]
            if args.limit:
                command += ['--limit', str(args.limit)]
            jobs[key] = dict(command=command, expected=min(args.limit, expected) if args.limit else expected,
                             benchmark=benchmark, model=model, weights=str(weights))
    audit = dict(protocol='audited-v1', paper='https://arxiv.org/html/2510.08558v3',
        AgentGym_commit=subprocess.check_output(['git','-C','external/AgentGym','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        verl_agent_commit=subprocess.check_output(['git','-C','external/verl-agent','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        AgentEval_revision='1f090d52ef01a83a889da02699e34a500e35a274',
        AgentEval_sciworld_sha256=hashlib.sha256((ROOT/'data/paper_eval/sciworld_test.json').read_bytes()).hexdigest(),
        manifests={e:hashlib.sha256((data/(e+'.json')).read_bytes()).hexdigest() for e in SPEC},
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [ROOT/'scripts/run_audited_paper_eval.py', ROOT/'scripts/paper_eval_tokenizers.py',
                           ROOT/'scripts/supervise_paper_eval.py', ROOT/'data/paper_eval/sources/2510.08558v3.pdf',
                           ROOT/'data/paper_eval/sources/science_expert_sample.jsonl']},
        temperature=.4, top_p=1.0, top_k={'vllm':-1, 'qwen38':0},
        repetition_penalty=1.0,
        qwen38_generation_defaults=json.loads((ROOT/'checkpoints/Qwen3.8-27B/generation_config.json').read_text()),
        prompt_limit=4096, generation_seed=0, environment_seed=1000,
        science_score_convention='Mean raw terminal SDK score, matching AgentGym rewards.mean(); failures may be -100. Full success = done and score == 100, reported separately.',
        limitations=[
            'Full heldout pools: ALFWorld valid_seen 140, AgentEval ScienceWorld 200, Verl WebShop goal indices 0:500. Paper exact val sampling IDs are unpublished; not necessarily its sampled 128-row validation denominator.',
            'ScienceWorld exact inference one-shot and horizon are unpublished. Use authors released expert TRAIN conversation first turn as fixed one-shot, Thought/Action grammar, SDK default horizon 100, recent 2 turns plus permanent task. No circuit action category removed.',
            'WebShop response limit 512 and horizon 15 come from referenced Verl example, not an explicit numeric WebShop paragraph in the paper. Small 1000-product catalog/synthetic goals match current referenced wrapper config.',
            'Qwen3.8-27B is a user-added baseline, not a model evaluated in the paper; hidden reasoning disabled. Transformers multi-GPU backend differs from 7B vLLM serving.',
            'Qwen3.8 top_k=0 and repetition_penalty=1.0 are sent through the supported generation_config JSON-string field, not unsupported top-level vLLM extensions.',
            '4096-token overflow drops oldest history; no task/action truncation. Unsupported oversized states become infrastructure errors, not benchmark failures.',
            'Full reproduction cannot be claimed without author inference prompt, horizon, validation IDs, and original framework revision.'
        ], jobs=jobs)
    save(out / 'audit.json', audit)
    if args.refresh_audit_only:
        print('Audit metadata refreshed; no jobs launched', flush=True)
        return
    children, restarts = {}, {k:0 for k in jobs}
    for key, job in jobs.items():
        children[key] = subprocess.Popen(job['command'], cwd=ROOT, env=child_env(),
            stdout=open(logs / (key + '.log'), 'a'), stderr=subprocess.STDOUT)
        print('Started', key, children[key].pid, flush=True)
    while True:
        stats = update(out, jobs, children)
        for key, process in list(children.items()):
            if process.poll() not in (None, 0) and restarts[key] < 3:
                restarts[key] += 1
                print('Restarting failed runner', key, 'exit=', process.returncode, flush=True)
                children[key] = subprocess.Popen(jobs[key]['command'], cwd=ROOT, env=child_env(),
                    stdout=open(logs / (key + '.log'), 'a'), stderr=subprocess.STDOUT)
        if all(p.poll() is not None for p in children.values()):
            update(out, jobs, children)
            print('All runners exited; inspect summary for errors/missing tasks.', flush=True)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
