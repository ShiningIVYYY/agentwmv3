"""Bounded/resumable50-task five-policy scheduler; accepted Episode AST reused."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.independent_dev import atomic
from evaluation.alfworld_dev50 import load_frozen, read, sha, require, completed, spent, schedule, maxima, active_workers, FREEZE, PROFILE
from scripts.run_alfworld_parallel import ReceiptLedger


def worker(run, attempt, index):
    from internal_signals.qwen35_actor import Qwen35Actor
    from experience.alfworld_memory import AlfExperienceMemory
    from evaluation.alfworld_episode import run_episode
    from scripts.analyze_alfworld_pilot import audit_episode
    p, split, freeze = load_frozen()
    manifest = read(run / 'manifest.json')
    proof = read(Path(manifest['preflight']) / 'complete.json')
    require(sha(Path(manifest['preflight']) / 'complete.json') == manifest['preflight_sha256'], 'Preflight receipt changed')
    require(sha(Path(manifest['preflight']) / 'goal_identity.json') == proof['goals_sha256'], 'Native identities changed')
    a = read(attempt / 'manifest.json')
    out = attempt / ('worker' + str(index))
    out.mkdir()
    def stop(*_):
        raise InterruptedError('Owned dev50 worker stop')
    signal.signal(signal.SIGTERM, stop)
    ledger = ReceiptLedger(out / 'ledger.json', run / 'STOP', manifest['deadline_unix'], a['worker_caps'][index], p['batch_limits'])
    started = time.perf_counter()
    try:
        actor = Qwen35Actor(ROOT / p['actor_config'], Path(manifest['model_dir']))
        actor.evaluation_left_truncation = False
        require(actor.fingerprint == freeze['actor_config_sha256'], 'Actor configuration changed')
        atomic(out / 'model.json', dict(model_id=actor.config['model_id'], revision=actor.config['model_revision'], config_sha256=actor.fingerprint,
            load_seconds=time.perf_counter()-started, loading_info=actor.loading_info, actor_reviewer_shared=True, pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        memory = AlfExperienceMemory(read(ROOT / p['memory_snapshot']), split)
        memory.model_fingerprint = actor.fingerprint
        gate = read(ROOT / p['gate_config'])
        goals = {r['task_id']: r for r in read(Path(manifest['preflight']) / 'goal_identity.json')}
        envargs = SimpleNamespace(env_python=manifest['env_python'], data_root=manifest['data_root'])
        episodes = []
        tasks_seen = []
        rows = {r['task_id']: r for r in split['evaluation_tasks']}
        batch_start = dict(ledger.state)
        for entry in a['schedule']:
            if entry['worker'] != index:
                continue
            load_frozen()
            require(sha(ROOT / FREEZE) == manifest['freeze_sha256'], 'Run freeze changed')
            require(sha(Path(manifest['preflight']) / 'goal_identity.json') == proof['goals_sha256'], 'Native identities changed during run')
            if entry['task_id'] not in tasks_seen:
                if tasks_seen and len(tasks_seen) % p['tasks_per_worker_batch'] == 0:
                    atomic(out / ('batch_' + str(len(tasks_seen)//5) + '.json'), dict(tasks=tasks_seen[-5:], completed=len(episodes),
                        cost={k: ledger.state[k]-batch_start[k] for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']}))
                    ledger.begin_batch()
                    batch_start = dict(ledger.state)
                tasks_seen.append(entry['task_id'])
            atomic(out / 'active_episode.json', dict(entry, started_at_ns=time.time_ns()))
            run_episode(row=rows[entry['task_id']], policy=entry['policy'], seed=entry['seed'], actor=actor, memory=memory,
                gate_config=gate, p=p, out=out, a=envargs, ledger=ledger, native_goals=goals, episodes=episodes)
            ep = out / 'episodes' / entry['task_id'] / entry['policy']
            result, _, _ = audit_episode(ep, entry, p, gate, memory, goals)
            atomic(ep / 'audit.json', dict(status='pass', steps_sha256=result['steps_sha256'], result=result))
        if tasks_seen:
            size=len(tasks_seen)%5 or 5
            atomic(out / ('batch_'+str((len(tasks_seen)+4)//5)+'.json'), dict(tasks=tasks_seen[-size:],completed=len(episodes),
                cost={k:ledger.state[k]-batch_start[k] for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']}))
        atomic(out / 'complete.json', dict(status='complete', episodes=len(episodes), seconds=time.perf_counter()-started, ledger=ledger.state))
    except BaseException as e:
        atomic(out / 'error.json', dict(type=type(e).__name__, message=str(e), ledger=ledger.state))
        raise


def launch(args):
    p, split, freeze = load_frozen()
    run = Path(args.run).absolute()
    preflight = Path(args.preflight).absolute()
    proof = read(preflight / 'complete.json')
    require(proof['status'] == 'pass' and proof['evaluation_identities'] == 50 and proof['disjoint_validation_comparisons'] == 6, 'Real preflight incomplete')
    require(proof['freeze_sha256'] == sha(ROOT / FREEZE) and proof['goals_sha256'] == sha(preflight / 'goal_identity.json'), 'Preflight identity changed')
    gpus = args.gpus.split(',')
    require(1 <= len(gpus) <= 4 and len(set(gpus)) == len(gpus), 'At most4 distinct GPUs')
    for gpu in gpus:
        free = int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())
        require(free >= 22000, 'GPU not sufficiently free; never displace another job')
    if not run.exists():
        require(not args.resume, 'Resume requires existing run')
        run.mkdir(parents=True)
    else:
        require(args.resume, 'Fresh Run ID required, or explicitly --resume')
    lock = (run / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not active_workers(run), 'Existing owned workers active; do not duplicate')
    if args.resume:
        manifest = read(run / 'manifest.json')
        require(manifest['freeze_sha256'] == sha(ROOT / FREEZE), 'Cannot resume with changed method')
        for k, v in dict(env_python=args.env_python, data_root=args.data_root, preflight=str(preflight), model_dir=str(Path(args.model_dir).absolute())).items():
            require(manifest[k] == v, 'Resume runtime differs: ' + k)
        require(not (run / 'complete.json').exists(), 'Already complete')
    else:
        manifest = dict(schema='alfworld-independent-dev50-run-v1', profile=p, freeze_sha256=sha(ROOT / FREEZE),
            profile_sha256=sha(ROOT / PROFILE), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            preflight=str(preflight), preflight_sha256=sha(preflight / 'complete.json'), model_dir=str(Path(args.model_dir).absolute()),
            env_python=args.env_python, data_root=args.data_root, started_ns=time.time_ns(), deadline_unix=time.time()+p['limits']['wall_seconds'],
            information_access='oracle-like native real-environment one-step lookahead', no_training=True, no_test_rollouts=True)
        atomic(run / 'manifest.json', manifest)
    prior = spent(run)
    prior['transition_queries'] = prior.get('transition_queries',0) + proof['transition_queries']
    prior['audit_capture_bundles'] = prior.get('audit_capture_bundles',0) + proof['audit_capture_bundles']
    keys = ['tokens','model_calls','transition_queries','audit_capture_bundles']
    remaining = {k: p['limits'][k]-prior.get(k,0) for k in keys}
    require(all(v>0 for v in remaining.values()), 'Resource budget exhausted')
    require(time.time() < manifest['deadline_unix'], 'Original deadline expired')
    done = completed(run)
    expected={(r['task_id'],policy,seed) for r in split['evaluation_tasks'] for policy in p['policies'] for seed in p['rollout_seeds']}
    require(set(done).issubset(expected),'Unknown completed task/policy/seed in run')
    entries = schedule(split, p['policies'], p['rollout_seeds'], done, len(gpus))
    require(bool(entries), 'No remaining work')
    attempt = run / ('attempt' + str(len(list(run.glob('attempt*')))).zfill(3))
    attempt.mkdir()
    if (run / 'STOP').exists():
        (run / 'STOP').unlink()
    caps = [{} for _ in gpus]
    raw = [Counter() for _ in gpus]
    for e in entries:
        raw[e['worker']].update(maxima(e,p))
    for key in keys:
        weight = 'model_calls' if key=='tokens' else key
        total = sum(r[weight] for r in raw)
        allocated = 0
        for i in range(len(gpus)):
            share = remaining[key]-allocated if i==len(gpus)-1 else int(remaining[key]*raw[i][weight]/total)
            caps[i][key] = share
            allocated += share
    atomic(attempt / 'manifest.json', dict(schedule=entries, worker_caps=caps, gpus=gpus, prior_spent=prior, started_ns=time.time_ns()))
    children, logs, error = [], [], None
    def stop(*_):
        raise InterruptedError('Owned dev50 scheduler stop')
    signal.signal(signal.SIGTERM, stop)
    try:
        for i, gpu in enumerate(gpus):
            log = (attempt / ('worker' + str(i) + '.log')).open('ab')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            child = subprocess.Popen([sys.executable,'-u',__file__,'--run',str(run),'--attempt',str(attempt),'--worker',str(i)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
        atomic(attempt / 'owned_workers.json', [dict(pid=c.pid,gpu=gpus[i]) for i,c in enumerate(children)])
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):
                raise RuntimeError('Worker failed; retain partial receipts, no outcome-based retry')
            if time.time() > manifest['deadline_unix']:
                raise TimeoutError('Original36h run deadline')
            n = len(list(run.glob('attempt*/worker*/episodes/*/*/audit.json')))
            atomic(run / 'progress.json', dict(stage='running', audited_completed=n, total=250, max_gpus=len(gpus), current_attempt=attempt.name,
                seconds_since_start=time.time()-manifest['started_ns']/1e9, deadline_unix=manifest['deadline_unix']))
            time.sleep(10)
        require(all(c.returncode==0 for c in children), 'Worker exit failure')
        require(len(completed(run))==250, 'Incomplete cohort; do not report final means')
        total = spent(run)
        total['transition_queries'] += proof['transition_queries']
        total['audit_capture_bundles'] += proof['audit_capture_bundles']
        require(all(total[k] <= p['limits'][k] for k in keys), 'Whole-run resource cap exceeded')
        atomic(run / 'complete.json', dict(status='complete', episodes=250, total_resource_spent=total, seconds=time.time()-manifest['started_ns']/1e9))
    except BaseException as e:
        error = dict(type=type(e).__name__, message=str(e))
        atomic(run / 'STOP', error)
        atomic(attempt / 'error.json', error)
        atomic(run / 'progress.json', dict(stage='error',error=error,completed=len(completed(run,False)),total=250))
        raise
    finally:
        if error:
            for c in children:
                try:
                    c.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    c.terminate()
        for c in children:
            try:
                c.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid, signal.SIGKILL)
                c.wait(timeout=10)
        from scripts.launch_alfworld_pilot import native_children
        residual = []
        for i in range(len(children)):
            for pid in native_children(attempt / ('worker'+str(i))):
                try:
                    os.killpg(pid,signal.SIGTERM)
                    residual.append(pid)
                except ProcessLookupError:
                    pass
        for log in logs:
            log.close()
        atomic(attempt / 'supervisor_complete.json', dict(error=error, workers_exited=[c.poll() is not None for c in children], owned_native_cleanup_pids=residual))
    from scripts.analyze_alfworld_dev50 import finalize
    finalize(run)
    atomic(run / 'progress.json', dict(stage='complete',completed=250,total=250,report='docs/ALFWORLD_DEV50_REPORT.md'))
    if args.auto_publish:
        from scripts.analyze_alfworld_dev50 import publish
        publish(run)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--env-python')
    ap.add_argument('--data-root')
    ap.add_argument('--model-dir', default=str(ROOT/'checkpoints/qwen35_9b'))
    ap.add_argument('--preflight', default='outputs/alfworld_dev50_preflight_v1')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--auto-publish', action='store_true')
    ap.add_argument('--worker', type=int)
    ap.add_argument('--attempt')
    a = ap.parse_args()
    if a.worker is not None:
        worker(Path(a.run), Path(a.attempt), a.worker)
    else:
        if not a.env_python or not a.data_root:
            ap.error('--env-python and --data-root required')
        launch(a)
