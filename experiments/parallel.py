"""Supervise independent evaluator processes and observe every native exit."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .matrix import read, save, statistics
from .task_queue import TaskQueue


def worker_command(argv, output, run, index, workers):
    args = list(argv)
    for flag in ('--workers', '--output'):
        position = args.index(flag)
        del args[position:position + 2]
    return [sys.executable] + args + ['--workers', '1', '--output',
        str(Path(output) / 'workers' / run / ('worker_%02d' % index)),
        '--task-queue', str(output), '--queue-run', run, '--queue-workers', str(workers)]


def run_parallel(args, root, argv=None):
    if args.task_queue or args.queue_run or args.queue_workers:
        raise RuntimeError('A worker cannot start a nested supervisor')
    if args.wm_backend == 'none' or args.generation_lock:
        raise RuntimeError('Parallel mode requires a WM and a concurrent model endpoint')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = read(args.manifest)['tasks']
    tasks = tasks[:args.limit] if args.limit else tasks
    run = 'run_' + uuid.uuid4().hex
    TaskQueue(out, args.manifest, tasks, run, args.workers)
    ids = [r['task_id'] for r in tasks]
    previous = read(out / 'parallel_state.json') if (out / 'parallel_state.json').exists() else {}
    history = previous.get('history', []) + ([previous] if previous else [])
    for record in history:
        record.pop('history', None)
    state = dict(run=run, workers=args.workers, started=time.time(), children={},
        worker_failures=previous.get('worker_failures', 0), history=history)
    processes = {}
    stopping = []
    exhausted = []
    handlers = {}

    def handle_signal(number, frame):
        stopping.append(number)

    def launch(index):
        record = state['children'].setdefault(str(index), dict(launches=0, exits=[]))
        command = worker_command(argv or sys.argv, out, run, index, args.workers)
        directory = out / 'workers' / run / ('worker_%02d' % index)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / 'runner.log', 'a') as log:
            process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONFAULTHANDLER='1'))
        processes[index] = process
        record.update(pid=process.pid, command=command, launches=record['launches'] + 1, status='running')
        print(json.dumps(dict(event='parallel_worker_started', worker=index, **record)), flush=True)

    def reap(index):
        process = processes[index]
        code = process.poll()
        if code is None:
            return False
        record = state['children'][str(index)]
        exit_record = dict(pid=process.pid, returncode=code, updated=time.time())
        record['exits'].append(exit_record)
        record.update(status='completed' if code == 0 else 'failed')
        if code != 0:
            state['worker_failures'] += 1
        del processes[index]
        print(json.dumps(dict(event='parallel_worker_exited', worker=index, **exit_record)), flush=True)
        return True

    def progress():
        workers = []
        for index, record in state['children'].items():
            path = out / 'workers' / run / ('worker_%02d' % int(index)) / 'progress.json'
            workers.append(dict(worker=int(index), pid=record['pid'], status=record['status'],
                progress=read(path) if path.exists() else {}))
        state['updated'] = time.time()
        save(out / 'parallel_state.json', state)
        save(out / 'progress.json', dict(state='parallel_running' if processes else 'finished',
            updated=state['updated'], worker_count=args.workers,
            active_workers=len(processes), workers=workers))

    with open(out / '.runner.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Do not start a second cohort if an interrupted supervisor left live
        # evaluator children; their kernel task locks also prevent duplication.
        from .scheduler import active_runners
        for runner in active_runners():
            command = runner['command']
            if '--task-queue' in command and command[command.index('--task-queue') + 1] == str(out):
                raise RuntimeError('Previous parallel worker still alive: %d' % runner['pid'])
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, handle_signal)
        try:
            for index in range(args.workers):
                launch(index)
            while processes and not stopping and not exhausted:
                for index in list(processes):
                    if reap(index):
                        record = state['children'][str(index)]
                        if record['status'] == 'failed' and record['launches'] < 3:
                            launch(index)
                        elif record['status'] == 'failed':
                            exhausted.append(index)
                progress()
                time.sleep(.5)
            if stopping:
                raise InterruptedError('Parallel supervisor received signal %d' % stopping[0])
            if exhausted:
                raise RuntimeError('Parallel worker retry budget exhausted: %s' % exhausted)
            # Task JSON alone cannot hide a SIGSEGV during native teardown.
            if any(r['status'] != 'completed' or r['exits'][-1]['returncode'] != 0
                   for r in state['children'].values()):
                raise RuntimeError('Parallel worker retry budget exhausted')
            if not statistics(out, ids)['complete']:
                raise RuntimeError('Parallel cohort ended with missing/error task results')
            state['status'] = 'completed'
            progress()
            return 0
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            deadline = time.monotonic() + 15
            while processes and time.monotonic() < deadline:
                for index in list(processes):
                    reap(index)
                if processes:
                    time.sleep(.1)
            for process in processes.values():
                if process.poll() is None:
                    process.kill()
            for index in list(processes):
                processes[index].wait()
                reap(index)
            progress()
            for number, handler in handlers.items():
                signal.signal(number, handler)
