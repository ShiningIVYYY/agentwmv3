"""Process-local environments with a shared, exclusively claimed task queue."""
import fcntl
import json
import os
from pathlib import Path
import time

from .matrix import read, save, sha


class TaskQueue:
    def __init__(self, output, manifest, tasks, run, workers):
        self.output = Path(output)
        self.directory = self.output / 'queue' / run
        self.tasks = tasks
        ids = [row['task_id'] for row in tasks]
        if len(ids) != len(set(ids)) or any(Path(i).name != i or i in ('.', '..') for i in ids):
            raise RuntimeError('Invalid or duplicate queue task IDs')
        self.workers = workers
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.output / 'tasks').mkdir(exist_ok=True)
        locks = self.output / 'task_locks'
        locks.mkdir(exist_ok=True)
        self.locks = locks
        config = dict(manifest_sha256=sha(manifest), task_ids=ids, workers=workers)
        with open(self.directory / '.config.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / 'config.json'
            if path.exists() and read(path) != config:
                raise RuntimeError('Task queue configuration changed')
            save(path, config)

    def publish_config(self, config):
        normalized = dict(config)
        for key in ('workers', 'task_queue', 'queue_run', 'queue_workers'):
            normalized.pop(key, None)
        normalized['output'] = str(self.output)
        normalized['execution'] = dict(mode='process-task-queue', workers=self.workers,
            partition='exclusive-task-locks', logs='independent-worker-directories')
        with open(self.output / '.config.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.output / 'config.json'
            if path.exists() and read(path) != normalized:
                raise RuntimeError('Parallel workers have different experiment configurations')
            save(path, normalized)

    def result_file(self, task_id):
        return self.output / 'tasks' / (task_id + '.json')

    def ready(self, worker):
        directory = self.directory / 'ready'
        directory.mkdir(exist_ok=True)
        save(directory / (worker + '.json'), dict(pid=os.getpid(), updated=time.time()))
        # All environments must initialize before the first cohort's work is
        # dispatched. This also makes a short preflight exercise all workers.
        while len(list(directory.glob('worker_*.json'))) < self.workers:
            time.sleep(.1)

    def claims(self):
        # Each claim's flock remains held across every action, retry and result
        # publication. The kernel releases it if a worker dies. A completed
        # result survives a crash between publication and receipt writing.
        while True:
            busy = False
            claimed = False
            for row in self.tasks:
                task_id = row['task_id']
                with open(self.locks / (task_id + '.lock'), 'a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        busy = True
                        continue
                    result = self.result_file(task_id)
                    receipt = self.directory / (task_id + '.json')
                    if result.exists() and read(result).get('status') == 'completed':
                        continue
                    if receipt.exists() and read(receipt).get('status') == 'error':
                        continue
                    save(receipt, dict(task_id=task_id, status='running', pid=os.getpid(), updated=time.time()))
                    yield row
                    final = read(result)
                    if final.get('task_id') != task_id or final.get('status') not in ('completed', 'error'):
                        raise RuntimeError('Worker did not publish a terminal task result')
                    save(receipt, dict(task_id=task_id, status=final['status'], pid=os.getpid(),
                        result_sha256=sha(result), updated=time.time()))
                    claimed = True
                    break
            if claimed:
                continue
            if not busy:
                return
            time.sleep(.25)
