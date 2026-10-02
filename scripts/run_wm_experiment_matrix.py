#!/usr/bin/env python3
"""Freeze baseline configs, validate each model wave, and run the full WM matrix."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.matrix import prepare, MODELS
from experiments.scheduler import MatrixScheduler


def launch(matrix):
    out = Path(matrix['output'])
    with open(out / '.launch.lock', 'a') as launching:
        fcntl.flock(launching, fcntl.LOCK_EX)
        with open(out / '.matrix.lock', 'a') as running:
            try:
                fcntl.flock(running, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('Matrix manager already running:', out)
                return 0
        argv = [sys.executable, str(ROOT / 'scripts/run_wm_experiment_matrix.py'), '--output', str(out), '--models'] + matrix['models']
        with open(out / 'manager.log', 'a') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1'))
        deadline = time.time() + 30
        while time.time() < deadline:
            if child.poll() is not None:
                raise RuntimeError('Manager startup failed; inspect ' + str(out / 'manager.log'))
            from experiments.matrix import read
            record = out / 'manager.json'
            if record.exists() and read(record).get('pid') == child.pid:
                print('Matrix manager PID=%d; output=%s' % (child.pid, out))
                return 0
            time.sleep(.5)
        raise RuntimeError('Manager startup not confirmed; inspect manager.log before retrying')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/wm_reflection/full_7b_v5')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=['qwen25','ee_il'])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--refresh-only', action='store_true')
    mode.add_argument('--launch', action='store_true', help='Launch or reuse the detached matrix manager')
    args = parser.parse_args()
    matrix = prepare(ROOT, args.output, args.models)
    if args.launch:
        return launch(matrix)
    scheduler = MatrixScheduler(matrix)
    if args.prepare_only or args.refresh_only:
        scheduler.refresh()
        print('Frozen %d experiments, %d full task instances: %s' % (
            matrix['experiment_count'], matrix['task_instances'], args.output))
        return 0
    try:
        return scheduler.run()
    except BlockingIOError:
        print('Matrix manager already running:', args.output)
        return 0
    except Exception as error:
        scheduler.state['manager_error'] = repr(error)
        scheduler.persist()
        scheduler.event('manager_error', error=repr(error))
        scheduler.refresh()
        raise


if __name__ == '__main__':
    sys.exit(main())
