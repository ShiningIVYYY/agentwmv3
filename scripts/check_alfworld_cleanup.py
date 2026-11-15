"""Read-only ownership check after the bounded pilot; never kill other jobs."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.independent_dev import atomic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', default='outputs/alfworld_final_cleanup.json')
    args = ap.parse_args()
    run = Path(args.run).absolute()
    manifest = json.loads((run / 'manifest.json').read_text())
    scopes = [run, Path(manifest['source_run'])] + [Path(f['run']) for f in manifest.get('failed_prior_runs', [])]
    scripts = ['scripts/alfworld_worker.py', 'scripts/run_alfworld_parallel.py', 'scripts/run_alfworld_pilot.py', 'scripts/launch_alfworld_pilot.py']
    ps = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    active = []
    for line in ps.splitlines():
        if any(s in line for s in scripts) and any(p.name in line for p in scopes):
            pid, command = line.strip().split(None, 1)
            active.append(dict(pid=int(pid), command=command))
    supervisor = json.loads((run / 'supervisor_complete.json').read_text())
    historical = subprocess.run(['git', '-C', str(ROOT.parent / 'agentwmv2'), 'status', '--short'], capture_output=True, text=True, check=False)
    receipt = dict(run=str(run.relative_to(ROOT)), checked_at_unix=time.time(),
        active_owned_processes=active, supervisor_workers_exited=supervisor['workers_exited'],
        historical_repo_check_exit=historical.returncode, historical_repo_status=historical.stdout,
        scope='Owned ALFWorld native/model/supervisor processes only; no other job stopped', process_kills_performed=0)
    atomic(Path(args.output), receipt)
    print(json.dumps(receipt, indent=2))
    if active or not all(supervisor['workers_exited']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
