#!/usr/bin/env python3
"""Check recorded real preflight evidence; never certify a full-state clone."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(directory, check_worktree=False):
    load = lambda name: json.loads((directory / name).read_text())
    probe = load('clone_preflight.json')
    metadata = load('qwen35_metadata.json')
    record = load('verification.json')
    require(probe['status'] == 'blocked_required_clone_not_validated', 'Do not promote this probe to clone acceptance')
    require(probe['isolated_clone_verified'] is False and probe['full_state_transfer_implemented'] is False,
            'This audit has no complete-state transfer implementation')
    require(record['full_state_clone_verified'] is False, 'Incorrect readiness label')
    require(probe['formal_environment_speculative_steps'] == 0, 'Formal-environment preview is forbidden')
    require(all(probe[k] == 0 for k in ('actor_calls', 'reviewer_calls', 'learned_wm_calls')),
            'No model inference is part of this prerequisite probe')
    require(metadata['metadata_only'] and not metadata['weights_downloaded']
            and not metadata['real_generation_verified'] and not metadata['thresholds_calibrated'],
            'Metadata must not be labelled as generation/calibration evidence')
    pairs = []
    pids = []
    for trial in probe['trials']:
        require(len(set(trial['distinct_jvm_pids'])) == 2, 'Instances must have distinct JVMs')
        pids.extend(trial['distinct_jvm_pids'])
        require(trial['initial_observable_projection_match'], 'Initial projections differ')
        for index, step in enumerate(trial['steps'], 1):
            require(step['reference_steps'] == step['worker_steps'] == index, 'Step accounting differs')
            require(step['full_snapshot_import_performed'] is False, 'No snapshot import was performed')
            require(step['reference_unchanged_during_worker_step'] and step['read_only_heap_witness_matches'],
                    'Reference isolation witness failed')
            fields = ('observation', 'raw_observation', 'reward', 'done', 'score', 'simulator_moves')
            require(all(step['worker'][k] == step['reference_normal_execution'][k] for k in fields),
                    'Recorded real transitions differ')
            pairs.append(step)
    require(len(set(pids)) == len(pids), 'Unexpected JVM reuse in this bounded probe')
    require(len(pairs) == record['paired_actions'] and 2 * len(pairs) == record['real_transition_calls'],
            'Cost accounting disagrees with real records')
    require(2 * len(probe['trials']) == record['initialization_step_calls'], 'Initialization calls omitted')
    for command in record['commands']:
        require(command['returncode'] == command['expected_returncode'], 'Unexpected command exit')
    require(record['commands'][0]['returncode'] == 42, 'Clone prerequisite must remain blocked')
    for name, expected in record['source_hashes'].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'Changed audited source: ' + name)
    for key, name in [('split_manifest_sha256', 'docs/phase3/split_manifest.json'),
                      ('development_plan_sha256', 'docs/phase5/development_plan.json')]:
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == probe[key], 'Changed data identity: ' + name)
    if check_worktree:
        changed = subprocess.check_output(['git', 'diff', record['baseline_commit'], '--name-only', '--']
                                         + record['protected_paths'], cwd=str(ROOT), text=True)
        require(not changed.strip(), 'Protected runtime/config/evidence changed: ' + changed)
    return dict(evidence_consistent=True, full_state_clone_verified=False, phase_status='blocked',
                paired_real_actions=len(pairs), model_generation_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-directory', type=Path, default=ROOT / 'docs/phase57')
    parser.add_argument('--check-worktree', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.evidence_directory, args.check_worktree), indent=2))


if __name__ == '__main__':
    main()
