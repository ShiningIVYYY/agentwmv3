"""Metadata-only50-task preregistration; no outcome-based task selection."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.independent_dev import atomic
from evaluation.alfworld_dev50 import choose_tasks, validate_split, sha, PROFILE, FREEZE
from scripts.prepare_alfworld_pilot import catalog
from scripts.run_alfworld_parallel import verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    a = ap.parse_args()
    if (ROOT / FREEZE).exists():
        raise ValueError('Preregister once; never overwrite the selected cohort')
    source = ROOT / 'outputs/alfworld_five_policy_v1'
    old = verify(source)
    prior = json.loads((ROOT / 'docs/alfworld_pilot/results/alfworld_preliminary_overlap_audit.json').read_text())
    train = catalog(Path(a.data_root), 'train')
    test = catalog(Path(a.data_root), 'valid_seen')
    old_split = old['split']
    if test != old_split['final_valid_seen']:
        raise ValueError('Final Test catalog changed')
    selected, validation = choose_tasks(train, old_split, prior)
    split = dict(schema='alfworld-independent-dev50-v1', evaluation_tasks=selected, environment_validation_tasks=validation,
        previous_pilot_evaluation_tasks=old_split['evaluation_tasks'], preliminary_probe_tasks=prior,
        memory_tasks=old_split['memory_tasks'], calibration_tasks=old_split['calibration_tasks'], final_valid_seen=test,
        assets=old_split['assets'], selection='Exclude all prior pilot/memory/calibration/preliminary goals and final140 canonical goals; SHA256 family round-robin, reserve6 disjoint physics-validation goals BEFORE selecting50; one fixed instance per goal; no outcome selection',
        source_train_instances=len(train), no_test_environment_rollout=True, no_method_selection=True)
    validate_split(split)
    atomic(ROOT / 'docs/alfworld_dev50_v1/split_manifest.json', split)
    p = dict(old['profile'])
    p.update(schema='alfworld-frozen-five-policy-independent-dev50-v1', split_manifest='docs/alfworld_dev50_v1/split_manifest.json',
        split_sha256=sha(ROOT / 'docs/alfworld_dev50_v1/split_manifest.json'),
        limits=dict(tokens=32000000, model_calls=16000, transition_queries=120000, audit_capture_bundles=140000, wall_seconds=129600),
        tasks_per_policy=50, total_episodes=250, recalibration=False, maximum_gpus=4, tasks_per_worker_batch=5,
        batch_limits=dict(tokens=4100000, model_calls=1600, transition_queries=12000, audit_capture_bundles=14000),
        gate_config='docs/alfworld_pilot/results/gate_config.json', memory_snapshot='docs/alfworld_pilot/memory_snapshot.json',
        frozen_calibration='docs/alfworld_pilot/results/calibration.json', seed_scope='one fixed rollout seed0 per independent task; not8-seed final benchmark')
    atomic(ROOT / PROFILE, p)
    names = set(old['source_hashes'])
    names.update(json.loads((ROOT / 'docs/alfworld_pilot/results/summary.json').read_text())['unchanged_shared_transitive_source_hashes'])
    names.update(['evaluation/alfworld_episode.py', 'scripts/build_alfworld_episode.py', 'scripts/run_alfworld_parallel.py',
        'scripts/analyze_alfworld_pilot.py', 'evaluation/alfworld_dev50.py', 'scripts/prepare_alfworld_dev50.py',
        'scripts/preflight_alfworld_dev50.py', 'scripts/run_alfworld_dev50.py', 'scripts/analyze_alfworld_dev50.py',
        p['gate_config'], p['memory_snapshot'], p['frozen_calibration']])
    freeze = dict(schema='alfworld-dev50-method-freeze-v1', baseline_profile=old['profile'],
        baseline_runtime_commit=old['source_commit'], baseline_report_commit='68ad762cf0aa4e5da9a02ad8133b2d576b4715dc',
        profile_sha256=sha(ROOT / PROFILE), method_hashes={n: sha(ROOT / n) for n in sorted(names)},
        actor_config_sha256=json.loads((source / 'model.json').read_text())['config_sha256'],
        reuse_prior_gate_thresholds_and_memory=True, no_reviewer_prompt_change=True, no_outcome_selection=True,
        known_prompt_limitation='Shared frozen Action placeholder retains ScienceWorld word; ALF guidance/native commands otherwise unchanged, no silent prompt repair')
    atomic(ROOT / FREEZE, freeze)
    print(json.dumps(dict(tasks=len(selected), episodes=p['total_episodes'], families=dict(Counter(r['family'] for r in selected)),
        independent_validation_tasks=len(validation), limits=p['limits']), indent=2))


if __name__ == '__main__':
    main()
