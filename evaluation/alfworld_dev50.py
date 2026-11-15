"""Frozen-method, task-level scheduling helpers for the authorized50-game dev run."""
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'configs/evaluation/alfworld_dev50_v1.json'
FREEZE = 'docs/alfworld_dev50_v1/freeze.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def choose_tasks(train, old, preliminary, count=50):
    excluded = {r['goal_group'] for r in old['final_valid_seen'] + old['memory_tasks'] + old['calibration_tasks'] + old['evaluation_tasks'] + preliminary}
    candidates = defaultdict(list)
    for row in train:
        if row['source_split'] == 'train' and row['goal_group'] not in excluded:
            candidates[row['goal_group']].append(row)
    families = defaultdict(list)
    key = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
    for group, values in candidates.items():
        families[values[0]['family']].append(group)
    for family in families:
        families[family].sort(key=lambda g: key(['alfworld-dev50-frozen-v1', g]))
    validation = []
    for family in sorted(families):
        require(bool(families[family]), 'Missing validation family')
        g = families[family].pop(0)
        validation.append(dict(min(candidates[g], key=lambda r: key(['alfworld-dev50-instance-v1', r['game']])), reserved_role='environment_validation'))
    selected = []
    while len(selected) < count and any(families.values()):
        for family in sorted(families):
            if families[family] and len(selected) < count:
                g = families[family].pop(0)
                selected.append(dict(min(candidates[g], key=lambda r: key(['alfworld-dev50-instance-v1', r['game']])), reserved_role='independent_development_evaluation'))
    require(len(selected) == count and len(validation) == 6, 'Insufficient disjoint goals')
    return selected, validation


def validate_split(split):
    roles = ['memory_tasks', 'calibration_tasks', 'evaluation_tasks', 'environment_validation_tasks', 'previous_pilot_evaluation_tasks', 'final_valid_seen']
    sets = [{r['goal_group'] for r in split[k]} for k in roles]
    require(len(split['evaluation_tasks']) == 50 and len(sets[2]) == 50, 'Need50 distinct evaluation goals')
    require(len(split['final_valid_seen']) == 140, 'Final valid_seen identity differs')
    for i in range(len(sets)):
        for j in range(i):
            require(not sets[i] & sets[j], 'Role goal leakage: ' + roles[i] + '/' + roles[j])
    prior = {r['goal_group'] for r in split['preliminary_probe_tasks']}
    require(not prior & (sets[2] | sets[3] | sets[5]), 'Prior probe enters evaluation/validation/Test')
    require(all(r['source_split'] == 'train' for k in roles[:-1] for r in split[k]), 'Non-TRAIN source')


def load_frozen():
    freeze = read(ROOT / FREEZE)
    require(sha(ROOT / PROFILE) == freeze['profile_sha256'], 'Dev50 profile changed')
    for name, digest in freeze['method_hashes'].items():
        require(sha(ROOT / name) == digest, 'Frozen method changed: ' + name)
    p = read(ROOT / PROFILE)
    for k, v in freeze['baseline_profile'].items():
        if k not in ['schema', 'split_manifest', 'split_sha256', 'limits']:
            require(p[k] == v, 'Pilot method/config changed: ' + k)
    require(sha(ROOT / p['split_manifest']) == p['split_sha256'], 'Dev50 split changed')
    split = read(ROOT / p['split_manifest'])
    validate_split(split)
    return p, split, freeze


def completed(run, verify_hash=True):
    found = {}
    for path in sorted(Path(run).glob('attempt*/worker*/episodes/*/*/complete.json')):
        result = read(path)
        key = (result['task_id'], result['policy'], result['rollout_seed'])
        require(key not in found, 'Duplicate completed Episode; do not average retries')
        if verify_hash:
            require(sha(path.parent / 'steps.jsonl') == result['steps_sha256'], 'Completed trajectory changed')
        found[key] = dict(path=str(path.parent), result=result)
    return found


def spent(run):
    total = Counter()
    for p in Path(run).glob('attempt*/worker*/ledger.json'):
        v = read(p)
        require(v['pending_model'] is None and v['pending_transition'] is None, 'Interrupted request has unknown receipt; audit before resume: ' + str(p))
        for k in ['tokens', 'model_calls', 'transition_queries', 'audit_capture_bundles']:
            total[k] += v[k]
    return dict(total)


def schedule(split, policies, seeds, done, workers):
    require(1 <= workers <= 4, 'At most4 workers')
    result = []
    for i, row in enumerate(split['evaluation_tasks']):
        for seed in seeds:
            for policy in policies:
                key = (row['task_id'], policy, seed)
                if key not in done:
                    result.append(dict(task_id=row['task_id'], policy=policy, seed=seed, worker=i % workers, task_ordinal=i))
    return result


def maxima(entry, p):
    h = p['maximum_interaction_turns']
    b = p['wm_invocation_budgets'][entry['policy']]
    return dict(model_calls=h+b, transition_queries=1+h+sum(i+2 for i in range(h-b, h)),
                audit_capture_bundles=1+h+sum(i+4 for i in range(h-b, h)))


def active_workers(run):
    rows = []
    for line in subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines():
        if 'scripts/run_alfworld_dev50.py' in line and '--worker' in line and str(Path(run).absolute()) in line:
            pid, args = line.strip().split(None, 1)
            rows.append(dict(pid=int(pid), args=args))
    return rows
