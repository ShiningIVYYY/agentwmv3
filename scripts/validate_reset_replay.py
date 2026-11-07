#!/usr/bin/env python3
"""Real SDK-only reset/replay tests on approved dev IDs, no model or gold path."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experience.splits import validate_manifest
from worldmodel.ground_truth_environment import ReplayScienceEnvironment, GroundTruthEnvironmentWM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError('New Run ID required')
    out.mkdir(parents=True)
    import hashlib
    provenance = dict(command=sys.argv, adapter_sha256=hashlib.sha256((ROOT/'worldmodel/ground_truth_environment.py').read_bytes()).hexdigest(),
                      validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      protocol='public_native_step_without_valid_enumeration/post_load_seed/strict_projection')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    manifest = json.loads((ROOT/'docs/phase3/split_manifest.json').read_text())
    validate_manifest(manifest, require_native=True)
    dev = {r['task_id']: r for r in manifest['partitions']['development']}
    ids = ['sciworld_128', 'sciworld_142', 'sciworld_2417', 'sciworld_3070']
    trials = []
    wm = GroundTruthEnvironmentWM()
    started = time.perf_counter()
    for task_id in ids:
        row = dev[task_id]
        for seed in (0, 1, 7):
            spec = dict(taskName=row['taskName'], variationIdx=row['variationIdx'], simplifications='',
                        seed=seed, environment_step_limit=21)
            env = ReplayScienceEnvironment(spec)
            try:
                for length in range(9):
                    # Same-state validation: trial B first, then normal A action.
                    action = ['look around', 'inventory', 'this is an invalid action', 'wait',
                              'open door to kitchen', 'go kitchen', 'look around', 'inventory', 'wait'][length]
                    if length in (0, 2, 4, 8):
                        result = wm.simulate_one_step(env, action)
                        actual = env.step(action)
                        match = result['prediction_available'] and all(
                            result['candidate_transition'][k] == actual[k]
                            for k in ('observation', 'reward', 'done', 'score', 'moves'))
                        trial = dict(task_id=task_id, variation=row['variationIdx'], seed=seed,
                                     history_length=length, candidate=action, simulation=result,
                                     actual=actual, transition_match=bool(match))
                        trials.append(trial)
                        with (out/'trials.jsonl').open('a') as f:
                            f.write(json.dumps(trial)+'\n')
                    else:
                        env.step(action)
                    if env.current['done']:
                        break
                (out/'progress.json').write_text(json.dumps(dict(completed_task_seed_pairs=len({(t['task_id'],t['seed']) for t in trials}),trials=len(trials)))+'\n')
            finally:
                env.close()
    summary = dict(trials=len(trials), verified=sum(t['simulation']['prediction_available'] for t in trials),
                   matched=sum(t['transition_match'] for t in trials),
                   reconstruction_failures=sum(not t['simulation']['prediction_available'] for t in trials),
                   verified_but_different=sum(t['simulation']['prediction_available'] and not t['transition_match'] for t in trials),
                   all_formal_unchanged=all(t['simulation']['formal_unchanged'] for t in trials),
                   full_hidden_state_equivalence_claimed=False, seconds=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    if summary['verified_but_different'] or not summary['all_formal_unchanged']:
        raise SystemExit(2)
    if not summary['verified']:
        raise SystemExit(42)


if __name__ == '__main__':
    main()
