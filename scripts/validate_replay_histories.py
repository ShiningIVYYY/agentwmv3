#!/usr/bin/env python3
"""Real multi-JVM validation using previously recorded, lawful dev action histories.

Commands are fixed from earlier real Full trajectories; this is an infrastructure
regression, not a new Actor evaluation or comparison against the old physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experience.splits import validate_manifest, file_hash
from worldmodel.ground_truth_environment import ReplayScienceEnvironment, GroundTruthEnvironmentWM, state_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'docs/phase3/split_manifest.json').read_text())
    validate_manifest(manifest, require_native=True)
    dev = {r['task_id']: r for r in manifest['partitions']['development']}
    source = Path(a.reference)/'evaluation_steps.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    sequences = {}
    for task in ('sciworld_128','sciworld_2417','sciworld_3070'):
        seq = sorted([r for r in rows if r['task_id']==task and r['policy']=='full'],key=lambda r:r['step'])
        if len(seq)!=20 or [r['step'] for r in seq]!=list(range(20)):
            raise ValueError('Expected complete recorded dev episode')
        sequences[task] = [r['final_action'] for r in seq]
    (out/'manifest.json').write_text(json.dumps(dict(reference_sha256=file_hash(source),
        adapter_sha256=file_hash(ROOT/'worldmodel/ground_truth_environment.py'),
        simulator_profile='scienceworld130', seeds=[0,1,7],
        source='previous_real_full_actions_reexecuted_as_fixed_commands_not_new_actor_results',
        full_hidden_state_equivalence_claimed=False),indent=2)+'\n')
    wm=GroundTruthEnvironmentWM(); trials=[]; negative=[]; started=time.perf_counter()
    for task, commands in sequences.items():
        row=dev[task]
        for seed in (0,1,7):
            env=ReplayScienceEnvironment(dict(taskName=row['taskName'],variationIdx=row['variationIdx'],
                simplifications='',seed=seed,environment_step_limit=21,simulator_profile='scienceworld130'))
            try:
                for step,action in enumerate(commands):
                    simulation=None
                    if step in (0,1,4,8,12,16,19):
                        simulation=wm.simulate_one_step(env,action)
                        if not simulation['prediction_available']:
                            raise RuntimeError('Real-history reconstruction failed: '+json.dumps(simulation))
                        if step==0:
                            again=wm.simulate_one_step(env,action)
                            if not again['prediction_available'] or again['candidate_state_sha256']!=simulation['candidate_state_sha256']:
                                raise RuntimeError('Repeated independent workers disagree')
                            negative.append(dict(test='repeated_independent_worker',task_id=task,seed=seed,
                                first_worker=simulation['worker_pid'],second_worker=again['worker_pid'],passed=True,
                                simulation=again))
                    actual=env.step(action)
                    if simulation:
                        exact=simulation['candidate_state_sha256']==state_fingerprint(env.capture())
                        public=all(simulation['candidate_transition'][k]==actual[k]
                            for k in ('observation','reward','done','score','moves'))
                        trial=dict(task_id=task,seed=seed,history_length=step,action=action,
                            simulation=simulation,actual=actual,public_match=public,projection_match=exact)
                        trials.append(trial)
                        with (out/'trials.jsonl').open('a') as f:f.write(json.dumps(trial)+'\n')
                        if not exact or not public:raise RuntimeError('Accepted worker and formal transition differ')
                    if step==0 and not actual['done']:
                        original=env.history[0]['after']['object_tree_sha256']
                        env.history[0]['after']['object_tree_sha256']='intentional_corrupted_journal_for_negative_test'
                        try:failed=wm.simulate_one_step(env,'look around')
                        finally:env.history[0]['after']['object_tree_sha256']=original
                        safe=(not failed['prediction_available'] and failed['status']=='reconstruction_failed'
                            and 'candidate_transition' not in failed and 'predicted_observation' not in failed
                            and failed['simulation_cost']['candidate_step_calls']==0 and failed['formal_unchanged'])
                        negative.append(dict(test='corrupt_evidence_rejected',task_id=task,seed=seed,passed=safe,simulation=failed))
                        if not safe:raise RuntimeError('Reconstruction failure leaked prediction')
                    if actual['done']:break
                (out/'progress.json').write_text(json.dumps(dict(task=task,seed=seed,trials=len(trials)))+'\n')
            finally:env.close()
    (out/'diagnostics.json').write_text(json.dumps(negative,indent=2)+'\n')
    simulations=[t['simulation'] for t in trials]+[t['simulation'] for t in negative]
    result=dict(trials=len(trials),verified=len(trials),public_matched=sum(t['public_match'] for t in trials),
        full_projection_matched=sum(t['projection_match'] for t in trials),
        repeated_worker_checks=sum(t['test']=='repeated_independent_worker' for t in negative),
        corrupted_journal_checks=sum(t['test']=='corrupt_evidence_rejected' for t in negative),
        negative_checks_passed=all(t['passed'] for t in negative),
        all_formal_unchanged=all(t['formal_unchanged'] for t in simulations),
        costs={k:sum(t['simulation_cost'][k] for t in simulations) for k in
            ('jvm_initializations','initialization_step_calls','replay_step_calls','candidate_step_calls')},
        simulation_latency=sum(t['simulation_latency'] for t in simulations),seconds=time.perf_counter()-started,
        full_hidden_state_equivalence_claimed=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
