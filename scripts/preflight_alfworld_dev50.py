"""Real initial identities and replay controls, no model generation or Test reset."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.independent_dev import atomic, append
from evaluation.alfworld_dev50 import load_frozen, read, sha, require, FREEZE
from worldmodel.alfworld_environment import AlfEnvironment, AlfGroundTruthWM, fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-python', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    out = Path(a.run).absolute()
    require(not out.exists(), 'Fresh preflight Run ID required')
    out.mkdir(parents=True)
    p, split, freeze = load_frozen()
    from transformers import AutoTokenizer
    from internal_signals.hf_actor import HFInternalActor
    from internal_signals.alfworld_prompts import draft_messages
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'checkpoints/qwen35_9b', local_files_only=True)
    actor = HFInternalActor.__new__(HFInternalActor)
    actor.tokenizer = tokenizer
    started = time.perf_counter()
    identities = []
    queries = captures = 0
    previous_goals = {r['goal'].strip().casefold() for r in read(ROOT / 'docs/alfworld_pilot/native_goal_identity.json')}
    for row in split['environment_validation_tasks'] + split['evaluation_tasks']:
        env = AlfEnvironment(row, a.env_python, a.data_root, out / 'native')
        queries += 1
        try:
            messages = draft_messages(env.goal, env.current['observation'], [], env.current['admissible_commands'])
            prompt_tokens = actor.count(messages)
            require(prompt_tokens <= 2048, 'Initial mandatory prompt overflow; do not replace task or silently truncate')
            require(env.goal.strip().casefold() not in previous_goals, 'Native goal overlaps prior13 tasks')
            r = dict(task_id=row['task_id'], goal=env.goal, initial_sha256=fingerprint(env.initial), formal_pid=env.pid,
                     initial_prompt_tokens=prompt_tokens, reserved_role=row['reserved_role'])
            identities.append(r)
            append(out / 'identity_receipts.jsonl', r)
            captures += env.capture_count
        finally:
            env.close()
    require(len({r['goal'].strip().casefold() for r in identities}) == 56, 'Native goal duplicates among50 eval and6 validation tasks')
    atomic(out / 'goal_identity.json', identities)
    comparisons = []
    for row in split['environment_validation_tasks']:
        env = AlfEnvironment(row, a.env_python, a.data_root, out / 'native')
        queries += 1
        try:
            env.step('look')
            rejected = env.step('open nonexistent container')
            queries += 2
            require(rejected['observation'] == 'Nothing happens.', 'Invalid-action replay control differs')
            simulation = AlfGroundTruthWM().simulate(env, 'look')
            sc = simulation['simulation_cost']
            queries += sc['initializations'] + sc['replay_steps'] + sc['candidate_steps']
            require(simulation['prediction_available'] and simulation['formal_unchanged'], 'Native replay reconstruction failed')
            actual = env.step('look')
            queries += 1
            require(actual == simulation['candidate_transition'], 'Controlled real transition differs')
            require(simulation['formal_pid'] != simulation['worker_pid'], 'Process isolation failed')
            captures += env.capture_count + sc['state_capture_bundles']
            r = dict(task_id=row['task_id'], family=row['family'], seed=row['seed'], matched=True, simulation=simulation, actual=actual,
                scope='Separate validation task, not an evaluation outcome or memory evidence')
            comparisons.append(r)
            append(out / 'replay_controls.jsonl', r)
        finally:
            env.close()
    load_frozen()
    atomic(out / 'complete.json', dict(status='pass', native_identities=len(identities), evaluation_identities=50,
        disjoint_validation_comparisons=len(comparisons), transition_queries=queries, audit_capture_bundles=captures,
        passive_inspection_queries=captures-queries, model_calls=0, tokens=0, no_test_resets=True,
        goals_sha256=sha(out / 'goal_identity.json'), freeze_sha256=sha(ROOT / FREEZE), seconds=time.perf_counter()-started))
    print((out / 'complete.json').read_text())


if __name__ == '__main__':
    main()
