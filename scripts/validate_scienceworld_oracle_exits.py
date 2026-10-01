#!/usr/bin/env python3
"""Compare independent SDK branches against HTTP execution at unordered exits."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldmodel.backends import OracleWorldModel, same_transition, science_observation_key
from worldmodel.contexts import ScienceActionView


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    fixture = json.loads(args.replay_prefix.read_text())
    rows = json.loads((ROOT / 'data/paper_eval/audited/scienceworld_context_v2.json').read_text())['tasks']
    row = next(r for r in rows if r['task_id'] == fixture['task_id'])
    spec = importlib.util.spec_from_file_location('evaluator', ROOT / 'scripts/run_audited_paper_eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    from types import SimpleNamespace
    backend = ev.Science(SimpleNamespace(science_url='http://127.0.0.1:36001'))
    view = ScienceActionView()
    eid = ev.http('POST', backend.url + '/create')['id']
    try:
        rst = ev.http('POST', backend.url + '/reset', json=dict(id=eid, data_idx=row['index']))
        initial = backend.transition(view.observe(rst))
        wm = OracleWorldModel(lambda: backend.branch(row), initial)
        def actual_step(action):
            rst = ev.http('POST', backend.url + '/step', json=dict(id=eid, action=view.action(action)))
            return backend.transition(view.observe(rst))
        # Reconstruct the saved true prefix. predict() verifies every one of
        # these transitions in each independent JVM before predicting the exit.
        for action in fixture['actions']:
            actual = actual_step(action)
            wm.history.append((action, actual))
        predictions = [wm.predict(fixture['candidate']) for _ in range(args.repetitions)]
        actual = actual_step(fixture['candidate'])
        assert '\n\n You also see: A door to ' in actual['observation'], 'Fixture did not reach a room exit list'
        assert all(same_transition(p, actual) for p in predictions), 'Oracle still differs from HTTP execution'
        wm.commit(fixture['candidate'], actual)
        # Genuine changes must still fail, even when the door text is reordered.
        bad = dict(actual, score=actual['score'] + 1)
        assert not same_transition(predictions[-1], bad)
        text = actual['observation']
        changed = text.replace('(that is closed)', '(that is open)', 1)
        if changed == text:
            changed = text.replace('(that is open)', '(that is closed)', 1)
        assert changed != text
        assert not same_transition(predictions[-1], dict(actual, observation=changed, raw_observation=changed))
        report = dict(passed=True, fixture_only=True, task_id=row['task_id'],
            verified_prefix_steps=len(fixture['actions']), candidate=fixture['candidate'],
            independent_predictions=len(predictions), strict_text_matches=[
                p['observation'] == actual['observation'] for p in predictions],
            semantic_matches=[same_transition(p, actual) for p in predictions],
            genuine_score_change_rejected=True, genuine_door_change_rejected=True,
            prediction_observations=[p['observation'] for p in predictions],
            actual_observation=actual['observation'],
            comparison_key=science_observation_key(actual['observation']))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        ev.http('POST', backend.url + '/close', json=dict(id=eid))


if __name__ == '__main__':
    main()
