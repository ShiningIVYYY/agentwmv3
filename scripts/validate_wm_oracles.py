#!/usr/bin/env python3
"""Real simulator oracle isolation checks; fixture actions are NOT model scores."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldmodel.backends import OracleWorldModel, same_transition, RandomIsolation

spec = importlib.util.spec_from_file_location('evaluator', ROOT / 'scripts/run_audited_paper_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def validate(env):
    manifest_name = dict(alfworld='alfworld', scienceworld='scienceworld_context_v2', webshop='webshop_index_v2')[env]
    rows = json.loads((ROOT / 'data/paper_eval/audited' / (manifest_name + '.json')).read_text())['tasks']
    row = rows[0]
    args = SimpleNamespace(science_url='http://127.0.0.1:36001', max_steps=4)
    backend = {'alfworld':ev.Alf, 'scienceworld':ev.Science, 'webshop':ev.Web}[env](args)
    if env == 'webshop':
        raw, _ = backend.env.reset(session=row['index'])
        task = backend.env.instruction_text
        factory = lambda: backend.branch(row, task)
        actions = ['search[laptop]', 'click[back to search]', 'search[shoes]']
    else:
        factory = lambda: backend.branch(row)
        actions = ['look', 'inventory', 'look'] if env == 'alfworld' else [
            'look around', 'focus on book', '0', 'look around']
    with RandomIsolation():
        step, close, initial = factory()
    wm = OracleWorldModel(factory, initial)
    report = []
    try:
        for action in actions:
            first = wm.predict(action)
            second = wm.predict(action)
            assert same_transition(first, second), 'Repeat oracle prediction changed'
            actual = step(action)
            assert same_transition(first, actual), 'Oracle prediction differed from real execution'
            wm.commit(action, actual)
            report.append(dict(action=action, repeat_equal=True, actual_equal=True,
                               verified_prefix_steps=first['verified_prefix_steps']))
        if env == 'webshop':
            # Target-aware purchase fixture only exercises terminal/reset semantics.
            goal = backend.env.server.goals[row['index']]
            target = goal['asin'].lower()
            fixture = ['click[back to search]', 'search[<q> ' + goal['query'] + ']']
            for action in fixture:
                prediction = wm.predict(action)
                wm.commit(action, step(action))
            for _ in range(100):
                prediction = wm.predict('click[' + target + ']')
                if prediction['observation'] != wm.history[-1][1]['observation']:
                    break
                next_action = 'click[next >]'
                prediction = wm.predict(next_action)
                wm.commit(next_action, step(next_action))
            else:
                raise RuntimeError('Target fixture not reachable')
            wm.commit('click[' + target + ']', step('click[' + target + ']'))
            for option in goal['goal_options'].values():
                action = 'click[' + option.lower() + ']'
                prediction = wm.predict(action)
                wm.commit(action, step(action))
            prediction = wm.predict('click[buy now]')
            actual = step('click[buy now]')
            wm.commit('click[buy now]', actual)
            assert actual['done'] and actual['reward'] == 1.0
            report.append(dict(action='click[buy now]', terminal=True, actual_equal=True, reward=1.0))
    finally:
        close()
    result = dict(env=env, passed=True, fixture_only=True, checks=report)
    out = ROOT / 'outputs/wm_reflection_setup' / (env + '_oracle_validation.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['alfworld','scienceworld','webshop'], required=True)
    validate(parser.parse_args().env)
