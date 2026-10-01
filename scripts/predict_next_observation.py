#!/usr/bin/env python3
"""Predict one action after an optional verified real action prefix."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldmodel.backends import RandomIsolation
from worldmodel.runtime import make_controller

spec = importlib.util.spec_from_file_location('evaluator', ROOT / 'scripts/run_audited_paper_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['alfworld','scienceworld','webshop'], required=True)
    parser.add_argument('--wm-backend', choices=['oracle','word2world'], required=True)
    parser.add_argument('--wm-config')
    parser.add_argument('--task-index', type=int, default=0, help='Index within the frozen manifest')
    parser.add_argument('--action', required=True)
    parser.add_argument('--prefix-actions', type=Path, help='JSON list of already executed action strings')
    parser.add_argument('--tokenizer-url', default='http://127.0.0.1:8105')
    parser.add_argument('--science-url', default='http://127.0.0.1:36001')
    parser.add_argument('--compare', action='store_true', help='Also execute on an isolated real instance and compare')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.reflection_max_rounds = 3
    name = {'alfworld':'alfworld','scienceworld':'scienceworld_context_v2','webshop':'webshop_index_v2'}[args.env]
    rows = json.loads((ROOT / 'data/paper_eval/audited' / (name + '.json')).read_text())['tasks']
    if not 0 <= args.task_index < len(rows):
        parser.error('--task-index is outside the frozen task manifest')
    row = rows[args.task_index]
    backend = {'alfworld':ev.Alf,'scienceworld':ev.Science,'webshop':ev.Web}[args.env](args)
    if args.env == 'webshop':
        raw, _ = backend.env.reset(session=row['index'])
        task = backend.env.instruction_text
        factory = lambda: backend.branch(row, task)
    else:
        task = ''
        factory = lambda: backend.branch(row)
    with RandomIsolation():
        step, close, initial = factory()
    try:
        if args.env == 'alfworld':
            task = initial['observation'].split('Your task is to: ',1)[1].strip()
        elif args.env == 'scienceworld':
            # The frozen goal is already embedded in the initial user context by
            # the server; obtain the exact text from a separate reset, not a test answer.
            eid = ev.http('POST', args.science_url + '/create')['id']
            try:
                task = ev.http('POST', args.science_url + '/reset', json=dict(id=eid,data_idx=row['index']))['task_description']
            finally:
                ev.http('POST', args.science_url + '/close', json=dict(id=eid))
        controller = make_controller(args, args.env, row, task, initial, factory, None, ROOT)
        prefix = json.loads(args.prefix_actions.read_text()) if args.prefix_actions else []
        if not isinstance(prefix, list) or not all(isinstance(a, str) for a in prefix):
            raise ValueError('Prefix must be a JSON list of actions')
        for action in prefix:
            if args.wm_backend == 'oracle':
                controller.wm.predict(action)
            actual = step(action)
            controller.wm.commit(action, actual)
        prediction = controller.wm.predict(args.action)
        result = dict(task_id=row['task_id'], action=args.action, prefix=prefix, prediction=prediction)
        if args.compare:
            actual = step(args.action)
            controller.wm.commit(args.action, actual)
            result.update(actual=actual, exact_match=prediction['observation'].strip() == actual['observation'].strip())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        close()


if __name__ == '__main__':
    main()
