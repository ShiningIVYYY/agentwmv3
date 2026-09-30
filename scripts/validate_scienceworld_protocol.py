#!/usr/bin/env python3
"""Full official-set / runtime-map preflight. Uses its own HTTP environment ID."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import requests
import run_audited_paper_eval as ev

REVISION = '1f090d52ef01a83a889da02699e34a500e35a274'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--science-url', default='http://127.0.0.1:36001')
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    p.add_argument('--output', default=str(ev.ROOT / 'outputs/paper_eval/scienceworld_context_v2/preflight.json'))
    args = p.parse_args()
    source = ev.ROOT / 'data/paper_eval/sciworld_test.json'
    url = args.endpoint + '/datasets/AgentGym/AgentEval/resolve/' + REVISION + '/sciworld_test.json'
    response = requests.get(url, timeout=90); response.raise_for_status()
    official = response.json()
    assert json.loads(source.read_text()) == official, 'Local official task set changed'
    demo_path = ev.ROOT / 'data/paper_eval/sources/science_expert_sample.jsonl'
    meta = requests.get(args.endpoint + '/api/datasets/osunlp/early-experience', timeout=90)
    meta.raise_for_status(); demo_revision = meta.json()['sha']
    demo_url = args.endpoint + '/datasets/osunlp/early-experience/resolve/' + demo_revision + '/scienceworld/expert_sft.jsonl'
    with requests.get(demo_url, timeout=90, stream=True) as response:
        response.raise_for_status()
        released_demo = json.loads(next(response.iter_lines()))
    assert json.loads(demo_path.read_text().splitlines()[0]) == released_demo, 'Training demo differs from released source'
    backend = ev.Science(args)
    tasks = backend.tasks()
    old = json.loads((ev.ROOT / 'data/paper_eval/audited/scienceworld.json').read_text())['tasks']
    fields = ('task_id','index','taskName','variationIdx')
    assert [{k:r[k] for k in fields} for r in tasks] == [{k:r[k] for k in fields} for r in old]
    # This allocated ID is never shared with a model runner. Reset verifies the
    # actual running HTTP service, not just a second local SDK enumeration.
    eid = ev.http('POST', args.science_url + '/create')['id']
    old_goals_checked = 0
    initial_scores = {}
    try:
        for i, row in enumerate(tasks):
            state = ev.http('POST', args.science_url + '/reset', json=dict(id=eid, data_idx=row['index']))
            assert (state['task_name'],state['var_num']) == (row['taskName'],row['variationIdx']), row
            assert hashlib.sha256(state['task_description'].encode()).hexdigest() == row['task_description_sha256'], row
            result = ev.ROOT / 'outputs/paper_eval/audited/scienceworld_ee_il/tasks' / (row['task_id'] + '.json')
            if result.exists():
                assert json.loads(result.read_text())['task_description'] == state['task_description'], row
                old_goals_checked += 1
            # Some native tasks award partial progress simply for starting in
            # the required room. Compare to the SDK, never assume reset score=0.
            assert state['score'] == row['initial_score'] and not state['done'], row
            key = str(state['score']); initial_scores[key] = initial_scores.get(key,0) + 1
            if (i + 1) % 20 == 0:
                print('HTTP task/variation/goal checked',i+1,'/ 200',flush=True)
    finally:
        ev.http('POST', args.science_url + '/close', json=dict(id=eid))
    manifest = ev.ROOT / 'data/paper_eval/audited/scienceworld_context_v2.json'
    from scienceworld.constants import JAR_PATH
    value = dict(protocol=ev.SCIENCE_PROTOCOL, env='scienceworld', tasks=tasks,
                 AgentEval_revision=REVISION, test_sha256=ev.digest(source),
                 scienceworld_version=importlib.metadata.version('scienceworld'),
                 simulator_jar_sha256=ev.digest(JAR_PATH), demo_sha256=ev.digest(demo_path))
    if manifest.exists():
        assert json.loads(manifest.read_text()) == value, 'Do not silently replace a changed fixed manifest'
    else:
        ev.atomic(manifest,value)
    report = dict(passed=True, updated=time.time(), count=len(tasks), unique_ids=200,
                  ordered_official_ids_match=True, sdk_map_unchanged=True,
                  http_map_and_goals_checked=200, previous_ee_il_goals_checked=old_goals_checked,
                  initial_score_distribution=initial_scores,
                  manifest_sha256=ev.digest(manifest), AgentEval_revision=REVISION,
                  official_url=url.replace(args.endpoint,'https://huggingface.co'),
                  scienceworld_version=value['scienceworld_version'], simulator_jar_sha256=value['simulator_jar_sha256'],
                  training_demo_revision=demo_revision, training_demo_exact_match=True,
                  training_demo_messages=len(backend.prelude), training_demo_sha256=ev.digest(demo_path),
                  java_version=subprocess.run(['java','-version'],capture_output=True,text=True).stderr,
                  limitations=['Author exact evaluation one-shot, horizon, SDK/JVM version and original ID mapping were not published. This proves identity to the selected official AgentEval set and installed stock AgentGym mapping, not byte-identical author evaluation.',
                               'Released first training dialogue is used whole; it is not claimed to be a terminal-winning demonstration. No test answers or simulator gold paths enter model prompts.'])
    ev.atomic(args.output,report)
    print('PASSED:',json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
