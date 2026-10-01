#!/usr/bin/env python3
"""Official heldout one-step fixtures, separate from Agent benchmark scores."""
import argparse
import json
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldmodel.backends import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['alfworld','scienceworld','webshop'], required=True)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--data-root', type=Path, default=Path('/data1/shn/Word2World/data/llama_factory'))
    args = parser.parse_args()
    config = load_config(ROOT / 'configs/worldmodel' / (args.env + '.json'), ROOT, args.env)
    name = {'alfworld':'alfworld_test_with_env_195.json','scienceworld':'sciworld_test_with_env_195.json',
            'webshop':'webshop_test_109.json'}[args.env]
    items = json.loads((args.data_root / name).read_text())[:args.samples]
    if not items:
        raise ValueError('No official fixture samples')
    rows = []
    for item in items:
        messages = item['messages'][:2]
        target = item['messages'][2]['content']
        response = requests.post(config['url'] + '/v1/chat/completions', json=dict(
            model=config['served_model_name'], messages=messages, temperature=0, top_p=1, max_tokens=2000), timeout=180)
        response.raise_for_status()
        result = response.json()
        choice = result['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise RuntimeError('Official fixture prediction truncated')
        prediction = choice['message']['content']
        rows.append(dict(messages=messages, ground_truth=target, prediction=prediction,
                         exact_match=target.strip() == prediction.strip(), usage=result.get('usage')))
    report = dict(env=args.env, source=str(args.data_root / name), fixture_only=True,
        provenance=config['provenance'], sampling=dict(temperature=0,top_p=1,max_tokens=2000),
        completed=len(rows), exact_matches=sum(r['exact_match'] for r in rows), rows=rows)
    out = ROOT / 'outputs/wm_reflection_setup' / (args.env + '_official_fixtures.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','provenance')},indent=2))


if __name__ == '__main__':
    main()
