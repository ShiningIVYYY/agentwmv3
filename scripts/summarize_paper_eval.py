#!/usr/bin/env python3
"""Write an always-current success-rate table for paper_eval trajectories."""
import json
from pathlib import Path

root = Path('/data1/shn/agentwmv2/outputs/paper_eval/formal')
rows = []
for env in ('alfworld', 'scienceworld', 'webshop'):
    for model in ('qwen25', 'qwen38', 'ee_il'):
        path = root / f'{env}_{model}' / 'trajectories.jsonl'
        data=[]
        if path.exists():
            for line in path.open():
                try: data.append(json.loads(line))
                except json.JSONDecodeError: pass
        wins=sum(bool(x.get('success', False)) for x in data)
        rate='—' if not data else f'{100*wins/len(data):.1f}%'
        rows.append((env, model, len(data), wins, rate))
out=['# Paper-style evaluation status', '', '| Benchmark | Model | Completed | Successes | Success rate |', '|---|---:|---:|---:|---:|']
out += [f'| {e} | {m} | {n} | {w} | {r} |' for e,m,n,w,r in rows]
(root/'results.md').write_text('\n'.join(out)+'\n')
