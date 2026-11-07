#!/usr/bin/env python3
"""Numerical portability of two actual dev prompts, not new benchmark outcomes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy
import scienceworld
import torch
import transformers
from internal_signals.qwen35_actor import Qwen35Actor
from worldmodel.ground_truth_environment import ReplayScienceEnvironment,GroundTruthEnvironmentWM


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--calibration-reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('New numerical-validation Run ID required')
    out.mkdir(parents=True)
    for module in (numpy,scienceworld,torch,transformers):
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError('Dependency still inherited outside this venv: '+module.__name__)
    old=[json.loads(s) for s in (Path(a.reference)/'evaluation_steps.jsonl').read_text().splitlines()]
    draft=json.loads((Path(a.calibration_reference)/'calibration_steps.jsonl').read_text().splitlines()[0])['metadata']
    review=next(t['review_metadata'] for t in old if t['review_metadata'])
    started=time.perf_counter()
    actor=Qwen35Actor(ROOT/'configs/internal_signals/qwen35_9b_replay_v1.json',a.model_dir,a.device)
    result=dict(purpose='same-public-prompts numerical compatibility only; no environment/action/utility outcome',
                independent_dependency_paths=True,model_revision=actor.config['model_revision'],
                model_loading_info=actor.loading_info,model_load_seconds=time.perf_counter()-started,
                packages={k:importlib.metadata.version(k) for k in ('torch','transformers','tokenizers','huggingface-hub','scienceworld','numpy')},
                tests=[],model_calls=0,total_tokens=0,environment_calls=0)
    for role,meta,scored in [('draft',draft,True),('review',review,False)]:
        record=meta['generation_record']
        seed=meta.get('actor_metadata',{}).get('generation_seed',record.get('request_seed'))
        output,new=actor.generate_role(record['messages'],role,seed,scored)
        exact_ids=new['generation_record']['response_ids']==record['response_ids']
        nll_equal=None;max_delta=None
        if scored:
            expected=record['token_nll'];actual=new['generation_record']['token_nll']
            max_delta=max(abs(x-y) for x,y in zip(expected,actual)) if len(expected)==len(actual) else None
            nll_equal=actual==expected
        result['tests'].append(dict(role=role,response_ids_identical=exact_ids,token_nll_identical=nll_equal,
                                   max_nll_delta=max_delta,metadata=new))
        result['model_calls']+=1;result['total_tokens']+=new['usage']['total_tokens']
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if not exact_ids or (scored and not nll_equal):
            raise ValueError('Independent environment differs numerically; do not silently reuse thresholds')
    env=ReplayScienceEnvironment(dict(taskName='use-thermometer',variationIdx=8,simplifications='',seed=0,environment_step_limit=21))
    try:
        simulation=GroundTruthEnvironmentWM().simulate_one_step(env,'look around')
        actual=env.step('look around')
        equal=simulation['prediction_available'] and all(simulation['candidate_transition'][k]==actual[k] for k in ('observation','reward','done','score','moves'))
        result.update(native_sdk_replay_smoke=dict(task_id='sciworld_128',simulation=simulation,actual=actual,transition_match=equal),
                      environment_calls=4)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if not equal:raise ValueError('Native SDK in independent environment failed replay smoke')
    finally:env.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('tests','native_sdk_replay_smoke')},indent=2))


if __name__=='__main__':main()
