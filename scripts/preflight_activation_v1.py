"""Real native identities and disjoint replay controls; never reset final Test."""
import argparse,json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from experience.splits import file_hash
from worldmodel.alfworld_environment import AlfEnvironment,AlfGroundTruthWM,fingerprint
from internal_signals.activation_protocol import draft_messages


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    out=Path(a.run).absolute()
    if out.exists():raise ValueError('Fresh preflight only')
    out.mkdir(parents=True);started=time.perf_counter()
    cfg=json.loads((ROOT/'configs/evaluation/activation_v1.json').read_text())
    split=json.loads((ROOT/cfg['split']).read_text());data=os.path.expandvars(cfg['data_root']);python=os.path.expandvars(cfg['env_python'])
    if '$' in data or '$' in python:raise ValueError('Missing ALFWORLD_DATA/ALFWORLD_PYTHON')
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256']:raise ValueError('Frozen split changed')
    from transformers import AutoTokenizer
    from internal_signals.hf_actor import HFInternalActor
    actor=HFInternalActor.__new__(HFInternalActor)
    actor.tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/qwen35_9b',local_files_only=True)
    old=[]
    for name in ['docs/alfworld_pilot/native_goal_identity.json','docs/alfworld_dev50_v1/native_goal_identity.json']:
        old.extend(json.loads((ROOT/name).read_text()))
    exposed={r['goal'].strip().casefold() for r in old};seen=set();identities=[];queries=captures=0
    for role,rs in split['partitions'].items():
        for row in rs:
            env=AlfEnvironment(row,python,data,out/'native');queries+=1
            try:
                goal=env.goal.strip().casefold()
                if goal in exposed or goal in seen:raise ValueError('Native goal overlap: '+row['task_id'])
                seen.add(goal)
                n=actor.count(draft_messages(env.goal,env.current['observation'],[],env.current['admissible_commands']))
                if n>2048:raise ValueError('Initial mandatory prompt overflow; no task replacement')
                receipt=dict(task_id=row['task_id'],role=role,goal=env.goal,goal_group=row['goal_group'],seed=row['seed'],
                    initial_sha256=fingerprint(env.initial),initial_prompt_tokens=n,versions=env.versions)
                append(out/'identities.jsonl',receipt);identities.append(receipt);captures+=env.capture_count
            finally:env.close()
    for row in split['partitions']['environment_validation']:
        env=AlfEnvironment(row,python,data,out/'native');queries+=1
        try:
            env.step('look');bad=env.step('open nonexistent container');queries+=2
            if bad['observation']!='Nothing happens.':raise ValueError('Invalid action control differs')
            sim=AlfGroundTruthWM().simulate(env,'look');cost=sim['simulation_cost']
            queries+=cost['initializations']+cost['replay_steps']+cost['candidate_steps']
            if not sim['prediction_available'] or not sim['formal_unchanged']:raise ValueError('Replay failed')
            actual=env.step('look');queries+=1
            if actual!=sim['candidate_transition'] or sim['formal_pid']==sim['worker_pid']:raise ValueError('Control transition/isolation mismatch')
            captures+=env.capture_count+cost['state_capture_bundles']
            append(out/'controls.jsonl',dict(task_id=row['task_id'],simulation=sim,actual=actual,matched=True))
        finally:env.close()
    result=dict(status='pass',identities=len(identities),roles={k:len(v) for k,v in split['partitions'].items()},
        replay_controls=len(split['partitions']['environment_validation']),transition_queries=queries,audit_capture_bundles=captures,
        tokens=0,model_calls=0,no_test_resets=True,no_prior_native_goal_overlap=True,split_sha256=file_hash(ROOT/cfg['split']),
        seconds=time.perf_counter()-started,scope='Exact public transitions and private fact digest, not full hidden RNG serialization')
    atomic(out/'complete.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
