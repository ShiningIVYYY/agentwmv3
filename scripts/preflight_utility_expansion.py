"""New TRAIN identities and independent replay controls, zero model inference."""
import argparse,json,os,sys,time
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
    out.mkdir(parents=True);start=time.perf_counter();cfg=json.loads((ROOT/'configs/evaluation/utility_expansion_v3.json').read_text())
    split=json.loads((ROOT/cfg['split']).read_text())
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256']:raise ValueError('Frozen split changed')
    python=os.path.expandvars(cfg['env_python']);data=os.path.expandvars(cfg['data_root'])
    if '$' in python or '$' in data:raise ValueError('Missing environment paths')
    from transformers import AutoTokenizer
    from internal_signals.hf_actor import HFInternalActor
    actor=HFInternalActor.__new__(HFInternalActor);actor.tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/qwen35_9b',local_files_only=True)
    old=[]
    for f in ['docs/alfworld_pilot/native_goal_identity.json','docs/alfworld_dev50_v1/native_goal_identity.json','docs/activation_v1/native_identities.json']:
        old.extend(json.loads((ROOT/f).read_text()))
    oldgoals={r['goal'].strip().casefold() for r in old};seen=set();identities=[];queries=captures=0
    trainids=set(split['expansion_train_task_ids']);controlids=set(split['new_native_validation_task_ids'])
    lookup={r['task_id']:r for rs in split['partitions'].values() for r in rs}
    for task in split['expansion_train_task_ids']+split['new_native_validation_task_ids']:
        row=lookup[task]
        if row['source_split']!='train':raise ValueError('Illegal source split')
        env=AlfEnvironment(row,python,data,out/'native');queries+=1
        try:
            goal=env.goal.strip().casefold()
            if goal in oldgoals or goal in seen:raise ValueError('Native goal overlap; no silent replacement: '+task)
            seen.add(goal);n=actor.count(draft_messages(env.goal,env.current['observation'],[],env.current['admissible_commands']))
            if n>2048:raise ValueError('Mandatory initial prompt overflow')
            receipt=dict(task_id=task,role=row['role'],goal_group=row['goal_group'],goal=env.goal,initial_sha256=fingerprint(env.initial),initial_prompt_tokens=n,versions=env.versions)
            append(out/'identities.jsonl',receipt);identities.append(receipt);captures+=env.capture_count
        finally:env.close()
    for task in split['new_native_validation_task_ids']:
        row=lookup[task];env=AlfEnvironment(row,python,data,out/'native');queries+=1
        try:
            env.step('look');bad=env.step('open nonexistent container');queries+=2
            if bad['observation']!='Nothing happens.':raise ValueError('Invalid-action control differs')
            sim=AlfGroundTruthWM().simulate(env,'look');cost=sim['simulation_cost'];queries+=cost['initializations']+cost['replay_steps']+cost['candidate_steps']
            if not sim['prediction_available'] or not sim['formal_unchanged'] or sim['worker_pid']==sim['formal_pid']:raise ValueError('Independent replay failed')
            actual=env.step('look');queries+=1
            if actual!=sim['candidate_transition']:raise ValueError('One-step transition mismatch')
            captures+=env.capture_count+cost['state_capture_bundles'];append(out/'replay_controls.jsonl',dict(task_id=task,matched=True,simulation=sim,actual=actual))
        finally:env.close()
    receipt=dict(status='pass',new_train_identities=len(trainids),new_control_identities=len(controlids),identities=len(identities),replay_controls=len(controlids),
        split_sha256=file_hash(ROOT/cfg['split']),transition_queries=queries,audit_capture_bundles=captures,tokens=0,model_calls=0,
        no_test_or_independent_dev_resets=True,no_prior_native_goal_overlap=True,seconds=time.perf_counter()-start,
        scope='Exact public transitions/private fact digest, not full hidden RNG serialization')
    atomic(out/'complete.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
