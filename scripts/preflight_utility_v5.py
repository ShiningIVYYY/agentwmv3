"""Native identity verification of fixed TRAIN/Val/control tasks, never Dev/Test."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from experience.splits import file_hash
from worldmodel.alfworld_environment import AlfEnvironment,AlfGroundTruthWM,fingerprint
from internal_signals.activation_protocol import draft_messages
from scripts.run_recovery_v4 import all_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run).absolute()
    if out.exists():raise ValueError('Fresh preflight only')
    out.mkdir(parents=True);start=time.perf_counter()
    cfg=json.loads((ROOT/'configs/evaluation/utility_expansion_v5.json').read_text());split=json.loads((ROOT/cfg['split']).read_text())
    if file_hash(ROOT/cfg['split'])!=cfg['split_sha256']:raise ValueError('Split changed')
    from transformers import AutoTokenizer
    from internal_signals.hf_actor import HFInternalActor
    actor=HFInternalActor.__new__(HFInternalActor);actor.tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/qwen35_9b',local_files_only=True)
    oldcfg=json.loads((ROOT/'configs/evaluation/recovery_v4.json').read_text());cs,_,_=all_records(ROOT/'outputs/recovery_v4',oldcfg)
    oldgoals={c['goal'].strip().casefold() for c in cs}
    for path in ['docs/alfworld_pilot/native_goal_identity.json','docs/alfworld_dev50_v1/native_goal_identity.json','docs/activation_v1/native_identities.json','docs/utility_expansion_v3/native_identities.json']:
        oldgoals|={r['goal'].strip().casefold() for r in json.loads((ROOT/path).read_text())}
    oldgoals|={json.loads(line)['goal'].strip().casefold() for line in (ROOT/'outputs/recovery_v4_preflight/identities.jsonl').read_text().splitlines()}
    lookup={r['task_id']:r for rs in split['partitions'].values() for r in rs};seen=set();queries=captures=0
    ids=split['new_train_task_ids']+split['new_validation_task_ids']+split['new_native_control_task_ids']
    if set(ids)&set(split['fresh_independent_dev_task_ids']):raise ValueError('Fresh Dev reset forbidden')
    python=os.path.expandvars(cfg['env_python']);data=os.path.expandvars(cfg['data_root'])
    if '$' in python or '$' in data:raise ValueError('Missing environment configuration')
    for task in ids:
        row=lookup[task]
        if row['source_split']!='train':raise ValueError('Illegal data source')
        env=AlfEnvironment(row,python,data,out/'native');queries+=1
        try:
            goal=env.goal.strip().casefold()
            if goal in oldgoals or goal in seen:raise ValueError('Native goal collision; no silent replacement: '+task)
            seen.add(goal);n=actor.count(draft_messages(env.goal,env.current['observation'],[],env.current['admissible_commands']))
            if n>2048:raise ValueError('Initial mandatory prompt overflow')
            append(out/'identities.jsonl',dict(task_id=task,role=row['role'],goal_group=row['goal_group'],goal=env.goal,
                initial_prompt_tokens=n,initial_sha256=fingerprint(env.initial),versions=env.versions))
            captures+=env.capture_count
            atomic(out/'progress.json',dict(verified=len(seen),target=len(ids),transition_queries=queries,audit_capture_bundles=captures))
        finally:env.close()
    for task in split['new_native_control_task_ids']:
        env=AlfEnvironment(lookup[task],python,data,out/'native');queries+=1
        try:
            env.step('look');bad=env.step('open nonexistent container');queries+=2
            if bad['observation']!='Nothing happens.':raise ValueError('Invalid-action control mismatch')
            sim=AlfGroundTruthWM().simulate(env,'look');c=sim['simulation_cost'];queries+=c['initializations']+c['replay_steps']+c['candidate_steps']
            if not sim['prediction_available'] or not sim['formal_unchanged'] or sim['worker_pid']==sim['formal_pid']:raise ValueError('Replay not isolated')
            actual=env.step('look');queries+=1
            if actual!=sim['candidate_transition']:raise ValueError('Replay transition mismatch')
            captures+=env.capture_count+c['state_capture_bundles'];append(out/'replay_controls.jsonl',dict(task_id=task,passed=True,simulation=sim,actual=actual))
        finally:env.close()
    receipt=dict(status='pass',identities=len(ids),new_train=80,new_validation=40,controls=5,replay_controls=5,
        split_sha256=file_hash(ROOT/cfg['split']),transition_queries=queries,audit_capture_bundles=captures,
        tokens=0,model_calls=0,no_test_or_fresh_dev_resets=True,no_prior_native_goal_overlap=True,seconds=time.perf_counter()-start,
        scope='Public transitions and private-fact digest; not full hidden RNG serialization')
    atomic(out/'complete.json',receipt);atomic(ROOT/'docs/utility_expansion_v5/native_preflight.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
