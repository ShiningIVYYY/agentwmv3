"""Retire unknown receipts conservatively; preserve completed tasks and reserve40 goals."""
import json,os,sys,time,subprocess
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic
from scripts.run_utility_sampling import read,rows,check,phase_cost
from scripts.prepare_alfworld_pilot import catalog,digest
from experience.splits import file_hash


def main():
    out=ROOT/'docs/recovery_v4';out.mkdir(parents=True,exist_ok=True)
    if (out/'recovery_receipt.json').exists():raise ValueError('Never overwrite a registered recovery')
    failed=ROOT/'outputs/utility_expansion_v3';oldm=read(failed/'manifest.json');oldcfg=oldm['profile'];check(oldm)
    parent=read(ROOT/oldcfg['split']);completed=set();sourcehash={};unknown=[];known=Counter();allcases=[];alllabels=[]
    for p in failed.glob('*_owned_workers.json'):
        for r in read(p):
            ps=subprocess.run(['ps','-p',str(r['pid']),'-o','args='],text=True,capture_output=True)
            if ps.returncode==0 and str(failed) in ps.stdout:raise ValueError('Prior owned worker still active')
    for w in sorted(failed.glob('train_pool/worker*')):
        completed|={r['task_id'] for r in rows(w/'completed_tasks.jsonl')} if (w/'completed_tasks.jsonl').exists() else set()
        ledger=read(w/'ledger.json');calls=rows(w/'generation_calls.jsonl')
        if len(calls)!=ledger['model_calls'] or sum(c['metadata']['usage']['total_tokens'] for c in calls)!=ledger['tokens']:raise ValueError('Interrupted journal needs manual reconciliation')
        if ledger['pending_transition']:raise ValueError('Unknown native transition cannot be retired as model-only cost')
        for k in ['tokens','model_calls','transition_queries','audit_capture_bundles']:known[k]+=ledger[k]
        if ledger['pending_model']:unknown.append(dict(worker=w.name,**ledger['pending_model'],actual_tokens=None,output_unavailable=True))
        allcases+=rows(w/'cases.jsonl');alllabels+=rows(w/'state_labels.jsonl')
        for p in w.iterdir():
            if p.is_file() and p.suffix in ['.json','.jsonl']:sourcehash[str(p.relative_to(ROOT))]=file_hash(p)
    for name in ['manifest.json','progress.json','error.json']:sourcehash[str((failed/name).relative_to(ROOT))]=file_hash(failed/name)
    kept=[c for c in allcases if c['task_id'] in completed];labels=[l for l in alllabels if l['task_id'] in completed]
    ids={c['case_id']:c for c in kept}
    if any(l['case_id'] not in ids for l in labels):raise ValueError('Unmatched completed label')
    for task in completed:
        episodes=[e for w in failed.glob('train_pool/worker*') for e in rows(w/'source_episodes.jsonl') if e['task_id']==task]
        if {e['policy'] for e in episodes}!={'never','always'}:raise ValueError('Completed task lacks both source policies')
    original=set(parent['expansion_train_task_ids']);remaining=[t for t in parent['expansion_train_task_ids'] if t not in completed]
    if not completed<=original or len(completed)+len(remaining)!=24:raise ValueError('Original task accounting changed')
    charge=dict(known);charge['tokens']+=sum(u['max_tokens'] for u in unknown);charge['model_calls']+=len(unknown)
    oldtrain={k:sum(b[k] for b in read(failed/'train_pool_budgets.json')) for k in known}
    residual={k:oldtrain[k]-charge[k] for k in known}
    receipt=dict(failed_run=str(failed.relative_to(ROOT)),source_hashes=sourcehash,completed_task_ids=sorted(completed),remaining_original_task_ids=remaining,
        retained_case_count=len(kept),retained_label_count=len(labels),retained_help_count=sum(l['label']=='help' for l in labels),
        excluded_partial_case_count=len(allcases)-len(kept),excluded_partial_label_count=len(alllabels)-len(labels),
        feature_hashes={c['feature_file']:file_hash(ROOT/c['feature_file']) for c in kept},
        known_completed_cost=dict(known),unreceipted_requests=unknown,unknown_token_upper_bound=sum(u['max_tokens'] for u in unknown),
        budget_charge_upper_bound=charge,remaining_original_train_limits=residual,original_deadline_unix=oldm['deadline_unix'],
        raw_pending_ledgers_modified=False,selection_rule='Retain all fully completed tasks irrespective of outcome; restart every incomplete task at its fixed seed; exclude partial-task rows from final training to prevent duplicates',
        known_cost_scope='Confirmed receipts only; unknown outputs are charged separately by their reservation upper bounds')
    protected=set(parent['excluded_goal_groups'])|{r['goal_group'] for rs in parent['partitions'].values() for r in rs}|{r['goal_group'] for r in parent['final_test']}
    groups=defaultdict(list)
    for r in catalog(Path(os.environ['ALFWORLD_DATA']),'train'):
        if r['goal_group'] not in protected:groups[r['goal_group']].append(r)
    families=defaultdict(list)
    for g,rs in groups.items():families[rs[0]['family']].append(g)
    for f in families:families[f].sort(key=lambda g:digest(['recovery-v4-additional-training',g]))
    ordered=[]
    while any(families.values()):
        for f in sorted(families):
            if families[f]:
                g=families[f].pop(0);ordered.append(min(groups[g],key=lambda r:digest(['recovery-v4-instance',r['game']])))
    if len(ordered)<45:raise ValueError('Insufficient untouched goals')
    newtrain=[dict(r,role='probe_train') for r in ordered[:40]];controls=[dict(r,role='environment_validation') for r in ordered[40:45]]
    split=json.loads(json.dumps(parent));split['schema']='utility-recovery-expansion-v4';split['partitions']['probe_train']+=newtrain;split['partitions']['environment_validation']+=controls
    split.update(parent_manifest=oldcfg['split'],parent_sha256=file_hash(ROOT/oldcfg['split']),new_train_task_ids=[r['task_id'] for r in newtrain],
        new_native_control_task_ids=[r['task_id'] for r in controls],remaining_original_task_ids=remaining,retained_failed_task_ids=sorted(completed),
        untouched_goals_available=len(groups),total_train_goals=100,original_roles_preserved=True)
    seen=set()
    for role,rs in split['partitions'].items():
        gs={r['goal_group'] for r in rs}
        if len(gs)!=len(rs) or seen&gs:raise ValueError('Goal overlap')
        seen|=gs
    if seen&{r['goal_group'] for r in split['final_test']}:raise ValueError('Test overlap')
    atomic(out/'split_manifest.json',split);atomic(out/'recovery_receipt.json',receipt)
    cfg=json.loads(json.dumps(oldcfg));cfg.update(schema='utility-recovery-v4',split='docs/recovery_v4/split_manifest.json',split_sha256=file_hash(out/'split_manifest.json'),gpus=[0,1,6,7],
        failed_run=str(failed.relative_to(ROOT)),total_training_goals=100,additional_training_goals=40,alignment_version='actual-generated-id-alignment-v1',
        global_limits={k:2*oldcfg['global_limits'][k]-charge[k] for k in known},wall_seconds=172800,
        recovery_limits=residual,expansion_limits=dict(tokens=24000000,model_calls=40000,transition_queries=200000,audit_capture_bundles=240000),
        original_recovery_deadline=oldm['deadline_unix'],budget_semantics='V3+recovery+extra expansion bounded by72M tokens, including conservative charge for2 missing outputs; V2 sunk corpus separate',
        no_automatic_gate_deployment=True)
    cfg['global_limits']['wall_seconds']=172800
    cfg['old_training_task_count']=47;cfg['new_training_task_count']=53;cfg['total_training_task_count']=100
    cfg['source_tasks']=dict(probe_train=100,retained_training=47,recover_original=13,additional_training=40,probe_validation=12,gate_calibration=12)
    atomic(ROOT/'configs/evaluation/recovery_v4.json',cfg)
    print(json.dumps(dict(retained_tasks=len(completed),remaining_original=len(remaining),additional_training=40,total_training=100,
        confirmed_cost=dict(known),unknown_requests=len(unknown),unknown_token_upper_bound=receipt['unknown_token_upper_bound'],
        family_counts=dict(Counter(r['family'] for r in newtrain))),indent=2))


if __name__=='__main__':main()
