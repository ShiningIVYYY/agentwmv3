"""Preregister one bounded extension; never overwrite a selected cohort."""
import json,os,sys,time,subprocess
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic
from experience.splits import file_hash
from scripts.prepare_alfworld_pilot import catalog,digest
from scripts.run_recovery_v4 import all_records


def allocate_roles(selected):
    """Stratify within the fixed goal set; avoid role/family cycle aliasing."""
    pools=defaultdict(list);new=defaultdict(list)
    for r in selected:pools[r['family']].append(r)
    for f in pools:pools[f].sort(key=lambda r:digest(['utility-v5-role-within-family',r['goal_group']]))
    for role,n in [('environment_validation',5),('reserved_independent_dev_v5',12),('probe_train',80),('probe_validation',40)]:
        total=sum(len(v) for v in pools.values())
        quotas={f:len(v)*n//total for f,v in pools.items()}
        priority=sorted(pools,key=lambda f:(-(len(pools[f])*n%total),digest(['utility-v5-quota',role,f])))
        for f in priority[:n-sum(quotas.values())]:quotas[f]+=1
        for f in sorted(pools):
            chosen=pools[f][:quotas[f]];pools[f]=pools[f][quotas[f]:]
            new[role].extend(dict(r,role=role) for r in chosen)
    if any(pools.values()):raise ValueError('Unassigned selected goals')
    return new


def main():
    out=ROOT/'docs/utility_expansion_v5'
    if out.exists():raise ValueError('Existing preregistration; do not redraw')
    base=json.loads((ROOT/'configs/evaluation/recovery_v4.json').read_text())
    parent=json.loads((ROOT/base['split']).read_text())
    summary=json.loads((ROOT/'outputs/recovery_v4/summary.json').read_text())
    if summary['status']!='complete':raise ValueError('Prior batch incomplete')
    cs,ls,ps=all_records(ROOT/'outputs/recovery_v4',base,verify=True)
    if len(cs)!=9628 or len(ls)!=782:raise ValueError('Prior denominator changed')
    for phase in ['recover_original','expand_train','heldout_pools']:
        if not all(json.loads((ROOT/'outputs/recovery_v4'/(phase+'_supervisor.json')).read_text())['workers_exited']):
            raise ValueError('Prior workers not released')
    blocked=set(parent['excluded_goal_groups'])|{r['goal_group'] for rs in parent['partitions'].values() for r in rs}|{r['goal_group'] for r in parent['final_test']}
    groups=defaultdict(list)
    for r in catalog(Path(os.environ['ALFWORLD_DATA']),'train'):
        if r['goal_group'] not in blocked:groups[r['goal_group']].append(r)
    families=defaultdict(list)
    for g,rs in groups.items():families[rs[0]['family']].append(g)
    for f in families:families[f].sort(key=lambda g:digest(['utility-v5-goal',g]))
    order=[]
    while any(families.values()):
        for f in sorted(families):
            if families[f]:
                g=families[f].pop(0);order.append(min(groups[g],key=lambda r:digest(['utility-v5-instance',r['game']])))
    if len(order)<137:raise ValueError('Need80Train/40Val/5controls/12freshDev; do not relax exclusions')
    new=allocate_roles(order[:137])
    split=json.loads(json.dumps(parent));split['schema']='utility-expansion-v5'
    for role,rs in new.items():split['partitions'].setdefault(role,[]).extend(rs)
    split.update(parent_manifest=base['split'],parent_sha256=file_hash(ROOT/base['split']),
        new_train_task_ids=[r['task_id'] for r in new['probe_train']],
        new_validation_task_ids=[r['task_id'] for r in new['probe_validation']],
        new_native_control_task_ids=[r['task_id'] for r in new['environment_validation']],
        fresh_independent_dev_task_ids=[r['task_id'] for r in new['reserved_independent_dev_v5']],
        old_roles_preserved=True,untouched_goals_before_selection=len(groups),
        new_family_counts={role:dict(Counter(r['family'] for r in rs)) for role,rs in new.items()},
        selection='Metadata-only canonical-group family round-robin SHA256 order;proportional within-family role quotas;no outcomes')
    seen=set()
    for role,rs in split['partitions'].items():
        gs={r['goal_group'] for r in rs}
        if len(gs)!=len(rs) or seen&gs:raise ValueError('Task-group role overlap')
        seen|=gs
    if seen&{r['goal_group'] for r in split['final_test']}:raise ValueError('Test leakage')
    source_hashes={}
    paths=['configs/evaluation/recovery_v4.json','docs/recovery_v4/split_manifest.json',
        'outputs/recovery_v4/summary.json','docs/recovery_v4/recovery_receipt.json','docs/utility_expansion_v3/prior_corpus.json']
    for run in ['utility_sampling_v2','utility_expansion_v3','recovery_v4']:
        for p in (ROOT/'outputs'/run).glob('*/worker*/*'):
            if p.is_file() and p.suffix in ['.json','.jsonl']:paths.append(str(p.relative_to(ROOT)))
    for p in paths:source_hashes[p]=file_hash(ROOT/p)
    features={c['feature_file']:file_hash(ROOT/c['feature_file']) for c in cs}
    receipt=dict(schema='utility-v5-prior-corpus',cases=len(cs),labels=len(ls),pairs=len(ps),
        source_hashes=source_hashes,feature_hashes=features,known_cumulative_cost=summary['known_cost_all_three_sampling_runs'],
        conservative_cumulative_cost=summary['cost_upper_bound_for_budget'],
        unknown_prior_receipts=summary['unknown_prior_receipts'],all_old_roles_preserved=True,
        reuse='All prior records; no source regeneration for100 old Train goals; at most4 newly labelled real revised states per old task')
    out.mkdir(parents=True);atomic(out/'split_manifest.json',split);atomic(out/'prior_corpus.json',receipt)
    cfg={k:base[k] for k in ['actor_config','data_root','env_python','memory_snapshot','memory_sha256','sampling_nll_thresholds','probe_training_profile']}
    cfg.update(schema='utility-expansion-v5',split='docs/utility_expansion_v5/split_manifest.json',split_sha256=file_hash(out/'split_manifest.json'),
        gpus=[4,6],wall_seconds=259200,minimum_free_disk_bytes=50*1024**3,maximum_run_bytes=90*1024**3,
        extra_train_states_per_task=4,validation_states_per_window=4,
        global_limits=dict(tokens=64000000,model_calls=120000,transition_queries=600000,audit_capture_bundles=660000),
        phase_limits=dict(densify_train=dict(tokens=12000000,model_calls=24000,transition_queries=150000,audit_capture_bundles=165000),
            new_train=dict(tokens=32000000,model_calls=60000,transition_queries=300000,audit_capture_bundles=330000),
            new_validation=dict(tokens=20000000,model_calls=36000,transition_queries=150000,audit_capture_bundles=165000)),
        old_train_goals=100,new_train_goals=80,new_validation_goals=40,
        source_rollout_seed=1,reviewer_seed=0,continuation_seeds=[1,2,3],confirmation_seeds=[4,5],
        outcome_filter='None for new revision enrichment; accepted change is a candidate only, never a Help label',
        no_actor_or_reviewer_training=True,no_memory_or_gate_changes=True,no_final_test_rollouts=True,
        no_fresh_independent_dev_rollouts=True,no_automatic_gate_deployment=True,
        budget_semantics='NEW64M-token/72h batch; old48.7M known costs and1890-token unknown upper charge remain separately included in cumulative reporting')
    atomic(ROOT/'configs/evaluation/utility_expansion_v5.json',cfg)
    print(json.dumps(dict(status='preregistered',counts={k:len(v) for k,v in new.items()},families=split['new_family_counts'],remaining_unassigned=len(order)-137),indent=2))


def amend_roles():
    """Explicit pre-outcome amendment; same selected137 goals, no redraw."""
    out=ROOT/'docs/utility_expansion_v5';path=out/'split_manifest.json'
    if (ROOT/'outputs/utility_expansion_v5').exists() or (ROOT/'outputs/utility_expansion_v5_preflight').exists():raise ValueError('Cannot change allocation after native/model data')
    if (out/'allocation_amendment.json').exists():raise ValueError('Do not amend twice')
    before=json.loads(path.read_text());parent=json.loads((ROOT/before['parent_manifest']).read_text())
    if '2Train/1Val slots' not in before['selection']:raise ValueError('Only original cyclic-allocation draft may be amended')
    ids=set(before['new_train_task_ids']+before['new_validation_task_ids']+before['new_native_control_task_ids']+before['fresh_independent_dev_task_ids'])
    selected=[r for rs in before['partitions'].values() for r in rs if r['task_id'] in ids]
    if len(selected)!=137:raise ValueError('Selected identity mismatch')
    new=allocate_roles(selected);after=json.loads(json.dumps(parent));after.update({k:v for k,v in before.items() if k!='partitions'})
    after['partitions']=json.loads(json.dumps(parent['partitions']))
    for role,rs in new.items():after['partitions'].setdefault(role,[]).extend(rs)
    after.update(new_train_task_ids=[r['task_id'] for r in new['probe_train']],new_validation_task_ids=[r['task_id'] for r in new['probe_validation']],
        new_native_control_task_ids=[r['task_id'] for r in new['environment_validation']],fresh_independent_dev_task_ids=[r['task_id'] for r in new['reserved_independent_dev_v5']],
        new_family_counts={role:dict(Counter(r['family'] for r in rs)) for role,rs in new.items()},
        selection='Same fixed137 metadata-only goals; proportional within-family role quotas; pre-outcome amendment of role/family cycle aliasing')
    (out/'split_manifest_v0_superseded.json').write_text(path.read_text());oldsha=file_hash(path);atomic(path,after)
    cpath=ROOT/'configs/evaluation/utility_expansion_v5.json';cfg=json.loads(cpath.read_text());cfg['split_sha256']=file_hash(path);atomic(cpath,cfg)
    amendment=dict(reason='Three-family cycle and2Train/1Val cycle aliased, concentrating Validation in one family. Proportional within-family quotas remove this metadata design defect.',
        same137goal_identities=True,no_native_or_model_data_before_amendment=True,old_sha256=oldsha,new_sha256=file_hash(path),
        before=before['new_family_counts'],after=after['new_family_counts'],registered_unix=time.time())
    atomic(out/'allocation_amendment.json',amendment);print(json.dumps(amendment,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--amend-fixed-goal-roles',action='store_true');a=p.parse_args()
    if a.amend_fixed_goal_roles:amend_roles()
    else:main()
