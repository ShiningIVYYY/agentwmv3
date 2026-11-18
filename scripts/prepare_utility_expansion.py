"""Reserve new TRAIN-only goal groups; preserve all prior role assignments."""
import argparse,copy,json,sys
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_alfworld_pilot import catalog,digest
from evaluation.independent_dev import atomic
from experience.splits import file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);a=ap.parse_args()
    dest=ROOT/'docs/utility_expansion_v3';dest.mkdir(parents=True,exist_ok=True)
    target=dest/'split_manifest.json'
    if target.exists():raise ValueError('Do not overwrite preregistered expansion')
    parent_path=ROOT/'docs/activation_v1/split_manifest.json';parent=json.loads(parent_path.read_text())
    protected=set(parent['excluded_goal_groups'])|{r['goal_group'] for rs in parent['partitions'].values() for r in rs}|{r['goal_group'] for r in parent['final_test']}
    groups=defaultdict(list)
    for r in catalog(Path(a.data_root),'train'):
        if r['goal_group'] not in protected:groups[r['goal_group']].append(r)
    families=defaultdict(list)
    for g,rs in groups.items():families[rs[0]['family']].append(g)
    for f in families:families[f].sort(key=lambda g:digest(['utility-expansion-v3-goal',g]))
    ordered=[]
    while any(families.values()):
        for f in sorted(families):
            if families[f]:
                g=families[f].pop(0);ordered.append(min(groups[g],key=lambda r:digest(['utility-expansion-v3-instance',r['game']])))
    if len(ordered)<29:raise ValueError('Insufficient untouched goals; do not relax exclusions')
    train=[dict(r,role='probe_train') for r in ordered[:24]];controls=[dict(r,role='environment_validation') for r in ordered[24:29]]
    split=copy.deepcopy(parent);split['schema']='utility-expansion-goal-split-v3';split['partitions']['probe_train']+=train;split['partitions']['environment_validation']+=controls
    split.update(parent_manifest=str(parent_path.relative_to(ROOT)),parent_sha256=file_hash(parent_path),expansion_train_task_ids=[r['task_id'] for r in train],
        new_native_validation_task_ids=[r['task_id'] for r in controls],new_goal_count=29,new_train_count=24,untouched_groups_available=len(groups),
        expansion_selection='TRAIN only; exclude every prior reserved/exposed canonical goal and final140; SHA256 family round-robin before model outcomes',
        inherited_roles_unchanged=True,scope='Expanded60 Train groups; original12 Val/12 Cal/12 exposed Dev stay in their original roles; no Dev/Test rollouts')
    seen=set()
    for role,rs in split['partitions'].items():
        gs={r['goal_group'] for r in rs}
        if len(gs)!=len(rs) or seen&gs:raise ValueError('Duplicate/overlapping split goal')
        seen|=gs
    if seen&{r['goal_group'] for r in split['final_test']}:raise ValueError('Test leakage')
    atomic(target,split)
    cfg=json.loads((ROOT/'configs/evaluation/utility_sampling_v2.json').read_text())
    cfg.update(schema='utility-expansion-v3',split=str(target.relative_to(ROOT)),split_sha256=file_hash(target),gpus=[5,6,7],prior_run='outputs/utility_sampling_v2',
        old_training_task_count=36,new_training_task_count=24,total_training_task_count=60,label_schema='utility-label-v3-all-zero-neutral',
        global_limits=dict(tokens=36000000,model_calls=60000,transition_queries=300000,audit_capture_bundles=360000,wall_seconds=129600),
        source_tasks=dict(new_probe_train=24,inherited_probe_train=36,probe_validation=12,gate_calibration=12),
        train_pool_budget_fraction=.5,previous_result_reuse='Reuse original real features and corrected derived labels without rerunning old36; preserve original raw outputs',
        expansion_rule=dict(min_repeated_help_states=5,min_train_tasks=3,use_combined_old_new_training=True),automatic_gate_deployment=False)
    atomic(ROOT/'configs/evaluation/utility_expansion_v3.json',cfg)
    print(json.dumps(dict(new_train=24,new_controls=5,untouched_available=len(groups),family_counts=dict(Counter(r['family'] for r in train))),indent=2))


if __name__=='__main__':main()
