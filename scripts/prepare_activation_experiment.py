"""Preregister fresh goal groups; exposed50-task cohort remains regression-only."""
import argparse,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_alfworld_pilot import catalog,digest
from evaluation.independent_dev import atomic
from experience.splits import file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--env-python',required=True);a=ap.parse_args()
    dest=ROOT/'docs/activation_v1';dest.mkdir(exist_ok=True)
    if (dest/'split_manifest.json').exists():raise ValueError('Never overwrite preregistered split')
    old=json.loads((ROOT/'docs/alfworld_dev50_v1/split_manifest.json').read_text())
    exposed=[r for k in ['evaluation_tasks','environment_validation_tasks','previous_pilot_evaluation_tasks','memory_tasks','calibration_tasks','preliminary_probe_tasks'] for r in old[k]]
    excluded={r['goal_group'] for r in exposed+old['final_valid_seen']}
    groups=defaultdict(list)
    for r in catalog(Path(a.data_root),'train'):
        if r['goal_group'] not in excluded:groups[r['goal_group']].append(r)
    families=defaultdict(list)
    for g,rs in groups.items():families[rs[0]['family']].append(g)
    for family in families:families[family].sort(key=lambda g:digest(['activation-v1-unseen-goal',g]))
    ordered=[]
    while any(families.values()):
        for family in sorted(families):
            if families[family]:
                g=families[family].pop(0);ordered.append(min(groups[g],key=lambda r:digest(['activation-v1-instance',r['game']])))
    sizes=dict(probe_train=36,probe_validation=12,gate_calibration=12,independent_dev=12,environment_validation=5)
    if len(ordered)<sum(sizes.values()):raise ValueError('Insufficient untouched goals')
    partitions={};offset=0
    for role,n in sizes.items():
        partitions[role]=[dict(r,role=role) for r in ordered[offset:offset+n]];offset+=n
    allgroups=[r['goal_group'] for rs in partitions.values() for r in rs]
    assert len(set(allgroups))==len(allgroups) and not set(allgroups)&excluded
    split=dict(schema='activation-goal-split-v1',partitions=partitions,excluded_goal_groups=sorted(excluded),
        source_available_groups=len(groups),final_test=old['final_valid_seen'],memory_tasks=old['memory_tasks'],
        family_counts={k:dict(Counter(r['family'] for r in rs)) for k,rs in partitions.items()},
        missing_families=sorted(set(r['family'] for r in old['final_valid_seen'])-{r['family'] for r in ordered}),
        scope='TRAIN-only new goals; previous50 and all prior calibration/memory/debug excluded; templates/layouts may overlap')
    atomic(dest/'split_manifest.json',split)
    actor=json.loads((ROOT/'configs/internal_signals/qwen35_9b_alfworld_v1.json').read_text())
    actor.update(prompt_schema='activation-pce-v2',action_format='strict_three_line_native_alfworld',purpose='AP0v2 frozen backbone + supervised probes, new prompt and new calibration')
    atomic(ROOT/'configs/internal_signals/qwen35_9b_activation_v1.json',actor)
    config=dict(schema='activation-experiment-v1',actor_config='configs/internal_signals/qwen35_9b_activation_v1.json',split='docs/activation_v1/split_manifest.json',
        split_sha256=file_hash(dest/'split_manifest.json'),data_root='${ALFWORLD_DATA}',env_python='${ALFWORLD_PYTHON}',collection_steps=12,utility_step=2,
        train_seeds=[0,1,2],evaluation_seed=0,budget=5,horizon=50,utility_labels='one_predeclared_step2_per_task; single paired continuation seed0',
        gpus=[0,6,7],stage_limits=dict(tokens=4000000,model_calls=4000,transition_queries=30000,audit_capture_bundles=40000),
        global_limits=dict(tokens=24000000,model_calls=24000,transition_queries=180000,audit_capture_bundles=240000,wall_seconds=129600),
        memory_snapshot='docs/alfworld_pilot/memory_snapshot.json',memory_sha256=file_hash(ROOT/'docs/alfworld_pilot/memory_snapshot.json'),
        no_actor_or_reviewer_training=True,format_failure='agent_abort_preserve_last_native_score_no_invented_action',
        reviewer_guard='reject_nonadmissible_replacement_keep_original',teacher_forced_activation_forward=True,
        utility_deployment_requirement='Both help and harm heads must have sufficient independent training/validation labels; otherwise no fake utility policy')
    atomic(ROOT/'configs/evaluation/activation_v1.json',config)
    print(json.dumps(dict(partitions=split['family_counts'],available=len(groups),missing_families=split['missing_families']),indent=2))


if __name__=='__main__':main()
