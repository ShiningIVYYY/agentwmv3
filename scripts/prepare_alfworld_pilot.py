"""Metadata-only split; never inspect walkthroughs or train a model."""
import argparse,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic
from experience.splits import file_hash


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
TYPES=['pick_and_place_simple','look_at_obj_in_light','pick_clean_then_place_in_recep','pick_heat_then_place_in_recep','pick_cool_then_place_in_recep','pick_two_obj_and_place']


def catalog(root,split):
    rows=[]
    for f in sorted((root/'json_2.1.1'/split).glob('*/*/game.tw-pddl')):
        rel=str(f.relative_to(root));meta=f.parent/'traj_data.json'
        if not meta.exists() or 'movable' in rel or 'Sliced' in rel:continue
        x=json.loads(meta.read_text());g=json.loads(f.read_text())
        if x['task_type'] not in TYPES or g.get('solvable') is not True:continue
        # Only public task parameters/identity are retained; expert plans are not used.
        params={k:x['pddl_params'][k] for k in ['object_target','parent_target','toggle_target','mrecep_target','object_sliced']}
        relevant=['object_target','toggle_target'] if x['task_type']=='look_at_obj_in_light' else ['object_target','parent_target']
        semantic={k:('' if params[k] is None or str(params[k]).casefold()=='none' else str(params[k]).strip().casefold()) for k in relevant}
        group=digest(dict(task_type=x['task_type'],goal_parameters=semantic,object_sliced=bool(params['object_sliced'])))
        rows.append(dict(task_id='alf-'+digest(rel)[:16],game=rel,game_sha256=file_hash(f),metadata_sha256=file_hash(meta),
                         family=x['task_type'],goal_group=group,params=params,source_split=split,seed=1000+len(rows)))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);a=ap.parse_args();root=Path(a.data_root)
    train=catalog(root,'train');test=catalog(root,'valid_seen')
    if len(test)!=140:raise ValueError('Official valid_seen identity count differs')
    test_goals={r['goal_group'] for r in test};groups=defaultdict(list)
    for row in train:
        if row['goal_group'] not in test_goals:groups[row['goal_group']].append(row)
    families=defaultdict(list)
    for g,rs in groups.items():families[rs[0]['family']].append(g)
    for family in families:families[family].sort(key=lambda g:digest(['alf-pilot-goal-v1',g]))
    ranked=[]
    while any(families.values()):
        for family in sorted(families):
            if families[family]:ranked.append(families[family].pop(0))
    if len(ranked)<13:raise ValueError('Insufficient train-only disjoint goals')
    selected=[]
    for i,g in enumerate(ranked[:13]):
        row=min(groups[g],key=lambda r:digest(['alf-pilot-instance-v1',r['game']]))
        selected.append(dict(row,reserved_role='experience' if i<2 or i>=11 else 'calibration' if i<8 else 'development_evaluation'))
    out=ROOT/'docs/alfworld_pilot';out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema='alfworld-train-goal-disjoint-pilot-v1',source_train_count=len(train),final_valid_seen_count=len(test),
        final_valid_seen=test,source_filtered_instances=sum(len(v) for v in groups.values()),source_filtered_goals=len(groups),
        memory_tasks=selected[:2]+selected[11:13],calibration_tasks=selected[2:8],evaluation_tasks=selected[8:11],
        selection='Canonical task-parameter goals; exclude valid_seen goals; SHA256 family round-robin; first2 memory,next6 calibration,next3 evaluation; next2 memory reserves activated after insufficient evidence, BEFORE any model evaluation',
        assets=dict(domain_sha256=file_hash(root/'logic/alfred.pddl'),grammar_sha256=file_hash(root/'logic/alfred.twl2')),
        no_test_environment_rollout=True,expert_trajectories_used=False,cross_domain_memory=False,
        limitations=['Only exact task-parameter duplicate goals excluded; task templates/house layouts may overlap','Tiny3-game development comparison, not official140 valid_seen performance'])
    atomic(out/'split_manifest.json',manifest)
    base=json.loads((ROOT/'configs/internal_signals/qwen35_9b_replay_v1.json').read_text())
    base.update(max_prompt_tokens=2048,prompt_schema='alfworld-premise-consequence-action-v1',purpose='Frozen9B ALFWorld pilot; new prompt identity and calibration; no Science thresholds')
    atomic(ROOT/'configs/internal_signals/qwen35_9b_alfworld_v1.json',base)
    profile=dict(schema='alfworld-five-gate-development-v1',actor_config='configs/internal_signals/qwen35_9b_alfworld_v1.json',
        model_id=base['model_id'],model_revision=base['model_revision'],actor_reviewer_shared=True,benchmark='alfworld',
        maximum_interaction_turns=50,history_length=2,prompt_limit=2048,response_limit=512,temperature=.4,top_p=1.,top_k=0,
        rollout_seeds=[0],policies=['never','always','internal','experience','full'],
        wm_invocation_budgets=dict(never=0,always=50,internal=5,experience=5,full=5),simulation_depth=1,review_rounds=1,
        reviewer_protocol='original-three-line-public-evidence-alfworld-v1',reviewer_max_new_tokens=512,
        calibration_quantile=.75,calibration_max_steps=8,memory_task_action_cap=30,
        memory_correction=False,allow_confirmed_action_skip=False,final_metric='native_won_success_rate',
        score_display='100*won auxiliary only; not ScienceWorld partial score',
        split_manifest='docs/alfworld_pilot/split_manifest.json',split_sha256=file_hash(out/'split_manifest.json'),
        cpu_versions=dict(alfworld='0.4.2',textworld='1.7.0'),
        limits=dict(tokens=2000000,model_calls=1100,transition_queries=7000,audit_capture_bundles=20000,wall_seconds=7200),
        inherited_science_configs_preserved=True,no_training=True,no_learned_wm=True,
        native_equivalence='Exact public transitions/admissible commands and private fact digest; not full hidden RNG serialization')
    atomic(ROOT/'configs/evaluation/alfworld_pilot_v1.json',profile)
    print(json.dumps(dict(train=len(train),test=len(test),eligible_groups=len(groups),selected=[(r['reserved_role'],r['family'],r['task_id']) for r in selected]),indent=2))


if __name__=='__main__':main()
