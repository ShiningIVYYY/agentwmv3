"""Freeze source-filtered development extension BEFORE any new model outcome."""
import hashlib,json,sys,re
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.splits import file_hash,goal_key,validate_manifest,digest
from evaluation.independent_dev import atomic


def goal_from_dialogue(row):
    msgs=row['messages']
    if len(msgs)<3 or msgs[2]['role']!='user':raise ValueError('Unknown public training row')
    text=msgs[2]['content']
    boundary=re.search(r'\nThis (?:room|outside location) is called ',text)
    if not text.startswith('Your task is to ') or boundary is None:
        raise ValueError('Unverified public goal boundary')
    return text[:boundary.start()]


def main():
    out=ROOT/'docs/sr_integration';out.mkdir(parents=True,exist_ok=True)
    groups=set();sources=[]
    for name,expected in [('expert_sft.jsonl','096895094fe44005b03037b44414388bc4935dcc0224fcc678f9811545cc2708'),
                          ('reflection_sft.jsonl','8020d0c6fe9dd86105870fc4bfe7c9ec45fcedb555318c9979c0e485016eb4ba')]:
        path=ROOT/'outputs/ee_sr_audit'/name
        assert file_hash(path)==expected
        local=set();count=0
        for line in path.open():local.add(goal_key(goal_from_dialogue(json.loads(line))));count+=1
        groups|=local;sources.append(dict(file=name,sha256=expected,rows=count,goal_groups=len(local),bytes=path.stat().st_size))
    split=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text());validate_manifest(split,require_native=True)
    memory=json.loads((ROOT/'docs/phase5/memory_snapshot_v2.json').read_text())
    lookup={r['task_id']:r for rs in split['partitions'].values() for r in rs}
    memory_ids={r['task_id'] for r in memory['cases']};memory_goals={lookup[i]['goal_group'] for i in memory_ids}
    test_goals={r['goal_group'] for r in split['partitions']['test']};test_ids={r['task_id'] for r in split['partitions']['test']}
    exposure=json.loads((ROOT/'docs/phase59/exposure_audit.json').read_text())
    excluded_ids=set(exposure['observed_ids'])|set(exposure['declared_ids'])|memory_ids|test_ids
    # Later evaluation sources use the original development partition, so all of it
    # is excluded. This extension only reserves previously-unused TRAIN goals.
    excluded_goals=memory_goals|test_goals|{r['goal_group'] for r in split['partitions']['development']}
    for name in ['configs/evaluation/phase59_independent_dev_v1.json','configs/evaluation/phase510_reviewer_repair_v1.json','configs/evaluation/ee_sr_actor_trial_v1.json']:
        d=json.loads((ROOT/name).read_text())
        for r in d.get('tasks',[]):excluded_ids.add(r['task_id']);excluded_goals.add(r['goal_group'])
    candidates=[r for r in split['partitions']['train'] if r['goal_group'] not in groups|excluded_goals and r['task_id'] not in excluded_ids]
    by_goal=defaultdict(list)
    for r in candidates:by_goal[r['goal_group']].append(r)
    by_family=defaultdict(list)
    for g,rs in by_goal.items():by_family[rs[0]['category']].append(g)
    for family in by_family:by_family[family].sort(key=lambda g:digest('sr-public-source-isolation-v1|'+g))
    ranked=[]
    while any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family]:ranked.append(by_family[family].pop(0))
    if len(ranked)<9:raise ValueError('Insufficient source-filtered groups; no substitution with Test tasks')
    chosen=[]
    for g in ranked[:9]:
        row=min(by_goal[g],key=lambda r:digest('sr-public-source-instance-v1|'+r['task_id']))
        chosen.append(dict(row,original_partition='train',reserved_role='calibration' if len(chosen)<6 else 'evaluation'))
    assert len({r['goal_group'] for r in chosen})==9
    assert not {r['goal_group'] for r in chosen}&groups
    protected=json.loads((ROOT/'configs/evaluation/ee_sr_actor_trial_v1.json').read_text())['protected_hashes']
    receipt=dict(schema='sr-public-source-isolation-v1',model_revision='4b8f0835a3f613e40541619462ae906db8dba95c',
        corpus_revision='7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce',sources=sources,published_goal_groups=len(groups),
        eligible_instances=len(candidates),eligible_groups=len(by_goal),eligible_tasks=candidates,
        calibration_tasks=chosen[:6],evaluation_tasks=chosen[6:],selection='SHA256-ranked family round-robin; one instance per whole goal group; first6 calibration/next3 evaluation; no outcomes',
        old_split_sha256=file_hash(ROOT/'docs/phase3/split_manifest.json'),memory_sha256=file_hash(ROOT/'docs/phase5/memory_snapshot_v2.json'),
        excluded_memory_source_ids=sorted(memory_ids),excluded_all_final_test_ids=True,excluded_all_final_test_goals=True,
        old_development_partition_excluded=True,old_manifest_and_memory_unchanged=True,
        optimizer_training_membership_verified=False,complete_checkpoint_training_isolation_claimed=False,
        interpretation='Exact normalized-goal disjoint from BOTH published SR sources; not proof of unpublished checkpoint training isolation or semantic-template isolation',
        published_source_goal_keys=sorted(groups))
    atomic(out/'isolation_manifest.json',receipt)
    base=json.loads((ROOT/'configs/evaluation/phase510_reviewer_repair_v1.json').read_text())
    keep=['maximum_interaction_turns','history_length','prompt_limit','response_limit','temperature','top_p','top_k','simplifications',
          'environment_step_limit','task_seed','rollout_seeds','rng_seed_policy','sdk_transition_policy','reconstruction_failure',
          'wm_invocation_budgets','memory_snapshot','reviewer_feedback_fields','full_hidden_state_equivalence_claimed','simulator_profile','simulator_seed','seed_policy','engine','simulation_depth','review_rounds']
    profile={k:base[k] for k in keep}
    profile.update(schema='sr-source-filtered-method-integration-v1',purpose='public_source_goal_disjoint_paired_development_not_final_test',
        models=['sr','qwen35'],policies=['never','always','internal','full'],calibration_tasks=chosen[:6],tasks=chosen[6:],
        calibration_max_actions=8,calibration_quantile=.75,isolation_manifest='docs/sr_integration/isolation_manifest.json',
        isolation_sha256=file_hash(out/'isolation_manifest.json'),sr_actor_config='configs/internal_signals/ee_sr_structured_v1.json',
        qwen35_actor_config='configs/internal_signals/qwen35_9b_replay_v1.json',reviewer_version='compact-json-conservative-v2',reviewer_cap=192,
        actor_reviewer_share_weights_per_model=True,sr_only_syntax_adaptation=True,prompt_content_unchanged=True,
        thresholds='Model-specific q.75 on shared isolated calibration; old9B/SR trial scores never reused',
        limits=dict(tokens=1000000,model_calls=700,transition_queries=6000,audit_capture_bundles=40000,wall_seconds=3600),protected_hashes=protected)
    atomic(ROOT/'configs/evaluation/sr_isolated_integration_v1.json',profile)
    c=json.loads((ROOT/'configs/internal_signals/ee_scienceworld_sr_v1.json').read_text())
    c.update(backend='hf_ee_sr_structured_raw_bf16_v1',format_constraint='three-line-and-compact-json-fsm-v1',alignment_method='actual_generated_token_stream_offsets_v1',
             purpose='Frozen SR Actor/Reviewer method integration, model-specific raw NLL calibration')
    atomic(ROOT/profile['sr_actor_config'],c)
    print(json.dumps(dict(eligible=len(candidates),groups=len(by_goal),calibration=[r['task_id'] for r in chosen[:6]],evaluation=[r['task_id'] for r in chosen[6:]],sources=sources),indent=2))


if __name__=='__main__':main()
