#!/usr/bin/env python3
"""Outcome-blind cohort/freeze manifest before any Phase5.9 generation."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.splits import validate_manifest,file_hash
from worldmodel.ground_truth_environment import simulator_identity
ID=re.compile(r'sciworld_\d+')
ACTUAL={'draft_action','executed_action','final_action','environment_feedback','actual','candidate','transition','prefix','formal_environment_actions','real_feedback','committed_actions'}


def rank(text):return hashlib.sha256(('phase59-independent-v1|'+text).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def actual_identities(value,ids,pairs,active=False):
    if isinstance(value,dict):
        active=active or bool(ACTUAL&value.keys())
        if active:
            for key in ('task_id','taskId','item_id'):
                if isinstance(value.get(key),str):ids.update(ID.findall(value[key]))
            name=value.get('taskName',value.get('task_name'))
            variation=value.get('variationIdx',value.get('variation_id'))
            if isinstance(name,str) and isinstance(variation,int):pairs.add((name,variation))
        for child in value.values():actual_identities(child,ids,pairs,active)
    elif isinstance(value,list):
        for child in value:actual_identities(child,ids,pairs,active)


def select(rows,count=12):
    # One round-robin representative per family, prioritizing fresh goal groups;
    # then at most two distinct variations per goal. No score/feedback is read.
    families=sorted({r['taskName'] for r in rows},key=rank)
    selected=[];goals=Counter();used=set()
    for cap in (1,2):
        while len(selected)<count:
            changed=False
            for family in families:
                choices=[r for r in rows if r['taskName']==family and r['task_id'] not in used and goals[r['goal_group']]<cap]
                if not choices:continue
                row=min(choices,key=lambda r:(goals[r['goal_group']],rank(r['goal_group']),rank(r['task_id'])))
                selected.append(row);used.add(row['task_id']);goals[row['goal_group']]+=1;changed=True
                if len(selected)==count:break
            if not changed:break
    if len(selected)!=count:raise ValueError('Insufficient independent task instances under frozen selection rule')
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='configs/evaluation/phase59_independent_dev_v1.json')
    p.add_argument('--audit',default='docs/phase59/exposure_audit.json');a=p.parse_args()
    out=ROOT/a.output;auditpath=ROOT/a.audit
    if out.exists() or auditpath.exists():raise ValueError('Refuse to reselect an already frozen cohort')
    split=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text());validate_manifest(split,require_native=True)
    observed=set();pairs=set();sources=[]
    for path in sorted((ROOT/'outputs').rglob('*')):
        if not path.is_file() or path.suffix not in ('.json','.jsonl'):continue
        if any(s.startswith('phase59') for s in path.relative_to(ROOT/'outputs').parts):continue
        if path.name=='phase59_exposure_precheck.json':continue
        text=path.read_text();ids=set();native=set()
        if path.suffix=='.jsonl':
            ids.update(ID.findall(str(path)+'\n'+text))
            for line in text.splitlines():
                if line.strip():actual_identities(json.loads(line),ids,native)
        else:
            actual_identities(json.loads(text),ids,native)
        if ids or native:
            observed.update(ids);pairs.update(native)
            sources.append(dict(path=str(path.relative_to(ROOT)),sha256=file_hash(path),ids=sorted(ids),native_pairs=sorted(native)))
    declared=set()
    for base in ('scripts','configs'):
        for path in (ROOT/base).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.sh') and 'phase59' not in path.name and path.name!='prepare_independent_dev_eval.py':
                declared.update(ID.findall(path.read_text()))
    plan=ROOT/'docs/phase5/development_plan.json';declared.update(ID.findall(plan.read_text()))
    allrows=[r for rows in split['partitions'].values() for r in rows]
    observed.update(r['task_id'] for r in allrows if (r['taskName'],r['variationIdx']) in pairs)
    dev=split['partitions']['development']
    exposed_goals={r['goal_group'] for r in dev if r['task_id'] in observed}
    candidates=[r for r in dev if r['task_id'] not in observed|declared and r['goal_group'] not in exposed_goals]
    selected=select(candidates)
    forbidden_ids={r['task_id'] for s in ('train','test') for r in split['partitions'][s]}
    assert not ({r['task_id'] for r in selected}&forbidden_ids)
    assert not ({r['goal_group'] for r in selected}&exposed_goals)
    # Freeze every method component; only the evaluation driver/supervisor is new.
    paths=set()
    for folder in ('internal_signals','gates','experience'):
        paths.update(p for p in (ROOT/folder).glob('*.py'))
    paths.update(ROOT/p for p in ['reflection/replay_review.py','worldmodel/ground_truth_environment.py',
        'worldmodel/contexts.py','worldmodel/backends.py','evaluation/profile.py',
        'configs/internal_signals/qwen35_9b_replay_v1.json','configs/gates/qwen35_replay_pilot_v1.json',
        'docs/phase5/memory_snapshot_v2.json','docs/phase3/split_manifest.json','docs/PAPER_ALIGNED_EVAL.md'])
    _,engine=simulator_identity({'simulator_profile':'scienceworld130'})
    profile=json.loads((ROOT/'configs/evaluation/phase57_replay_dev_v2.json').read_text())
    profile={k:v for k,v in profile.items() if k not in ('calibration_tasks','evaluation_tasks','calibration_max_actions','calibration_quantile','requires_fresh_calibration')}
    profile.update(schema='phase59-frozen-independent-development-v1',purpose='independent_dev_evaluation_not_final_benchmark',
        method_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tasks=selected,round1_task_ids=[r['task_id'] for r in selected[:5]],rollout_seeds=[0,1],
        simulator_seed=0,seed_policy='phase57-compatible request SHA256(role,task,turn,rollout); native task seed remains0',
        gate_config='configs/gates/qwen35_replay_pilot_v1.json',actor_config='configs/internal_signals/qwen35_9b_replay_v1.json',
        engine=engine,method_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},
        exposure_audit=a.audit,selection='family round-robin / SHA256 ranks / max2 variants per unseen normalized goal',
        threshold_calibration_in_this_phase=False,freeze_before_first_generation=True,
        limits=dict(max_workers=2,max_llm_tokens=4000000,max_model_calls=2650,max_transition_queries=12500,
            max_audit_capture_bundles=96000,max_wall_seconds=28800,max_batches=9,max_groups_per_batch=3),
        worker_limits=dict(tokens=2000000,model_calls=1325,transition_queries=6250,audit_capture_bundles=48000),
        batch_limits=dict(tokens=500000,model_calls=330,transition_queries=1512,audit_capture_bundles=12000),
        analysis_plan=dict(bootstrap_seed=202659,bootstrap_resamples=2000,confidence=.95,cluster='goal_group',
            paired_comparisons=['full-never','full-always','full-internal'],seed_sd='sample_ddof1',
            aggregation='instance_weighted',influence='leave_one_task_and_goal_out',no_significance_claim=True),
        round1_acceptance='engineering checks only; do not condition continuation on scores or Full advantage',
        continuation='same12 tasks x2 seeds x4 policies; reuse first5/seed0 results without regeneration')
    audit=dict(schema='phase59-exposure-audit-v1',source_files=sources,observed_ids=sorted(observed),
        declared_ids=sorted(declared),native_pairs=sorted(pairs),exposed_dev_goal_groups=sorted(exposed_goals),
        counts=dict(dev_pool=len(dev),observed_dev_ids=sum(r['task_id'] in observed for r in dev),
            exposed_goal_groups=len(exposed_goals),eligible_instances=len(candidates),eligible_goal_groups=len({r['goal_group'] for r in candidates})),
        eligible_families=dict(Counter(r['taskName'] for r in candidates)),selected_ids=[r['task_id'] for r in selected],
        independent_of_prior_outcomes=True,final_test_used=False,
        limits=['Initial identity/catalog metadata was previously available; it is not an untouched research process.',
                'Shared task templates/layouts persist; semantic near-duplicate independence not established.',
                'Only8 untouched normalized goal groups remain; repeated variants are clustered in uncertainty analysis.',
                'Previously planned representative IDs excluded; unused goal-group names in pool metadata alone are not executed debugging.'])
    write(auditpath,audit);profile['exposure_audit_sha256']=file_hash(auditpath);write(out,profile)
    print(json.dumps(dict(counts=audit['counts'],selected=[(r['task_id'],r['taskName'],r['variationIdx']) for r in selected],
        selected_goal_groups=len({r['goal_group'] for r in selected}),profile_sha256=file_hash(out)),indent=2))


if __name__=='__main__':main()
