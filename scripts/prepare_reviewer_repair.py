#!/usr/bin/env python3
"""Freeze bounded regression cases/new-ID known-goal variants; never select by new results."""
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows
from experience.splits import file_hash,validate_manifest
from scripts.audit_independent_dev import recorded_seed
from scripts.audit_phase57_replay import feedback_status
from scripts.export_phase57_evidence import compact
from reflection.replay_review import review_messages
from gates.core import RuleGate,GateInput,ExternalInput
from experience.memory import ExperienceMemory
from experience.schema import ExternalQuery
from experience.collection import fields


def rank(s):return hashlib.sha256(('phase510-regression-v1|'+s).encode()).hexdigest()

def events():
    files=[ROOT/'outputs/replay_fix_130_e2e_v1/evaluation_steps.jsonl']+sorted((ROOT/'outputs/phase59_independent_v2/episodes').rglob('steps.jsonl'))
    result=[];sources={}
    for f in files:
        sources[str(f.relative_to(ROOT))]=file_hash(f);groups=defaultdict(list)
        for i,line in enumerate(f.read_text().splitlines()):
            t=json.loads(line);key=(t['task_id'],t['policy'],t.get('rollout_seed',0));past=groups[key]
            if t.get('review_metadata'):
                assert t['simulation']['prediction_available'] and t['simulation']['source']=='ground_truth_environment'
                prompt=review_messages(t['goal'],t['observation'],past,t['draft_action'],t['simulation'])
                meta=t['review_metadata'];assert prompt==meta['generation_record']['messages']
                prompt_id=rank(json.dumps(dict(prompt=prompt,seed=recorded_seed(meta)),sort_keys=True))
                ref=dict(source=str(f.relative_to(ROOT)),line_index=i,line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                    task_id=t['task_id'],policy=t['policy'],seed=t.get('rollout_seed',0),step=t['step']+1,prompt_id=prompt_id)
                result.append(dict(ref=ref,t=t,past=list(past),legacy='replay_fix_130' in str(f)))
            past.append(dict(observation=t['observation'],action=t['final_action']))
    return result,sources


def source_case(ref):
    f=ROOT/ref['source'];lines=f.read_text().splitlines();line=lines[ref['line_index']]
    assert hashlib.sha256(line.encode()).hexdigest()==ref['line_sha256']
    t=json.loads(line);past=[];actions=[]
    for l in lines[:ref['line_index']]:
        row=json.loads(l)
        if (row['task_id'],row['policy'],row.get('rollout_seed',0))==(t['task_id'],t['policy'],t.get('rollout_seed',0)):
            past.append(dict(observation=row['observation'],action=row['final_action']));actions.append(row['final_action'])
    assert len(actions)==t['step'];return t,past,actions


def main():
    output=ROOT/'configs/evaluation/phase510_reviewer_repair_v1.json'
    if output.exists():raise ValueError('Refuse to replace frozen cases/config')
    ev,sources=events();recent=[e for e in ev if not e['legacy']];invalid=[e for e in recent if e['t']['review']['parsed']['decision']=='INVALID']
    counts=Counter((e['t']['review']['parsed']['decision'],e['t']['review_metadata']['finish_reason']) for e in recent)
    failures=[]
    for e in invalid:
        t=e['t'];m=t['review_metadata'];text=t['review']['raw'];r=m['generation_record']
        failures.append(dict(**e['ref'],finish_reason=m['finish_reason'],tokens=m['usage']['completion_tokens'],
            recorded_truncated=r['truncated'],response_ids=len(r['response_ids']),final_action_field_present='\nFinal Action:' in text,
            blank_before_final='\n\nFinal Action:' in text,reason_words=len(text.partition('Reason:')[2].partition('Final Action:')[0].split()),
            raw=text,parser=t['review']['parsed'],template_tail=r['prompt_text'][-85:]))
    audit=dict(reviewer_requests=len(recent),format_failures=len(invalid),counts={str(k):v for k,v in counts.items()},
        length_failures=sum(x['finish_reason']=='length' and x['recorded_truncated'] and x['tokens']==512 for x in failures),
        stopped_blankline_failures=sum(x['finish_reason']=='stop' and x['blank_before_final'] for x in failures),
        generated_think_markers=sum('<think>' in e['t']['review']['raw'] or '</think>' in e['t']['review']['raw'] for e in recent),
        empty_closed_think_prefix=sum(e['t']['review_metadata']['generation_record']['prompt_text'].endswith('<think>\n\n</think>\n\n') for e in recent),
        original_is_json_protocol=False,reviewer_retry_requests=0,failures=failures,source_hashes=sources)
    assert audit['length_failures']==50 and audit['stopped_blankline_failures']==3
    atomic(ROOT/'docs/phase510/original_format_audit.json',audit)
    # Select20 unique old debug requests, balanced by original diagnostic category.
    selected=[];used=set()
    def category(e):
        t=e['t'];r=t['review']['parsed']
        if r['decision']=='INVALID':return 'format_failure'
        if t['final_action']==t['draft_action']:return 'unchanged'
        return feedback_status(t['actual']['observation'])
    for cat in ('format_failure','explicit_rejection','accepted_feedback','unchanged'):
        choices=sorted([e for e in ev if e['legacy'] and category(e)==cat],key=lambda e:rank(e['ref']['prompt_id']))
        n=0
        for e in choices:
            if e['ref']['prompt_id'] in used:continue
            selected.append(dict(**e['ref'],panel='old_debug',stratum=cat));used.add(e['ref']['prompt_id']);n+=1
            if n==5:break
        if n!=5:raise ValueError('Insufficient prespecified old debug stratum '+cat)
    # All53 failure events, deduplicated by EXACT original prompt+seed.
    regression=[]
    for e in sorted(invalid,key=lambda e:rank(e['ref']['prompt_id'])):
        matching=next((x for x in regression if x['prompt_id']==e['ref']['prompt_id']),None)
        if matching:matching['source_events'].append(e['ref']);continue
        regression.append(dict(**e['ref'],panel='phase59_format_failure',source_events=[e['ref']]))
    #12 quality-only Full regressions; outcome-targeted, NOT independent performance evidence.
    changed=[e for e in recent if e['t']['policy']=='full' and e['t']['final_action']!=e['t']['draft_action']]
    quality=[];qused=set()
    for label,predicate,n in [('negative_delta',lambda t:t['one_step_score_delta_actual_minus_trial_draft']<0,2),
        ('positive_delta',lambda t:t['one_step_score_delta_actual_minus_trial_draft']>0,1),
        ('unknown',lambda t:feedback_status(t['actual']['observation'])=='unknown',3),
        ('rejected',lambda t:feedback_status(t['actual']['observation'])=='explicit_rejection',3),
        ('accepted',lambda t:feedback_status(t['actual']['observation'])=='accepted_feedback',3)]:
        candidates=sorted([e for e in changed if predicate(e['t']) and e['ref']['prompt_id'] not in qused],key=lambda e:rank(e['ref']['prompt_id']))
        assert len(candidates)>=n
        for e in candidates[:n]:quality.append(dict(**e['ref'],panel='phase59_full_quality',stratum=label));qused.add(e['ref']['prompt_id'])
    split=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text());validate_manifest(split,require_native=True)
    old=json.loads((ROOT/'configs/evaluation/phase59_independent_dev_v1.json').read_text());exp=json.loads((ROOT/'docs/phase59/exposure_audit.json').read_text())
    exposed_goals=set(exp['exposed_dev_goal_groups'])|{r['goal_group'] for r in old['tasks']}
    unused_goals={r['goal_group'] for r in split['partitions']['development']}-exposed_goals
    assert not unused_goals
    usedids=set(exp['observed_ids'])|set(exp['declared_ids'])|{r['task_id'] for r in old['tasks']}
    candidates=[r for r in split['partitions']['development'] if r['task_id'] not in usedids and r['goal_group'] not in exp['exposed_dev_goal_groups']]
    tasks=[]
    for cat in sorted({r['category'] for r in candidates}):
        choices=[r for r in candidates if r['category']==cat];tasks.append(min(choices,key=lambda r:rank(r['task_id'])))
    assert len(tasks)==3
    badids={r['task_id'] for s in ('test','train') for r in split['partitions'][s]};assert not badids&{r['task_id'] for r in tasks}
    profile={k:v for k,v in old.items() if k not in ('round1_task_ids','limits','worker_limits','batch_limits','analysis_plan','exposure_audit_sha256','selection','continuation','round1_acceptance')}
    profile.update(schema='phase510-bounded-reviewer-repair-v1',purpose='regression_and_new_id_known_goal_diagnostic_not_independent_generalization',
        tasks=tasks,rollout_seeds=[0],original_base_commit='327b46374e7489c91debf5ccb78ad30d7ca3ba07',
        fixed_reviewer='compact-json-conservative-v2',reviewer_config='configs/reviewers/compact_json_conservative_v2.json',
        old_debug_panel=selected,format_regression_panel=regression,quality_panel=quality,
        source_hashes=sources,unused_dev_goal_groups=0,unused_variant_candidates=len(candidates),
        limits=dict(tokens=650000,model_calls=600,transition_queries=6500,audit_capture_bundles=25000),
        batch_limits=dict(tokens=200000,model_calls=150,transition_queries=1800,audit_capture_bundles=8000),
        max_wall_seconds=10800,max_gpu_workers=1,
        episode_variants=[dict(policy='never',reviewer='original'),dict(policy='always',reviewer='original'),dict(policy='always',reviewer='compact-json-conservative-v2'),dict(policy='full',reviewer='original'),dict(policy='full',reviewer='compact-json-conservative-v2')],
        design_rules=['no threshold/memory/gate change','offline generation before posthoc branch feedback','same-state one-step utility only','new IDs chosen before new Reviewer outcomes; known goal groups','no final test; no Phase6; no output-based task reselection'])
    atomic(ROOT/'docs/phase510/task_isolation.json',dict(candidates=candidates,selected=tasks,unused_goal_groups=[],
        previously_exposed_goal_groups=len(exposed_goals),prior_phase59_ids=old['tasks'],source_audit_sha256=file_hash(ROOT/'docs/phase59/exposure_audit.json'),
        limitation='No pristine goal-level dev holdout remains; new-ID variants are regression/within-goal checks only.'))
    # Gate analysis only. Re-evaluate applicability on the same historical public query.
    memory=ExperienceMemory(json.loads((ROOT/old['memory_snapshot']).read_text()),split);gate=json.loads((ROOT/old['gate_config']).read_text());extras=[];triggers=Counter()
    for e in [e for e in recent if e['t']['policy']=='full']:
        t=e['t'];sig=t['metadata']['internal_signals'];l=sig['local_uncertainty']>gate['tau_local'];g=sig['global_uncertainty']>gate['tau_global'];triggers['both' if l and g else 'local_only' if l else 'global_only' if g else 'neither']+=1
        context=GateInput(t['gate']['decision_id'],t['gate']['budget_remaining'],sig,ExternalInput(**t['resolved_external']),t['gate']['budget_remaining'],t['metadata']['decision_format_valid'])
        counter=RuleGate(dict(gate,policy='internal')).decide(context)
        if t['gate']['call_wm'] and not counter['call_wm']:
            premise,consequence,draft=fields(t['draft_output']);q=ExternalQuery(benchmark='scienceworld',task_id=t['task_id'],episode_id=t['gate']['decision_id'].rsplit('/',1)[0],step_index=t['step']+1,available_at_ns=t['decided_at_ns'],goal=t['goal'],observation=t['observation'],draft_action=draft,premise=premise,expected_consequence=consequence)
            fresh=memory.evaluate(q);assert fresh['conflict']=='true'
            assert sorted(fresh['supporting_evidence_ids'])==sorted(t['resolved_external']['evidence_ids'])
            extras.append(dict(ref=e['ref'],trace=compact(t),applicability_recheck=fresh,trial_confirms_closed_door_rule=t['simulation']['predicted_observation']=='The door is not open.'))
    atomic(ROOT/'docs/phase510/gate_audit.json',dict(external_added_calls=extras,count=len(extras),full_state_trigger_overlap=dict(triggers),
        no_gate_tuning=True,no_downstream_causal_utility_claim=True,experience_hash=file_hash(ROOT/old['memory_snapshot'])))
    atomic(output,profile)
    print(json.dumps(dict(old_debug=len(selected),unique_failure_prompts=len(regression),failure_events=len(invalid),quality=len(quality),tasks=tasks,unused_goal_groups=0),indent=2))


if __name__=='__main__':main()
