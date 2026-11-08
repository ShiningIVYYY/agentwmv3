#!/usr/bin/env python3
"""Post-run reviewer/cost diagnostics on completed real episodes; no method updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows
from experience.splits import file_hash
from scripts.audit_phase57_replay import feedback_status
from scripts.export_phase57_evidence import compact


def ranks(values):
    # Average tied ranks; deterministic and no new scipy dependency.
    order=np.argsort(values,kind='stable');out=np.empty(len(values),dtype=float);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out


def correlation(x,y):
    if len(x)<2 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output)
    if not (run/'complete.json').exists():raise ValueError('Complete run required')
    if json.loads((run/'final_audit.json').read_text())['status']!='engineering_checks_pass':raise ValueError('Engineering audit required')
    if out.exists():raise ValueError('New output required')
    traces=[];episodes=[]
    for f in sorted((run/'episodes').rglob('steps.jsonl')):
        traces.extend(read_rows(f));episodes.append(json.loads((f.parent/'complete.json').read_text()))
    if len(episodes)!=96:raise ValueError('All96 required')
    results=[];examples=[]
    for policy in ('never','always','internal','full'):
        ts=[t for t in traces if t['policy']==policy];es=[e for e in episodes if e['policy']==policy]
        sims=[t for t in ts if t['simulation']];reviews=[t for t in ts if t['review_metadata']]
        changed=[t for t in reviews if t['final_action']!=t['draft_action']]
        deltas=[t['one_step_score_delta_actual_minus_trial_draft'] for t in changed]
        kept=[t for t in sims if t['simulation']['prediction_available'] and t['final_action']==t['draft_action']]
        nll=[t['metadata']['internal_signals'] for t in ts if 'internal_signals' in t['metadata']]
        pairs=[(s['local_uncertainty'],s['global_uncertainty']) for s in nll if not s['missing_span'] and s['local_uncertainty'] is not None and s['global_uncertainty'] is not None]
        x=np.array([p[0] for p in pairs]);y=np.array([p[1] for p in pairs])
        byrole={}
        for role,metadata in [('actor',[t['metadata'] for t in ts]),('reviewer',[t['review_metadata'] for t in reviews])]:
            byrole[role]=dict(calls=len(metadata),input_tokens=sum(m['usage']['prompt_tokens'] for m in metadata),
                output_tokens=sum(m['usage']['completion_tokens'] for m in metadata),total_tokens=sum(m['usage']['total_tokens'] for m in metadata),
                generation_seconds=sum(m['internal_cost']['generation_seconds'] for m in metadata))
        results.append(dict(policy=policy,episode_count=len(es),formal_steps=len(ts),role_cost=byrole,
            review_decisions=dict(Counter(t['review']['parsed']['decision'] for t in reviews)),
            format_errors=sum(t['review']['parsed']['decision']=='INVALID' for t in reviews),
            format_errors_at_512_tokens=sum(t['review']['parsed']['decision']=='INVALID' and t['review_metadata']['usage']['completion_tokens']==512 for t in reviews),
            format_error_examples=[dict(task_id=t['task_id'],seed=t['rollout_seed'],step=t['step']+1,
                response_tokens=t['review_metadata']['usage']['completion_tokens'],parse=t['review']['parsed'],
                raw_head=t['review']['raw'][:120],raw_tail=t['review']['raw'][-400:])
                for t in reviews if t['review']['parsed']['decision']=='INVALID'][:3],
            replace_same_action=sum(t['review']['parsed']['decision']=='REPLACE' and t['draft_action']==t['final_action'] for t in reviews),
            actual_revisions=len(changed),revision_feedback=dict(Counter(feedback_status(t['actual']['observation']) for t in changed)),
            one_step_revision_score_delta=dict(positive=sum(d>0 for d in deltas),zero=sum(d==0 for d in deltas),negative=sum(d<0 for d in deltas),sum=sum(deltas)),
            replay_attempts=len(sims),replay_verified=sum(t['simulation']['prediction_available'] for t in sims),
            replay_rejections=dict(Counter(t['simulation']['status'] for t in sims if not t['simulation']['prediction_available'])),
            formal_unchanged_checks_passed=sum(t['simulation']['formal_unchanged'] for t in sims),
            same_action_next_projection_checks=len(kept),same_action_next_projection_passed=sum(t['kept_draft_transition_match'] is True for t in kept),
            draft_parse_errors=sum(not t['metadata']['decision_format_valid'] for t in ts),
            scored_steps=len(nll),missing_internal=sum(s['missing_span'] for s in nll),
            local_global_pearson=correlation(x,y),local_global_spearman=correlation(ranks(x),ranks(y)),
            local_quantiles=[float(v) for v in np.quantile(x,[0,.25,.5,.75,1])] if len(x) else [],
            global_quantiles=[float(v) for v in np.quantile(y,[0,.25,.5,.75,1])] if len(y) else [],
            episode_termination=dict(Counter('success_score100' if e['success'] else 'native_done_non_success' if e['done'] else 'outer_horizon20' for e in es)),
            full_hidden_state_equivalence_claimed=False,downstream_per_call_causal_utility_established=False))
        for label,predicate in [('positive',lambda t:t['one_step_score_delta_actual_minus_trial_draft']>0),
            ('negative',lambda t:t['one_step_score_delta_actual_minus_trial_draft']<0),
            ('explicit_rejection',lambda t:feedback_status(t['actual']['observation'])=='explicit_rejection')]:
            for t in [t for t in changed if predicate(t)][:2]:examples.append(dict(label=label,rollout_seed=t['rollout_seed'],**compact(t)))
    atomic(out,dict(policies=results,representative_revisions=examples,
        source_script_sha256=file_hash(Path(__file__)),run_manifest_sha256=file_hash(run/'manifest.json'),
        score_delta_scope='formal final action next score minus isolated draft trial next score; not downstream causal return',
        validity_scope='conservative public-feedback acceptance/rejection/Unknown; not exhaustive grammar validity or task correctness',
        correlation_scope='realized policy states, not threshold re-calibration or independent per-step statistical samples'))
    print(out)


if __name__=='__main__':main()
