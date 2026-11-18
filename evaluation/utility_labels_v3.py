"""Strict implementation of the already declared Utility-v2 label definition.

Repairs the legacy `zero_count == 3` error on five-seed mixed outcomes.
Actor/capture/sampling/positive thresholds remain unchanged.
"""
from collections import Counter

SCHEMA='utility-label-v3-all-zero-neutral'


def aggregate(deltas,changed,unchanged_verified=False):
    if not changed:
        if unchanged_verified and deltas==[0]:
            return dict(label='neutral',utility_help=0,utility_harm=0,reliability='verified_equal_action_state_and_seed1_continuation',mean_delta=0.,
                label_schema=SCHEMA,note='Observed equal action/state/continuation, not semantic action correctness; no unrun seeds invented.')
        return dict(label='unknown',utility_help=None,utility_harm=None,reliability='unverified_unchanged_action',mean_delta=None,label_schema=SCHEMA)
    if len(deltas) not in [3,5] or any(d not in [-1,0,1] for d in deltas):
        return dict(label='unknown',utility_help=None,utility_harm=None,reliability='incomplete_repeated_pairs',mean_delta=None,label_schema=SCHEMA)
    c=Counter(deltas)
    if len(deltas)==5 and c[1]>=3 and c[-1]==0:label='help'
    elif len(deltas)==5 and c[-1]>=3 and c[1]==0:label='harm'
    elif all(d==0 for d in deltas):label='neutral'
    else:label='uncertain'
    return dict(label=label,utility_help=int(label=='help') if label!='uncertain' else None,
        utility_harm=int(label=='harm') if label!='uncertain' else None,reliability=str(len(deltas))+'_paired_continuation_seeds' if label!='uncertain' else 'seed_sensitive',
        mean_delta=sum(deltas)/len(deltas),label_schema=SCHEMA,
        note='Fixed Reviewer seed0 and Never-WM continuation; finite-seed diagnostic rule, not statistical proof.')


def relabel(record):
    result=dict(record)
    if record.get('deltas') is not None:
        update=aggregate(record['deltas'],record['changed'],not record['changed'])
        result.update(update)
        if update['label']!=record['label']:
            result['legacy_label']=record['label'];result['correction']='Mixed five-seed outcomes must be Uncertain, not Neutral'
    return result
