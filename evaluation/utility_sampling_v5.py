"""Versioned offline selection: TRAIN enrichment, outcome-blind Validation."""
import hashlib,json,random
from evaluation.recovery_collection import select_pair

WINDOWS=((0,9),(10,29),(30,49))


def rank(*parts):
    return hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()


def context_key(c):
    return rank(c['goal_group'],c['metadata']['generation_record']['prompt_text'],c['draft_action'])


def changed_candidates(cases,role,task,blocked,limit=4):
    if role!='probe_train':raise ValueError('Reviewer-outcome enrichment is TRAIN only')
    pools=[];seen=set(blocked)
    for lo,hi in WINDOWS:
        pool=[]
        for c in sorted(cases,key=lambda x:rank('v5-changed',task,x['case_id'])):
            key=context_key(c)
            if not c.get('feature_valid',True) or key in seen:continue
            if c['source_policy']!='always' or c['executed_action']==c['draft_action']:continue
            if not lo<=c['step']<=hi:continue
            seen.add(key);pool.append(c)
        pools.append(pool)
    selected=[]
    while len(selected)<limit and any(pools):
        for pool in pools:
            if pool and len(selected)<limit:
                c=pool.pop(0)
                selected.append(dict(case_id=c['case_id'],step=c['step'],kind='train_real_accepted_revision_v5',
                    inclusion_probability=None,selection_uses_future_outcomes=True,
                    selection_scope='Offline TRAIN scheduling only; no success or utility filter'))
    return selected


def train_selection(never,always,task,limit=4):
    base,first=select_pair(never,always,'probe_train',task)
    lookup={c['case_id']:c for c in never+always};keys=set();selected=[]
    for s in base:
        key=context_key(lookup[s['case_id']])
        if key not in keys:keys.add(key);selected.append(s)
    return selected+changed_candidates(always,'probe_train',task,keys,limit),first


def validation_selection(cases,task,per_window=4):
    """Use only pre-action identity/context and feature validity, never outcomes."""
    selected=[];seen=set()
    for lo,hi in WINDOWS:
        unique={}
        for c in cases:
            if c.get('feature_valid',True) and lo<=c['step']<=hi:
                key=context_key(c)
                if key not in seen:unique.setdefault(key,c)
        seen.update(unique)
        pool=[unique[k] for k in sorted(unique)];n=len(pool);k=min(n,per_window)
        seed=int(rank('v5-validation-uniform',task,lo,hi)[:16],16)
        for i in random.Random(seed).sample(range(n),k):
            c=pool[i]
            selected.append(dict(case_id=c['case_id'],step=c['step'],window=[lo,hi],
                kind='validation_uniform_without_replacement_v5',pool_size=n,
                inclusion_probability=k/n,selection_uses_future_outcomes=False))
    return selected
