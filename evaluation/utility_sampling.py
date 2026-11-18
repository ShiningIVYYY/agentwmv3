"""Pre-outcome stratified sampling and conservative repeated utility labels."""
import hashlib,json,random,math
from collections import Counter

WINDOWS=((0,9),(10,29),(30,49))


def seed_for(*parts):return int(hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()[:16],16)


def public_flags(action,commands,observation,past,external,local,global_,thresholds):
    repeat=len(past)>=2 and all(p['action']==action and p['observation']==observation for p in past[-2:])
    conflict=external['conflict']=='true' and bool(external['supporting_evidence_ids']) and external['reliability']=='supported'
    high=local is not None and global_ is not None and (local>thresholds['tau_local'] or global_>thresholds['tau_global'])
    flags=dict(nonadmissible=action not in commands,repeated_unchanged_context=repeat,verified_external_conflict=conflict,high_nll=high)
    return dict(flags=flags,weight=1+4*flags['nonadmissible']+3*repeat+4*conflict+int(high))


def select_cases(cases,task):
    selected=[]
    for lo,hi in WINDOWS:
        # Exact Actor-context/action duplicates within a window count once.
        unique={}
        for c in cases:
            if lo<=c['step']<=hi:
                key=json.dumps([c['observation'],c['commands'],c['draft_action'],c['past'][-2:]],sort_keys=True)
                unique.setdefault(key,c)
        pool=sorted(unique.values(),key=lambda c:c['step']);n=len(pool)
        if not n:continue
        rng=random.Random(seed_for('utility-v2',task,lo,hi))
        uniform=rng.randrange(n);chosen=[(uniform,'uniform')]
        weights=[c['sampling']['weight'] for c in pool];total=sum(weights)
        if n>1:
            remaining=[i for i in range(n) if i!=uniform];u=rng.random()*(total-weights[uniform]);running=0
            enriched=remaining[-1]
            for i in remaining:
                running+=weights[i]
                if u<running:enriched=i;break
            chosen.append((enriched,'public_difficulty_weighted'))
        for i,kind in chosen:
            inclusion=1. if n==1 else 1/n+sum(weights[i]/(total-weights[j])/n for j in range(n) if j!=i)
            if not 0<inclusion<=1+1e-9:raise ValueError('Invalid selection inclusion probability')
            selected.append(dict(case_id=pool[i]['case_id'],step=pool[i]['step'],window=[lo,hi],kind=kind,pool_size=n,
                inclusion_probability=min(1.,inclusion),weight=weights[i],selection_uses_future_outcomes=False))
    return selected


def aggregate(deltas,changed,unchanged_verified=False):
    if not changed:
        if unchanged_verified and deltas==[0]:
            return dict(label='neutral',utility_help=0,utility_harm=0,reliability='verified_equal_action_state_and_seed1_continuation',mean_delta=0.,
                note='One actual paired continuation; no fabricated unrun seed outcomes. Same post-action Actor context has no persistent review information.')
        return dict(label='unknown',utility_help=None,utility_harm=None,reliability='unverified_unchanged_action',mean_delta=None)
    if len(deltas) not in [3,5] or any(d not in [-1,0,1] for d in deltas):
        return dict(label='unknown',utility_help=None,utility_harm=None,reliability='incomplete_repeated_pairs',mean_delta=None)
    counts=Counter(deltas)
    if len(deltas)==5 and counts[1]>=3 and counts[-1]==0:label='help'
    elif len(deltas)==5 and counts[-1]>=3 and counts[1]==0:label='harm'
    elif counts[0]==3:label='neutral'
    else:label='uncertain'
    return dict(label=label,utility_help=int(label=='help') if label not in ['unknown','uncertain'] else None,
        utility_harm=int(label=='harm') if label not in ['unknown','uncertain'] else None,
        reliability=str(len(deltas))+'_paired_continuation_seeds' if label not in ['unknown','uncertain'] else 'seed_sensitive',
        mean_delta=sum(deltas)/len(deltas),note='Conditional on fixed Reviewer seed0 and frozen Never-WM continuation; repeated candidates, not universal utility or a significance test.')
