"""ALFWorld-only real-feedback rules. No ScienceWorld cards or future query fields."""
import json,re,time
from gates.core import ExternalInput
from experience.memory import bm25


def match_rule(kind,observation,action,commands):
    if action in commands:return None
    if kind=='repeat_open':
        m=re.fullmatch(r'open (.+)',action)
        if not m:return None
        container=m[1]
        if re.search(r'(?i)(?:You open the '+re.escape(container)+r'\.|The '+re.escape(container)+r' is open\b)',observation):return dict(container=container)
    if kind=='closed_take':
        m=re.fullmatch(r'take (.+) from (.+)',action)
        if not m:return None
        container=m[2]
        if re.search(r'(?i)(?:You close the '+re.escape(container)+r'\.|The '+re.escape(container)+r' is closed\b)',observation):return dict(container=container,object=m[1])
    return None


def synthesize(cases,source_tasks):
    allowed={r['task_id'] for r in source_tasks};cards=[]
    for c in cases:
        if c['task_id'] not in allowed or c['source_split']!='train' or c['wm_called']:raise ValueError('Illegal ALFWorld evidence source')
        if not 0<c['decision_at_ns']<c['outcome_available_at_ns']:raise ValueError('Noncausal evidence')
    for kind in ['repeat_open','closed_take']:
        related=[c for c in cases if c.get('probe_kind')==kind]
        supports=[c for c in related if c['environment_feedback'].strip()=='Nothing happens.']
        counter=[c for c in related if c not in supports]
        independent={c['episode_id'] for c in supports}
        if len(independent)<2 or len({c['task_id'] for c in supports})<2 or counter:continue
        cards.append(dict(card_id='alf-'+kind,kind=kind,applicability_conditions='Draft repeats opening an observably open container' if kind=='repeat_open' else 'Draft takes an object through an observably closed container',
            observed_rule='This exact precondition failure returned Nothing happens in independent training episodes.',
            failure_pattern='Nonadmissible command with explicit visible container-state contradiction',
            suggested_correction='Do not repeat the open command' if kind=='repeat_open' else 'Open the visible closed container before attempting retrieval',
            supporting_cases=[c['case_id'] for c in supports],counterexamples=[c['case_id'] for c in counter],
            reliability='supported',evidence_type='observed_training_feedback',source_task_ids=sorted({c['task_id'] for c in supports}),
            available_at_ns=max(c['outcome_available_at_ns'] for c in supports),wm_utility='unknown'))
    return dict(schema='alfworld-experience-snapshot-v1',cases=cases,cards=cards,built_at_ns=time.time_ns(),
        method='deterministic_independent_episode_support; no LLM synthesis',offline_model_calls=0,offline_tokens=0,cross_domain_memory=False)


class AlfExperienceMemory:
    def __init__(self,snapshot,manifest):
        if snapshot['schema']!='alfworld-experience-snapshot-v1' or snapshot['cross_domain_memory']:raise ValueError('Wrong memory/domain')
        self.snapshot=snapshot;self.cards=snapshot['cards'];self.cases=snapshot['cases'];self.built=snapshot['built_at_ns']
        allowed={r['task_id'] for r in manifest['memory_tasks']}
        if any(c['task_id'] not in allowed for c in self.cases):raise ValueError('Experience split leakage')
        lookup={r['task_id']:r for r in manifest['memory_tasks']}
        if any(c['goal_group']!=lookup[c['task_id']]['goal_group'] for c in self.cases):raise ValueError('Experience goal identity changed')
        if self.cards!=synthesize(self.cases,manifest['memory_tasks'])['cards']:raise ValueError('Card differs from actual evidence synthesis')
        if any(c['outcome_available_at_ns']>self.built for c in self.cases):raise ValueError('Snapshot predates its evidence')
        ids={c['case_id']:c for c in self.cases}
        for card in self.cards:
            if card['counterexamples'] or card['reliability']!='supported':raise ValueError('Unsupported card')
            support=[ids[i] for i in card['supporting_cases']]
            if len({c['episode_id'] for c in support})<2 or len({c['task_id'] for c in support})<2:raise ValueError('Repeated steps/tasks are not independent evidence')
    def evaluate(self,query):
        if set(query)!={'task_id','episode_id','goal','observation','draft_action','commands','available_at_ns'}:raise ValueError('Unexpected current/future query fields')
        now=query['available_at_ns']
        if now<self.built:raise ValueError('Snapshot not yet available')
        start=time.perf_counter();text=' '.join([query['goal'],query['observation'],query['draft_action']])
        eligible=[c for c in self.cards if c['available_at_ns']<=now and query['task_id'] not in c['source_task_ids']]
        ranked=bm25(eligible,text,5);retrieval=time.perf_counter()-start;judge=time.perf_counter()
        applicable=[c for c,score in ranked if match_rule(c['kind'],query['observation'],query['draft_action'],query['commands'])]
        evidence=sorted({i for c in applicable for i in c['supporting_cases']})
        reliable=bool(applicable and evidence)
        signal=dict(coverage='true' if reliable else 'unknown',conflict='true' if reliable else 'unknown',sufficiency='false' if reliable else 'unknown',
            reliability='supported' if reliable else 'unknown',matched_experience_ids=[c['card_id'] for c in applicable],supporting_evidence_ids=evidence,
            reason='Observed container-state conflict supported by independent TRAIN feedback' if reliable else 'No reliable applicable ALFWorld evidence; Unknown abstention',
            retrieval_latency=retrieval,judge_latency=time.perf_counter()-judge,additional_tokens=0,judge_model_calls=0,retrieval_calls=1,
            query_at_ns=now,wm_utility='unknown',action_modified=False,source_scope='alfworld_training_only_frozen',
            retrieved_cards=[dict(experience_id=c['card_id'],score=score,applicable=c in applicable) for c,score in ranked])
        resolved=ExternalInput(coverage=signal['coverage'],conflict=signal['conflict'],sufficiency=signal['sufficiency'],reliability=signal['reliability'],
            evidence_ids=tuple(evidence),matched_experience_ids=tuple(signal['matched_experience_ids']),evidence_verified=reliable,
            action_status='unresolved_conflict' if reliable else 'unknown')
        return signal,resolved
