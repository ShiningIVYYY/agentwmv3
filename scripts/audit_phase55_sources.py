#!/usr/bin/env python3
"""Read ALL Phase5 source trajectories; outcomes never tune any diagnostic rule."""
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gates.core import RuleGate,GateInput,ExternalInput,fingerprint
from experience.splits import file_hash
from worldmodel.contexts import checkpoint_observation

def load(p):return json.loads(Path(p).read_text())

def main():
    root=ROOT/'outputs/phase5_episodes_v1';episodes=load(root/'episodes.json');summaries=[];reviews=Counter();examples={};forecast=[];duplicates={};external=[];raw_bindings=0
    config=load(ROOT/'configs/gates/dev_distribution_phase5_v1.json');shadow=RuleGate(config,policy='internal')
    for e in episodes:
        ep=root/e['episode_path'];rows=[json.loads(s) for s in (ep/'steps.jsonl').read_text().splitlines()]
        previous=0;changes=[]
        for t in rows:
            native=t['native_receipt']['response'];assert float(native['score'])==float(t['score'])
            if native['score']!=previous:changes.append({'step':t['step_index'],'before':previous,'after':native['score'],'action':t['action'],'feedback':t['next_observation']})
            previous=native['score']
            reflection=t.get('reflection',{})
            predictions=reflection.get('predictions',[])
            for j,p in enumerate(predictions):
                last_match=max((i for i,v in enumerate(predictions) if v['action']==t['action']),default=-1)
                executed=(p['action']==t['action'])
                # All alternatives share the same pre-action real prefix. No
                # claim about unexecuted actions without a real state branch.
                row={'task_id':t['task_id'],'policy':t['policy'],'step':t['step_index'],'action':p['action'],
                    'executed':executed,'final_logged_prediction':j==last_match,
                    'exact_match':p['observation'].strip()==t['next_observation'].strip() if executed else None,
                    'prediction':p['observation'],'actual':t['next_observation'] if executed else None}
                forecast.append(row)
            for rev in reflection.get('reviews',[]):
                kind=rev['decision']['decision'];reviews[kind]+=1
                parsed=json.loads(rev['output']);reviews['valid_json']+=1
                if kind=='invalid':reviews[rev['error_kind']]+=1
                if kind=='replace':reviews['replacement_api_enumerated' if rev['decision']['action'] in t['admissible_actions'] else 'replacement_api_unknown']+=1
                category=('same_action_replace_violation' if rev['error_kind']=='invalid_format' else 'unsupported_replacement' if rev['error_kind']=='invalid_action' else kind)
                # Preserve complete original inputs/outputs for representative real cases.
                examples.setdefault(category,dict(task_id=t['task_id'],policy=t['policy'],step=t['step_index'],goal=t['goal'],observation=t['observation'],draft_action=t['gate_query']['draft_action'],
                    candidate=rev['candidate'],wm_predictions=predictions,reviewer=rev,final_action=t['action'],environment_feedback=t['next_observation'],score=t['score']))
            if t['policy']=='full':
                context=GateInput(**dict(t['gate_input'],external=ExternalInput(**t['gate_input']['external'])))
                internal=shadow.decide(context);full=t['gate_decision']
                if internal['call_wm']!=full['call_wm'] or full['experience_conflict']=='true':
                    external.append(dict(task=t['task_id'],step=t['step_index'],full_call=full['call_wm'],same_state_internal_call=internal['call_wm'],
                        requested_differs=internal['verification_requested']!=full['verification_requested'],full_reason=full['reason'],full_budget=full['budget_remaining'],local=full['local_uncertainty'],global_nll=full['global_uncertainty']))
        assert e['metrics']['task_score']==float(rows[-1]['native_receipt']['response']['score'])
        reqs=[json.loads(s) for s in (ep/'wm_http_private.jsonl').read_text().splitlines()] if (ep/'wm_http_private.jsonl').exists() else []
        for i,t in enumerate(rows):
            observed=[q for q in reqs if q['decision_id']==t['gate_decision']['decision_id']]
            ps=[v for v in t.get('reflection',{}).get('predictions',[]) if not v.get('cached',False)]
            assert len(ps)==len(observed)
            history=[turn for prior in rows[:i] for turn in ({'role':'user','content':prior['action']},{'role':'assistant','content':prior['next_observation']})]
            for prediction,req in zip(ps,observed):
                payload=req['request'];raw=req['response']['choices'][0]['message']['content']
                assert raw==prediction['raw_output']
                assert payload['messages'][1:-1]==history
                assert payload['messages'][-1]=={'role':'user','content':prediction['action']}
                assert checkpoint_observation('scienceworld',raw.split(' [SUCCESS]',1)[0],t['goal'])==prediction['observation']
                assert t['gate_decision']['decided_at_ns']<req['started_at_ns']<req['finished_at_ns']<t['native_receipt']['started_at_ns']
                raw_bindings+=1
        seen=set();repeat=[];per_decision=Counter()
        for req in reqs:
            key=(req['decision_id'],fingerprint(req['request']));per_decision[req['decision_id']]+=1
            if key in seen:repeat.append(req)
            seen.add(key)
        duplicates[e['policy']]=duplicates.get(e['policy'],dict(requests=0,repeated_identical_same_decision_requests=0,repeated_usage_tokens=0))
        duplicates[e['policy']]['requests']+=len(reqs);duplicates[e['policy']]['repeated_identical_same_decision_requests']+=len(repeat)
        duplicates[e['policy']]['repeated_usage_tokens']+=sum(x['usage']['total_tokens'] for x in repeat)
        summaries.append(dict(task_id=e['task_id'],policy=e['policy'],score=e['metrics']['task_score'],steps=len(rows),
            termination=e['metrics']['termination_reason'],score_changes=changes,last_action=rows[-1]['action'],last_feedback=rows[-1]['next_observation'],
            budget_invocations=0 if e['policy']=='never' else 20 if e['policy']=='always' else 5,cost=e['costs'],
            forecasts_per_invocation_histogram=dict(Counter(per_decision.values())),
            source_steps_sha256=file_hash(ep/'steps.jsonl')))
    final_forecasts=[p for p in forecast if p['final_logged_prediction']]
    result=dict(schema='phase55-original-source-audit-v1',source_commit='f6e864690fea7866a1112a250832619eba2e25a6',
        original_episodes=summaries,reviewer_counts=dict(reviews),reviewer_examples=examples,
        wm_final_executed_action_forecasts=dict(n=len(final_forecasts),exact_matches=sum(p['exact_match'] for p in final_forecasts),
            denominator='only executed final candidate, selection-biased; unexecuted alternatives Unknown'),
        forecasts=forecast,repeated_requests=duplicates,full_same_state_internal_shadow=external,
        raw_model_response_and_causal_past_binding_verified=raw_bindings,
        no_rules_thresholds_or_memory_changed=True)
    out=ROOT/'outputs/phase55_setup';out.mkdir(parents=True,exist_ok=True);(out/'source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('reviewer_counts','wm_final_executed_action_forecasts','repeated_requests','full_same_state_internal_shadow')},indent=2))

if __name__=='__main__':main()
