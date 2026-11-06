#!/usr/bin/env python3
"""Actual-score/causal/cost audit for all four diagnostics and five branch pairs."""
from collections import Counter
import json
from pathlib import Path
import re
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.metrics import costs
from evaluation.profile import canonicalize_observation
from gates.core import fingerprint
from experience.splits import file_hash
from internal_signals.decision import messages
from analysis.reviewer_metrics import termination_causes,verified_pair_utility

def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def check(v,reason):
    if not v:raise ValueError(reason)
def domain(rows):return [(t['action'],t['observation'],t['next_observation'],t['score'],t['generation_record']['response_ids']) for t in rows]


def main():
    setup=ROOT/'outputs/phase55_setup';plan=load(setup/'plan.json');expected={t['task_id'] for t in plan['tasks']}
    check(load(setup/'cohort_finished.json')['complete'],'Cohort incomplete')
    source=load(setup/'source_audit.json');all_rows={};variant_results={};cost_tables={};termination=[];guards=[];raw_counts=Counter();total_http=0;artifact_hashes={}
    frozen=load(setup/'never_diagnostic_manifest.json')['frozen_inputs']
    for name,digest in frozen.items():check(file_hash(ROOT/name)==digest,'Frozen inputs changed: '+name)
    for variant in plan['variants']:
        root=ROOT/('outputs/phase55_'+variant+'_v1');episodes=load(root/'episodes.json')
        check({e['task_id'] for e in episodes}==expected and len(episodes)==5,'Wrong cohort')
        provenance=load(root/'diagnostic_manifest.json')
        check(provenance['variant']==variant and provenance['frozen_inputs']==frozen,'Variant/source changed')
        for name,digest in provenance['source_hashes'].items():check(file_hash(root/'executed_diagnostic_source'/name)==digest,'Diagnostic source archive changed')
        for name,digest in load(root/'manifest.json')['signature']['source_hashes'].items():check(file_hash(root/'executed_source'/name)==digest,'Base source archive changed')
        predictions=[];review_counts=Counter();review_history=[];wm_prompts=[];review_prompts=[];steps_total=0
        for e in episodes:
            ep=root/e['episode_path'];ts=lines(ep/'steps.jsonl');all_rows[variant,e['task_id']]=ts
            actor=load(ep/'actor_calls.json');http=load(ep/'wm_requests_public.json');private=lines(ep/'wm_http_private.jsonl') if (ep/'wm_http_private.jsonl').exists() else []
            check(costs(ts,actor,http,e['costs']['episode_total_latency_seconds'])==e['costs'],'Cost arithmetic differs')
            check(e['metrics']['task_score']==ts[-1]['native_receipt']['response']['score'],'Final raw score differs')
            check(len(http)==len(private)==e['costs']['wm_calls'],'HTTP/accounting mismatch');total_http+=len(http)
            previous_score=0
            for i,t in enumerate(ts,1):
                steps_total+=1;receipt=t['native_receipt'];native=receipt['response'];decision=t['gate_decision'];execution=t['gate_execution']
                check(native['actor_interaction_steps']==i,'WM advanced Actor action count')
                check(t['messages']==messages(t['goal'],t['observation'],ts[:i-1],min(2,i-1)),'Draft prompt/history changed')
                check(decision['decided_at_ns']<execution['started_at_ns']<=receipt['started_at_ns']<receipt['available_at_ns'],'Future-feedback leak')
                check(t['signal_computation']['internal'] is False,'Unneeded NLL extraction')
                expected_call=variant!='never';check(decision['call_wm']==expected_call,'Diagnostic policy not followed')
                observed=[q for q in private if q['decision_id']==decision['decision_id']]
                ps=[q for q in t.get('reflection',{}).get('predictions',[]) if not q.get('cached',False)]
                check(len(ps)==len(observed)==execution['wm_calls'],'Forecast dispatch mismatch')
                for pred,req in zip(ps,observed):
                    check(req['response']['choices'][0]['message']['content']==pred['raw_output'],'Not a real model response')
                    check(decision['decided_at_ns']<req['started_at_ns']<req['finished_at_ns']<receipt['started_at_ns'],'Prediction after feedback')
                    past=[turn for prior in ts[:i-1] for turn in ({'role':'user','content':prior['action']},{'role':'assistant','content':prior['next_observation']})]
                    check(req['request']['messages'][1:-1]==past,'Hypothetical/current future inserted into WM history')
                    wm_prompts.append(req['usage']['prompt_tokens'])
                reflection=t.get('reflection',{})
                if variant=='prediction_only':
                    check(len(observed)==1 and not reflection['reviews'] and t['action']==t['gate_query']['draft_action'],'Prediction-only changed actions or reviewed')
                if variant=='never':check(not observed and t['action']==t['gate_query']['draft_action'],'Never made WM call')
                if reflection:
                    final=reflection['predictions'][-1];check(final['action']==t['action'],'Final prediction is for another action')
                    predictions.append(dict(task=e['task_id'],step=i,action=t['action'],prediction=final['observation'],feedback=t['next_observation'],exact=final['observation'].strip()==t['next_observation'].strip(),
                        profile_canonical_match=canonicalize_observation(final['observation']).strip()==canonicalize_observation(t['next_observation']).strip(),
                        score_delta=t['score']-previous_score,negative_native_score=t['score']<0))
                    for rev in reflection['reviews']:
                        raw=json.loads(rev['output']);review_counts[rev['decision']['decision']]+=1
                        if rev['error_kind']:review_counts[rev['error_kind']]+=1
                        if raw['decision']=='keep':review_counts['keep_original_draft' if rev['candidate']==reflection['initial_action'] else 'keep_revised_candidate']+=1
                        check(rev['generation']['generation_record']['response']==rev['output'],'Reviewer output was rewritten')
                        review_history.append(rev['history_length']);review_prompts.append(rev['generation']['usage']['prompt_tokens'])
                    if variant=='validity_checked':
                        proposed=reflection['proposed_final_action'];listed=proposed in t['admissible_actions']
                        check(reflection['official_action_validity']==('api_enumerated' if listed else 'unknown'),'Unknown validity reinterpreted')
                        blocked=proposed!=reflection['initial_action'] and not listed
                        check(reflection['guard_blocked_unknown_replacement']==blocked,'Guard not applied as declared')
                        check(t['action']==(reflection['initial_action'] if blocked else proposed),'Guard changed action beyond declared rule')
                        guards.append(dict(task=e['task_id'],step=i,blocked=blocked,proposed=proposed,executed=t['action'],score=t['score'],feedback=t['next_observation']))
                previous_score=t['score']
            last=ts[-1]['native_receipt']['response'];causes=termination_causes(last)
            check(bool(causes),'Unexplained termination')
            termination.append(dict(variant=variant,task=e['task_id'],steps=len(ts),score=last['score'],simulator_moves=last['simulator_moves'],causes=causes))
            artifact_hashes[str((ep/'steps.jsonl').relative_to(ROOT))]=file_hash(ep/'steps.jsonl')
        total=lambda key:sum(e['costs'][key] for e in episodes)
        usage={role:{key:sum(e['costs'][role+'_usage'][key] for e in episodes) for key in ('prompt_tokens','completion_tokens','total_tokens')} for role in ('draft','reflection_revision','wm','total')}
        cost_tables[variant]=dict(model_calls=total('total_model_calls'),draft_calls=total('draft_calls'),review_calls=total('reflection_revision_calls'),wm_requests=total('wm_calls'),tokens=usage,
            seconds=total('episode_total_latency_seconds'),wm_seconds=total('wm_prediction_http_seconds'),review_seconds=total('reflection_revision_generation_seconds'),
            reviewer_history_counts=dict(Counter(review_history)),review_prompt_range=[min(review_prompts),max(review_prompts)] if review_prompts else [],wm_prompt_range=[min(wm_prompts),max(wm_prompts)] if wm_prompts else [])
        variant_results[variant]=dict(instances=5,mean_score=sum(e['metrics']['task_score'] for e in episodes)/5,success_rate=sum(e['metrics']['success_score_ge100'] for e in episodes)/5,
            environment_steps=steps_total,scores={e['task_id']:e['metrics']['task_score'] for e in episodes},reviewer_counts=dict(review_counts),final_forecasts=predictions,
            final_forecast_exact_matches=sum(p['exact'] for p in predictions),final_forecast_count=len(predictions))
        variant_results[variant]['final_forecast_profile_canonical_matches']=sum(p['profile_canonical_match'] for p in predictions)
    parity={}
    original_episodes=load(ROOT/'outputs/phase5_episodes_v1/episodes.json')
    for variant,policy in (('never','never'),('original','always')):
        for e in original_episodes:
            if e['policy']==policy:
                previous=lines(ROOT/'outputs/phase5_episodes_v1'/e['episode_path']/'steps.jsonl')
                parity[variant+'/'+e['task_id']]=domain(previous)==domain(all_rows[variant,e['task_id']])
    negative_control={task:domain(all_rows['never',task])==domain(all_rows['prediction_only',task]) for task in expected}
    check(all(negative_control.values()),'Prediction-only changed the environment/Actor trajectory')
    branch_root=ROOT/'outputs/phase55_branches_v1';branch_summary=load(branch_root/'summary.json');check(branch_summary['completed_pairs']==5,'Missing branch pairs')
    for pair in branch_summary['pairs']:
        skip=load(branch_root/pair['task_id']/'skip/summary.json');call=load(branch_root/pair['task_id']/'call/summary.json')
        check(verified_pair_utility(skip,call)==pair['score_utility_call_minus_skip'],'Unmatched utility or score arithmetic')
        a=lines(branch_root/pair['task_id']/'skip/steps.jsonl');b=lines(branch_root/pair['task_id']/'call/steps.jsonl')
        check(a[0]['generation_record']['response_ids']==b[0]['generation_record']['response_ids'],'Initial Actor differs')
        check(a[0]['observation']==b[0]['observation'],'Branch initial state differs')
        for branch,ts in (('skip',a),('call',b)):
            info=load(branch_root/pair['task_id']/branch/'summary.json')
            check(info['restore_provenance']['heap_matches'],'Heap restore failed')
            check(info['final_score']==ts[-1]['native']['score'],'Branch score extraction wrong')
            check(sum(t['native']['reward'] for t in ts)==info['final_score']-info['initial_score'],'Reward not telescoping')
            reqpath=branch_root/pair['task_id']/branch/'wm_http_private.jsonl';http=lines(reqpath) if reqpath.exists() else []
            total_http+=len(http)
            check(len(http)==info['wm_requests'],'Branch model cost count differs')
            check(sum(t['treatment_called'] for t in ts)==(branch=='call'),'More than one treatment or wrong branch')
            check(all(q['finished_at_ns']<ts[0]['environment_step_started_at_ns'] for q in http),'WM saw post-action branch feedback')
            check(len(ts)<=info['remaining_actor_budget'],'Remaining budget exceeded')
    service_log=(setup/'wm.txt').read_text();server_count=sum('"POST /v1/chat/completions HTTP/1.1" 200 OK' in s for s in service_log.splitlines())
    captured=setup/'wm_metrics.txt'
    if captured.exists():metrics=captured.read_text()
    else:
        with urllib.request.urlopen('http://127.0.0.1:18212/metrics',timeout=15) as response:metrics=response.read().decode()
        captured.write_text(metrics)
    success=sum(float(x) for x in re.findall(r'^vllm:request_success_total\{[^}]*\} ([0-9.e+]+)',metrics,re.M))
    check(server_count==success==total_http,'Fresh owned service counters differ')
    result=dict(engineering_audit_passed=True,baseline_reproduction=parity,prediction_only_never_full_trajectory_parity=negative_control,
        variants=variant_results,costs=cost_tables,termination=termination,guards=guards,branch_summary=branch_summary,
        original_source_audit=source,server_wire=dict(server_http_requests=server_count,server_success_counter=success,client_requests=total_http,raw_metrics_sha256=file_hash(captured)),
        frozen_inputs_verified=frozen,raw_trace_sha256=artifact_hashes,
        scope='fixed5 development tasks,seed0; fixed5 first-invocation states; no final-test inference or rule selection')
    (setup/'final_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,baseline_parity=parity,negative_control_parity=negative_control,
        variants={v:{k:x for k,x in data.items() if k in ('mean_score','success_rate','environment_steps')} for v,data in variant_results.items()},
        branch_utilities=[(p['task_id'],p['score_utility_call_minus_skip']) for p in branch_summary['pairs']],server_requests=total_http),indent=2))


if __name__=='__main__':main()
