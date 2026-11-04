#!/usr/bin/env python3
"""Replay completed real episodes without model calls or outcome-based tuning."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from analysis.phase5_audit import internal_diagnostic,external_diagnostic,wm_utility
from evaluation.metrics import costs,episode_metrics
from evaluation.seeds import request_seed
from experience.memory import ExperienceMemory
from experience.schema import ExternalQuery,validate_case
from experience.splits import file_hash,validate_manifest,goal_key
from internal_signals.decision import messages
from internal_signals.provider import TokenNLLProvider
from internal_signals.hidden_states import response_alignment
from internal_signals.token_nll import summarize_fields
from gates.core import RuleGate,GateInput,ExternalInput,fingerprint
from gates.evidence import resolve_external
from gates.budget import BudgetLedger


def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def check(value,reason):
    if not value:raise ValueError(reason)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',default='outputs/phase5_episodes_v1')
    p.add_argument('--calibration',default='outputs/phase5_calibration_v1')
    p.add_argument('--output',default='outputs/phase5_setup/final_audit.json')
    a=p.parse_args();root=Path(a.episodes);plan=load('outputs/phase5_setup/development_plan.json')
    config=load('configs/gates/dev_distribution_phase5_v1.json')
    manifest=load('docs/phase3/split_manifest.json');validate_manifest(manifest,require_native=True)
    memory=ExperienceMemory(load('outputs/phase5_setup/memory_v2.json'),manifest)
    old_memory=ExperienceMemory(load('docs/phase3/memory_snapshot.json'),manifest)
    run=load(root/'manifest.json');episodes=load(root/'episodes.json')
    check(load(root/'progress.json')['state']=='finished','Incomplete validation cohort')
    expected={(p,t['task_id'],0) for p in plan['policies'] for t in plan['validation_tasks']}
    check({(e['policy'],e['task_id'],e['rollout_seed']) for e in episodes}==expected and len(episodes)==20,'Wrong fixed cohort')
    check(file_hash('outputs/phase5_setup/memory_v2.json')==run['signature']['memory_sha256'],'Memory changed')
    for name,digest in run['signature']['source_hashes'].items():
        check(file_hash(root/'executed_source'/name)==digest,'Executed source archive differs: '+name)
        check(file_hash(ROOT/name)==digest,'Online source changed after run: '+name)
    check(file_hash('configs/gates/dev_distribution_phase5_v1.json')==run['signature']['threshold_sha256'],'Threshold changed')
    check(not ({t['goal_group'] for t in plan['calibration_tasks']} & {t['goal_group'] for t in plan['validation_tasks']}),'Calibration/validation group overlap')
    train_cases=memory.snapshot['cases'] if hasattr(memory,'snapshot') else load('outputs/phase5_setup/memory_v2.json')['cases']
    for case in train_cases:validate_case(case,manifest)
    tokenizer=AutoTokenizer.from_pretrained('checkpoints/qwen3-4b',local_files_only=True)
    provider=TokenNLLProvider(tokenizer,load('configs/internal_signals/scienceworld_token_nll_v1.json'))
    counts={'steps':0,'calls':0,'skips':0,'wm_http_requests':0,'scored_drafts':0,'unknown_experience':0,'budget_blocked':0,'missing_internal':0,'prompt_left_truncations':0,'review_prompt_left_truncations':0,'goal_not_fully_visible':0,'draft_format_failures':0,'review_format_or_action_errors':0}
    structured_tokens={policy:0 for policy in plan['policies']}
    firsts={};summaries=[];examples={}
    for e in episodes:
        ep=root/e['episode_path'];traces=lines(ep/'steps.jsonl');actor=load(ep/'actor_calls.json');http=load(ep/'wm_requests_public.json')
        private=lines(ep/'wm_http_private.jsonl') if (ep/'wm_http_private.jsonl').exists() else []
        check(len(private)==len(http),'Private/public actual WM request count mismatch')
        c=dict(config,policy=e['policy'],wm_invocation_budget_per_episode=plan['budget'][e['policy']],wm_forecast_budget_per_episode=4*plan['budget'][e['policy']])
        gate=RuleGate(c);budget=BudgetLedger(c['wm_invocation_budget_per_episode'],c['wm_forecast_budget_per_episode'])
        check(e['configuration_signature']==fingerprint(run['signature']),'Episode configuration signature differs')
        firsts.setdefault(e['task_id'],[]).append((e['initial_observation_sha256'],e['first_draft_response_ids']))
        for i,t in enumerate(traces,1):
            counts['steps']+=1;d=t['gate_decision'];q=ExternalQuery(**t['gate_query']);receipt=t['native_receipt'];record=t['generation_record'];execution=t['gate_execution']
            check(memory.built_at<q.available_at_ns,'Shared memory not frozen before validation')
            check(q.task_id==e['task_id'] and goal_key(q.goal)==e['goal_group'],'Query task/goal identity differs')
            check(q.step_index==i and t['step_index']==i,'Step mismatch')
            check(q.observation==t['observation'] and q.goal==t['goal'],'Gate consumed another state')
            check(q.available_at_ns<=d['decided_at_ns']<execution['started_at_ns']<=receipt['started_at_ns']<receipt['available_at_ns'],'Noncausal dispatch')
            check(config['calibrated_at_ns']<q.available_at_ns,'Threshold not frozen before validation')
            check(execution['actual_committed_at_ns']>=receipt['available_at_ns'],'Commit precedes real feedback')
            check(receipt['request']['action']==t['action'] and receipt['response']['observation']==t['next_observation'],'Feedback/action mismatch')
            check(t['messages']==messages(t['goal'],t['observation'],traces[:i-1],min(2,i-1)),'Prompt/history includes wrong steps')
            check(t['history_length']==min(2,i-1),'History length differs')
            full_ids=tokenizer(record['prompt_text'],add_special_tokens=False)['input_ids'];removed=record.get('left_truncated_prompt_tokens',0)
            check(record['prompt_ids']==full_ids[removed:] and len(record['prompt_ids'])<=4096,'Prompt suffix not exact')
            check(t['seed_metadata']['request_seed']==request_seed(e['task_id'],i-1,e['rollout_seed']),'Draft seed mismatch')
            actual_seed=t.get('actor_metadata',{}).get('generation_seed',record.get('request_seed'))
            check(actual_seed==t['seed_metadata']['request_seed'],'Model generation used another seed')
            check(record['dtype']=='bfloat16' and record['model_revision']==config['internal_signature']['model_revision'],'Actor revision/dtype mismatch')
            text,offsets=response_alignment(tokenizer,record['prompt_text'],record['prompt_ids'],record['response_ids'],left_truncated_tokens=removed)
            fields=summarize_fields(text,offsets,[1.0]*len(offsets))
            structured_tokens[e['policy']]+=sum(f['token_count'] for f in fields['field_spans'])
            counts['draft_format_failures']+=not t['decision_format_valid']
            check(t['signal_computation']['internal']==gate.needs_internal,'Unneeded extraction or missing requested signal')
            if gate.needs_internal:
                recomputed=asdict(provider.extract(record));saved=t['internal_signals'];counts['scored_drafts']+=1
                for key in ('local_uncertainty','global_uncertainty','premise_uncertainty','consequence_uncertainty','missing_span','status','errors'):
                    check(recomputed[key]==saved[key],'NLL replay mismatch: '+key)
                counts['missing_internal']+=saved['missing_span']
            elif 'internal_signals' in t:check(t['internal_signals'] is None,'Unused policy extracted signals')
            raw=memory.evaluate(q) if gate.needs_external else None
            ext=resolve_external(raw,q,memory) if gate.needs_external else ExternalInput()
            if raw:
                counts['unknown_experience']+=raw['coverage']=='unknown'
                for key in ('coverage','conflict','sufficiency','supporting_evidence_ids','matched_experience_ids'):
                    check(raw[key]==t['external_signals'][key],'External replay mismatch: '+key)
            context=GateInput(**dict(t['gate_input'],external=ExternalInput(**t['gate_input']['external'])))
            check(fingerprint(asdict(context.external))==fingerprint(asdict(ext)),'Evidence resolution differs')
            check(context.remaining_wm_budget==budget.invocations and context.remaining_forecast_budget==budget.forecasts,'Budget before mismatch')
            decision=gate.decide(context)
            for key,value in decision.items():
                if key!='gate_latency':check(d[key]==value,'Gate replay mismatch: '+key)
            check(d['input_sha256']==fingerprint(asdict(context)),'Gate input fingerprint differs')
            observed=[x for x in http if x['decision_id']==d['decision_id']]
            check(len(observed)==execution['wm_calls'],'Actual WM HTTP mismatch')
            for request in observed:
                check(d['decided_at_ns']<request['started_at_ns']<=request['finished_at_ns']<receipt['started_at_ns'],'WM request precedes Gate/follows step')
            if d['call_wm']:
                counts['calls']+=1;counts['wm_http_requests']+=len(observed)
                check(execution['branch']=='call' and observed,'No real WM invocation')
                budget.reserve(d['decision_id'],4);budget.settle(d['decision_id'],len(observed))
                check(len(t['reflection']['reviews'])==execution['review_calls'],'Review count mismatch')
                for j,review in enumerate(t['reflection']['reviews']):
                    generation=review['generation'];expected_seed=request_seed(e['task_id'],i-1,e['rollout_seed'],role='review',subcall=j)
                    check(generation['seed_metadata']['request_seed']==expected_seed,'Review request seed differs')
                    check(generation['generation_record']['request_seed']==expected_seed,'Review model seed differs')
                    rr=generation['generation_record'];cut=rr.get('left_truncated_prompt_tokens',0)
                    ids=tokenizer(rr['prompt_text'],add_special_tokens=False)['input_ids']
                    check(ids[cut:]==rr['prompt_ids'] and len(rr['prompt_ids'])<=4096,'Review prompt suffix/limit differs')
                    counts['review_prompt_left_truncations']+=cut>0
                    counts['review_format_or_action_errors']+=review['error_kind'] is not None
            else:
                counts['skips']+=1
                check(not observed and execution['branch']=='skip' and t['action']==q.draft_action,'Skip called WM/changed draft')
            counts['budget_blocked']+=d['verification_unavailable'];counts['prompt_left_truncations']+=removed>0
            counts['goal_not_fully_visible']+=not t['goal_fully_visible_in_model_input']
            check(execution['budget_after']==budget.snapshot(),'Budget after mismatch')
            kind=e['policy']+('/call' if d['call_wm'] else '/skip')
            if d['verification_unavailable']:kind=e['policy']+'/budget_blocked'
            if d['experience_conflict']=='true':kind=e['policy']+'/experience_conflict'
            if kind not in examples:
                examples[kind]={k:t.get(k) for k in ('task_id','step_index','policy','goal','observation','draft_output','internal_signals','external_signals','gate_decision','gate_execution','action','next_observation','score')}
        metric=episode_metrics(dict(load(ep/'result.json'),steps=traces),traces[-1]['native_receipt']['response'])
        check(metric==e['metrics'],'Native metrics differ')
        check(metric['task_score']==float(traces[-1]['native_receipt']['response']['score']),'Final raw score differs')
        recomputed=costs(traces,actor,http,e['costs']['episode_total_latency_seconds'])
        check(recomputed==e['costs'],'Cost replay differs')
        check(len([c for c in actor if c['role']=='draft'])==len(traces),'Draft request count differs')
        check(len([c for c in actor if c['role']=='review'])==sum(t['gate_execution']['review_calls'] for t in traces),'Review request count differs')
        summaries.append({k:e[k] for k in ('task_id','policy','metrics','costs')})
    parity={k:all(v==values[0] for v in values) for k,values in firsts.items()}
    check(all(parity.values()),'Same-task initial observation/first draft differs between policies')
    service_log=Path('outputs/phase5_setup/wm_service_resume.txt').read_text()
    server_lines=[line for line in service_log.splitlines() if '"POST /v1/chat/completions HTTP/1.1" 200 OK' in line]
    saved_metrics=Path('outputs/phase5_setup/wm_final_metrics.txt')
    if saved_metrics.exists():
        # The first audit captured the real owned service counter before shutdown.
        # Subsequent offline replay must not require keeping a GPU service alive.
        metrics=saved_metrics.read_text()
    else:
        with urllib.request.urlopen('http://127.0.0.1:18212/metrics',timeout=15) as response:metrics=response.read().decode()
    successes=sum(float(v) for v in re.findall(r'^vllm:request_success_total\{[^}]*\} ([0-9.e+]+)',metrics,re.M))
    check(len(server_lines)==counts['wm_http_requests']==successes,'Independent WM service/observer request counts differ')
    Path('outputs/phase5_setup/wm_http_access_lines.txt').write_text('\n'.join(server_lines)+'\n')
    Path('outputs/phase5_setup/wm_final_metrics.txt').write_text(metrics)
    wire={'fresh_owned_service':True,'server_successful_http_requests':len(server_lines),'server_prometheus_successes':successes,
        'client_observed_requests':counts['wm_http_requests'],'all_counts_match':True,
        'service_job':load('outputs/phase5_setup/wm_service_resume.json'),
        'access_lines_sha256':file_hash('outputs/phase5_setup/wm_http_access_lines.txt'),
        'captured_service_metrics_sha256':file_hash('outputs/phase5_setup/wm_final_metrics.txt')}
    Path('outputs/phase5_setup/server_wire_audit.json').write_text(json.dumps(wire,indent=2)+'\n')
    result={'engineering_audit_passed':True,'counts':counts,'initial_state_and_first_draft_parity':parity,
        'premise_consequence_output_tokens_by_policy':structured_tokens,
        'structured_token_definition':'value-intersecting tokens only; included in actual total draft usage; no action-only counterfactual savings claimed',
        'source_archive_verified':True,'memory_train_only':True,'threshold_frozen_before_validation':True,
        'complete_fixed_cohort':True,'internal_audit':internal_diagnostic(a.calibration,config),
        'external_audit':external_diagnostic(a.calibration,a.episodes,memory,old_memory,config),
        'wm_utility_audit':wm_utility(a.episodes),'episodes':summaries,'representative_traces':examples,
        'causal_wm_utility_not_established':True,'formal_test_rollouts':0}
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'counts':counts},indent=2))


if __name__=='__main__':main()
