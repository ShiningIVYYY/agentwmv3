#!/usr/bin/env python3
"""Publish bounded real diagnostics after original and corrective audits pass."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.initialization import observed_room,VERSION
from evaluation.metrics import costs
from experience.splits import file_hash
from analysis.reviewer_metrics import feedback_kind

def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def check(v,reason):
    if not v:raise ValueError(reason)


def audit_fixed(setup):
    root=ROOT/'outputs/phase55_fixed_initializer_original_v1';job=load(setup/'fixed_initializer_job.json')
    check(job.get('returncode')==0,'Corrective model run incomplete')
    episodes=load(root/'episodes.json');check(len(episodes)==5,'Wrong corrective cohort')
    provenance=load(root/'diagnostic_manifest.json')
    check(provenance['initializer_policy']==VERSION,'Wrong initialization policy')
    for name,digest in provenance['initializer_files'].items():check(file_hash(ROOT/name)==digest,'Frozen initializer changed')
    forecast=[];requests_total=0
    for e in episodes:
        ep=root/e['episode_path'];ts=lines(ep/'steps.jsonl');calls=load(ep/'actor_calls.json');http=load(ep/'wm_requests_public.json');private=lines(ep/'wm_http_private.jsonl')
        check(costs(ts,calls,http,e['costs']['episode_total_latency_seconds'])==e['costs'],'Corrective costs mismatch')
        check(ts[-1]['score']==e['metrics']['task_score']==ts[-1]['native_receipt']['response']['score'],'Corrective score extraction mismatch')
        check(len(private)==len(http)==e['costs']['wm_calls'],'Corrective HTTP mismatch');requests_total+=len(private)
        system=load(ep/('wm_context_'+e['task_id'])/'context.json')
        for i,t in enumerate(ts):
            check(t['gate_decision']['call_wm'],'Corrective Always skipped WM')
            observed=[q for q in private if q['decision_id']==t['gate_decision']['decision_id']]
            past=[turn for prior in ts[:i] for turn in ({'role':'user','content':prior['action']},{'role':'assistant','content':prior['next_observation']})]
            for request in observed:
                check(request['request']['messages'][0]['content']==system,'Initializer varies within episode')
                check(request['request']['messages'][1:-1]==past,'Future/hypothetical feedback in corrective history')
                check(t['gate_decision']['decided_at_ns']<request['started_at_ns']<request['finished_at_ns']<t['native_receipt']['started_at_ns'],'Corrective noncausal request')
            p=t['reflection']['predictions'][-1]
            forecast.append(dict(task=e['task_id'],step=i+1,exact=p['observation'].strip()==t['next_observation'].strip()))
    access=(setup/'fixed_wm.txt').read_text().count('"POST /v1/chat/completions HTTP/1.1" 200 OK')
    metrics=(setup/'fixed_wm_metrics.txt').read_text();counter=sum(float(x) for x in re.findall(r'^vllm:request_success_total\{[^}]*\} ([0-9.e+]+)',metrics,re.M))
    check(requests_total==access==counter,'Corrective server counter mismatch')
    return dict(audit_passed=True,summary=load(root/'summary.json'),episodes=episodes,server_requests=requests_total,
        server_counter=counter,forecast_rows=forecast,all_initializer_files_verified=True)


def examples_from_original():
    root=ROOT/'outputs/phase5_episodes_v1';result={}
    for e in load(root/'episodes.json'):
        ep=root/e['episode_path'];ts=lines(ep/'steps.jsonl');private=lines(ep/'wm_http_private.jsonl') if (ep/'wm_http_private.jsonl').exists() else []
        before=0
        for i,t in enumerate(ts):
            ref=t.get('reflection');delta=t['score']-before;before=t['score']
            if not ref:continue
            for rev in ref['reviews']:
                d=rev['decision'];labels=[]
                if d['decision']=='keep':labels.append('keep_original_draft' if rev['candidate']==ref['initial_action'] else 'keep_revised_candidate')
                elif rev['error_kind']=='invalid_format':labels.append('semantic_replace_same_action_violation')
                elif rev['error_kind']=='invalid_action':labels.append('replacement_rejected_by_grammar')
                elif d['decision']=='replace':labels.append('accepted_replacement_api_enumerated' if d['action'] in t['admissible_actions'] else 'accepted_replacement_validity_unknown')
                if d['decision']=='replace' and d['action']==t['action'] and delta<0:labels.append('executed_replacement_task_score_decrease')
                final=ref['predictions'][-1]
                if t['action']!=ref['initial_action'] and delta<0 and final['observation'].strip()==t['next_observation'].strip():labels.append('correct_text_forecast_task_harmful_revision')
                wm=[q for q in private if q['decision_id']==t['gate_decision']['decision_id']]
                for label in labels:
                    if label in result:continue
                    result[label]=dict(source=str((ep/'steps.jsonl').relative_to(ROOT)),task_id=t['task_id'],policy=t['policy'],step=t['step_index'],goal=t['goal'],observation=t['observation'],draft_action=t['gate_query']['draft_action'],
                        wm_inputs=[dict(system_sha256=hashlib.sha256(q['request']['messages'][0]['content'].encode()).hexdigest(),last_action=q['request']['messages'][-1]['content'],real_past_turns=len(q['request']['messages'])-2,temperature=q['request']['temperature'],max_tokens=q['request']['max_tokens']) for q in wm],
                        wm_predictions=ref['predictions'],reviewer_input=rev['messages'],reviewer_output=rev['output'],reviewer_decision=rev['decision'],
                        proposed_action_executed=d['action']==t['action'],final_action=t['action'],feedback_for_final_action=t['next_observation'],score=t['score'],score_delta=delta,
                        outcome_kind=feedback_kind(t['next_observation'],delta),following_observed_scores=[dict(step=k['step_index'],action=k['action'],score=k['score']) for k in ts[i+1:i+4]],
                        unexecuted_proposal_feedback_not_invented=True)
    return result


def initialization_drift():
    old=load(ROOT/'outputs/phase5_episodes_v1/episodes.json');new=load(ROOT/'outputs/phase55_original_v1/episodes.json');rows=[]
    for e in new:
        o=next(x for x in old if x['policy']=='always' and x['task_id']==e['task_id'])
        a=lines(ROOT/'outputs/phase5_episodes_v1'/o['episode_path']/'wm_http_private.jsonl');b=lines(ROOT/'outputs/phase55_original_v1'/e['episode_path']/'wm_http_private.jsonl')
        check(len(a)==len(b),'Original request count changed')
        da=[];bodies=True;outputs=True
        for x,y in zip(a,b):
            p=x['request'];q=y['request'];bodies &= p['messages'][1:]==q['messages'][1:] and {k:v for k,v in p.items() if k!='messages'}=={k:v for k,v in q.items() if k!='messages'}
            outputs &= x['response']['choices'][0]['message']['content']==y['response']['choices'][0]['message']['content']
            da.append(y['usage']['prompt_tokens']-x['usage']['prompt_tokens'])
        sx=a[0]['request']['messages'][0]['content'];sy=b[0]['request']['messages'][0]['content']
        def mislabeled(s):return [dict(label=name,actual=observed_room(look)) for name,look in re.findall(r'(?m)^== Room: (.+) ==\n([^\n]+)',s) if observed_room(look)!=name]
        rows.append(dict(task=e['task_id'],system_same=sx==sy,old_system_sha256=hashlib.sha256(sx.encode()).hexdigest(),new_system_sha256=hashlib.sha256(sy.encode()).hexdigest(),
            only_system_changed=bodies,model_responses_same=outputs,per_request_prompt_delta_values=sorted(set(da)),sum_prompt_token_delta=sum(da),
            old_mislabeled_sections=mislabeled(sx),new_mislabeled_sections=mislabeled(sy)))
    return rows


def main():
    setup=ROOT/'outputs/phase55_setup';out=ROOT/'docs/phase55';out.mkdir(parents=True,exist_ok=True)
    x=load(setup/'final_analysis.json');check(x['engineering_audit_passed'],'Initial audit failed');fixed=audit_fixed(setup)
    for name,digest in x['frozen_inputs_verified'].items():check(file_hash(ROOT/name)==digest,'Frozen protected input changed')
    save(out/'results.json',{k:v for k,v in x.items() if k!='original_source_audit'})
    save(out/'original_source_audit.json',x['original_source_audit']);save(out/'fixed_initializer_audit.json',fixed)
    save(out/'reviewer_examples.json',examples_from_original());drift=initialization_drift();save(out/'initialization_drift.json',drift)
    idle_seconds=max(0,(setup/'cleanup.json').stat().st_mtime-load(setup/'branches_job.json')['finished_at_ns']/1e9)
    save(out/'service_lifecycle.json',dict(post_experiment_resident_service_seconds_approx=idle_seconds,
        post_experiment_resident_service_hours_approx=idle_seconds/3600,
        end_evidence='cleanup.json original filesystem mtime',start_evidence='successful branches_job finished_at_ns',
        cause='network outage prevented client cleanup after detached experiments; fixed corrective supervisor uses finally cleanup',
        not_inference_latency=True,not_converted_to_monetary_cost=True))
    for name in ('plan.json','cohort_jobs.json','branches_job.json','test_results.json','cleanup.json','checkpoint_probe.json','old_failure_branch_probes.json','initializer_failure_probe.json','fixed_initializer_job.json','fixed_cleanup.json'):
        shutil.copyfile(setup/name,out/name)
    shutil.copyfile(setup/'validated_initializers/summary.json',out/'validated_initializer_summary.json')
    frozen_dir=out/'validated_initializers';frozen_dir.mkdir(exist_ok=True)
    for p in (setup/'validated_initializers').glob('sciworld_*.json'):
        if '.attempt' not in p.name:shutil.copyfile(p,frozen_dir/p.name)
    for p in setup.glob('test_*.txt'):shutil.copyfile(p,out/p.name)
    shutil.copyfile(setup/'final_analysis.txt',out/'final_analysis.txt')
    save(out/'wm_metrics_excerpt.json',dict(server_wire=x['server_wire'],original_counter_lines=[s for s in (setup/'wm_metrics.txt').read_text().splitlines() if s.startswith('vllm:request_success_total')],fixed_counter_lines=[s for s in (setup/'fixed_wm_metrics.txt').read_text().splitlines() if s.startswith('vllm:request_success_total')]))
    hashes={}
    for variant in x['variants']:
        root=ROOT/('outputs/phase55_'+variant+'_v1')
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.py'):hashes[str(p.relative_to(ROOT))]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    for folder in ('outputs/phase55_branches_v1','outputs/phase55_fixed_initializer_original_v1','outputs/phase55_setup/validated_initializers','outputs/phase55_setup/validated_initializers_failed_exact_alias_v1'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.py'):hashes[str(p.relative_to(ROOT))]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    save(out/'raw_artifact_manifest.json',hashes)
    table='| Variant | Mean final Task Score | Success | Env actions | WM requests | Model calls | Total tokens | Episode seconds |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for name in x['variants']:
        v=x['variants'][name];c=x['costs'][name];table+=f"| {name} | {v['mean_score']:.1f} | {v['success_rate']:.0%} | {v['environment_steps']} | {c['wm_requests']} | {c['model_calls']} | {c['tokens']['total']['total_tokens']:,} | {c['seconds']:.3f} |\n"
    f=fixed['summary']['policies']['always'];table+=f"| original + validated initializer (separate corrective run) | {f['mean_task_score']:.1f} | {f['success_rate']:.0%} | {f['total_environment_steps']} | {f['total_wm_calls']} | {f['total_online_model_calls']} | {f['total_tokens']:,} | {f['total_latency_seconds']:.3f} |\n"
    branches=[b for pair in x['branch_summary']['pairs'] for b in pair['branches']]
    totals=dict(tokens=sum(c['tokens']['total']['total_tokens'] for c in x['costs'].values())+f['total_tokens']+sum(b['total_tokens'] for b in branches),
        model_calls=sum(c['model_calls'] for c in x['costs'].values())+f['total_online_model_calls']+sum(b['actor_calls']+b['wm_requests'] for b in branches),
        actor_actions=sum(v['environment_steps'] for v in x['variants'].values())+f['total_environment_steps']+sum(b['continuation_actions'] for b in branches),
        wm_requests=x['server_wire']['client_requests']+fixed['server_requests'],
        episode_and_branch_seconds=sum(c['seconds'] for c in x['costs'].values())+f['total_latency_seconds']+sum(b['seconds'] for b in branches),
        offline_validated_initializer_seconds=load(setup/'validated_initializers/summary.json')['seconds'],
        snapshot_state_setup_seconds=sum(p['shared_state_reset_prefix_checkpoint_seconds'] for p in x['branch_summary']['pairs']),
        snapshot_restore_seconds=sum(b['restore_seconds'] for b in branches))
    check(totals['actor_actions']<=600,'Diagnostic action cap exceeded');save(out/'cost_totals.json',totals)
    bt='| Task / intervention step | Skip final score | Call final score | Call−Skip | Skip tokens | Call tokens |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for pair in x['branch_summary']['pairs']:
        a,b=pair['branches'];bt+=f"| {pair['task_id']} / {pair['source_step']} | {a['final_score']} | {b['final_score']} | {pair['score_utility_call_minus_skip']} | {a['total_tokens']:,} | {b['total_tokens']:,} |\n"
    source=x['original_source_audit'];rc=source['reviewer_counts'];examples=load(out/'reviewer_examples.json')
    reviewer=f'''# Reviewer audit — Phase5.5

All278 Phase5 reviews are actual frozen Qwen3-4B generations, not fixtures. Classification axes must not be added together as disjoint categories.

| Per-review protocol category | Count |
| --- | ---: |
| Keep current candidate (may already be revised) | {rc['keep']} |
| Accepted template-level replacement | {rc['replace']} |
| Of accepted replacements: current API enumeration present | {rc['replacement_api_enumerated']} |
| Of accepted replacements: API enumeration absent, native validity Unknown | {rc['replacement_api_unknown']} |
| Replacement rejected by existing grammar validator | {rc['invalid_action']} |
| Semantic contract violation: replace returned same candidate | {rc['invalid_format']} |
| JSON decoding errors | 0 |

39 keep decisions are not necessarily39 keeps of the original Draft. Separate keep-original/keep-revised counters for the rerun are in [results.json](phase55/results.json). “Template accepted”, “API listed”, “environment accepts”, and “advances the task” are different properties. Absence from the incomplete SDK list remains Unknown. No future feedback is used by the diagnostic guard.

Real examples, with Goal/Observation/Draft, WM input hashes/actions, every relevant prediction, exact Reviewer input/output, final action/feedback and following observed scores: [reviewer_examples.json](phase55/reviewer_examples.json). Categories include {', '.join(examples)}. A rejected unexecuted proposal is not assigned invented environment feedback; the displayed feedback is explicitly for the final executed action.

The original Reviewer sees generic ScienceWorld action templates, not a grounded finite enumeration of all valid current object-action combinations. It receives predicted observation text, not native future rewards/scores or an Oracle. It can select an absent object, misuse a descriptive referent, violate a task focus constraint, repeat the same action under replace, or generate unsupported commands. Current state versus predicted state is explicitly distinguished by the prompt, but model compliance is unreliable. These are model/protocol failures; no JSON transport corruption is found.

The default three-round policy can cycle candidates, repeat a failed review, and execute a final replacement even when its final forecast is negative. The last forecast has no final veto. We preserve this original policy rather than introducing answer-specific repairs.

## Original two failures and missing task-utility channel

Fresh shared-heap probes of dev1709 and349 reproduce all six distinct stored candidate feedback texts. The absent jar and descriptive “substance called air” fail resolution. Crucially, `focus on glass cup` and `focus on bee hive` return the correctly predicted **You focus…** text but yield native-100. Successful parsing/interaction is not task success. See [old_failure_branch_probes.json](phase55/old_failure_branch_probes.json) and [Phase5 WM failure audit](WM_FAILURE_AUDIT.md). The old hidden heap was not saved; these are new controlled states with matching observable context, not asserted historical hidden-state identity.

All107 selected final-action forecasts in the old cohort match raw real feedback text; this is selection-biased and often dominated by easy failure strings. New Prediction-Only provides an action-unmodified denominator: strict text {x['variants']['prediction_only']['final_forecast_exact_matches']}/70, existing profile-canonical equivalence {x['variants']['prediction_only']['final_forecast_profile_canonical_matches']}/70. The strict mismatch is object-list order, not an established fact error. Validity-Checked has the same kind of order-only mismatch. Raw forecasts are preserved; normalization is analysis-only using the pre-existing official observation canonicalizer. General WM accuracy and the causal effect of an incorrect prediction on Reviewer decisions remain unverified.

## Independent diagnostic guard

The original Reviewer runs unchanged. Only its final replacement is checked against the already available official current action list. An unlisted replacement is Unknown and reverts to the original Draft, whose safety is also not invented. Raw model outputs/reviews are not rewritten. A previously generated draft forecast is logged as cached when reversion needs its commit comparison; no synthetic model response or duplicate inference is inserted. The guard blocked {sum(g['blocked'] for g in x['guards'])}/65 proposed final replacements, but cannot reject an API-listed goal-damaging focus action. It does not guarantee fewer raw invalid model proposals.

{table}
The guard's small partial-score difference is not proof of benefit, and independently collected WM initialization strings are a confound described in the main report. The variant is not silently enabled for Always, Internal or Full.
'''
    (ROOT/'docs/REVIEWER_AUDIT.md').write_text(reviewer)
    utility=f'''# WM utility diagnostics — Phase5.5

## Matched states, not similar states

Five points were frozen before these experiments: first Full WM invocation in each original development task, independent of reward selection. A fresh SDK world replays the real prefix and verifies observations/reward/score/done. One live-JVM checkpoint then restores **the same** captured heap (world, parser, RNG/object IDs) plus Python reward state and wrapper Actor-step counter for both branches. Original historical hidden-state identity is not claimed. Scope is these new reconstructed shared states.

Snapshot probe captures826,857 fields/34,598 arrays and reproduces look/wait/focus feedback, rewards and clocks. A stochastic greenhouse wait also repeats after restore. Both branches reproduce identical first Draft token IDs, including the original stored Draft. Model weights/configuration and remaining Actor budgets match. Branch A executes the Draft; Branch B invokes the original WM/Reviewer once. All subsequent decisions follow the same frozen **Never WM** Actor policy with common task-turn seeds. No future branch result is visible when choosing the first action; no branch output trains/tunes a Gate.

{bt}
Observed utility: **{x['branch_summary']['positive']} positive, {x['branch_summary']['negative']} negative, {x['branch_summary']['zero']} zero**. In3070 the Call branch focuses on a finger painting and ends at-100, whereas Skip ends at0. In742 verification delays a fatal focus but both branches still end at-100; delaying failure is not positive final-score utility. These are one-seed sample-path effects of one intervention followed by Never, not the effect of Always or an expected utility estimate over all states. No task success or positively useful final-score WM intervention was established in these five pairs.

## Prediction versus useful revision

The model can predict literal feedback accurately in this small distribution. That does not establish its ability to predict task reward or termination. Reviewer receives only the hypothetical observation field, not Oracle score/done labels. A truthful “You focus…” forecast may accompany a catastrophic task failure. Predicted feedback accuracy and Reviewer goal reasoning must be evaluated separately. Additional unexecuted alternatives remain Unknown unless tested in a real isolated branch.

Prediction-Only exactly reproduces Never's full actions/observations/scores in all five tasks. It is a negative control: model serving itself does not mutate the real environment or consume extra Actor steps. It also cannot improve actions when action modification is disabled.

## Initialization engineering defect and corrective run

The released initializer treats every root ObjectTree child as a room; terminals and sewer nodes appear among them. It ignores failed/ambiguous teleports and labels the unchanged current observation with the requested target. A terminal ambiguity poisons later room queries; the isolated failure/restore reproducer confirms this. JVM traversal order makes the wrong room sections and token counts vary. Local Python RandomIsolation does not make the entire JVM/parser/alias pipeline deterministic.

The independent repair uses official admissible teleport commands, verifies actual room identity, restores the same helper heap before every room view, excludes unsupported root nodes and freezes a source-hashed real description. It preserves the WM backbone, original review prompt and rounds. All five tasks have10 verified rooms; two independent collections reproduce room observation facts. SDK object aliases can still vary; both samples are retained and the first is frozen **before** model evaluation, without reward selection. This is replay of actual helper observations, not invented room text or a claim of universal cold-JVM byte equality.

Five corrective Episodes use the same original tasks/seed/settings, with new initializer only. Mean score **{f['mean_task_score']:.1f}**, success **{f['success_rate']:.0%}**, tokens **{f['total_tokens']:,}**. Original results remain unchanged. Exact initializer/provider/model-request/score/cost audit: [fixed_initializer_audit.json](phase55/fixed_initializer_audit.json). The strict five pairs used the original initializer; corrected initialization was tested in full episodes, not new matched pairs, so the old pair utilities are not silently transferred to that new input version. The repair is explicit opt-in; legacy Always/Full definitions are preserved. Separately created helper-world hidden facts are not proven identical to every actual Actor world; this remains an initialization limitation requiring controlled future protocol design.

Full large logs, checkpoints' verification metadata, all real model requests and executed sources remain under `outputs/phase55_*`. Compact hashes: [raw_artifact_manifest.json](phase55/raw_artifact_manifest.json). No final200-task inference, training, threshold calibration or experience expansion occurred.
'''
    (ROOT/'docs/WM_UTILITY_DIAGNOSTICS.md').write_text(utility)
    oldcost='| Phase5 policy | Draft tokens | Review tokens | WM tokens | WM requests | Identical same-decision repeat requests |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for policy in ('never','always','internal','full'):
        es=[e for e in source['original_episodes'] if e['policy']==policy]
        oldcost+=f"| {policy} | {sum(e['cost']['draft_usage']['total_tokens'] for e in es):,} | {sum(e['cost']['reflection_revision_usage']['total_tokens'] for e in es):,} | {sum(e['cost']['wm_usage']['total_tokens'] for e in es):,} | {sum(e['cost']['wm_calls'] for e in es)} | {source['repeated_requests'][policy]['repeated_identical_same_decision_requests']} |\n"
    total_old=sum(v['repeated_identical_same_decision_requests'] for v in source['repeated_requests'].values())
    tests=sum(int(re.search(r'Ran (\d+) tests',p.read_text()).group(1)) for p in setup.glob('test_*.txt') if re.search(r'Ran (\d+) tests',p.read_text()))
    report=f'''# Phase5.5 report — WM verification and Reviewer effectiveness

Complete diagnostic phase; **do not start Phase6**. Engineering correctness does not demonstrate useful verification. The current original WM+Reviewer pipeline has no established positive final-score utility in the five strict pairs and can harm a task.

## A. Results audit and direct answers

1. Never's mean advantage is one100-point difference over five instances:3070 ends0 under Never versus-100 under all three original WM strategies. The revision to `focus on finger painting` is syntactically accepted but task-damaging. Other task scores are the same; all success rates are0. This is not a statistically established ranking over the benchmark.
2. No native score extraction, double-penalty or extra-real-step bug was found. SDK score is rounded100 times server score; reward is its delta and is not summed as Task Score. Negative task score forces SDK termination. `wait` can advance11 simulator ticks; Never2417 ends at simulator move23 after6 Actor actions.20 Actor turns, SDK clock21 and model calls are separate counters. Earlier native_terminal labels mean SDK done, not necessarily simulator task completion. **A real WM initializer defect was found and independently repaired**, with old results retained.
3. The learned WM does predict real feedback correctly in this small set: all107 old selected final predictions match exactly. Raw HTTP replies, pre-action timestamps and committed-only past prefixes bind all315 old requests; no feedback is filled back into predictions. Prediction-Only strict69/70 versus profile-canonical70/70 shows only list-order mismatch here. This is not a general accuracy result or task-utility probability.
4. Reviewer has valid JSON but34 replace-same-action violations and50 unsupported-action proposals. Generic templates provide weak current-object grounding; correct predicted feedback can be ignored, and the final-round policy has no negative-forecast veto. Public task constraints are not reliably followed.
5. Some revisions resolve an immediate invalid action, but positive downstream task value is not established. API validity and successful interaction text are insufficient for goal correctness. The guard cannot prevent a listed but wrong focus action.
6. No positive final-score WM intervention was found in the5 strict pairs:0 positive/1 negative/4 zero. The negative case is3070 (Call−Skip=-100). Do not treat delay, action changes or unchanged score as benefit.
7. There are {total_old} exactly repeated WM payloads at the same decision/prefix (69 Always/16 Internal/15 Full), caused by invalid-review retries and candidate cycles. These are genuine served requests, not duplicate token accounting or duplicate environment execution. Memoization could remove repeated inference without changing depth, but is not silently enabled. Server prefix caching already affects compute, so repeated logical tokens are not a measured FLOP/dollar saving.
8. Never and Original Always reproduce full recorded Actor trajectories/scores. The independent guard yields mean-28 versus original-29, no success; corrected initialization yields mean{f['mean_task_score']:.1f}, success{f['success_rate']:.0%}. Original inputs/results remain versioned. Guard and initializer changes are separate; no favorable result is substituted for a baseline.
9. The framework runs and literal-feedback prediction is usable for diagnosis, but the original verification policy does **not** meet a stable positive-utility prerequisite. Actor task/action errors, missing utility information, weak review compliance and initialization flaws all matter; Reviewer is not presumed the sole cause.
10. **Hold Phase6 large sweeps.** First validate a generic goal-grounded Reviewer/verification criterion and common frozen initializer on independent development evidence, then agree on shared configuration and budgets. No learned Gate/probe training is proposed.

## B. Fixed cohort, variants and fairness

Same five approved development IDs/variations, seed0, H20/SDK21, last2 Draft history, prompt4096/response512, T0.4/top_p1/top_k0, frozen nonthinking Qwen3-4B BF16 and independent frozen Word2World Qwen2.5-7B. WM forecasts remain depth1, up to3 review rounds/final forecast. No train/dev/test reassignment,200-task rollout or experience/threshold change. The original source implementation is unchanged.

{table}
All tasks within each variant receive equal instance weight; one seed has no across-seed sample SD. Later trajectories are not assumed equal. Prediction-Only has one forecast/no Reviewer/no action change; Validity-Checked adds an explicit post-review API guard. Neither is relabelled as original Always.

**Initialization fairness correction:** fixed model/configuration does not guarantee byte-identical initialized WM input. Old versus rerun initialization differs for3/5 tasks; Draft/Reviewer tokens and Actor trajectories match, but WM system strings account for the token drift (old original900,153 versus rerun914,019). Original and guard therefore are not a fully isolated causal guard comparison. [initialization_drift.json](phase55/initialization_drift.json) records exact hashes, wrong labels and per-request deltas. The independent repair freezes verified actual descriptions; promotion to all future baselines requires an explicit new common protocol. Matching observable initial states does not by itself prove historical hidden heaps identical.

## C. Costs and redundancy

{oldcost}
Never has one Draft call per real action. Original Always uses65 Drafts/168 reviews/190 forecasts (423 model calls); Full uses73 Drafts/55 reviews/62 forecasts (190 model calls). Reviews fit at most4096 tokens; their legacy history can include up to19 past steps, whereas Draft history is2. WM retains full real history under32768 context. Initial world text and long review instructions repeat across actual requests. Changing review history or shortening depth would be an algorithm change, not an unnoticed optimization.

All billed/logical input and output tokens come from actual generation/service usage and are summed once. No raw prediction list is re-added to HTTP cost. The diagnostic guard's cached draft comparison incurs no extra model call. Episode latency includes real online generation, prediction and context preparation; component durations overlap and must not be summed twice. Native/helper setup, JVM checkpoints and offline initializer collection are separate. A network outage left the original WM service resident for approximately **{idle_seconds/3600:.2f}h after experiments completed** until reconnection; this resource reservation is explicitly separate from inference latency and is not claimed free. [service_lifecycle.json](phase55/service_lifecycle.json) gives timing evidence. The corrective supervisor now cleans its own service groups in finally even if the SSH client disconnects. No reliable electricity/dollar conversion is available.

Total new Phase5.5 model work (four diagnostics, ten branch continuations, five corrective episodes): **{totals['tokens']:,} tokens / {totals['model_calls']} model calls / {totals['wm_requests']} WM requests / {totals['actor_actions']} Actor actions**, {totals['episode_and_branch_seconds']:.3f}s summed episode/branch wall time. Validated room collection separately used0 model calls and{totals['offline_validated_initializer_seconds']:.3f}s CPU/SDK wall time. JVM checkpoint/reset/restore and model/service startup remain separately recorded; the failed first collection was not fully timed and is not claimed free. [cost_totals.json](phase55/cost_totals.json) retains the measured components.

## D. External experience kept frozen

Same3-card/54-TRAIN-case snapshot and thresholds throughout. Full's three conflict signals change the same-state Internal decision on two Full2417 steps8/9 (the other conflict also has high NLL). This pure Gate shadow comparison is not an environment counterfactual. Limited coverage and shared finite budget mean the trajectories can differ while final scores and invocation totals match. Sufficiency skip remains disabled and has no natural positive; these data cannot refute external complementarity or redundancy avoidance generally.

## E. Files, tests and reproduction

- `diagnostics/reviewer.py`: independent Prediction-Only and final API guard, raw review preservation.
- `diagnostics/initialization.py`: validated room collector/frozen WM-only input provider; original upstream code unmodified.
- `scripts/run_reviewer_diagnostics.py`: reuse original Science.run/Gate dispatcher, named variants and source archives.
- `scripts/serve_scienceworld_diagnostics.py`, `diagnose_wm_branches.py`: same native scoring plus clock/snapshot receipts; matched branch diagnostic only.
- `analysis/reviewer_metrics.py`, `audit_phase55_sources.py`, `analyze_phase55.py`, `report_phase55.py`: outcome/causal/cost/response-binding audit and compact reports.
- `prepare_validated_initializers.py`, `probe_initializer_failure.py`, `test_reviewer_diagnostics.py`: actual collector/ambiguity reproducer, independent real sampling and synthetic unit contracts.
- `docs/phase55/*`, reports, Implementation Plan/README/progress: compact real evidence and resumable status.

{tests} unique CPU unit/regression cases passed in the recorded four suites; synthetic fixtures are tests only. Real20-episode diagnostic,10 strict branch episodes and5 corrective episodes completed. Snapshot restoration, same first Drafts/budgets, complete Actor/cost/score and owned-server counts are verified. Actual commands are saved in [cohort_jobs.json](phase55/cohort_jobs.json), [branches_job.json](phase55/branches_job.json), [fixed_initializer_job.json](phase55/fixed_initializer_job.json). Core CPU commands:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-spanuq/bin/python scripts/test_reviewer_diagnostics.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-spanuq/bin/python scripts/audit_phase55_sources.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-spanuq/bin/python scripts/analyze_phase55.py
PYTHONPATH=. .venv-spanuq/bin/python scripts/report_phase55.py
```

Large raw requests/private runtime contexts and executed source remain on lab under ignored `outputs/phase55_*`; hashes/sizes are committed. Five compact **development-only** frozen helper descriptions are intentionally included under docs/phase55/validated_initializers for exact WM-input replay; these are real source data, not Experience Cases or final-test answers. Copy them to outputs/phase55_setup/validated_initializers when replaying the corrected diagnostic. Actual server dependency/model paths in job records are provenance, not a verified fresh-server installation guide. No model weights/large caches/credentials are committed. Failed exact-full-description preflight is retained; valid SDK alias differences are explicitly recorded, not hidden by discarding weights/data or fabricating room text.

## F. Scientific limits and next design

Five tasks/one seed and first-invocation state selection do not establish universal harms or absence of beneficial verification. Literal-text prediction misses reward/termination meaning; the critic/Actor can execute legal but goal-wrong actions. Independent helper worlds are not proven equal to actual historical Actor hidden states. The strict pairs support only their shared current heap and fixed one-intervention continuation policy. The new initializer removes verified wrong labels/parser contamination but does not repair model reasoning by construction.

Next work should separately test generic current-state/goal grounding and Unknown-safe review, freeze one shared initialization protocol, and consider exact-request caching with actual call accounting. Do not implement task-specific focus exceptions, tune on final test outcomes, enlarge Memory or retrain a Router here. Keep original/corrected versions for honest comparisons.

## G. Stop and Git

Phase5.5 complete; Phase6 NOT authorized. Baseline f6e864690fea7866a1112a250832619eba2e25a6. Final commit/push is reported at handoff. User must confirm the next verification design before any large evaluation. Both owned service groups are cleaned up; unrelated jobs and agentwmv2 remain untouched.
'''
    (ROOT/'docs/PHASE5_5_REPORT.md').write_text(report)
    print(json.dumps(dict(report_written=True,tests=tests,fixed_mean=f['mean_task_score'],fixed_tokens=f['total_tokens'],branch_positive=x['branch_summary']['positive']),indent=2))


if __name__=='__main__':main()
