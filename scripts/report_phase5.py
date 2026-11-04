#!/usr/bin/env python3
"""Publish compact, source-bound Phase5 development diagnostics after audit passes."""
import json
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.splits import file_hash

def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    out=ROOT/'docs/phase5';out.mkdir(parents=True,exist_ok=True)
    setup=ROOT/'outputs/phase5_setup';run=ROOT/'outputs/phase5_episodes_v1'
    audit=load(setup/'final_audit.json');assert audit['engineering_audit_passed']
    summary=load(run/'summary.json');episodes=load(run/'episodes.json');distribution=load(setup/'calibration_distribution.json')
    for filename in ('development_plan.json','calibration_distribution.json','paper_profile_audit.json','memory_coverage.json','wm_failure_audit.json','horizon_probe.json','episodes_job.json','wm_service_resume.json','environment_service_resume.json','regression_resume.json','server_wire_audit.json','gpu_observation.json','cleanup.json','reviewer_quality.json','wm_http_access_lines.txt'):
        shutil.copyfile(setup/filename,out/filename)
    # Preserve the unmodified full Prometheus response privately; publish only
    # the counter evidence needed for request accounting, with its raw-file hash.
    obsolete=out/'wm_final_metrics.txt'
    if obsolete.exists():obsolete.unlink()
    save(out/'wm_service_metrics_excerpt.json',{
        'raw_file':'outputs/phase5_setup/wm_final_metrics.txt',
        'raw_sha256':file_hash(setup/'wm_final_metrics.txt'),
        'success_counter_lines':[s for s in (setup/'wm_final_metrics.txt').read_text().splitlines() if s.startswith('vllm:request_success_total')]})
    shutil.copyfile(setup/'memory_v2.json',out/'memory_snapshot_v2.json')
    for p in setup.glob('test_*_resume.txt'):shutil.copyfile(p,out/p.name)
    shutil.copyfile(setup/'profile_tests_resume.txt',out/'profile_tests_resume.txt')
    shutil.copyfile(setup/'final_audit.txt',out/'final_audit.txt')
    save(out/'results.json',{'summary':summary,'episodes':episodes})
    save(out/'acceptance.json',{k:v for k,v in audit.items() if k not in ('representative_traces','episodes')})
    save(out/'representative_traces.json',audit['representative_traces'])
    provenance={}
    for folder in ('outputs/phase5_calibration_v1','outputs/phase5_episodes_v1','outputs/phase5_canary_v1','outputs/phase5_canary_v2'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and (p.name in ('steps.jsonl','native_receipts.jsonl','manifest.json','actor_calls.json','wm_requests_public.json','result.json','failure.json','summary.json') or 'executed_source' in p.parts):
                provenance[str(p.relative_to(ROOT))]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    save(out/'raw_artifact_manifest.json',provenance)
    policies=summary['policies'];count=audit['counts'];external=audit['external_audit'];utility=audit['wm_utility_audit']
    table='| Policy | Mean final Task Score | Success (score>=100) | Env steps | Gate calls | WM forecasts | Total model calls | Total tokens | Online seconds |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for policy in ('never','always','internal','full'):
        rows=[e for e in episodes if e['policy']==policy];s=policies[policy]
        table+=f"| {policy} | {s['mean_task_score']:.2f} | {s['success_rate']:.0%} | {s['total_environment_steps']} | {sum(e['costs']['gate_call_steps'] for e in rows)} | {s['total_wm_calls']} | {s['total_online_model_calls']} | {s['total_tokens']} | {s['total_latency_seconds']:.3f} |\n"
    tasktable='| Development task | Never | Always | Internal | Full |\n| --- | ---: | ---: | ---: | ---: |\n'
    for task in load(setup/'development_plan.json')['validation_tasks']:
        scores=[next(e['metrics']['task_score'] for e in episodes if e['task_id']==task['task_id'] and e['policy']==p) for p in ('never','always','internal','full')]
        tasktable+='| '+task['task_id']+' | '+' | '.join(str(s) for s in scores)+' |\n'
    costkeys=('draft_calls','reflection_revision_calls','wm_calls','internal_signal_seconds','retrieval_calls','retrieval_seconds','external_judge_seconds','gate_rule_seconds','signals_evidence_gate_seconds','wm_prediction_http_seconds','wm_context_setup_seconds','agent_generation_seconds','reflection_revision_generation_seconds','episode_total_latency_seconds')
    costtable='| Cost component (all5 tasks) | Never | Always | Internal | Full |\n| --- | ---: | ---: | ---: | ---: |\n'
    for key in costkeys:
        values=[sum(e['costs'][key] for e in episodes if e['policy']==p) for p in ('never','always','internal','full')]
        costtable+='| '+key+' | '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in values)+' |\n'
    for role in ('draft','reflection_revision','wm','total'):
        for kind in ('prompt_tokens','completion_tokens','total_tokens'):
            values=[sum(e['costs'][role+'_usage'][kind] for e in episodes if e['policy']==p) for p in ('never','always','internal','full')]
            costtable+='| '+role+'_'+kind+' | '+' | '.join(str(v) for v in values)+' |\n'
    all_tokens=sum(e['costs']['total_usage']['total_tokens'] for e in episodes);all_calls=sum(e['costs']['total_model_calls'] for e in episodes)
    all_seconds=sum(e['costs']['episode_total_latency_seconds'] for e in episodes)
    peak=max(e['costs']['actor_peak_allocated_bytes'] for e in episodes)/2**30
    cal=load(ROOT/'outputs/phase5_calibration_v1/episodes.json')
    cal_tokens=sum(e['costs']['total_usage']['total_tokens'] for e in cal)
    totals={'online_tokens':all_tokens,'online_model_calls':all_calls,'online_episode_seconds':all_seconds,'actor_peak_allocated_gib':peak,'calibration_tokens':cal_tokens}
    save(out/'cost_totals.json',totals)
    small=f'''# Phase 5 small-scale ScienceWorld results

This is fixed **development functional validation**, not the final200-task benchmark. Five goal-disjoint held-out development tasks, rollout seed0, four Gates, complete episodes up to20 interactions. No across-seed sample SD can be estimated from one seed. Means weight five task instances equally, including early failures; neither batches nor task categories are reweighted.

{table}
{tasktable}
Native raw negative scores are retained. Success is the auxiliary final score>=100 diagnostic. Native termination and horizon truncation are recorded separately per episode in [results.json](phase5/results.json). Initial state and first Draft tokens match across all four policies for each task. Later state sequences are not assumed equal.

Budgets were frozen before outcomes: Never0, Always20, Internal5, Full5 **invocations per episode**, with4-forecast reservation per invocation and actual forecast settlement. These deliberately different functional budgets do not constitute a matched-cost efficacy comparison. Correction and confirmed-action skip are off for all Gates; unused signal modules do not run.

## Online inference costs

{costtable}
The full20-episode diagnostic used **{all_tokens:,} tokens / {all_calls} model calls / {all_seconds:.3f}s** summed episode wall time. Model load/hash, WM service startup, environment validation, historical training-data collection and distribution calibration are separate. Whole runner wall time: {summary['whole_run_seconds']:.3f}s; Actor load/hash: {summary['actor_load_and_hash_seconds']:.3f}s. Actor peak torch allocated memory: {peak:.3f}GiB; the dedicated vLLM WM service separately reserves its KV cache and model memory. See server GPU observations/provenance; torch allocation is not nvidia-smi total VRAM.

Agent generation time already includes review generation; reflection generation time is a subset and must not be added twice. Internal extraction includes raw-logit reduction/field scoring with zero extra backbone forward passes. Signals/evidence/Gate time overlaps retrieval and Gate rule time; component times are descriptive, not additive wall-time terms. Total tokens include complete structured Drafts, labels and Action, all reviewer prompts/replies and every WM forecast. Additional premise/consequence value tokens by policy: `{json.dumps(audit['premise_consequence_output_tokens_by_policy'])}`; these are included, not added twice. No action-only paired generation was run, so a net incremental reasoning-token saving/cost is not claimed.

Offline old rule synthesis:0 LLM calls/0 tokens,0.005727s. Phase5 train-only third-card extension:0 new cases/0 new model calls/0 tokens,0.022964s. Historical Phase3 collection:48 Actor calls /1,726 output tokens including EOS /59.252s generation /0.1812s extraction; six controlled training probes add0 model calls and0.7163s native step-request time. Historical input-token totals, failed-first-run cost and complete startup overhead were not fully profiled; they are unmeasured, not zero. See PHASE3_REPORT.md. Development distribution calibration:273 actual Actor calls, {cal_tokens:,} tokens,426.011280s whole run, no WM. Canary/probes are setup diagnostics, excluded from these20-episode metrics. The WM remained resident on its dedicated GPU during this sequential cohort, including Never; no idle reservation is converted into a fictitious per-request inference charge. No reliable monetary tariff or electrical-energy measurement is available.

## Causal and scientific diagnostics

- Replay checks {count['steps']} real decisions: {count['calls']} Call / {count['skips']} Skip, {count['wm_http_requests']} observed WM HTTP forecasts. Native receipts, query timestamps, evidence resolution, exact Gate outputs, budgets, score/cost recomputation and source archives agree. Independent server wire counts are in [server_wire_audit.json](phase5/server_wire_audit.json).
- Internal: Pearson0.893067 / Spearman0.898846 on273 development decisions. After threshold freeze, explicit native failure feedback appears in47/68 above-Local-threshold decisions and113/205 at/below threshold. Other feedback is **not** a correctness label. These dependent in-sample observations are preliminary, not an error-probability calibration or a test-set detector AUROC.
- External: {len(external['old_snapshot_low_nll_conflict_shadow_cases'])} low-Local-NLL development decisions conflict with the already-existing closed-route evidence in retrospective replay. This shadow analysis did not drive those Actor-only trajectories. Live Full: coverage true {external['live_full_coverage_true']}/{external['live_full_steps']}, conflict true {external['live_full_conflict_true']}, sufficiency true {external['live_full_sufficiency_true']}. New-snapshot retrospective natural sufficiency cases: {len(external['new_snapshot_natural_sufficiency_retro_cases'])}. Do not backdate the new card into calibration-time online claims.
- WM: {utility['changed_actions']}/{utility['calls']} invocations changed the action; {utility['revised_actions_with_real_failure_feedback']} changed actions received explicit real failure feedback. Final forecasts unavailable/incomplete: {utility['unavailable_or_incomplete_final_predictions']}. An action change or failed revised action alone does not establish causal benefit/harm: its unexecuted alternative outcome is unknown. No same-state Call/Skip counterfactual rollout was performed.
- Reviewer: all278 responses are syntactically valid JSON.34 `invalid_format` labels mean a semantic protocol violation (replace chose the same candidate), not JSON decoding failure;50 `invalid_action` labels reject unsupported replacement commands. Full examples/reasons: [reviewer_quality.json](phase5/reviewer_quality.json). These errors and weak grounding remain visible; no ad hoc critic repair is used.

Representative true decisions: [representative_traces.json](phase5/representative_traces.json). Full model/native trajectories and executed source remain on lab under `outputs/phase5_episodes_v1`; immutable hashes/sizes: [raw_artifact_manifest.json](phase5/raw_artifact_manifest.json). Compact acceptance/scientific diagnostics: [acceptance.json](phase5/acceptance.json). No synthetic fixture is included in these results.
'''
    (ROOT/'docs/SMALL_SCALE_EVAL.md').write_text(small)
    report=f'''# Phase 5 report — end-to-end ScienceWorld integration

Phase5 development engineering validation completed; Phase6 is not started. **Engineering correctness, model capability and scientific effectiveness are distinct.** The existing system executes real complete episodes with real learned-WM Call/Skip. Reflection remains weak, experience coverage is narrow, and these data do not establish either research hypothesis.

## A. Completed work and configuration

Actor: frozen `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`. BF16 HF SDPA, torch2.8.0+cu128 / Transformers4.57.3, nonthinking, T0.4/top_p1/top_k0/repetition_penalty1, max input4096/response512, history2. Actor GPU0 and WM GPU1 on RTX4090; old environments remain unchanged.

Independent WM: `X1AOX1A/WorldModel-Sciworld-Qwen2.5-7B`, revision `6196b3531c5634a8229af6b2cd111e02a70c13b5`, existing weights read-only. vLLM0.11.0 BF16, greedy T0/top_p1/max512, context32768, max-seqs1, eager. Existing helper source commit `e8dc240269fb222de3c242b09c70bfbf0b4ac1c9`; existing JSON reflection up to3 rounds, final replacement forecast, max4 one-step forecasts of the same real prefix per invocation. Simulation alternatives do not increment real interaction steps. WM-only initialization includes existing all-room context; it never enters Actor Draft/Internal/External/Gate input. Review receives predicted observations, not the private world initialization.

Internal backend `hf_generation_raw_bf16_v1` and base Actor configuration SHA256 `980d1f3f3f07e0df1061a842e80d42926e573e1d1810a2663c13fbefe001504b` are preserved. The frozen BF16 model supplies raw pre-temperature generation logits; NLL reduction uses FP32: Local=max(mean Premise-value NLL, mean Expected-Consequence-value NLL); Global=mean all textual response tokens incl field labels/Action, excl prompt/EOS/padding. No extra forward. Opt-in profile adds verified official per-turn RNG and exact suffix input truncation; the profile hash binds new thresholds. Fifteen old real Phase2 records replay identically. Legacy behavior stays default-off.

Pipeline: observation → structured Draft → requested Internal/External signals → rule Gate → either learned WM prediction/unchanged reflection or genuine Skip → exact final Action → real SDK feedback → commit. Reflection/WM tokenizers are independent of Actor tokenizer; plain-text transport is compatible. Score/accounting and source-bound real replay are verified.

## B. Files and purpose

| Files | Purpose |
| --- | --- |
| evaluation/profile.py, seeds.py, observer.py, metrics.py | Explicit paper numeric/presentation profile, official seed derivation, actual WM request observation, raw instance metrics and full costs |
| configs/evaluation/paper_aligned_v1/scienceworld.json | Independent opt-in setting; legacy configs retained |
| configs/gates/dev_distribution_phase5_v1.json | Frozen dev-only quantiles; separate invocation/forecast budgets |
| scripts/serve_scienceworld_profile.py | Owned H20/SDK21 thin environment service; reuse installed wrapper, do not edit v2 |
| scripts/prepare_phase5_plan.py, calibrate_phase5.py | Goal-disjoint frozen development cohort; distribution-only calibration |
| scripts/run_scienceworld_validation.py | Bounded runner using existing Science.run, source archiving, role RNG, native receipts and per-episode resume |
| scripts/audit_wm_failures.py, audit_scienceworld_profile.py, audit_experience_coverage.py | Original failures, pinned200-task metadata and train evidence audit |
| scripts/prepare_phase5_memory.py; experience/memory.py, synthesis.py; gates/evidence.py | Third narrow same-room observation card from existing approved TRAIN cases; preserve old card behavior and reliability requirements |
| internal_signals/hf_actor.py, hidden_states.py, provider.py; gates/generation.py | Opt-in seeded generation/exact left truncation with verified response alignment; unchanged raw NLL formula and old config |
| scripts/run_audited_paper_eval.py | Small profile hooks for fixed last2 history/native menu IDs, preserving original loop and other profiles |
| scripts/test_scienceworld_validation.py, audit_phase5.py; analysis/phase5_audit.py |37 CPU integration contracts and source/causal/metric/cost real replay; read-only scientific diagnostics |
| scripts/report_phase5.py, docs/phase5/*, WM_FAILURE_AUDIT.md, SMALL_SCALE_EVAL.md | Compact true results, reproducible reports and provenance; full large logs remain ignored |
| README, IMPLEMENTATION_PLAN.md, PAPER_ALIGNED_EVAL.md, PROGRESS.md, PHASE5_WORKLOG.md | Current state, intentional paper differences, risks and next-phase boundary |

## C. WM failure investigation / actual fix

See [WM_FAILURE_AUDIT.md](WM_FAILURE_AUDIT.md). Phase4 `focus on glass jar`: jar absent from bathroom; reviewer ignored negative forecast and selected the absent goal object. `focus on substance called air`: descriptive phrase does not resolve as the canonical command; fresh probe `focus on air` works. Both final failures were predicted correctly; valid JSON and last-Action parsing preserved. The final-round execution policy is an existing capability/design limitation, not a transport/parser defect. Seven independently reset real development probes retained; no answer-specific action patch or evaluation-driven Gate change.

Actual deterministic engineering bug: new observer keyword binding initially failed on requests `url=`. It was fixed with regression coverage before the successful two-step canary. Initial failure and executed source preserved. No model inference or environment action from that failed canary is counted as research data. In the complete cohort all278 reviews are valid JSON;34 same-action replace violations and50 unsupported replacements are model/protocol failures, not corrupted transport. See [reviewer_quality.json](phase5/reviewer_quality.json).

## D. Paper profile / data / thresholds / memory

Exact ordered200 test task IDs/template/variation mapping and prior native goal hashes checked against pinned official source/current SDK catalog; no test rollout. SDK1.2.3/JAR hash pinned, H20 validated. Package identity does not prove identical historical authors' binary. Structured Qwen3-4B HF Actor, native Action parser and WM addition are intentional deviations from paper Qwen3.5/action-only/backend. [Protocol](PAPER_ALIGNED_EVAL.md) preserves earlier records and details these differences. No RL loss, training invalid penalty or reward normalization enters evaluation.

Train1292 / dev197 / test200 instances, with conservative normalized-goal and instance isolation, frozen source manifests.30 dev goal groups for calibration;5 **different** dev goal groups for full validation. Chemistry/lifespan each have only one available dev goal group, allocated to calibration; complete-episode validation covers Measure, Conductivity and Find only. Template families can be shared across partitions; goal/instance overlap is excluded, not a claimed template-held-out split.

273 real dev decisions,244 unique prompt-response states,0 missing; repeated steps remain real workload but not independent evidence. Predeclared q0.75 linear quantiles: Local **0.12081397919163489**, Global **0.053253251821409955**, strict>. Expected Internal requests on Actor-only calibration states85/273=31.1355% before budgets/external feedback. Sensitivity and all points in [calibration_distribution.json](phase5/calibration_distribution.json); leave-one-episode-out ranges Local0.119247–0.121801 / Global0.052727–0.055246. No outlier removal, task-outcome selection or test tuning. This is distribution calibration, **not proof of multi-seed stability, useful thresholds or optimal WM budget**.

Memory v2 freezes3 cards, same54 verified TRAIN cases; old2-card snapshot preserved. Only added same-room `look around` fact with2 independent episodes/instances/goal groups and no counterexamples. Train inventory: navigation16/object interaction16/preconditions20/failure29 cases, but broad rules/corrections/subgoal progress lack enough independent supporting evidence. WM utility remains Unknown. Sufficiency is not inferred from BM25, and no reliability criterion was relaxed. Natural sufficient confirmations: live Full {external['live_full_sufficiency_true']}, retrospective new-snapshot dev {len(external['new_snapshot_natural_sufficiency_retro_cases'])}. Correction and confirmation Skip remain off. Frozen snapshot shared by all policies; unused modules cost0 online.

## E. Tests, actual results and costs

{table}
Actual counts: {count['steps']} real decisions; {count['calls']} Call/{count['skips']} Skip; {count['wm_http_requests']} WM forecasts. Budget-blocked {count['budget_blocked']}; missing Internal {count['missing_internal']}; Draft format failures {count['draft_format_failures']}; reviewer parse/action validation errors {count['review_format_or_action_errors']}. Invalid native actions are retained rather than repaired. All5 initial-state/first-Draft parity checks pass. Native termination, horizon, actual server requests, independent evidence, NLL recomputation, role seeds, budget settlement, exact prompt history and costs pass replay.

37 new CPU contracts +217 prior regression tests =254 passed. Unit fixtures are not research results. Real complete20-episode cohort and2-step canary passed; prior7 single-action probes andH20 horizon probe retained. Old Phase4 byte-identity source audit is historical; optional Phase5 RNG/truncation hooks are explicitly versioned and validated by golden-score and real replay rather than misreported as byte-identical source.

All online runs: **{all_tokens:,} actual input/output tokens, {all_calls} model calls, {all_seconds:.3f}s episode wall time**, Actor peak allocated{peak:.3f}GiB plus dedicated WM reservation. Full per-policy costs, service/setup/calibration/offline separation and scientific audits: [SMALL_SCALE_EVAL.md](SMALL_SCALE_EVAL.md). Judge/router learned-model calls0; raw NLL uses no extra forward. No reliable dollar conversion.

Core actual commands (from repo root):

```bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python scripts/test_scienceworld_validation.py
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python -u scripts/run_scienceworld_validation.py --mode episodes --plan outputs/phase5_setup/development_plan.json --model-dir checkpoints/qwen3-4b --wm-checkpoint /data1/shn/agentwmv2/checkpoints/word2world/scienceworld --science-python /home/shn/miniconda3/envs/ee-scienceworld/bin/python --threshold-config configs/gates/dev_distribution_phase5_v1.json --snapshot outputs/phase5_setup/memory_v2.json --output outputs/phase5_episodes_v1
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-spanuq/bin/python scripts/audit_phase5.py
PYTHONPATH=. .venv-spanuq/bin/python scripts/report_phase5.py
```

These are actual server provenance paths, not portable installation defaults. The new profile/SDK service and separate WM must be running. Commands to reconstruct service/model/development inputs are recorded in docs/phase5 and PHASE5_WORKLOG; no paths imply permission to modify v2. Use a **new output Run ID** for a rerun. `--resume` only skips completed episodes with identical configuration/source signatures and permits one explicit infrastructure retry; it never merges partial steps or replaces unfavorable valid outcomes.

## F. Limits / Phase6 recommendation

Engineering preparation for Phase6 is available: fixed IDs/profile, complete episodes, true Call/Skip, strict accounting and frozen train-only memory/dev-only thresholds. This is **not** scientific acceptance of Gate benefit or a reason to launch200x8 automatically. Reviewer can ignore accurate negative forecasts; two NLL features are strongly correlated; many low-NLL actions fail; memory coverage remains narrow and sufficiency lacks a natural positive. Five tasks/one seed/three families cannot rank methods reliably. If changing the reviewer/final-veto/memory, freeze a new shared protocol and repeat appropriate development validation before formal comparisons; do not choose it on final-test results.

Phase6 should first agree on critic-policy stability and shared memory version, predeclare matched-budget grids and correction groups, then obtain explicit user authorization for resources/sweeps. Across8 rollout seeds, aggregate fixed task instances within each seed and report mean±sample SD across seed means. Actor remains Qwen3-4B. Cross-benchmark/fresh-server portability work remains later. No new training, final200x8 evaluation or Phase6 work was performed. **Stop here and await user confirmation.**

## G. Version control and stop

Baseline commit: `7a039fa78688e1caf36451dd2d3e599c215d6dfe`. Final Phase5 commit and verified push are reported in the delivery message. Only agentwmv3 source/docs/compact evidence are committed; model weights, environments, private WM initialization and complete large trajectories remain excluded. The original paper document is preserved verbatim as a prefix plus the explicit Phase5 addendum. Phase5 is complete; no Phase6 work is authorized by this report.
'''
    (ROOT/'docs/PHASE5_REPORT.md').write_text(report)
    print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
