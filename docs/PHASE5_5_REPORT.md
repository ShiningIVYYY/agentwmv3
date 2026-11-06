# Phase5.5 report — WM verification and Reviewer effectiveness

Complete diagnostic phase; **do not start Phase6**. Engineering correctness does not demonstrate useful verification. The current original WM+Reviewer pipeline has no established positive final-score utility in the five strict pairs and can harm a task.

## A. Results audit and direct answers

1. Never's mean advantage is one100-point difference over five instances:3070 ends0 under Never versus-100 under all three original WM strategies. The revision to `focus on finger painting` is syntactically accepted but task-damaging. Other task scores are the same; all success rates are0. This is not a statistically established ranking over the benchmark.
2. No native score extraction, double-penalty or extra-real-step bug was found. SDK score is rounded100 times server score; reward is its delta and is not summed as Task Score. Negative task score forces SDK termination. `wait` can advance11 simulator ticks; Never2417 ends at simulator move23 after6 Actor actions.20 Actor turns, SDK clock21 and model calls are separate counters. Earlier native_terminal labels mean SDK done, not necessarily simulator task completion. **A real WM initializer defect was found and independently repaired**, with old results retained.
3. The learned WM does predict real feedback correctly in this small set: all107 old selected final predictions match exactly. Raw HTTP replies, pre-action timestamps and committed-only past prefixes bind all315 old requests; no feedback is filled back into predictions. Prediction-Only strict69/70 versus profile-canonical70/70 shows only list-order mismatch here. This is not a general accuracy result or task-utility probability.
4. Reviewer has valid JSON but34 replace-same-action violations and50 unsupported-action proposals. Generic templates provide weak current-object grounding; correct predicted feedback can be ignored, and the final-round policy has no negative-forecast veto. Public task constraints are not reliably followed.
5. Some revisions resolve an immediate invalid action, but positive downstream task value is not established. API validity and successful interaction text are insufficient for goal correctness. The guard cannot prevent a listed but wrong focus action.
6. No positive final-score WM intervention was found in the5 strict pairs:0 positive/1 negative/4 zero. The negative case is3070 (Call−Skip=-100). Do not treat delay, action changes or unchanged score as benefit.
7. There are 100 exactly repeated WM payloads at the same decision/prefix (69 Always/16 Internal/15 Full), caused by invalid-review retries and candidate cycles. These are genuine served requests, not duplicate token accounting or duplicate environment execution. Memoization could remove repeated inference without changing depth, but is not silently enabled. Server prefix caching already affects compute, so repeated logical tokens are not a measured FLOP/dollar saving.
8. Never and Original Always reproduce full recorded Actor trajectories/scores. The independent guard yields mean-28 versus original-29, no success; corrected initialization yields mean-29.0, success0%. Original inputs/results remain versioned. Guard and initializer changes are separate; no favorable result is substituted for a baseline.
9. The framework runs and literal-feedback prediction is usable for diagnosis, but the original verification policy does **not** meet a stable positive-utility prerequisite. Actor task/action errors, missing utility information, weak review compliance and initialization flaws all matter; Reviewer is not presumed the sole cause.
10. **Hold Phase6 large sweeps.** First validate a generic goal-grounded Reviewer/verification criterion and common frozen initializer on independent development evidence, then agree on shared configuration and budgets. No learned Gate/probe training is proposed.

## B. Fixed cohort, variants and fairness

Same five approved development IDs/variations, seed0, H20/SDK21, last2 Draft history, prompt4096/response512, T0.4/top_p1/top_k0, frozen nonthinking Qwen3-4B BF16 and independent frozen Word2World Qwen2.5-7B. WM forecasts remain depth1, up to3 review rounds/final forecast. No train/dev/test reassignment,200-task rollout or experience/threshold change. The original source implementation is unchanged.

| Variant | Mean final Task Score | Success | Env actions | WM requests | Model calls | Total tokens | Episode seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| never | -9.0 | 0% | 70 | 0 | 70 | 27,242 | 95.407 |
| original | -29.0 | 0% | 65 | 190 | 423 | 914,019 | 542.843 |
| prediction_only | -9.0 | 0% | 70 | 70 | 140 | 237,009 | 149.952 |
| validity_checked | -28.0 | 0% | 65 | 164 | 381 | 972,306 | 569.777 |
| original + validated initializer (separate corrective run) | -29.0 | 0% | 65 | 190 | 423 | 804,134 | 522.052 |

All tasks within each variant receive equal instance weight; one seed has no across-seed sample SD. Later trajectories are not assumed equal. Prediction-Only has one forecast/no Reviewer/no action change; Validity-Checked adds an explicit post-review API guard. Neither is relabelled as original Always.

**Initialization fairness correction:** fixed model/configuration does not guarantee byte-identical initialized WM input. Old versus rerun initialization differs for3/5 tasks; Draft/Reviewer tokens and Actor trajectories match, but WM system strings account for the token drift (old original900,153 versus rerun914,019). Original and guard therefore are not a fully isolated causal guard comparison. [initialization_drift.json](phase55/initialization_drift.json) records exact hashes, wrong labels and per-request deltas. The independent repair freezes verified actual descriptions; promotion to all future baselines requires an explicit new common protocol. Matching observable initial states does not by itself prove historical hidden heaps identical.

## C. Costs and redundancy

| Phase5 policy | Draft tokens | Review tokens | WM tokens | WM requests | Identical same-decision repeat requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| never | 27,242 | 0 | 0 | 0 | 0 |
| always | 27,811 | 266,958 | 605,384 | 190 | 69 |
| internal | 36,241 | 103,746 | 211,703 | 63 | 16 |
| full | 34,596 | 102,539 | 212,207 | 62 | 15 |

Never has one Draft call per real action. Original Always uses65 Drafts/168 reviews/190 forecasts (423 model calls); Full uses73 Drafts/55 reviews/62 forecasts (190 model calls). Reviews fit at most4096 tokens; their legacy history can include up to19 past steps, whereas Draft history is2. WM retains full real history under32768 context. Initial world text and long review instructions repeat across actual requests. Changing review history or shortening depth would be an algorithm change, not an unnoticed optimization.

All billed/logical input and output tokens come from actual generation/service usage and are summed once. No raw prediction list is re-added to HTTP cost. The diagnostic guard's cached draft comparison incurs no extra model call. Episode latency includes real online generation, prediction and context preparation; component durations overlap and must not be summed twice. Native/helper setup, JVM checkpoints and offline initializer collection are separate. A network outage left the original WM service resident for approximately **51.32h after experiments completed** until reconnection; this resource reservation is explicitly separate from inference latency and is not claimed free. [service_lifecycle.json](phase55/service_lifecycle.json) gives timing evidence. The corrective supervisor now cleans its own service groups in finally even if the SSH client disconnects. No reliable electricity/dollar conversion is available.

Total new Phase5.5 model work (four diagnostics, ten branch continuations, five corrective episodes): **3,073,665 tokens / 1590 model calls / 628 WM requests / 462 Actor actions**, 2100.222s summed episode/branch wall time. Validated room collection separately used0 model calls and56.831s CPU/SDK wall time. JVM checkpoint/reset/restore and model/service startup remain separately recorded; the failed first collection was not fully timed and is not claimed free. [cost_totals.json](phase55/cost_totals.json) retains the measured components.

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

164 unique CPU unit/regression cases passed in the recorded four suites; synthetic fixtures are tests only. Real20-episode diagnostic,10 strict branch episodes and5 corrective episodes completed. Snapshot restoration, same first Drafts/budgets, complete Actor/cost/score and owned-server counts are verified. Actual commands are saved in [cohort_jobs.json](phase55/cohort_jobs.json), [branches_job.json](phase55/branches_job.json), [fixed_initializer_job.json](phase55/fixed_initializer_job.json). Core CPU commands:

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
