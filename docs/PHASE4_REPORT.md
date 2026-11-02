# Phase 4 report — Training-Free Internal–External Gate

## A. Completed scope and acceptance distinction

Engineering acceptance is met: seven common-interface Gates, source-bound external evidence, deterministic missing/budget rules, development-only provisional quantile tooling, actual dispatch through the old learned-WM reflection flow, cost logs and12 real committed ScienceWorld steps. No new parameter training, Memory Correction, test-dependent selection or full benchmark was performed. Phase2 Internal Signals/config, Phase3 Memory snapshot and PAPER_ALIGNED_EVAL.md are unchanged.

Scientific validity is NOT established. Only two narrow door rules exist, there is no natural positive Sufficiency example, calibration is provisional, and both Full Gate calls ended in unsuccessful revised actions. Changing an action is not proof of helpful WM use; no success/cost advantage or hypothesis confirmation is claimed.

Design contract: GATE_DESIGN.md. Real compact evidence: phase4/. Current configuration: configs/gates/provisional_full_v1.json. Stop before Phase5 pending confirmation.

## B. Code and documentation

| Files | Purpose |
| --- | --- |
| gates/core.py | Typed GateInput/ExternalInput, seven policies, scoring identity, priority/missing/tri-state rules and explanations |
| gates/budget.py | Verification-invocation budget + actual-forecast cap with worst-case reservation, settlement and duplicate protection |
| gates/evidence.py | Bind current query/Card/Case/snapshot/applicability/time; distinguish confirmed/correctable/unresolved/unknown |
| gates/generation.py | Unscored same-model drafts for baselines and reviews; scored drafts reuse Phase2 verbatim |
| gates/runtime.py | Optional pre-WM decision, lazy original controller, Call/Skip dispatch and real-history commit on both branches |
| gates/worldmodel_setup.py | Original author WM initialization in existing SDK Python; local checkpoint chat-template token counting |
| scripts/calibrate_gate.py | Approved-dev NLL quantiles/distribution/correlation only; provisional output and no outcome optimization |
| scripts/diagnose_gate.py | Small approved-development real end-to-end diagnostics, actual usage/cost and pre-outcome traces |
| scripts/audit_gate_inputs.py; audit_gate_diagnostic.py | Recheck54 training cases/config identity, causal input/evidence/budget/dispatch/prompt/source replay |
| scripts/test_gates.py |63 CPU cases; fake models are limited to unit tests, never research outcomes |
| scripts/run_audited_paper_eval.py | Default-off Gate seam, lazy controller and committed real-feedback updates; legacy no-Gate behavior retained |
| worldmodel/runtime.py | Default-off exact context/count adapters; existing forecasting/review/revision unchanged |
| README.md; IMPLEMENTATION_PLAN.md; PROGRESS.md; GATE_DESIGN.md; phase4/ | Status, contracts, measured results, source versions, logs and reproducibility |

No agentwmv2 files/dependencies/services were modified. Existing WM weights were read-only reused, all weight/tokenizer hashes verified against revision6196b3531c5634a8229af6b2cd111e02a70c13b5. Only132KB of pinned existing helper SOURCE was copied into v3's ignored external directory. No model download or package installation occurred.

## C. Answers1–6: rules, semantics, calibration and baselines

1. **Final rule.** Full defaults to BudgetAvailable AND (ReliableConflict OR HighLocal OR HighGlobal OR MissingRequiredInternal OR InvalidDraftFormat). Strict > thresholds. Optional confirmed-action override is implemented but disabled in the actual configuration. Budget exhaustion skips WM while explicitly marking the draft unverified; it never implies the action is correct.
2. **Combination.** Reliable evidence-backed conflict takes priority over internal confidence and any sufficient claim. Unknown/inapplicable/unreliable external evidence falls back to the same internal rule. Local and Global are OR-ed; neither is treated as an independent calibrated error probability.
3. **Missing/Unknown/conflict.** Current external tri-states remain true/false/unknown. Missing local uses valid Global against tau_global. Missing required Global/both scores, truncation/alignment/numeric failure or malformed draft requests verification subject to budget. Backend/dtype/revision/scoring/config mismatch rejects the run, not silently falls back to a different scale. Contradictory conflict=true/sufficiency=true cannot skip. Unverified external assertions cannot trigger confirmation.
4. **When can Sufficiency skip?** Only explicit opt-in plus verified evidence confirming the CURRENT unchanged valid draft, coverage=true/conflict=false/sufficiency=true and action_status=confirmed_action. Correctable conflict is NOT resolved while Memory Correction is off, and still calls. Current door evidence confirms only a bounded immediate effect, not task utility; therefore real configuration disables this override. The branch is unit-tested, not a demonstrated useful real skip.
5. **Threshold status.** Q=.75 linear development quantiles: Local .13648279807529134, Global .06599500845849053. Input25 Phase3 diagnostic steps/5 goal groups, including exposed targeted examples. Pearson .9588707683, Spearman .9484615385. The source is too small/biased for stable calibration. Config/report explicitly say provisional_functional_only; --allow-provisional required. No success/reward/test result selected thresholds or budgets. Budget1 verification/four forecast slots per Episode was preset, not optimized. Local/Global are highly correlated. A larger independently partitioned dev-calibration/dev-validation study is still required.
6. **Uniform baselines.** never, always, local, global, internal, experience, full all implement RuleGate.decide(GateInput). Never/Always omit NLL extraction and retrieval; Experience-only omits NLL; internal-only policies omit retrieval. Same Actor/config/structured draft, WM/protocol and Memory Correction=false. A real raw-logit-on/off same-prompt check returned identical response token IDs. Local/Global reuse the same accepted provider, including its small CPU aggregate bookkeeping; actual cost is recorded. Random/Periodic scheduling interface is designed in GATE_DESIGN.md; active policies/full experiments remain Phase6 work.

The internal identity is unchanged: hf_generation_raw_bf16_v1, frozen Qwen3-4B @1cfa9a7208912126459214e8b04321603b3df60c, BF16 generation/FP32 raw-logit reduction. Local=max(two field-value means). Global includes all textual response labels/newlines/Action but excludes prompt/EOS/padding/special tokens. There is no rescoring forward or mixed-precision threshold reuse.

Source audit confirms all54 Cases belong to the approved train pool;6 controlled probe steps are from train1667/1545, not development/test. Snapshot hash remains c6c2dc4dc6ebecf6ef9d03020b7721748d79afa6379a6e2779a5a127e01c25d7. Actual matched Cards are checked for current-state/action applicability and source IDs, not just BM25 similarity. WM Utility stays Unknown.

## D. Answers7–9: actual branches, examples, tests and cost

### Actual runtime

The Actor ran on GPU3. A temporary owned vLLM0.11.0 Word2World service ran on GPU6, port18212, using existing ScienceWorld Qwen2.5-7B weights, BF16, temperature0/top_p1/response512/context32768. It used the ORIGINAL JSON reflection controller with max3 review rounds and its final-replacement prediction. Simulation remains one step from the same real prefix. No Oracle backend was substituted. Original privileged initialization is available only to WM, not to Memory/Gate/Actor draft.

Full diagnostic: previously exposed development1709/349, max5 committed Actor turns each, budget1 verification/four forecasts per Episode. Actual2 Call steps,8 Skip steps,8 forecast RPCs,6 review Actor calls;3 later requests blocked by exhausted budget. All6 reviews parsed; no structural draft errors or missing spans were observed. Each Call used the original three reviews plus final forecast; there was no duplicate Gate dispatch. Original review protocols may revisit candidate actions; every actual forecast is counted, not hidden behind a single invocation count.

Separate one-turn Never and Always checks on approved dev1709 validated the unscored baseline path. Never:1 draft,0 NLL extraction,0 retrieval,0 WM/reviews,0 WM context setup. Always:1 draft,0 NLL extraction/retrieval,4 predictions and3 reviews. This is not a performance comparison with Full's different horizon. Actual owned-server HTTP log total12 = Full8 + Always4 + Never0. All12 local WM prompt-token counts equal served API usage. Gate decisions/query hashes precede both WM predictions and the environment HTTP step.

Skip retains the draft exactly. After Skip, committed REAL feedback is still added to an existing WM's history without a prediction. A controller created after earlier Skip steps teacher-forces those steps before predicting. These history paths are CPU integration-tested with explicitly labelled mock models, not presented as additional real experiment results.

### Five representative Full traces

Full state/decision/evidence/WM forecasts/reviews/native receipts are in phase4/representative_traces.json and phase4/phase4_full_v1/steps.jsonl. Post-step feedback below is audit evidence, never current Gate input.

| Task/step | Pre-action signals | Gate/actual branch | Final action and TRUE feedback |
| --- | --- | --- | --- |
|1709/1 | Local .029114, Global .014681; currently closed kitchen route; supported Conflict=true, four source Cases | Call despite low NLL;4 real forecasts/3 reviews | WM pipeline revised go kitchen -> focus on glass jar; No known action matches that input. |
|349/1 | Local .303201, Global .111833; external Unknown | Call through high Local/Global internal fallback;4 forecasts/3 reviews | Revised focus on thermometer -> focus on substance called air; No known action matches that input. |
|1709/2 | Local .017606, Global .007515; sparse feedback, experience Unknown | Skip, no active internal trigger;0 forecasts/reviews | Unchanged go kitchen; The door is not open. |
|1709/5 | Local .157944 > threshold; external Unknown; budget0 | Requested verification but Skip due budget;0 forecasts/reviews | Unchanged go kitchen; The door is not open. |
|349/5 | Local .029543, Global .021542; explicit currently closed hallway route; supported Conflict=true; budget0 | Requested verification but Skip due budget;0 forecasts/reviews | Unchanged go hallway; The door is not open. |

Natural positive Sufficiency, Missing Internal Signal, contradictory external evidence and correctable-conflict inputs did not arise live; their rule paths are unit-tested only. No artificial benchmark feedback or deliberately relabelled success was used to create those branches. Both real Full WM-revised actions failed. This is a negative functional observation, not a causal proof that WM harmed the task: no matched same-state Call/Skip outcome comparison was run.

### Measured cost

| Full10-step component | Actual measurement |
| --- | --- |
| Draft Actor |10 calls;3,913 input +338 output =4,251 tokens |
| Review/revision Actor |6 calls;4,278 input +330 output =4,608 tokens;11.8250s generation |
| WM |8 predictions;20,435 input +63 output =20,498 tokens;2.74335s prediction time |
| Total online model work |24 model calls;28,626 input +731 output =29,357 tokens |
| All Actor generation |24.39270s, including draft and review generation |
| Internal |10 extractions,.049314s;0 extra backbone forwards |
| Retrieval |10 calls,.003167s |
| Deterministic external Judge |.000298s;0 Judge model calls/tokens |
| Gate rule alone |.000211s total; not used as the system-cost claim |
| External retrieval/proof/Gate/logging combined |.010481s total |
| Instrumented online inference components |27.19584s = Actor generation + Internal extraction + external/evidence/Gate + WM prediction |
| Lazy original WM initialization |10.57231s, separate from online inference components |
| Entire diagnostic driver |58.23856s, including Actor loading/hash checks, setup, environment/I/O and separate parity test |
| Separate generation-parity validation |1 additional Actor call,391 tokens,1.26853s; excluded from online policy-call count, explicitly retained |
| Recorded Actor per-call GPU peak |7.742GiB allocated; not the entire system/WM peak |

Never one-turn:1 model call/391 tokens,1.85750s instrumented inference,12.70593s whole driver,0 WM setup. Always one-turn:8 online model calls/12,309 tokens,8.67421s instrumented inference,3.71965s WM initialization,23.39878s whole driver. Do not infer relative performance from these different-horizon functional checks. Across the three diagnostics there are33 online model calls plus1 explicitly separate parity call.

Offline reused Memory is not free: Phase3's48 training Actor calls/1,726 generated tokens/59.2515s generation/.1812s NLL extraction;6 controlled true environment steps/.7163s request time; deterministic54-Case->2-Card synthesis5.727ms,0 synthesis-model calls/tokens. It was not rebuilt or charged repeatedly per online step in Phase4. Report this one-time cost/amortization in later comparisons; Never/Always do not need to construct Memory for their policy.

Existing WM weight/tokenizer verification took16.9306s. Owned service startup/cold serving residency is separate from driver-online timings; startup logs are retained but no complete cold-start GPU-energy or monetary measure is claimed. The WM reserved approximately21.46GiB at the recorded service snapshot; this is a residency observation, not a measured allocator peak or attribution to Gate. No reliable monetary tariff is available. Whole-driver wall time and component times have different boundaries and must not be added as if independent. The service was stopped afterward and GPU6 released. Unrelated jobs were untouched.

### Commands and verification

All commands were run in agentwmv3. Portable variable-based equivalents are in GATE_DESIGN.md. Actual core commands:

~~~bash
.venv-spanuq/bin/python scripts/calibrate_gate.py --trace docs/phase3/phase3_dev_v1/steps.jsonl --trace docs/phase3/phase3_targeted_dev_v1/steps.jsonl --manifest docs/phase3/split_manifest.json --actor-config configs/internal_signals/scienceworld_token_nll_v1.json --quantile .75 --allow-provisional --output-config configs/gates/provisional_full_v1.json --output-report outputs/phase4_setup/calibration.json
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python -u scripts/diagnose_gate.py --manifest docs/phase3/split_manifest.json --snapshot docs/phase3/memory_snapshot.json --gate-config configs/gates/provisional_full_v1.json --policy full --task-id sciworld_1709 --task-id sciworld_349 --steps 5 --model-dir checkpoints/qwen3-4b --wm-checkpoint /data1/shn/agentwmv2/checkpoints/word2world/scienceworld --science-python /home/shn/miniconda3/envs/ee-scienceworld/bin/python --allow-provisional --validate-unscored-parity --output outputs/phase4_full_v1
# Same real driver/asset/config arguments for these separate baseline runs:
# --policy never --task-id sciworld_1709 --steps 1 --output outputs/phase4_never_v1
# --policy always --task-id sciworld_1709 --steps 1 --output outputs/phase4_always_v1
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python scripts/test_gates.py
.venv-spanuq/bin/python scripts/audit_gate_inputs.py
.venv-spanuq/bin/python scripts/audit_gate_diagnostic.py --manifest docs/phase3/split_manifest.json --snapshot docs/phase3/memory_snapshot.json --run-dir outputs/phase4_full_v1 --output outputs/phase4_setup/phase4_full_v1_audit.json
~~~

63 Gate tests +154 regression cases =217 passing cases across9 suites. Gate tests cover seven-policy interface, signature/threshold boundaries, valid zero, missing/invalid/Unknown, conflict-sufficiency priority, correction-disabled behavior, optional confirmation, budget caps/reservation/failures/duplicates, future-input rejection, source/query/evidence/applicability binding, real-dev calibration restrictions and Call/Skip history integration. Unit-model outputs are never research data. Input audit plus all three actual-run causal/replay audits pass. Captured server requests, action receipts, Actor prompts, evidence, budgets and decisions agree; no skip made a prediction/review or altered its draft. No full ScienceWorld/ALFWorld/WebShop sweep, threshold search, new training or fresh-server installation was run.

At the user's request, final CPU tests/audits were launched with start_new_session=True on lab, PID3903604, independent of SSH/local screen state. They completed with state=finished/all_passed=true. Logs/progress remain at outputs/phase4_setup/background_progress.json and are compactly committed as phase4/background_checks.json plus per-suite .txt logs. No GPU work is left running for this phase.

## E. Remaining issues and failure disclosure

- Provisional thresholds use only25 exposed diagnostic decisions and strongly correlated NLLs. They are not validated deployment settings, uncertainty calibration, a true WM-utility estimator or evidence for Hypotheses1/2.
- Two door Cards leave most observed states uncovered. Sparse current feedback intentionally abstains rather than reconstructing state from hidden information. No natural confirmation or reliable Memory correction is demonstrated.
- Actual WM revisions did not solve the immediate action failures. Future WM/action/prompt compatibility and useful-verification diagnostics are needed; no backbone/review-depth change was made to hide this result.
- Small targeted previously exposed dev tasks are functional checks, not unbiased task metrics. Template/layout and semantic near-duplicate limitations from Phase3 remain. Test tasks/outcomes never calibrated Gate.
- First WM launch failed because a shared GPU became occupied. The failure and service command are retained; the retry used a free GPU. No other user's process was killed and no dependency was changed.
- After the real runs, input type guards, an explicit signal_validity output, a rule-scope confirmation guard and configurable recorded WM GPU metadata were added. These do not change any observed branch;217 CPU/regression cases and exact archived trace replay pass. Original executed source versions are included for differences. No additional GPU run was performed solely for these validation/metadata extensions.
- Generic baseline/full benchmark CLI profiles are not enabled. Opt-in injection is exposed through the Phase4 diagnostic; production cohort rollout/profile hardening belongs to Phase5. Monetary/GPU-energy pricing and full deployment cold-start profiling remain unverified.

## F. Answer10: Phase5 readiness

Engineering prerequisites for Phase5 functional integration are satisfied: real learned-WM Call/Skip, unchanged scored provider and prompt, trustworthy source/tri-state binding, disabled correction/confirmation override, explicit invocation/forecast budgets, real feedback committed on skips and complete measured usage categories. This does not authorize full benchmark evaluation or demonstrate scientific usefulness.

After user confirmation: harden the named ScienceWorld profile without overwriting legacy settings; establish independent group-wise dev calibration/validation and more representative decisions; diagnose WM/revision compatibility with real evidence; retain conservative Unknown and disabled unvalidated confirmation skip; validate per-run resume/error/cost bookkeeping. Only later run a properly calibrated and fair evaluation cohort. Actor remains Qwen3-4B, WM remains current checkpoint, and paper-aligned settings/seeds/scoring stay governed by the unchanged protocol.

## G. Stop and Git

Phase4 complete; Phase5 NOT started or authorized. Await user confirmation. Baseline was2b6fde0c084ff2b38ccca8f913eed95f871168cc. Actual working-tree source hashes and executed versions are preserved in each run manifest/archive. Final Git commit and verified GitHub push are reported at handoff. Models, full large logs, WM-only private initialization context, caches and sensitive configuration are not committed. agentwmv2 remains clean and unchanged.
