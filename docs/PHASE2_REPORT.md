# Phase 2 report — Training-Free Internal Signal Extraction

## A. Completion and acceptance

Phase 2 functional acceptance is met for the selected Token NLL route. Three true ScienceWorld training instances produced 15 real decisions, each with both field scores and local/global NLL. No Memory, Gate, threshold search, training, full benchmark or Phase 3 work was done. This establishes an auditable functioning module, not validated error detection or improved task performance.

Formal contract is docs/INTERNAL_SIGNALS.md. Current backend is hf_generation_raw_bf16_v1, frozen Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c, Transformers4.57.3/torch2.8.0+cu128, BF16 parameters with FP32 raw-logit/NLL reduction, SDPA, non-thinking. No FP32 or rescored likelihood is mixed into signals.

## B. Code changes

| File | Purpose |
| --- | --- |
| internal_signals/base.py | Provider-neutral typed result/interface |
| internal_signals/provider.py | Official TokenNLLProvider, metadata consistency guards and configurable null/global-only missing policy |
| internal_signals/hf_actor.py | Frozen local HF generation behind existing Agent.count/generate; raw logits, provenance and cost |
| internal_signals/decision.py | Versioned structured Science prompt, observable-only recent history, exact native action-tail helper |
| configs/internal_signals/scienceworld_token_nll_v1.json | New opt-in diagnostic/scoring configuration; old configs untouched |
| scripts/run_audited_paper_eval.py | Small opt-in delegation/prompt/format branches; original Science.run step/feedback loop retained |
| scripts/diagnose_scienceworld_internal.py | Deterministic disjoint training task selection, 15-step real test and separate numerical diagnostics |
| scripts/audit_internal_diagnostic.py | Offline causal-input and arithmetic verification of actual traces |
| scripts/test_internal_signals.py | 14 new unit tests, including real tokenizer, missing policies, mismatch guards and legacy default-off |
| README, plan/progress, Internal Signal docs | Official route/status and reproducible instructions |
| docs/phase2/ | Small trace/evidence archive |

Hidden-state extraction, strict SpanUQ adapter, official SpanUQ code, v2 and existing WM settings were not changed. No new dependencies/model downloads were required. The old HTTP Actor implementation and legacy one-shot prompt remain default when internal_signals_config is absent. Baseline CLI/default configurations are not switched to the new prompt. Future comparisons must explicitly share the new Actor prompt and provider.

## C. Definitions and data isolation

Field scores are mean raw next-token NLL in the Premise and Expected Consequence values; Local is their maximum. Global averages ALL response text tokens, including labels/newlines/Action. Prompt, template prefix, trailing EOS, padding and unexpected special tokens are excluded. The boundary policy counts intersecting whole tokens and logs overhangs. Null is used for missing signals; valid computed zero remains distinct. Default global_only fallback is explicit and does not replace the local field.

The pinned source has 2120 training IDs, 200 evaluation IDs and 61 overlaps. All evaluation IDs are removed before diagnostic selection. A fixed SHA256 ranking selects one training instance per named task family:
- sciworld_349: use-thermometer, variation229.
- sciworld_1709: test-conductivity, variation273.
- sciworld_2997: find-living-thing, variation61.

Each runs at most five real actions, reusing the existing HTTP create/reset/admissible/step/close route. Native task name/variation are checked by the existing loop. These instances are reserved for diagnostic-development, not future untouched validation. No test success/trajectory was used. Exact manifest/source hashes are in phase2/manifest.json. The inspected service launch uses ee-scienceworld, package1.2.3, JAR SHA256 f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7. The installed deployment JAR hash is recorded; this is not a claim of remote attestation.

Only existing current observations/goals and committed past observation/action pairs enter prompts. Expected Consequence is model prediction, never actual future feedback. The offline trace audit rebuilt all 15 prompts exactly and rechecked NLL aggregates. Feedback and native scores are logged only after executing the candidate.

## D. Actual results

- 15/15 decisions: valid structured format.
- Premise missing: 0/15; consequence missing: 0/15.
- Parse failure/missing-span rates: 0%; alignment/numeric errors: 0.
- Response score-token counts: 23–43; 494 online generation tokens including EOS across 15 calls.
- Local NLL range: 0.029114–0.303201; Global range: 0.014509–0.126063 nats/token.
- No NaN/Inf/negative anomaly or truncation observed. This small sample cannot establish a population missing rate or score calibration.
- Native environment feedback included 2 no-known-action messages and 1 unknown-action response, plus closed-door and ambiguity outcomes. Valid structured formatting is NOT equivalent to executable/correct action selection; SDK admissible lists are not exhaustive.

Selected actual steps (full text and feedback are in phase2/steps.jsonl):

| Task / step | Action | Premise NLL | Consequence NLL | Local | Global |
| --- | --- | ---: | ---: | ---: | ---: |
| sciworld_349 / 1 | focus on thermometer | 0.303201 | 0.149456 | 0.303201 | 0.111833 |
| sciworld_1709 / 1 | go kitchen | 0.029114 | 0.018509 | 0.029114 | 0.014681 |
| sciworld_2997 / 2 | look around | 0.018543 | 0.143073 | 0.143073 | 0.050684 |
| sciworld_2997 / 3 | look around | 0.025596 | 0.137510 | 0.137510 | 0.048328 |

Manual checks:
1. sciworld_349/1 predicts focusing a thermometer; native feedback is No known action matches that input. This is a real failed proposal, not a parsing repair or fabricated WM return.
2. sciworld_1709/1 proposes go kitchen at Local=.029114, but feedback says the door is not open. Low model NLL did not ensure an executable action.
3. sciworld_2997/2 states the agent is in the kitchen (Premise NLL=.018543), while the current actual observation says art studio. The agent then looks around. This illustrates a fluent, low-surprisal unsupported premise, not a calibrated test of Hypothesis1.
4. sciworld_1709/4 gets a native duplicated-label numbered ambiguity menu; next step repeats the text command and receives Unknown action. The adapter did not collapse objects or invent an action. The prompt/parser work does not solve native disambiguation by itself.

No correlation/AUROC/success-rate benefit or WM-utility claim is made. Native scores and five-step partial trajectories are audit feedback, not formal benchmark results. Actor weakness/repeated look-around and consequence quality remain scientific limitations.

### Numeric consistency

Repeat of the first real decision uses the identical raw generation provider, IDs, dtype/config and seed. Token IDs and selected-token NLL match exactly: maximum difference0. One additional Actor call is counted (16 total calls for 15 real actions).

A separate teacher-forced BF16 diagnostic on that same original prompt and complete response uses the correct next-token shift. Max NLL drift is0.487322, mean0.024485 nats, extra forward latency about0.04584s. It is explicitly not used for signals and not asserted equivalent. There is no automatic switch to HF re-forward, FP32, vLLM or processed sampling log-probabilities. Prior Phase1B precision/backend findings remain applicable; metadata mismatch is tested as an error. Repeat equality on one real prompt is not a cross-hardware determinism guarantee.

### GPU and cost

GPU1 RTX4090, maximum recorded online-call peak allocated 7.7291 GiB. Mean generation 1.155167s/step; mean NLL extraction/alignment/aggregation 4.547ms/step. Online extraction adds zero backbone forward passes. The separate teacher-validation forward did not record its own peak memory; the reported number is not a measured whole-diagnostic-process maximum. Raw retained logit peak is about25.5MiB, plus temporary stack/reduction memory; these are not zero-cost operations.

The structured instruction adds34 prompt tokens versus an action-only instruction on the SAME recorded goal/state/history, counted without running a counterfactual Actor. Mean actual non-action output tokens (Premise/Consequence/labels/separators) are29.4 per step. This is not a measured output-token or latency delta against the old baseline. Across15 steps, generation time totals about17.328s and extraction about0.0682s; environment time/model loading are separate. Diagnostic repeat generation and one teacher-forced validation pass are additional validation cost, not normal online provider cost.

No WM, Judge, Router, Gate or retrieval call was made. The test process exited and GPU1 returned to3MiB; unrelated GPU0 service was unchanged.

### Commands actually executed

~~~bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 .venv-spanuq/bin/python scripts/test_internal_signals.py
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 .venv-spanuq/bin/python -u scripts/diagnose_scienceworld_internal.py --model-dir checkpoints/qwen3-4b --protocol-file /data1/shn/tmp/paper-aligned-audit/official/agent_system/scienceworld_protocol.py --train-ids /data1/shn/tmp/paper-aligned-audit/train_ids.json --test-ids /data1/shn/tmp/paper-aligned-audit/test_ids.json --science-python /home/shn/miniconda3/envs/ee-scienceworld/bin/python --device cuda:0 --output outputs/phase2_diagnostic_v1
.venv-spanuq/bin/python scripts/audit_internal_diagnostic.py --run-dir outputs/phase2_diagnostic_v1
.venv-spanuq/bin/python -m py_compile internal_signals/*.py scripts/diagnose_scienceworld_internal.py scripts/run_audited_paper_eval.py
~~~

All passed. Regression commands/results are preserved in phase2/regressions.json: test_token_nll(10), test_spanuq_integration(11), test_wm_reflection(33), test_agentwm_reflection(28), test_science_evidence(18), test_scienceworld_sr_context(3), all exit0. Together with14 new tests:117 passing unit cases. Mock fixtures are only unit tests, not scientific outputs.

The full original1.6MiB run remains under outputs/phase2_diagnostic_v1 on lab. The committed steps preserve exact prompts, response IDs/offsets/NLL, observations, actions, true feedback and scoring/cost metadata. Only large SDK admissible-action lists, which never entered prompts, are replaced by count+hash in the compact archive; complete originals remain on the server. No failed step was removed.

After the run, a progress-publication-only fix marks the driver finished. phase2/post_run_progress_fix.patch records the exact difference between executed and final driver source; inference/scoring code is unchanged and all other recorded source hashes match. The final driver compiles; no second GPU rollout was performed solely for this bookkeeping change.

### Portable rerun requirements

The model/env come from Phase1 setup. Provide an existing ScienceWorld1.2.3 AgentGym HTTP service and its environment Python. Obtain the pinned AgentGym-RL-Data-ID train/sciworld_train.json and eval/sciworld_test.json at revision99d0b7923bf126d9c6cdea5f362d2d41ccb5d493, and the official task catalog from Self-Distillation-AgentRL commit35a44f5c5647b1785ad23ddf0ffbd196b695bf86. Only data identity is reused, never RL training code. The driver verifies all source hashes/commit before selecting tasks.

~~~bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 .venv-spanuq/bin/python scripts/diagnose_scienceworld_internal.py --model-dir "$QWEN3_MODEL_DIR" --protocol-file "$REFERENCE_REPO/agent_system/scienceworld_protocol.py" --train-ids "$SOURCE_IDS/train/sciworld_train.json" --test-ids "$SOURCE_IDS/eval/sciworld_test.json" --science-python "$SCIENCEWORLD_PYTHON" --science-url "$SCIENCEWORLD_URL" --output "outputs/$NEW_RUN_ID"
~~~

Every output path must be new; diagnostic implicit resume/retries are refused. The official backend is currently exposed by this diagnostic entrypoint, not enabled by default for full evaluation. A fresh-server installation is a Phase7 validation, not claimed here.

## E. Limits and remaining work

Token NLL measures lexical conditional likelihood, not truth/error probability, causal correctness or WM benefit. Relevant field selection is syntactic; fluent copying/unsupported premises can be very confident. Global includes formatting and Action tokens and can dilute local issues. Field semantic quality is inconsistent: sciworld_349/2 describes a need to locate the thermometer rather than the immediate effect of look around; sciworld_1709/4 predicts conductivity identification from looking at a jar. Formatting success does not guarantee a meaningful condition or forecast. These issues are retained rather than prompt-tuned away on this diagnostic. Fifteen steps over three families are insufficient for calibration or error-detection performance. Backend/precision differences are material and scores must not be mixed. A low-score rule has not been designed or tested.

Current module has no production streaming-logprob optimization; retained logits and duplicate model residency for a future HTTP/HF split would need separate cost validation. Native action executability and ambiguity handling are not solved by the uncertainty module. No missing-field fallback was triggered in the live sample; its behavior is covered by tests, not a measured real-world failure rate. The existing v3 external benchmark assets remain a later independent deployment requirement.

SpanUQ remains unavailable and preserved; its existing hidden-state pipeline, strict-loading guard and reports are unchanged. The structural provider interface can support a future validated replacement, but score scale/thresholds/field matching/cost need fresh development validation. No scores are labelled SpanUQ.

## F. Phase 3 readiness

Engineering prerequisites are satisfied for proceeding to development Experience Memory AFTER user confirmation: stable fixed provider, explicit missing semantics, auditable real observations/actions/feedback, disjoint task provenance and cost metadata. Scientific usefulness is not established and no values are sent to a Gate. Reserve these diagnostic instances as development and do not reuse their outcomes as untouched validation. Phase3 must continue to exclude all test trajectories/IDs from experience synthesis and threshold choice.

## G. Stop

Phase2 complete for the selected route. Await user confirmation before Phase3. Paper-Aligned Evaluation Protocol and all legacy benchmark configurations remain unchanged.
