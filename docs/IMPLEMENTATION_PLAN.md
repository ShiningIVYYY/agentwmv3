# Training-Free Internal–External Selective WM: implementation plan

## Phase control

Current phase: 2, explicitly authorized and completed using user-selected route B (Token NLL). Phase 1 SpanUQ remains shelved/unavailable; it no longer blocks this selected route. Phases 3–7 are NOT authorized. Each phase requires implementation, appropriate tests, an A–G report and explicit user confirmation before the next phase. See PROGRESS.md for restart state. Hypotheses 1 and 2 remain unverified. No Actor/WM/probe/router/judge training or task-specific parameter updates are permitted.

Read REPOSITORY_AUDIT.md and SPANUQ_AUDIT.md before modifying code. Existing components: audited Agent/benchmark loops, WM backends, reflection protocols, snapshot/replay, scheduler and task logs. Implemented additions: isolated Qwen3-4B inference, structured Science decisions and formal Token NLL signals. Still missing: validated optional SpanUQ integration, provenance-controlled cross-episode memory, pre-WM Gate and complete system-cost accounting/fresh-server validation.

## Proposed interfaces (design only)

- DraftDecision: raw output, environment action, premise/consequence text with token and character ranges, exact messages/input IDs, response IDs, model/tokenizer revisions. Preserve original action grammar.
- InternalSignals: local/global/premise/consequence uncertainty (nullable), detected spans, missing_span, missing_reason, validity/alignment metadata, extraction latency, forward passes/tokens/compute, checkpoint metadata. Missing is not zero.
- ExperienceQuery: benchmark, split/task/episode/step, goal/subgoal, current observable state and draft. It must not include current action's future feedback or private WM state.
- ExternalSignals: coverage/conflict/sufficiency, reliability, evidence/card/case IDs, reason, retrieval/judge latency and cost. Conflicting or unsupported evidence must not imply sufficiency.
- GateContext: observable decision + internal/external signals + predeclared budget state. GateDecision: call_wm, reason, triggered_signals, copied signal values, budget_remaining, gate latency and additional cost.
- ExperienceCase: draft/executed action, committed real feedback, outcome observation time, source IDs, split/protocol, called_wm. Cards distinguish observations, hypotheses, counterexamples and support. Future outcomes are allowed only in offline training/development cases with explicit availability timestamps.
- Budget counts actual forecasts as well as call steps: one Call may produce several predictions. Reserve the fixed review protocol's maximum cost or declare step-level budget and separately enforce forecast cap. Never silently shorten review depth for selected Gates.

## Planned file map

| File(s) | Phase / relationship |
| --- | --- |
| `internal_signals/{spanuq_adapter,hidden_states,schemas}.py` (new) | 1–2; thin frozen official inference wrapper, after upstream compatibility resolution |
| `scripts/spanuq_smoke.py` (new), `configs/internal_signals/qwen3_4b.json` (new) | 1; pinned isolated minimal reproduction and metadata |
| `environment/spanuq-inference.*` (new, exact format chosen after compatibility check) | 1; locked HF/probe environment, distinct generation/environment services |
| `scripts/test_spanuq_integration.py` (new) | 1; real checkpoint key/shape/boundary/finite-value tests, model-gated |
| `internal_signals/decision.py`, `scripts/test_internal_signals.py` (new) | 2; structured draft and missing-span behavior, real Agent trace review |
| `experience/{cases,cards,retrieval,signals,correction}.py` (new) | 3; real source-controlled memory, fixed-schema judge, optional correction |
| `scripts/build_experience.py`, `scripts/test_experience.py`, `configs/experience/*.json` (new) | 3; split/time/provenance checks and reproducible memory builds |
| `gates/{base,rules,baselines,budget}.py`, `scripts/test_gates.py`, `configs/gates/*.json` (new) | 4; one interface and CPU truth-table tests |
| `worldmodel/runtime.py`, `worldmodel/backends.py` (modify minimally) | 5; gated dispatch and real-feedback update on Skip; preserve forecast/commit path on Call |
| `scripts/run_audited_paper_eval.py`, tokenizer/service launcher scripts and new-cohort configs (modify/add) | 2/5; Qwen3 config, draft/signals seam, single-step trace and actual feedback sink; no broad runner rewrite |
| `experiments/matrix.py`, new v3 runner configuration (extend) | 6; keep legacy matrices as historical protocols |
| `analysis/{costs,success_cost,overconfidence,redundancy,complementarity,harmful_calls}.py` (new) | 6; trace-derived honest metrics, optional counterfactual analysis |
| README, `.env.example`, dependency pins, reproduction scripts/docs (add/update) | 7; only commands for implemented entrypoints |

Paths above are proposed, NOT present/implemented modules. Create directories only when their phase needs them. Phase 0 changes only docs, an inherited-README notice and ignore rules. External source patches, split manifests and setup provenance must be reconstructed independently inside v3; do not reuse writable v2 symlinks or modify v2's conda environments.

## Phase 1 — isolated Qwen3-4B / SpanUQ

Execution status: partial, not accepted. See [PHASE1_REPORT.md](PHASE1_REPORT.md). Independent environment and three real Qwen3 hidden-state samples passed; strict probe load fails with 4 unexpected u_conditioner keys and positional-buffer mismatch. All official code commits have the same model blob. Producer block convention [21,22,23] was verified against HF indices [22,23,24]. Do not enter Phase 2 or treat guarded scoring code as a usable probe.

1. Resolve official config/API/checkpoint version mismatch, layer indexing and training tokenizer provenance first. Inspect compatible upstream history; if necessary document a minimal compatibility patch without reimplementing SpanUQ. Unknown/missing head weights cause failure, not strict=False acceptance.
2. Create a new Python >=3.9 environment on the data volume (candidate Python 3.10/3.12, torch/transformers versions selected from validated compatibility; Qwen3 requires Transformers >=4.51). Exact versions locked only after validation. Keep ScienceWorld 3.8 and old vLLM untouched.
3. Download ONLY pinned Qwen3-4B and Qwen3-4B probe after Phase 1 authorization. Configure HF_ENDPOINT/HF_HOME/cache/revision via env/config. Do not use official download-all/data scripts.
4. Reproduce official inference on a few ordinary English inputs; retain outputs/spans/raw and calibrated scores, device peak VRAM, latency and metadata. If upstream demo is broken, save exact failure before any adapter.
5. Implement exact chat input/response boundary checks, selected hidden states and cache, test full vs response token offsets, missing/overflow/EOS and numerical legality.
Acceptance: real complete checkpoint load with checked keys/shapes; established layer convention; reproducible valid real spans; model versions and additional forward overhead recorded. If provenance cannot resolve layer convention or architecture, stop. No Agent integration before this acceptance.

## Phase 2 — decision uncertainty

Executed and accepted engineering scope: user selected frozen Qwen3-4B raw-generation Token NLL, not SpanUQ. See INTERNAL_SIGNALS.md and PHASE2_REPORT.md. Formal provider hf_generation_raw_bf16_v1 uses Premise/Expected Consequence mean NLL, their maximum and whole-response mean; missing values remain null with explicit configurable fallback. Original Agent/Science.run is reused through opt-in branches. 15 real disjoint training decisions, 0 missing/parse/alignment errors, repeat NLL delta0 and117 passing unit cases. A separate BF16 re-forward differs materially and is prohibited as a mixed provider. No Memory/Gate/calibration implemented.

Historical proposal below referring to SpanUQ is superseded for the current route; it remains relevant only to a future separately validated probe backend.

Reuse prompts, add minimal Premise/Expected Consequence while retaining native Action; same draft format across all new baselines. Count extra tokens/latency. Review relevant spans by field, compute nullable local max and premise/consequence scores; use validated native sequence score or explicitly named deterministic fallback. Missing-span fallback proposal: if global is valid, use global with an explicit fallback flag; if both missing, request WM subject to budget and report missing-signal rate. Do not treat fallback as calibrated risk. Collect a small REAL development ScienceWorld decision set with no future feedback in extraction. Acceptance: audited span/text/boundary report, missing/unrelated rates, descriptive error association only. Stop if SpanUQ is unsuitable for Agent text.

## Phase 3 — real experience memory

Before reading old results into memory, freeze split manifests and exclude test traces including past test-driven fixes. Extract cases only from real train/dev interaction sources with identity/availability timestamps. Build cards via rules/frozen LLM; require support/counterexamples and distinguish causal speculation. Single-case evidence remains low reliability. Use BM25 first or pinned existing embedding; fixed top-k, prompt/schema/temperature/revision for any judge. Log complete offline synthesis and online retrieval/judge costs. Correction is optional, explicit and shared across compared Gates. Acceptance: evidence-backed examples of applicability/conflict/sufficiency/correction plus split-overlap and time-leakage checks. Do not invent cards or trajectories when sources are absent.

## Phase 4 — deterministic Gate

Implement Call = B AND NOT S AND (C OR U_local > tau_local OR U_global > tau_global). Missing signals use the documented Phase 2 fallback. Conflicting cards force S=false; low-reliability cards alone do not assert C/S (record their presence for analysis). S=true requires applicable reliable evidence with concrete action confirmation/correction and no unresolved conflict. Conservative validity/schema failure produces no sufficiency. Define budget semantics and no remaining-budget fallback explicitly.

Calibrate thresholds or target rates using a frozen dev set only; record whether dev success was used in selection. No test-dependent matched-budget tuning. Acceptance: CPU truth-table/property tests for missing, invalid, contradictory and budget-exhausted cases; immutable configs, explanatory logs, no optimizer/training path.

## Phase 5 — ScienceWorld end to end

Follow-up evaluation specification: [PAPER_ALIGNED_EVAL.md](PAPER_ALIGNED_EVAL.md) is authoritative for the new opt-in paper-aligned cohort: Qwen3-4B, exact verified AgentGym 200 IDs, H=20, history=2, prompt=4096, response=512; preserve the old 100-step and other historical protocols. Implement profile-specific prompt/projection/reset/horizon/score/seed behavior only after phase authorization. This documentation update does not start Phase 5.

Reconstruct independent v3 runtime assets and local patches first. The new paper-aligned target is AgentGym + ScienceWorld 1.2.3, 200 fixed tasks and 20 agent turns; the inherited cohort remains 100 steps, and RLVMR/L0-L2/30 steps is not established. Exact source-ID/order/task-variation equivalence has now been checked; live JAR/reset/observation equivalence remains pending. No silent conversion. Freeze Actor and WM services, simulation/review depth, prompts, seed and manifest. Insert signals/memory/Gate between parsed draft and wm_runtime.select. For Skip, execute exactly once and update real WM history without invoking predict/reflection. For Call, preserve the selected current prediction/reflection/revision protocol and commit verification. Experience insertion occurs only after true feedback; never add rejected forecasts as facts.

Acceptance: a small real no-WM Qwen3 episode plus several Call/Skip traces; zero duplicate step/call and no hidden/future state leakage; non-thinking/action parsing verified; failures retained. Smoke tests do not validate hypotheses. No full benchmark at this stage.

## Phase 6 — baselines, ablations and analysis

Apply the independent paper-aligned profile from [PAPER_ALIGNED_EVAL.md](PAPER_ALIGNED_EVAL.md): rollout seeds 0–7 with per-task-turn request seeding, fixed task identity, T=.4/top_p=1/no top-k restriction/response=512, per-instance aggregation and sample SD across eight complete seed means. Science main outcome is raw final Task Score; ALF is SR; Web is SR plus graded score. Add meaningful unequal-batch/category and score-scale tests. Keep pure paper-prompt and structured-draft controls distinct. ALF targets 140 valid_seen/H=50/history=2/prompt=2048; Web targets fixed 128 goal indices/H=15/history=2/prompt=4096 after environment validation. No test-driven threshold or memory construction.

Implement Never, Always, Random and Periodic at predeclared matched-budget grids, action uncertainty, SpanUQ local, global, combined internal, experience-only and full Gate; retain Risk only if a real compatible implementation is later identified. Add Memory-only and full without sufficiency/conflict/local. Group memory correction on/off separately, sharing memory/correction for within-group Gate comparisons. Fixed Actor/WM, review depth, prompts, seeds/task set and environment; trajectories naturally diverge after decisions.

Log success, score/steps, all Actor/WM/review/replan/judge/router calls, input/output tokens, SpanUQ passes/compute, retrievals, wall latency, offline synthesis and online total cost. Monetary estimates only with declared trustworthy prices. Summarize Success–Cost curves; keep partial runs and infra errors explicit. Analyze low-U actual errors, redundant calls, internal/external overlap and harmful-call evidence; same-state counterfactual Call/Skip branches are post-hoc only and must pass state recovery checks. Action changes alone are not benefit. Acceptance: reproducible complete metric aggregation, budget/no-future tests, configuration parity checks and evidence-linked examples; no efficacy claim before data.

## Phase 7 — reproduction and wider benchmarks

Publish actual runnable setup/download/env/service/baseline/full/analysis commands. Pin source/checkpoint/tokenizer and external patches, manifests and random seeds; no personal absolute paths in new operational code. Independent run IDs, lock/retry/recovery and error logs, source/environment fingerprints, no weights/large data/results in Git. Validate on a fresh independent environment. After ScienceWorld acceptance, diagnose ALFWorld then WebShop separately; do not break working environments. Cross-domain memory requires a declared condition. Acceptance: fresh-server reproduction checklist actually executed where available, all unmet items labelled unverified, and complete cross-benchmark provenance.

## Phase 0 acceptance answers

1. Qwen3-4B switch: feasible via existing model/url/tokenizer flags and non-thinking support, but launchers/sampling/API payload need minimal changes; real inference unverified.
2. SpanUQ reuse: hardware plausibly adequate for short inference; current code/checkpoint integration confirmed incompatible and cannot yet be called runnable. New isolated environment and upstream compatibility resolution required.
3. Hidden states: HF teacher-forced forward of exact chat prompt and response, response-only [B,T,2560] on layers [21,22,23] after resolving HF-index vs hook-index ambiguity. vLLM+HF fallback adds cost.
4. WM: retain learned Qwen2.5 WM and fixed review flow. Selective Skip needs an explicit real-feedback update path; naive early return is unsafe.
5. Memory: query after Draft/internal before Gate; record new cases after committed environment feedback, filtered by split/time. Shared correction after evidence resolution and before final execution with consistent WM input.
6. Gate: new `gates/` modules called from central `worldmodel/runtime.py` seam; no generic gate currently present.
7. Leakage: existing test logs, test-informed policy/context, oracle metadata, future step outcomes, online episode ordering and cross-benchmark sharing are concrete risks; freeze independent splits and timestamp/provenance checks.
8. File map above is the scoped proposal; none of those Phase 1–7 runtime edits are authorized/executed now.

## Evaluation planning follow-up

Paper/source audit completed without changing inference integration or legacy runtime configs. See PAPER_ALIGNED_EVAL.md for pinned source, exact Science IDs/61 overlap exclusion, independent development proposal, configuration map and pending equivalence checks. Phase 3–4 memory/calibration must exclude all 200 test IDs; derive a separate dev split from the 2059 disjoint source-train pool before inspecting outcomes. This update grants no phase/experiment authorization.

## Phase 1B compatibility investigation and fallback planning

Authorized Phase 1B is complete. Conclusion B: official SpanUQ remains incompatible; strict loading is still rejected. See SPANUQ_COMPATIBILITY.md and the unposted SPANUQ_ISSUE_DRAFT.md. A standalone fixed-field frozen-Qwen token-NLL prototype/design is in INTERNAL_SIGNAL_FALLBACK.md; no Agent/Gate wiring or complete Phase 2 development occurred. BF16 generation/full-forward NLL equivalence failed the predeclared tolerance, while FP32 diagnostic passed; provider/dtype-specific scoring must be fixed before later calibration. User must select the route before Phase 2. The existing external memory, Gate and paper-aligned evaluation plan remain unchanged.

## Phase 2 handoff

Token NLL Internal Signals are implemented and real-development diagnosed. New source/data/cost provenance is under docs/phase2. Existing paper-aligned evaluation settings are not changed; this is a separate15-step diagnostic. Phase3 can build true training/development experience only after confirmation, reserving sciworld_349/1709/2997 from any untouched validation partition. Do not infer error calibration or WM utility from these NLL values.
