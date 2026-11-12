# Training-Free Internal–External Selective WM: implementation plan

Early Experience SR Actor trial COMPLETE: strict frozen Qwen2.5-7B SR load passes; native-H2 scores83/−100/5, mean−4/SR0 of3. Structured3/3 attempts fail;7/7 generations lack required spans. Matched9B native control has2 completed−100 and1 format failure; no complete-cohort ranking. Public training-corpus Goal overlap prevents clean-generalization claims. No Gate/Reviewer/Memory/threshold/Paper/default change, no Test rollout or Phase6. See [trial report](EE_SR_ACTOR_TRIAL.md).

Phase5.10 COMPLETE: compact-json-conservative-v2 has0/76 offline and0/47 online format errors; Original online4/44. All15 Episodes complete,91/91 lookaheads and29/29 KEEP projections verified. Every condition scores−66.67/SR0%; no task benefit. Always replacement acceptance improves, Full does not. Gate/thresholds/Memory/Actor/WM/Paper protocol unchanged; new-ID variants share known goals. All owned jobs stopped. Await user decision; do not start Phase6. See PHASE5_10_REPORT.md.

Phase5.9 COMPLETE: 12 frozen independent dev instances × seeds0/1 ×4 policies =96 real Episodes; 8 goal clusters. Engineering audit passes (531/531 lookaheads,187/187 KEEP projections). Mean scores Never−0.542/Always3.333/Internal−10.833/Full−19.375; successes0/1/0/0 of24. Full did not reproduce a benefit. Frozen method/Memory/thresholds unchanged. Read [Phase5.9 report](PHASE5_9_REPORT.md) and [independent protocol](INDEPENDENT_DEV_EVAL.md). Await user review; do not start Phase6.

## Phase5.10 acceptance and next decision

- Format engineering: bounded real regression76/76 and online47/47 parse;52 unique CPU tests pass.
- Environment/runtime:15 Episodes,91 verified one-step lookaheads,29 exact KEEP projections; all source/input/seed/budget/receipt checks pass. Diagnostic reader incidents and their full costs are preserved.
- Action quality: Always public execution acceptance improves; Full acceptance stays3/7 and rejected replacements persist. Same-state one-step Fixed−Skip is0 in32 selected states; no downstream causal return estimate.
- Scientific utility: five conditions all score−66.67/SR0% on3 new-ID/known-goal variants. No task improvement; no independent unseen-goal evidence. Original defaults and Gate/threshold/Memory/Actor/Paper protocol remain unchanged.

Await user review before further implementation. Recommended future scope: separately version format-only versus conservative-prompt effects; diagnose goal-incorrect but syntactically accepted revisions; evaluate paired continuations under explicit extra query/cost permissions. A new goal-level development partition could be reserved from unused, uncollected training goals only after a separately approved split audit excluding Memory/test overlap. Do not tune or relabel current regression tasks as independent. No Phase6 authorization.

## Follow-up: deterministic real-environment next-state repair

User authorized resolving true-simulator next-state prediction. Root cause:1.2.3 unstable object iteration changes in-place heat updates across independent JVMs. Use official native1.3.0 commit e8216d6044e8e39be9fcb185e3b2dfb602584b52, with a pinned hash and explicit new profile; no local Java/Scala patch or SDK replacement. Controlled48/48 and recorded-history63/63 transitions match;9 repeated worker and9 corrupt-evidence checks pass. New post-step fingerprints strengthen KEEP verification. See REAL_ENV_WM_REPLAY_FIX.md and REAL_ENV_WM_REPLAY_FIX_REPORT.md.

The bounded driver defaults to phase57_replay_dev_v2.json and recollects dev calibration; old v1 configs/reports/evidence remain for provenance. Native physics/door-reference behavior can differ from1.2.3, so this is a separate environment profile, not silent exact paper alignment. Memory, Gate rules, Reviewer, model/scoring and final task identities remain unchanged. New four-policy dev regression and source/cost audit are recorded in the repair report. Do not enter Phase6 without approval; reliable simulator access does not resolve Reviewer reliability or establish utility.

## Phase control (original Phase5.7 handoff)

**Phase5.7 amended Reset + Replay integration and bounded validation are complete. Await user confirmation; no Phase6 authorization.** Actor/Reviewer are official frozen Qwen3.5-9B; real independent ScienceWorld replay supplies one-step feedback only when verified. The initial full-JVM-clone prerequisite was superseded by the user and is archived in PHASE5_7_INITIAL_AUDIT.md. It is not an outstanding requirement for the accepted reset/replay design.

Read PHASE5_7_REPORT.md,REAL_ENVIRONMENT_WM.md,QWEN35_9B_INTEGRATION.md and PROGRESS.md. Engineering trace checks pass; general replay reliability,stable thresholds and scientific effectiveness do not follow. Old Qwen3-4B/Word2World plans below are historical and do not override the current architecture.

## Current implementation and evidence

- Independent-JVM reset,all committed commands replayed,public/state/RNG/tree checks,one candidate step,one shared9B Reviewer call,then normal formal execution. Failed verification means explicit failure/no Reviewer/unchanged draft,not fake prediction or post-action reflection. No Java/Scala edits or Qwen2.5 online fallback.
- Original Prompt/parser,TokenNLL reductions,Memory/evidence and Gate rule reused. One-round protocol reserves one forecast; legacy JSON review retains four. Memory correction/confirmed skip remain disabled.
-9B model/tokenizer/EOS/path verified;new config fingerprint and40 true dev decisions/6 tasks;prespecifiedq.75 Local.5749072272810143/Global.3332634086050418. No outcome selection or old4B threshold transfer. The Query zero/one-based interface bug was repaired without changing Memory rules; identical calibration was reused by hashes.
-12 real dev episodes/3 tasks/seed0:Never-61,Always-7.67,Internal-100,Full41.67 mean scores;all0 success.201 formal actions,75 lookahead attempts,21 verified,54 rejected;7/13 replacements receive explicit refusal. Raw logs and source revisions retained;compact evidence under docs/phase57_replay.
- Fully separate inference environment .venv-qwen35-clean,pip check0,91 tests;actual Actor/Reviewer prompt outputs identical to original run,Actor NLL delta0,and native replay smoke passes. Published cohort used prior overlay with same pinned core versions;not silently relabelled.
- New modules:worldmodel/ground_truth_environment.py,internal_signals/qwen35_actor.py,reflection/replay_review.py;bounded scripts/run_phase57_replay.py;independent configs/evaluation/phase57_replay_dev_v1.json and9B scoring/pilot Gate configs. Evidence auditor/exporter/minimal failure reproducer and setup/tests are implemented,not proposals.

## Before any Phase6 large evaluation

1. Use the explicit official1.3.0 replay profile and its new verified evidence; historical1.2.3 had21/75 availability. Continue validating broader lawful development states and simulator information permissions. Finite state/transition agreement is not complete hidden equality; preserve strict failure rejection and historical thermal evidence.
2. Independently validate Reviewer grounding/goal reasoning. Any generic action-validity guard or prompt change must be a named version shared by all compared Gates;retain the current results. No task-specific rules or hidden answers.
3. Expand/repeat only legal development calibration and diagnostic seeds under a predeclared plan. The40-step pilot is not stable calibration;no final-test threshold/rule selection. Keep budget independent of threshold and include all actual costs.
4. Audit horizon/negative-failure censoring:Full's partial scores/all20-turn endpoints are not task success or strict downstream WM utility. More positive/zero/negative evidence requires reliable same-state comparisons;no similar-state shortcuts.
5. Integrate the verified new environment boundary into a general evaluation matrix only after approval;current driver deliberately restricts tasks/seed/cohort. Preserve the original paper protocol and final200 IDs. Eight rollout seeds,matched-budget baselines/ablations and ALFWorld/WebShop remain future phases.
6. Continue Phase7 reproducibility work only in its authorized scope:clean lab setup is tested,another physical server is not;keep models/cache/logs out of Git and record failures/version/Run IDs.

## Historical Phase0–7 proposals and interfaces

The original plan is retained below for provenance. Several proposed paths were implemented with different filenames,SpanUQ remains shelved,and old model-specific instructions are historical. Current status/file map is above and in the phase reports.

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

Completed for the narrow rule-based engineering scope. Final split:1292 memory-train/197 development/200 fixed test, canonical IDs plus normalized goal-group isolation and native test identity audit. Memory:48 fresh Qwen steps plus6 separately labelled real environment probe steps ->54 Cases/two Cards; no v2 trajectory feedback. Source/case/card validation, BM25, temporal filtering, conservative tri-state ExternalSignalProvider and read-only pre-action hook are implemented. Phase2 backend/config/prompt and PAPER_ALIGNED_EVAL.md are byte-identical. First fixed dev15 all Unknown; separate previously exposed targeted dev10 yielded two evidence-backed blocked-door conflicts, no natural positive sufficiency.37 new +117 regression cases pass, causal/query/prompt/receipt replay25/25. See PHASE3_REPORT.md and phase3/ evidence. These diagnostic cohorts are development-exposed, not untouched validation; do not synthesize from them or use them as blind performance estimates. No Gate/correction/training/full benchmark. Stop before Phase4.

Next Phase4 must consume Unknown explicitly, preserve unresolved-conflict semantics (S=false when draft unchanged), define budgets/missing fallbacks and explanation logs, and unit-test combinations before development-only calibration. Positive sufficiency has only unit-test support; broader skip claims require separate real evidence. Rule/prompt/Judge expansion requires a new version/independent development validation, not retroactive adjustment of this frozen snapshot.

Historical Phase3 proposal below is retained for context; actual correction is intentionally disabled under the user's Phase3 scope.

Before reading old results into memory, freeze split manifests and exclude test traces including past test-driven fixes. Extract cases only from real train/dev interaction sources with identity/availability timestamps. Build cards via rules/frozen LLM; require support/counterexamples and distinguish causal speculation. Single-case evidence remains low reliability. Use BM25 first or pinned existing embedding; fixed top-k, prompt/schema/temperature/revision for any judge. Log complete offline synthesis and online retrieval/judge costs. Correction is optional, explicit and shared across compared Gates. Acceptance: evidence-backed examples of applicability/conflict/sufficiency/correction plus split-overlap and time-leakage checks. Do not invent cards or trajectories when sources are absent.

## Phase 4 — deterministic Gate (completed)

Implemented gates/{core,budget,evidence,generation,runtime,worldmodel_setup}.py and provisional configuration, calibration/diagnostic/audit/test scripts. Unified never/always/local/global/internal/experience/full; no extra extraction/retrieval for unused baseline signal types. Full default: budget AND (reliable unresolved/correctable conflict OR high Local OR high Global OR missing-required/invalid draft). Conflict always wins over sufficiency. A proposed correction is not an executed correction; correction remains off. Optional confirmed CURRENT action skipping is CPU-tested but disabled pending natural evidence. Unknown explicitly falls back/abstains, never implies no conflict or sufficient.

Budget counts verification invocations plus a separate actual-forecast cap: reserve the unchanged protocol's maximum4 before a call, settle actual usage, never shorten the3-round JSON review. Skip really avoids predict/review and teacher-forces committed real feedback. Pure decisions precede WM and environment calls. Existing Word2World backbone, one-step prediction, JSON reflection and final-replacement revision are preserved through opt-in adapters; legacy no-Gate paths stay default. Accepted Actor/Internal/config, frozen54-Case Memory and paper protocol remain unchanged.

Provisional Q=.75 thresholds from25 exposed dev steps/5 goal groups: Local=.13648279807529134, Global=.06599500845849053; Pearson=.95887/Spearman=.94846. No reward/success/test selection. Preset diagnostic budget1 invocation/4 predictions; no stable calibration or task-utility claim. Future calibration must use larger goal-group-separated development calibration/validation; the diagnostic data is already exposed.

217 CPU/regression cases pass. Actual Full10 steps:2 Calls/8 Skips,8 WM forecasts/6 review calls,3 budget-blocked requests. Never/Always one-step real checks omit internal/retrieval costs as required. All12 committed development traces pass causal/input/evidence/budget/prompt replay; actual owned server12 forecast requests match logs. Both Full WM revisions failed immediate execution: no benefit claim. Full online model cost29357 tokens/24 calls, plus1 separately recorded parity-validation call; all actor/internal/external/WM/revision/setup and offline-memory costs are recorded. Current confirmation has no natural positive example. See PHASE4_REPORT.md and docs/phase4.

Random/Periodic matched-budget scheduling interface is documented; active implementation/full experiments remain Phase6. Stop before Phase5.

## Phase 5 — ScienceWorld end to end

Completed. Independent opt-in profile: exact pinned200 evaluation IDs/template/variation metadata checked, H20/SDK21, history2, input4096/response512, temperature0.4/top_p1/top_k0, official per-task-turn rollout seed derivation. Old profiles preserved. SDK1.2.3/JAR hash pinned; exact historical authors' binary parity remains unverified. Intentional Qwen3-4B/structured prompt/HF backend/WM differences recorded in PAPER_ALIGNED_EVAL.md.

Thirty legal development goal groups produced273 real Actor-only scored decisions. q0.75 Local0.12081397919163489/Global0.053253251821409955 frozen before separate5-goal-group complete validation, no outcome-based selection. Pearson0.893067/Spearman0.898846: local/global strongly related. Calibration stability/generalization is not proven.

Five fixed held-out development tasks x Never/Always/Internal/Full, seed0, complete episodes to native termination orH20:280 true decisions;107 Call/173 Skip;315 real WM forecasts match owned server counters. Model revisions/scoring, original Science.run/JSON3-round protocol preserved. Strict source/causal/NLL/role-seed/budget/prompt/score/cost replay and254 CPU/regression tests passed. Whole cohort has0 successes; no effectiveness claim.

Both Phase4 final failures were forecast correctly; reviewer grounding/decision errors, not corrupt parser/transport. Phase5 reviews278/278 valid JSON;34 same-action replace violations/50 unsupported replacements. Fixed new observer URL-keyword binding bug before successful canary; do not confuse this with model errors. No task-specific repair or critic/backbone change.

Train-only Memory extension adds one same-room observation card from the existing54 cases,2 independent supports/no counters; base snapshot unchanged. Shared frozen snapshot, unused modules skipped. Natural Sufficiency positive still absent; Full coverage10/73 and conflict3/73. Eleven low-Local-NLL historical conflict shadow cases are exploratory, not gate-training inputs or utility proof.

See PHASE5_REPORT.md, SMALL_SCALE_EVAL.md, WM_FAILURE_AUDIT.md and docs/phase5 for actual commands, costs, per-task scores, complete raw-log hashes and remaining risks. All online diagnostic totals1,628,427 tokens/873 model calls/1212.106s episode wall time; setup/calibration/offline construction separately reported. Owned services stopped and GPUs released. **Stop before Phase6.** Full benchmark, matched-budget sweeps, cross-benchmark runs and fresh-server reproduction remain unauthorized/unperformed.

## Phase 5.5 — WM verification and Reviewer effectiveness audit

Completed. All20 Phase5 source trajectories/315 real WM replies/past-only prefixes audited. No score extraction or duplicate-main-step defect. The one3070 wrong-goal focus explains the mean-score gap. SDK clock termination is distinguished from task completion; PAPER_ALIGNED_EVAL.md contains an explicit clarification.

Four independent variants x5 original dev tasks/seed0/H20: Never-9, Original-29, Prediction-Only-9, Validity-Checked-28; all0 success. Never/Original trajectories reproduce Phase5 exactly; Prediction-Only matches Never.31 new unit contracts plus133 related regression cases pass (164 total).

Five strict paired branch points selected as first Full call per task, live-JVM heap/RNG/Python reward state restored, identical Actor prefixes/first Draft IDs/budgets/turn seeds, one intervention then Never.0 positive/1 negative/4 zero final-score utilities. New reconstructed shared states are not asserted identical to historical hidden heaps.

Actual initializer defect found: root terminals/sewer treated as rooms, failed/ambiguous teleports contaminate subsequent queries/labels. Independent repair uses official action enumeration, per-room snapshot isolation and observed-room verification. All5 tasks have10 verified rooms; repeat look facts match, aliases can vary. Freeze first actual descriptions before outcome measurement. Separate5 corrected-initializer/original-Reviewer Episodes remain-29/0 success,804134 tokens; no hidden baseline replacement. Upstream code, default Gate/reviewer, Actor/NLL config, Memory/thresholds and final test identities remain unchanged.

Original token drift between reruns is isolated to WM system strings; actual requests counted once.100 same-decision identical WM requests are redundancy by review retry/cycles, not duplicate environment execution. Snapshot/API validity/partial progress alone do not establish utility. The initial network outage caused idle WM service residency, disclosed separately; corrected owned-service supervisor cleans in finally. All owned services now stopped.

Read PHASE5_5_REPORT.md (ten answers), REVIEWER_AUDIT.md (real category examples), WM_UTILITY_DIAGNOSTICS.md (matched branches) and docs/phase55 (real results/costs/source hashes). Total462 Actor actions, within600 bound; no model training, Test rollout, threshold search, memory extension or Phase6 work. Recommend goal-grounded verification and one common validated initializer before formal matched-budget comparisons. Stop for user confirmation.

## Phase 6 — baselines, ablations and analysis

Apply the independent paper-aligned profile from [PAPER_ALIGNED_EVAL.md](PAPER_ALIGNED_EVAL.md): rollout seeds 0–7 with per-task-turn request seeding, fixed task identity, T=.4/top_p=1/no top-k restriction/response=512, per-instance aggregation and sample SD across eight complete seed means. Science main outcome is raw final Task Score; ALF is SR; Web is SR plus graded score. Add meaningful unequal-batch/category and score-scale tests. Keep pure paper-prompt and structured-draft controls distinct. ALF targets 140 valid_seen/H=50/history=2/prompt=2048; Web targets fixed 128 goal indices/H=15/history=2/prompt=4096 after environment validation. No test-driven threshold or memory construction.

Implement Never, Always, Random and Periodic at predeclared matched-budget grids, action uncertainty, current Token NLL local/global (SpanUQ deferred), combined internal, experience-only and full Gate; retain Risk only if a real compatible implementation is later identified. Add Memory-only and full without sufficiency/conflict/local. Group memory correction on/off separately, sharing memory/correction for within-group Gate comparisons. Fixed Actor/WM, review depth, prompts, seeds/task set and environment; trajectories naturally diverge after decisions.

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
