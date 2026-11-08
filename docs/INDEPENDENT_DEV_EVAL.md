# Phase 5.9 — frozen independent development evaluation

Status: preregistered protocol, not results. Only Phase5.9 is authorized. No final200-task sweep, training, method revision, memory expansion, outcome-based threshold selection or old Qwen2.5 WM is included.

## Frozen method

Method source base: `6e564390955f136ea80d57facffd2e50c8543d34`. The complete configuration and content hashes are [phase59_independent_dev_v1.json](../configs/evaluation/phase59_independent_dev_v1.json). Every worker and audit rejects method/config drift.

- One frozen official post-trained Qwen3.5-9B object per worker supplies BOTH Actor and Reviewer; revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Existing HF5.22.2/Torch2.8.0/cu128/BF16/reference kernels, official nonthinking template, structured Draft, EOS/token alignment and `hf_qwen35_generation_raw_bf16_v1` remain unchanged. Global includes field labels/Action but excludes prompt/EOS/padding/special tokens; Local=max(mean Premise content NLL,mean Expected-Consequence content NLL). No additional online forward pass.
- Native engine1.3.0, official commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52`, SHA256 `e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465`; the unchanged SDK1.2.3 Python client uses its supported serverPath argument. Independent JVM reset/replay, exact checked prefixes, one candidate step and one Reviewer maximum. Failure remains unavailable/KEEP-original; no formal preview/rollback.
- The existing `docs/phase5/memory_snapshot_v2.json`, evidence rules, tri-state Unknown handling, frozen Reviewer Prompt/parser and Gate rule are unchanged. Memory correction and confirmed-action sufficiency skipping stay disabled.
- Existing thresholds: Local0.5749072272810143, Global0.3332634086050418. Invocation budgets per Episode: Never0,Always20,Internal5,Full5. No recalibration in Phase5.9.
- H20/history2/prompt4096/response512/T=.4/top_p1/top_k0. Native tick cap21 and empty simplifications retained. Final raw Task Score includes negative values; success diagnostic remains done AND score100. No training penalty or reward transform.
- Rollout seeds0/1, per-task/turn/role request hashing extended from the frozen seed0 scheme. Simulator/task seed stays0, so initial task states match across policy/rollout seeds. The frozen selection seed/rank is separate from sampling seeds.

These are paper-aligned scalar limits with the already declared structured-prompt,9B and native1.3.0/Oracle-Like deviations. They are not exact paper reproduction. The historical Gate Design/model-version statements do not override these pinned current runtime files. `PAPER_ALIGNED_EVAL.md` remains unchanged.

## Cohort selected before generation

[Exposure audit](phase59/exposure_audit.json) inventories prior raw JSON/JSONL execution artifacts with hashes, native task/variation aliases, code-declared IDs and old selected representatives. Complete identity catalogs are not misclassified as executed episodes. No old trajectory feedback becomes Memory evidence.

Of197 lawful dev instances,39 IDs/35 normalized goal groups have prior real-use evidence. Excluding these, native aliases and declared old representative IDs leaves32 instances/8 never-run goal groups. The fixed selection uses family round-robin, SHA256 ranking and at most2 variations per fresh goal. Twelve instances from5 native task families/3 broad categories are frozen; the first5 cover all5 available families. Chemistry/lifespan have no untouched goal groups and are excluded rather than relaxed into the cohort.

Initial goal/catalog metadata was available previously. This independence claim concerns actual task/variation interactions, calibration/debugging trajectories and method-selection outcomes, not a perfectly pristine research process or unseen task templates. Physical layouts/templates and semantic near-duplicates remain possible. Eight goal clusters, not twelve fully independent goals, are the conservative analysis units.

The full12-task list and both seeds are frozen BEFORE Round1. Do not select tasks after seeing Full results, replace difficult tasks, or expand Memory. Final200 Test tasks and all training IDs are excluded. On each native initialization, goal identity is checked again; mismatch stops engineering validation instead of substituting a task.

## Two rounds and bounded execution

Round1: first5 frozen tasks × seed0 × four strategies =20 Episodes. Continue only after complete engineering audit of prompts/causality, budgets, scoring/termination, identical initial states, actual Call/Skip requests, strict KEEP state fingerprints and token/environment ledgers. Scores, success and Full advantage are NEVER acceptance conditions.

Round2: the same fixed12 tasks × seeds0/1 × four strategies =96 total Episodes. Reuse the first20 measurements; execute the remaining76 without repeating them. Different policies have different later states and calls. Never/Always omit NLL; only Full queries Memory. Every method shares its frozen Actor/Reviewer/WM/Prompt/settings.

At most2 simultaneous GPU workers, each with one persistent shared Actor/Reviewer instance. No vLLM/backend change or repeated model download. Workers retain their models during the short Round1 audit (up to10min), then reuse them for Round2. Each batch contains at most3 Task×Seed groups (12 Episodes); at most9 batches across both rounds/workers.

Hard run caps:4,000,000 LLM Tokens,2,650 model requests,12,500 native transition queries,96,000 state-audit capture bundles and8h wall. Each worker gets half the global caps (model cap1325,transition6250,capture48000), with per-batch500,000 Tokens/330 calls/1512 transition queries/12000 captures. Reservations use max-response512 and exact replay-length upper bounds; caps stop the run rather than alter a Gate or shorten verification.

Environment query units are explicit: formal initialization/action queries, worker initialization, replay commands and candidate commands are native TRANSITION queries. SDK state inspection and selected-RNG reflection are separately counted as capture BUNDLES; low-level Py4J RPC count is not claimed measured. All validation time is included in simulator latency. JVM/model setup and formal cleanup are recorded. Simulator transitions do not consume formal Actor turns but remain real additional environment access/cost.

Generation and transition reservations/receipts are persisted, along with per-step full raw traces and atomic completed-Episode checkpoints. `--resume` skips only verified completed Episodes under identical source/config hashes. Incomplete/unknown receipts and partial Episodes block silent retries and require engineering audit; raw partial costs are retained. There is no outcome-based retry. Unexpected errors stop both owned workers; only their own processes are cleaned up.

## Descriptive analysis fixed in advance

Report all Task×Seed×Policy rows, per-seed complete task means and mean ± sample SD of the TWO seed means (limited precision, no significance claim). Overall scores/SR are instance weighted. Do not average categories or unequal batches equally.

Pair Full–Never,Full–Always,Full–Internal at the same task/seed, retaining different realized trajectories. Provide score/SR differences, tokens, transition-query counts and latency. Descriptive95% percentile bootstrap intervals use2000 resamples,seed202659, with normalized-goal clusters resampled together (all variants and both seeds retained). Preserve the instance-weighted denominator within each resample. With only8 clusters and2 seeds, intervals are fragile, not proof of statistical significance or generalization. Show per-task contributions, positive-gain concentration and leave-one-task/goal-out sensitivity; no post-hoc removal of influential tasks.

For Full traces, re-evaluate the Internal Gate on the SAME recorded pre-action inputs and remaining budget. Log each External-added Call, its NLL/thresholds, evidence IDs, real trial feedback, Reviewer result and actual feedback. Keep Unknown and natural sufficiency positives visible. Acceptance of an action is not task correctness. Immediate actual-minus-trial score is one-step evidence only; no strict downstream Call/Skip utility is claimed without a separately validated paired continuation experiment. This phase does not silently run such an extra experiment.

Classify Reviewer KEEP/REPLACE/format failure, accepted-feedback/rejected/Unknown replacements, syntax/object/precondition/ambiguity rejection, and immediate score changes. Record replay rejects and formal-isolation failures even if scores look good. Full/Internal whole-episode paired differences are policy-level observations, not evidence that every individual extra WM call was useful.

## Commands

```bash
# Re-selection is deliberately refused if the frozen profile already exists.
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/prepare_independent_dev_eval.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' .venv-qwen35-clean/bin/python scripts/test_independent_dev.py

# Choose two genuinely free24GB GPUs; do not displace other users.
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/run_independent_dev.py \
 --model-dir checkpoints/qwen35_9b --gpus 6,7 --run outputs/phase59_independent_v1

# Only after engineering review of a clean between-Episode interruption:
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/run_independent_dev.py \
 --model-dir checkpoints/qwen35_9b --gpus 6,7 --run outputs/phase59_independent_v1 --resume
```

The supervisor verifies all13 local model/tokenizer assets against the previously pinned official checksums; the workers run offline. An optional online verify-only invocation encountered HF TLS EOF during preparation; no weights were downloaded or substituted. Local pinned SHA verification is the operational integrity check.

Raw traces remain under the ignored run directory; compact per-instance raw results, full decision signals, evidence examples and provenance are exported with the final report. Unit fixtures/Mock scores never enter research metrics. No training or final Benchmark experiment is part of this protocol.

No authoritative native difficulty labels were available; tasks are not retrospectively labelled easy/hard from scores. Use the isolated Python3.12 interpreter; system Python3.9 is incompatible with existing type hints.

## Recorded engineering repair before Round2

Round1 completed all20 Episodes. The first auditor incorrectly read request_seed from scored generation_record; scored metadata actually records actor_metadata.generation_seed. This is a post-run reader defect, not a sampling/Actor/Gate change. Original run/code/data remain preserved. The strict dual-schema fix rejects absent/conflicting seeds;6 new tests pass. Re-audit of all20 original byte-identical measurements passes:91/91 lookaheads and27 exact KEEP projections. Recovery imports cumulative ledgers and ALL20 results; no selected reruns. Remaining76 execute under the same frozen cohort/method/deadline. Safe stop/reload adds two model initializations, counted as engineering overhead. See phase59/engineering_repair.json.
