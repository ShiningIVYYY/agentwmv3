# Current status

Phase2 COMPLETE for user-selected route B. Phase3 NOT STARTED; await explicit confirmation. See PHASE2_REPORT.md and INTERNAL_SIGNALS.md. Earlier entries below are historical; SpanUQ remains shelved/unavailable.

# agentwmv3 progress

- Phase 0: COMPLETE. Phase 1: explicitly authorized and PARTIALLY EXECUTED; acceptance BLOCKED by official source/checkpoint mismatch. Phases 2–7: NOT STARTED.
- Audit/docs completed; Git/core-isolation/test checks are in docs/audit/final_checks.txt. GitHub push is verified separately at handoff; see audit/git_push.txt when present.
- Source v2 is clean; inherited 87 tracked files/history; no v2 core changes, environments, processes, weights or outputs altered.
- Plan: IMPLEMENTATION_PLAN.md. Detailed evidence: REPOSITORY_AUDIT.md, SPANUQ_AUDIT.md, audit/.
- Baseline regression: 7 suites / 147 cases passed (one needed PYTHONPATH=.); protocol suite unresolved: 10 errors from missing ignored external assets. No complete benchmark or real model inference.
- Confirmed blocker for Phase 1: SpanUQ published config/API mismatch; layer convention unresolved. No checkpoint weights loaded.
- Remote: user supplied https://github.com/ShiningIVYYY/agentwmv3.git; use GitHub SSH identity ShiningIVYYY. No bare fallback requested.
- STOP at Phase 1: no Phase 2. Phase 1 authorized the independent environment, two pinned weight downloads and three backbone samples. No training/full evaluation is authorized.
- Next required work: obtain matching official probe architecture/provenance, strictly load and validate real SpanUQ outputs. No guessed/unused modules or dropped weights.

## Session restart

1. `ssh lab`, then `cd /data1/shn/agentwmv3`.
2. Read this file, IMPLEMENTATION_PLAN.md, SPANUQ_AUDIT.md; inspect `git status` and current branch/remote.
3. Phase 1 was explicitly authorized but remains unaccepted. Read PHASE1_REPORT.md and do not proceed to Phase 2.

## Paper-aligned evaluation planning follow-up

- Completed paper Section 4/C.1/C.4/Table S2/S3 and official evaluation source audit; pinned official commit 35a44f5c5647b1785ad23ddf0ffbd196b695bf86.
- Added PAPER_ALIGNED_EVAL.md and updated Phase 5–6 plan. This is documentation-only; existing configurations, runtime code and SpanUQ files were not edited.
- Science 200 IDs/file hash/order/task-variation mapping match; 61 source-train overlaps independently confirmed, leaving 2059. Live simulator/asset equivalence remains pending.
- New planned cohort: Science 20/history2/prompt4096; ALF 50/history2/prompt2048; Web fixed128/15/history2/prompt4096; common response512/T=.4/top_p1/no top-k restriction/seeds0–7. Actor remains Qwen3-4B.
- Separate dev/memory isolation and instance-weighted mean±sample SD planned; no runtime config, dataset split, model inference or full experiment implemented/launched in this follow-up.
- Existing phase/SpanUQ status above is not advanced by this evaluation planning task.
- WebShop follow-up risk: exact goal identity also depends on stable slot seed 1000+i; selecting 128 indices from the legacy single-seed catalog is insufficient. This remains a Phase 5–6 preparation requirement.

## Phase 1 handoff

- New .venv-spanuq: Python3.12.0/torch2.8.0+cu128/Transformers4.57.3; lock captured, pip check passed. v2/old vLLM untouched.
- Only pinned Qwen3-4B/probe assets downloaded; published LFS hashes verified.
- Official history: 4 commits, one model blob; HF original/current probe identical. No compatible implementation found.
- Strict load fails: 4 absent u_conditioner keys; PE [1,16384,512] versus [1,4096,512]. No uncertainty scores produced.
- Three real backbone samples passed token/offset/shape/finite/layer/repeat checks. 11 CPU tests passed. Overall smoke exits 2; Phase 1 NOT accepted.
- Correction: official batch producer hooks blocks [21,22,23] = HF tuple [22,23,24]; only evaluator on-the-fly is off by one.
- PHASE1_REPORT.md and docs/phase1 contain commands/results/blockers. PAPER_ALIGNED_EVAL.md unchanged. Stop before Phase 2.

## Phase 1B handoff — complete investigation, route selection pending

- Conclusion B: no official compatible code/update found; GitHub API rate-limited, public issue/PR HTML shows zero results and no release; HF unchanged. No Issue posted.
- Full 134-vs-130 tensor comparison saved; PE length difference is deterministic (full/prefix max error 0), but conditioner forward remains unpublished. Checkpoint unchanged; strict=True still fails.
- New documents: SPANUQ_COMPATIBILITY.md, INTERNAL_SIGNAL_FALLBACK.md, SPANUQ_ISSUE_DRAFT.md. Small real evidence in docs/phase1b.
- Standalone fixed-field NLL prototype only: three synthetic prompts with real Qwen3-4B; 10 CPU tests pass. BF16 raw-vs-rescore equivalence fails 0.1-nat tolerance (exit 2); FP32 diagnostic passes (exit 0). Scores are surprisal, not error probabilities.
- Existing hidden pipeline, strict loader, v2, benchmark configuration and PAPER_ALIGNED_EVAL.md remain unchanged. No training or complete benchmark.
- STOP: await user choice before Phase 2; do not silently enable fallback or reuse SpanUQ thresholds.

## Phase 2 final handoff

- Official provider hf_generation_raw_bf16_v1: frozen Qwen3-4B, BF16 generation/raw-logit FP32 reduction. No rescore/FP32 scores mixed.
- Reused original Agent.count/generate and Science.run with default-off opt-in branches; baseline CLI/prompt unchanged.
- Three disjoint source-train instances (sciworld_349/1709/2997), 15 real actions; missing/parse/alignment errors0, repeat NLL drift0. Native action failures and a confidently incorrect premise are retained.
-117 passing unit cases, causal/arithmetic trace audit passes. Extra online forward0; mean extraction4.55ms, peak7.73GiB. Separate numerical validation: one extra generation/one teacher forward, re-forward max drift.4873, not used for signals.
- New docs and compact traces under docs/phase2; full outputs/phase2_diagnostic_v1 retained. No thresholds, Memory, Gate or full benchmark.
- STOP before Phase3. Preserve PAPER_ALIGNED_EVAL.md, v2 and old environments.
