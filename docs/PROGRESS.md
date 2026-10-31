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
