# agentwmv3 progress

- Phase 0 audit, implementation planning and checks: COMPLETE. Phases 1–7: NOT STARTED.
- Audit/docs completed; Git/core-isolation/test checks are in docs/audit/final_checks.txt. GitHub push is verified separately at handoff; see audit/git_push.txt when present.
- Source v2 is clean; inherited 87 tracked files/history; no v2 core changes, environments, processes, weights or outputs altered.
- Plan: IMPLEMENTATION_PLAN.md. Detailed evidence: REPOSITORY_AUDIT.md, SPANUQ_AUDIT.md, audit/.
- Baseline regression: 7 suites / 147 cases passed (one needed PYTHONPATH=.); protocol suite unresolved: 10 errors from missing ignored external assets. No complete benchmark or real model inference.
- Confirmed blocker for Phase 1: SpanUQ published config/API mismatch; layer convention unresolved. No checkpoint weights loaded.
- Remote: user supplied https://github.com/ShiningIVYYY/agentwmv3.git; use GitHub SSH identity ShiningIVYYY. No bare fallback requested.
- STOP: await explicit user approval before Phase 1. Do not download weights, install inference dependencies, train or run full evaluation on resume.
- Next authorized work after approval: resolve official SpanUQ compatibility/provenance, then isolated environment and real minimal inference only.

## Session restart

1. `ssh lab`, then `cd /data1/shn/agentwmv3`.
2. Read this file, IMPLEMENTATION_PLAN.md, SPANUQ_AUDIT.md; inspect `git status` and current branch/remote.
3. Do not infer Phase 1 permission from the existence of these plans. Obtain/locate explicit confirmation.
