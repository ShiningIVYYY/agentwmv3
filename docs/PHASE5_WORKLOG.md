# Phase 5 worklog

Status: Phase5 COMPLETE engineering validation; awaiting user Phase6 confirmation. Historical resume entry: User reported free GPUs. Owned environment PID1470273 / WM PID1470274 on GPU1 / episode PID1472573 on GPU0. Frozen5-task x4-policy cohort is running detached; see outputs/phase5_episodes_v1/progress.json. No automatic GPU waiter; no Phase6 authorization.

## Completed work
- Baseline commit: 7a039fa78688e1caf36451dd2d3e599c215d6dfe. Phase 5 changes are saved in the working tree; final validation/commit/push remain pending.
- Phase 4 WM failures: both final failures were predicted by WM; the JSON reviewer selected absent/misnamed objects. Seven real independent development reset/action probes retained in outputs/phase5_setup/wm_failure_audit.json. No answer-specific repair.
- Independent paper profile implemented: H20 / SDK21, history2, input4096 / response512, temperature0.4 / top_p1 / no top-k, official task-turn request seeds. Qwen3-4B structured draft / WM differences remain explicit. Old service36001 retained.
- Horizon probe: exactly20 real actions allowed; native_done=false / horizon_truncated=true;21st action blocked.
- Initial canary failed on observer URL-keyword handling before generation or environment actions. Fixed; two-step canary v2 passed. Logs retained.
- Calibration completed: outputs/phase5_calibration_v1;30 legal development tasks /273 real scored decisions, no WM. No missing fields. Pearson local/global0.893067, Spearman0.898846. No outcome-based selection.
- Frozen distribution thresholds, q0.75: local0.12081397919163489 / global0.053253251821409955. Config: configs/gates/dev_distribution_phase5_v1.json. Distribution-only estimate, not proven stable across seeds.
- Frozen new memory: outputs/phase5_setup/memory_v2.json;3 cards / same54 TRAIN cases. Added narrowly scoped same-room look-around fact supported by2 independent episodes. No new LLM calls, no dev/test feedback included, old Phase3 snapshot unchanged.
- Independent frozen plan: outputs/phase5_setup/development_plan.json. Calibration30 goal groups; validation5 distinct held-out groups. Shared policies never/always/internal/full, seed0, H20; invocation budgets0/20/5/5 configured independently of thresholds.
- Paper metadata-only profile audit retained in outputs/phase5_setup/paper_profile_audit.json.

## Pause and remaining work
- All GPUs became occupied by unrelated jobs. No Phase5 WM service was started and no complete five-task/four-policy cohort was run.
- User explicitly requested waiting for their next instruction. Do not poll/schedule GPU work automatically.
- Owned empty profile service36051 (PID892759) was stopped on pause; original36001 and other jobs were untouched. Restart owned profile service only when resuming.
- Calibration process907573 already finished and released its GPU.
- Remaining: latest CPU regression rerun, complete20 development episodes when authorized/resources available, causal/cost/WM/external audits, final reports/README/plan, Git commit and push.
- Full raw data and executed-source snapshots remain under ignored outputs/. No model weights, large logs or private WM context should be committed.
- No agentwmv2 modifications, new training, full200x8 sweep, or Phase6 authorization.

## Resume verification
-37 CPU profile tests and217 prior regression cases passed (254 total).
- Original pause was honored until user requested continuation. Original36001 service and unrelated jobs untouched.
- Never5 complete episodes have0 actual WM calls; independent server request and full causal replay audit will run after the20-episode cohort finishes.
- Read-only diagnostics found11 low-Local-NLL historical closed-route conflicts; no natural new-card sufficiency case. No outcome-based rule changes.

## Completion
Phase5 engineering validation complete; stop before Phase6.20 complete development episodes/280 decisions,107 Call/173 Skip,315 server-verified WM forecasts.254 CPU tests and offline full real replay pass. See PHASE5_REPORT.md/SMALL_SCALE_EVAL.md for actual results/limits. Owned services stopped; no queued work. Original pause and resume history above retained; this entry is the latest status.

## Reproduction entrypoints (no new runs authorized)
- Environment: $SCIENCE_PYTHON scripts/serve_scienceworld_profile.py --port 36051 (existing installed SDK1.2.3). This wraps the existing package read-only; Phase7 fresh-server setup is not yet verified.
- WM exact command/environment: docs/phase5/wm_service_resume.json; model weights remain ignored/read-only.
- Calibration actual job: outputs/phase5_setup/calibration_job.json; frozen compact points/plan in docs/phase5.
- Main actual bounded command: docs/phase5/episodes_job.json / PHASE5_REPORT.md. Use new output Run ID for reproduction; docs/phase5/development_plan.json and memory_snapshot_v2.json preserve the exact frozen inputs.
- CPU contracts: CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-spanuq/bin/python scripts/test_scienceworld_validation.py.
- Canonical captured-cohort replay: CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-spanuq/bin/python scripts/audit_phase5.py; saved real server metrics permit replay after shutdown, no model calls. Full raw artifacts must remain available.
- Compact reports: PYTHONPATH=. .venv-spanuq/bin/python scripts/report_phase5.py.
- Final raw trajectories/model calls/native receipts/source archive: outputs/phase5_episodes_v1. Historical failed canary preserved under outputs/phase5_canary_v1, successful canary under v2. No failed data enter metrics.
