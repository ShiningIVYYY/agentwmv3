# Frozen ALFWorld50-task development evaluation

User authorized50 Episodes PER policy: Never/Always/Internal/External/Full, total250, rollout seed0. This is an expansion of the completed3-task diagnostic, not Phase6 or the official140 valid_seen×8-seed test. No outcome-based tuning, training, learned WM or Qwen2.5 online path.

## Frozen method

Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, BF16/HF5.22.2/Torch2.8.0, shared Actor/Reviewer weights, nonthinking official template, T=.4/top_p1/top_k0/repetition1, response512/input2048/history2/H50. Same structured Actor prompt, original three-line Reviewer/one round/512 cap, native independent-process one-step Reset/Replay. Same budgets Never0,Always50,Internal5,External5,Full5. Same raw-generation TokenNLL provider, Local=max(field means), Global=all response text including labels/Action excluding prompt/EOS/padding. No extra backbone forward.

ALF thresholds remain Local0.5448528666853629/Global0.32670785406992164, selected earlier from48 calibration decisions. NO recalibration on the3-task outcomes or new50 tasks. Frozen31-case/two-card ALF Memory; no synthesis, correction, Judge or online writes. No new Reviewer validity guard, JSON constraint or prompt repair. The legacy ScienceWorld word in the shared Action placeholder is explicitly retained for method parity, as documented in the pilot. Actor/Reviewer errors remain measured outcomes; format fallback is explicit.

## Independent cohort

Metadata-only selection from cached supported solvable TRAIN games. Exclude all final140 IDs and canonical task-parameter goal groups; additionally exclude four Memory tasks, six calibration tasks, three prior evaluation tasks and both preliminary-probe goal groups. Reserve six additional goal-disjoint validation tasks, one per family, for physical replay controls. Select50 further unique goal groups by predeclared SHA256/family round-robin ordering, one fixed game per goal. No reward or success is read to choose the cohort. Fixed task IDs/game/metadata hashes/native seeds are in docs/alfworld_dev50_v1/split_manifest.json. Task templates/layouts and unknown model-pretraining exposure are not proven disjoint.

Before model generation: reset all56 selected/validation identities, record public native Goal/initial-state hash and actual chat-template prompt length. Check native Goal uniqueness and exclusion from prior13 observed goals. Six replay tests execute controlled invalid actions on the DISJOINT validation tasks only, compare independent-process candidate with actual formal transition, public state/admissible list/private fact digest, and check formal invariance. No final valid_seen reset; no evaluation feedback enters Memory or calibration. Private hashes/facts are audit-only, never Actor/Gate/Reviewer input. Test outputs are not evaluation success metrics.

## Resource budget and durable execution

At most4 GPUs authorized; one persistent9B replica per actually free GPU, shared Actor/Reviewer within worker. After CPU preflight, GPUs0/2/3/4 each had more than22000MiB free; launch uses those four cards. Earlier availability had allowed only GPU0; no method was changed when more cards became available. No other job/service is displaced. Expected wall time based on the tiny pilot is roughly4–6 hours on four4090s (about15–20 hours if serial), highly task/server-load dependent; this is an estimate, not a promise.

Global hard bounds:36h original calendar deadline,32M LLM tokens,16000 model calls,120000 native initialization/replay/candidate queries and140000 state receipts (including passive inspections). Preflight queries are charged once; original calibration/construction costs remain separately recorded historical reusable costs. Each worker saves a five-task batch receipt (25 Episodes with all five policies) and caps batch usage at4.1M tokens/1600 calls/12000 transitions/14000 state receipts. Caps halt rather than alter methods or silently shorten episodes.

Each step persists its trace and actual cost receipts; each completed Episode is immediately audited with the accepted strict auditor. A run lock and live-worker check prevent duplicate launches. Explicit --resume preserves checked complete Episodes and all previous-attempt costs, restarts only partial Episodes from their original task/seed in a new attempt directory, and inherits the original deadline. Unknown in-flight receipts block resume for audit. No automatic outcome-based retry or favorable-task replacement. STOP/worker failure/deadline stops owned workers and native processes; errors preserve logs and do not become fabricated zero-score results.

The supervisor runs detached, survives local SSH/screen disconnection, writes progress.json and partial per-Episode artifacts, and closes owned model/native processes after completion/error. Successful250-Episode completion triggers a CPU-only full audit and final report/results. --auto-publish commits/pushes only its new report/results if the worktree is clean; network/conflicting edits leave publication.json=needs_review and retain all scientific results. No constant assistant monitoring is required.

## Actual commands

Use existing isolated model/CPU environments; no dependencies or weights are installed/downloaded:

```bash
"$MODEL_PYTHON" scripts/test_alfworld_dev50.py
"$MODEL_PYTHON" scripts/prepare_alfworld_dev50.py --data-root "$ALFWORLD_DATA"
CUDA_VISIBLE_DEVICES='' "$MODEL_PYTHON" scripts/preflight_alfworld_dev50.py --env-python "$ALFWORLD_PYTHON" --data-root "$ALFWORLD_DATA" --run outputs/alfworld_dev50_preflight_v1
"$MODEL_PYTHON" scripts/run_alfworld_dev50.py --run outputs/alfworld_dev50_v1 --gpus 0,2,3,4 --env-python "$ALFWORLD_PYTHON" --data-root "$ALFWORLD_DATA" --preflight outputs/alfworld_dev50_preflight_v1 --auto-publish
```

The actual launch uses a detached subprocess with stdout/stderr redirected to outputs/alfworld_dev50_v1.log. Check outputs/alfworld_dev50_v1-supervisor.json, outputs/alfworld_dev50_v1/progress.json, attempt000/worker*/active_episode.json and ledgers. The outer detached wrapper records finalization/publication failures even when all model Episodes completed. Do not launch another copy while active. After a verified safe stop, add --resume to the same command; never reset an unknown receipt or extend the deadline silently.

## Final analysis

Native won Success Rate is primary;100×won is only auxiliary and must not be confused with ScienceWorld partial Task Score. Equal task-instance aggregation, per-task paired policy differences and per-family counts; one seed cannot provide across-seed sample SD. Wilson intervals are descriptive and do not establish independence/generalization. Report formal environment steps separately from lookahead/replay/passive queries; all Actor/Reviewer tokens, NLL, retrieval/Gate/simulation latency, valid/nonadmissible revisions and rejected/missing signals remain visible. Monetary/FLOP cost is not fabricated.

Full-vs-Internal direct current-Full-state shadow decisions and whole-trajectory comparisons are separate. Action revision alone is not positive long-horizon causal utility. Simulator access is oracle-like, not learned-WM prediction; an extra Reviewer-inference/no-feedback control and matched-budget Random/Periodic remain unrun. No threshold, Memory or Reviewer change is allowed in response to these50 outcomes. Final report: docs/ALFWORLD_DEV50_REPORT.md; raw logs remain under the unique run directory, compact results are published only after complete strict audit.
