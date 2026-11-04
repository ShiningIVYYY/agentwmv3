# Current status

Phase5 COMPLETE engineering validation; awaiting user confirmation before Phase6. Read [PHASE5_REPORT.md](PHASE5_REPORT.md), [WM_FAILURE_AUDIT.md](WM_FAILURE_AUDIT.md), [SMALL_SCALE_EVAL.md](SMALL_SCALE_EVAL.md) and [PAPER_ALIGNED_EVAL.md](PAPER_ALIGNED_EVAL.md). All work is in agentwmv3; agentwmv2 remains clean.

- 280 real dev decisions in20 complete episodes;107 Call/173 Skip;315 actual WM forecasts verified against owned server counters. All5 initial-state/first-Draft parity checks pass.
- 254 CPU/regression cases and exact causal/score/cost/NLL/budget/evidence replay passed. Full large raw trajectories/source snapshots stay under outputs/; compact provenance/results in docs/phase5.
- Actor Qwen3-4B BF16/nonthinking, accepted raw NLL backend/base configuration unchanged. Separate fixed Word2World Qwen2.5-7B WM and JSON3-round policy retained.
- q0.75 thresholds Local0.12081397919163489 / Global0.053253251821409955 from273 decisions/30 dev goal groups.5 held-out development groups for episodes, seed0 only. Distribution estimate, not stable optimal thresholds.
- Shared train-only Memory v2:3 cards/same54 cases. Narrow coverage; natural sufficient confirmation absent. Correction and confirmation Skip remain disabled.
- Mean final native Task Score: Never-9, Always-29, Internal-29, Full-29; all success rates0/5. Reviewer valid JSON278/278 but84 semantic/action violations. Model capability/scientific benefit are NOT established by engineering tests.
- Online total1,628,427 tokens/873 model calls/1212.106s episode wall; Actor peak allocated8.418GiB, dedicated WM reservation separate. Offline/setup/calibration costs disclosed.
- Owned Phase5 WM/environment services stopped after audit; no auto waiter or queued experiments. Other jobs/services untouched. Original pause was respected until explicit user resume.
- No training, final200x8 sweep, ALF/WebShop evaluation or Phase6 implementation. Final commit/push recorded in delivery.

Historical reports are phase-specific records, not current authorization.
