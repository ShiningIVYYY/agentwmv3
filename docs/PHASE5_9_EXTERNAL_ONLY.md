# Phase5.9 supplement: Experience-Only Gate

Status: PREREGISTERED; no result yet. User requested the missing pure External-rule arm alongside Never−0.5417,Always3.3333,Internal−10.8333,Full−19.375. Only24 new episodes are authorized here; original96 are retained and not regenerated. This is retrospective development ablation, not fresh independent generalization or Phase6.

## Frozen comparison

Actual source run `outputs/phase59_independent_v2`, source commit `830d2abceac788c5e8e69813c9fde5941fac7994`. Its exact `evaluation/independent_dev.py` is preserved as `docs/phase59_external/frozen_runtime.py`, SHA256 `1bfc3f5c2665be33c18c90080194f3351cd4071e016f84aa83a51348257ae791`, and used directly. Imported method/model/Memory/Paper sources are checked against the original method hashes. Later SR changes and compact-JSON Reviewer are excluded.

Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, shared frozen Actor/Reviewer weights; BF16/HF5.22.2/Torch2.8, original nonthinking template, unchanged structured Draft prompt. Original THREE-LINE Reviewer, response512, one round. No Phase5.10 compact JSON repair. Native ScienceWorld1.3.0, separate-JVM Reset+Replay, one-step candidate simulation. Same exact12 Task/Variation IDs and rollout seeds0/1, H20/nativecap21/history2/prompt4096/T=.4/p1/k0/simulator seed0. Raw final Task Score, negatives retained; original success definition retained.

Frozen Memory `docs/phase5/memory_snapshot_v2.json` and its3cards/54cases, BM25 retrieval and conservative evidence rules unchanged. Policy is existing `RuleGate(policy='experience')`: request WM only on reliable evidence-backed action conflict, subject to budget5 per episode, same as Internal/Full. Unknown abstains; no memory-only action correction or sufficiency skip is enabled. This is an External invocation Gate, not an autonomous rules-only Actor. Actor still produces every Draft; Reviewer acts only after verified Lookahead.

No Internal NLL extraction or extra forward is performed. Existing Gate threshold numbers remain as identity/config fields but cannot trigger this policy. No threshold calibration, experience expansion or training.

## Bounded execution and audit

Two RTX4090 workers, max24 completed episodes; limits650k totalTokens/650model calls/3500native transition queries/18000capture bundles/1h, each worker half caps. Model loaded once per worker, instances released in finally, owned process groups supervised. Resume can reuse completed episodes only; partial attempts remain charged and require receipt audit.

Before measurement, `scripts/prepare_external_only.py` verifies actual baseline profile equality and writes independent `configs/evaluation/phase59_external_only_v1.json`. `scripts/run_external_only.py` calls the exact old runtime and records full generation/decisions/feedback/cost. Final audit checks initial-state parity, causal order, raw receipts and consistency, no Internal computation, External evidence links,5-call cap, source hashes, per-task-seed metrics and paired differences against the four preserved arms. Invalid Reviewer outputs retain INVALID/fallback records as in Phase5.9, not silently repaired.

Commands from repository root:

```bash
.venv-qwen35-clean/bin/python scripts/prepare_external_only.py
.venv-qwen35-clean/bin/python scripts/run_external_only.py --run outputs/phase59_external_only_v1 --gpus 2,6 --model-dir checkpoints/qwen35_9b
```

No Qwen2.5 WM, SR model/new cohort, final200-test rollouts, extra model training or method tuning belongs to this supplement. Results and final commit will be added after the run and audit.
