# Legal utility-training expansion v3

Status: implementation/preflight for a fixed new training batch, not new task-performance evidence.

## Scope and frozen components

The user authorized additional lawful training samples after V2 found4 repeated-help candidates across3 training tasks. Reuse the original36 training goals,2651 real feature cases and229 state records without spending model calls to recreate them. Add24 new canonical training goals. The original12 Probe Validation,12 Gate Calibration,12 exposed independent-Dev and final140 valid_seen tasks retain their roles. Never move previously evaluated Dev/Test tasks into Probe Train.

Actor/Reviewer remain the same frozen Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, shared within each worker. No fine-tuning, RL, new Judge or WM model. BF16, raw-generation NLL backend, `activation-pce-v2` prompts/parser/guard, H50/history2/input2048/output512/temperature0.4/top-p1/top-k0 remain unchanged. The WM remains independent native ALFWorld0.4.2/TextWorld1.7.0 Reset/Replay. Experience Memory, Gate rules and NLL thresholds are unchanged; original Qwen2.5 services are not used. Only eligible small probes may train conditionally; no online deployment or Phase6 follows automatically.

## Dataset isolation and new identities

The new split copies prior role assignments exactly and appends24 TRAIN goals plus5 separate native-validation controls. Selection is a fixed SHA256 family round-robin over the remaining291 groups, excluding all prior77 reserved goals, older exposed/Memory/calibration goals, and final140 canonical Test goals. Freeze the complete new24 task list before any model output; finish the batch rather than stopping as soon as the fifth positive appears.

The selected new training family counts are6 simple placement,6 cleaning,6 cooling,1 heating and5 two-object placement. Heating and light-object coverage are limited under these exclusions; do not relax goal isolation to create a favorable cohort. Templates/layouts can overlap despite canonical-goal and exact native-goal checks.

CPU preflight resets only the24 new TRAIN tasks and5 new controls, checking their actual native goals against all existing identity receipts, model-template initial length and environment version. Five independent controls include an invalid action in the replay prefix and compare exact subsequent public transitions/private fact digests, score, termination and formal counters. No final Test, prior independent-Dev or Val/Cal task is reset by this preflight. Observable/private-digest agreement is not a claim of full hidden RNG serialization.

## Label aggregation correction

Corpus reuse exposed a deterministic V2 bug: `zero_count == 3` treated some mixed five-seed outcomes as Neutral, contrary to the stated all-zero rule. Five existing records are affected, e.g.[0,0,1,1,0] or[1,0,0,-1,0]. They should be Uncertain, not hard negatives. The corrected derived prior counts are **209 Neutral /16 Uncertain /4 Help**; all4 positive candidates and the prior stop decision are unchanged. No model was trained on the incorrect labels.

Keep original raw state records, vectors, features and historical results unchanged. `docs/utility_expansion_v3/label_corrections.json` records every correction. The new `utility-label-v3-all-zero-neutral` aggregator is imported only by the v3 path. The actual counterfactual execution function is AST-identical to V2; the Actor/Reviewer/sampler and positive criterion do not change. Exhaustive three-/five-seed tests ensure Neutral requires all-zero outcomes and mixed signs/weak evidence remain Uncertain.

The positive rule remains: changed action,5 actual paired continuation seeds, at least3 positive deltas and no negative deltas. An unchanged action needs an actual equal-state/equal-seed continuation check and is labelled zero intervention, never action correctness. A single lucky improvement is not a hard positive. The finite-seed rule is not a statistical significance guarantee.

## Sampling and paired task attribution

Reuse the exact V2 procedure: same-task Never/Always source Episodes use Actor seed1; one-step preview and Reviewer seed0 are fixed. Sample early/middle/late states using one uniform and one public-difficulty-weighted candidate per available window. Probe Train may additionally mine the first exact shared-state action divergence and up to2 actual revisions in Always-success/Never-failure tasks. Task success selects offline candidates, not positive labels. Val/Cal never use this outcome mining.

Each candidate is independently reconstructed to its full actual prefix/fingerprint. Skip executes the original draft; Call executes the real preview's fixed Reviewer action, followed by the identical frozen Never-WM policy and remaining horizon. Compare seeds1/2/3, extending4/5 whenever any nonzero difference appears. Reuse an actual observed Never seed1 suffix only when it really matches the specified Skip policy; an Always suffix is never substituted. Source capture stores detached features before any current preview/feedback. Future outcomes can supervise labels but cannot enter the feature API.

Source schemas retain the `utility-v2/...` case-ID prefix for compatibility; new task IDs and immutable run/feature paths prevent collisions. Do not duplicate old cases as new independent evidence or confuse states from the same task with independent tasks.

## Sequential stages and no deployment

1. Fully collect24 new TRAIN task pairs and selected state labels, retaining all outcomes and budget-skipped Unknowns.
2. Merge with the frozen, corrected old36-goal corpus. Apply the unchanged>=5 repeated-Help states across>=3 TRAIN-task criterion.
3. Only if qualified, collect the already fixed12 Probe Validation and12 Gate Calibration task pairs, without outcome enrichment. New independent-Dev/Test remains untouched.
4. Train only small heads that independently meet the original positive/negative/task/count requirements. Report unavailable heads explicitly. Three layers/Linear-MLP-EMA/three seeds and Train-only normalization are unchanged.
5. Mark all classifier artifacts **deployment_allowed=false**: stratified/outcome-enriched sampling does not directly yield calibrated natural-traffic probabilities or validated online Gate thresholds. No full baseline suite or learned-Gate evaluation is launched by this collector.

If the merged pool remains insufficient, stop with actual receipts. Do not lower the positive rule, relabel partial progress as task success, pick favorable Val tasks or automatically redesign Reviewer. Counterfactual utility remains conditional on fixed Reviewer seed0 and Never continuation; multi-call synergy may need a different explicitly defined future study.

## Resources, costs and commands

Use three free cards5/6/7, maximum4 authorized. New-run bounds:36h,36M tokens,60000 model calls,300000 transitions and360000 state receipts. New TRAIN receives at most half; optional Val/Cal share only the recorded remainder. Reserve the entire worst-case state/confirmation cost before starting. The V2 sunk cost is reported separately and once in the combined V2+V3 accounting; imported data does not incur new model calls. New CPU preflight queries, capture forward time/memory and conditional CPU probe-training time are separately retained. No weights, caches or large raw trajectories go into Git.

```bash
# Set existing ALFWORLD_DATA and ALFWORLD_PYTHON first; no new environments/packages.
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_utility_expansion.py -v
# One-time reservation/audit/preflight already executed; Run IDs cannot be reused.
nohup env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-qwen35-clean/bin/python -u scripts/run_utility_expansion.py \
  --run outputs/utility_expansion_v3 --auto-publish \
  > outputs/utility_expansion_v3.log 2>&1 < /dev/null &
```

Implementation: `prepare_utility_expansion.py`, `audit_utility_prior.py`, `preflight_utility_expansion.py`, `run_utility_expansion.py`, `evaluation/utility_labels_v3.py` and `utility_labeling_v3.py`. Existing V2 execution helpers and bounded executor are reused. Startup freezes new code/config/Git, prior corpus receipts and source/feature SHA256 hashes. Source checks and strict loading prevent silent model/version drift. Native environments are individually owned/cleaned; controller exceptions preserve raw evidence and stop owned workers. No blind resume or duplicate run is allowed.

Authoritative progress/errors: `outputs/utility_expansion_v3/{progress,error,feasibility,publication}.json`. Raw old data stay in `outputs/utility_sampling_v2`; new data/weights stay in `outputs/utility_expansion_v3`. The controller exports compact actual results to `docs/UTILITY_EXPANSION_V3_REPORT.md` and `docs/utility_expansion_v3/results/summary.json`. Git publication failure is recorded separately without changing a completed scientific run into a failed experiment. Check startup once, then leave it bounded in the background as authorized.
