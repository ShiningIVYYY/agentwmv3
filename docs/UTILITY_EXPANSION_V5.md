# Utility-label and independent Validation expansion v5

Status: preregistered;125 native identities and5 replay controls pass;15 CPU selection/provenance tests pass. Model startup is recorded separately.
User authorized continuing the complete approved Probe training plan and specifically
expanding reliable Help examples and independent validation data. Work is limited
to agentwmv3. No final Test, fresh independent-Dev policy evaluation or Phase6.

## Frozen method

Shared frozen Actor/Reviewer: Qwen/Qwen3.5-9B,
revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, BF16,
activation-pce-v2, no thinking, history2, prompt2048/response512,
temperature0.4/top_p1/top_k0, H50. Actual-generated-token alignment v1 and
hf_qwen35_generation_raw_bf16_v1 remain unchanged. Capture blocks10/15/21,
response-only post-token features,4096 dimensions. Same independent native
ALFWorld Reset/Replay, one-step lookahead, fixed Reviewer prompt/guard,
frozen31-case/two-card Memory, existing Gate and NLL thresholds.
No Qwen2.5 or foundation-model training. No new dependency/model download.

## Fixed cohort and role isolation

Reuse all prior9628 true decision cases and782 repeated-state labels. Preserve
100 existing Train goals,12 Validation,12 Calibration, all previously exposed
roles and140 final valid_seen identities. Do not regenerate old paired episodes.

Remaining217 untouched canonical goals span only simple placement, cleaning
and two-object placement. Before new native/model results, choose one fixed
137-goal metadata-only batch:80 new Train,40 new Validation,5 native controls,
12 reserved fresh independent-Dev. Remaining80 goals stay unassigned.
The reserved fresh-Dev goals are not reset or evaluated in this run.

An initial metadata role-allocation defect made the three-family round-robin
cycle coincide with the2Train/1Val role cycle. BEFORE any new native/model
outputs, amend allocation proportionally WITHIN THE SAME137 goals. Archive the
superseded draft and before/after hashes/family counts in allocation_amendment.json.
No goal is added, removed or chosen from observed outcomes. Old roles are untouched.

Task ID, canonical goal group and actual native-goal checks exclude prior/test
identities. Task templates/layouts may still overlap. The reduced family coverage
must not be described as broad cross-family generalization.

## Sampling and labels

1. Old Train densification: select at most4 unlabelled, distinct Actor-context/action
   states per old Train goal where the actual source Reviewer accepted a change.
   There are1147 raw candidates over94 old tasks before context deduplication.
   Use deterministic temporal-window scheduling, excluding every previously
   labelled context. Do not filter by source success or future utility.
2. New Train: paired Never/Always source episodes at Actor seed1. Retain existing
   temporal/public-risk base selection and exact common-state divergence; add
   at most4 genuine accepted-revision candidates independently of task outcome.
3. New Validation: paired fixed source policies, then uniform sampling without
   replacement of up to4 unique pre-action contexts per0-9/10-29/30-49 window.
   Do not read feedback, Reviewer output, task success or utility to select states.
   Record pool size/inclusion probability. Fixed40 tasks all remain in denominator.

TRAIN enrichment is explicitly post-decision offline scheduling. The cached
features were captured BEFORE real current feedback/review and never include
label outcomes. Accepted change does not itself constitute a Help label.
Validation sampling is outcome-blind, but mixes fixed source-policy trajectories;
it is not a claim of natural online prevalence. Prior weighted Validation and new
uniform Validation remain separately identifiable.

Actual labels use the existing same-state intervention, fixed Reviewer seed0,
Never-WM continuation seeds1/2/3, extending to4/5 whenever any nonzero difference
occurs. Help requires at least3/5 success improvements and no regressions; Harm
is symmetric. All-zero outcomes are Neutral. Mixed/incomplete outcomes remain
Uncertain/Unknown. No-change requires an actually matching continuation and
never implies the original action was correct. One state is one labelled example,
not five independent observations. Do not resample until a desired label appears.

## Bounded resources and accounting

Current two free4090 GPUs4/6, maximum4 authorized; do not displace others.
NEW batch upper limits:72h,64M tokens,120k model requests,600k native transitions,
660k state receipts. Phase token caps:12M old-state densification,32M new Train,
20M new Validation. Native preflight is included in new caps. Stop below50GiB
free disk or when new-run bytes exceed90GiB. No unbounded expansion/retry.

Previous confirmed cumulative48,699,305 tokens and two historical unknown output
receipts remain immutable. Preserve the1,890-token conservative charge separately
from actual tokens. New budget is an explicit extension, never a reset of prior
costs. Report new/cumulative costs, lookahead and replay queries, feature-forward
compute, model loading, and unsuccessful work. Raw outputs/caches/weights stay
outside Git; publish compact audits and metrics only.

## Conditional linear training

Complete all fixed sampling stages before checking readiness. Use the unchanged
linear_utility_v1 profile: at least500 Train states,50 Help over10 Train goals,
200 Validation states/20 Help,50 Calibration states. These are starting safeguards,
not statistical guarantees. Only linear Help probes, three layers, all3 seeds,
Train-only normalization/weights and grouped learning curves. Missing labels stop
fitting explicitly. No fake Harm head, missing-risk-zero interpretation or automatic
online deployment. Independent policy evaluation depends on future readiness;
no task-performance claim follows from collection or offline AP.

## Reproduction and lifecycle

Set ALFWORLD_DATA and ALFWORLD_PYTHON using the existing isolated environments.

```sh
.venv-qwen35-clean/bin/python scripts/prepare_utility_v5.py
# Existing preregistration must never be overwritten or redrawn.
# The archived initial allocation used the explicit pre-outcome amendment:
# .venv-qwen35-clean/bin/python scripts/prepare_utility_v5.py --amend-fixed-goal-roles
.venv-qwen35-clean/bin/python scripts/test_utility_v5.py
.venv-qwen35-clean/bin/python scripts/preflight_utility_v5.py --run outputs/utility_expansion_v5_preflight
.venv-qwen35-clean/bin/python scripts/run_utility_expansion_v5.py --run outputs/utility_expansion_v5 --auto-publish
```

Fresh-run guards, hash-frozen manifests, per-task/source/counterfactual receipts,
owned PID cleanup and progress/error/publication files prevent blind restarts.
Verify first real labels before leaving bounded background execution. Completion
publishes UTILITY_EXPANSION_V5_REPORT.md and compact results. An interrupted run
requires an audited new recovery receipt; do not silently retry or reset budgets.
