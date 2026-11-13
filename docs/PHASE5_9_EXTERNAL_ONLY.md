# Phase5.9 supplement: Experience-Only Gate

Status: COMPLETE;24/24 real Episodes and source/cost/causal audits pass. User requested the missing pure External-rule arm alongside Never−0.5417,Always3.3333,Internal−10.8333,Full−19.375. Only24 new episodes are authorized here; original96 are retained and not regenerated. This is retrospective development ablation, not fresh independent generalization or Phase6.

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


## Completed results

|Policy|Mean Task Score|SR|Total Tokens|WM Lookahead calls|
|---|---:|---:|---:|---:|
|Never, historical|−0.5417|0/24|163799|0|
|Always, historical|3.3333|1/24|424279|371|
|Internal, historical|−10.8333|0/24|219635|80|
|Full, historical|−19.3750|0/24|212548|80|
|Experience-Only, NEW|**−2.7917**|**0/24**|**173623**|**14**|

Seed0:9.4167,Seed1:−15.0; across2 seed means sampleSD17.2652. Goal-cluster descriptive bootstrap95%interval[−30.5592,27.0125],8clusters/2000resamples with frozen seed202659; not a significance claim or8-seed formal evaluation. Original four rows reused without rerun or alteration. Per-instance results and96 paired differences are in results/episodes.csv and results/paired_differences.csv.

Against Never: score−2.25,Tokens+6.00%; against Always: score−6.125,Tokens−59.08%; against Internal: score+8.0417,Tokens−20.95%; against Full: score+16.5833,Tokens−18.31%. These are paired policy outcomes on the reused development cohort, not proof of per-call causal utility. Always has budget20 while Experience/Internal/Full have5; this is not a matched-budget dominance experiment.

Only4/24 task×seed scores change from Never:2458/seed0−50,2438/seed1−4,2519/seed1+50,2458/seed1−50. Thus20/24 identical scores; the two2458 variants substantially affect the mean. At the task-ID level, one has positive and two have negative average differences. No clean independent generalization claim.

363 formal actions,14/14 lookaheads verified,0 replay rejections,14 Reviewer requests;13 REPLACE and1 INVALID. Of13 actual replacements,12 have accepted execution feedback and1 explicit rejection. This is not semantic correctness or downstream utility. INVALID output remains recorded with original unchanged-draft fallback. Natural Sufficiency positives0. Coverage41/363,conflict14/363,other349conflict judgments Unknown. No rule relaxation/Memory expansion. Reviewer instability from the original experiment is intentionally retained.

All24 initial-state hashes match original Never. Actual Actor output IDs and transitions match the original Never prefixes until the first changed action; post-intervention trajectories are not assumed identical. Gate decisions replay deterministically from evidence/budget; all evidence IDs belong to the frozen approved TRAIN cases. Query→Gate→isolated feedback→formal execution timestamps checked, private state absent from model prompts; original Reviewer Prompt/seed verified. One KEEP/fallback projection exactly matches simulated/actual state. All32 protected source/config hashes match. No Internal extraction/raw logits or extra forward cost; no LLM Judge.

NEW cost:377 model calls=363Actor+14Reviewer,155118input+18505output=173623Tokens;461native transition queries (363formal actions+24formal initializations+74extra lookahead initialization/replay/candidate queries),1349audit capture bundles. Retrieval0.1144s,rule-Judge0.00981s,Actor generation785.042s,Reviewer generation69.892s,simulation29.258s. Sum episode wall961.106s includes work/cleanup across two workers; supervisor wall570.718s includes parallel execution and setup. These are different scopes and must not be added together. Model loads are separate receipts; CPU/JVM queries are charged, monetary/energy cost unmeasured. Offline Memory construction unchanged; new synthesis/training cost0.

Resource caps respected; workers use originalBF16/HF/Torch path (including original reference-kernel fallbacks), not newly installed optimized kernels. Model loading strict and shared per worker. All raw generation/decision/environment logs retained in ignored outputs/phase59_external_only_v1; committed compact decisions and SHA256 raw-log index under docs/phase59_external/results. CPU preflight checks source/cohort/seeds/Unknown/conflict/budget passed; an initial shell-heredoc quoting error was corrected before any model run. Real final audit passes all24episodes. This is a diagnostic error, not excluded experiment data.

Analysis command:

```bash
.venv-qwen35-clean/bin/python scripts/analyze_external_only.py --run outputs/phase59_external_only_v1 --output outputs/phase59_external_analysis_v1
```

Experimental source commit `ee5e9c4` preregistered the new arm; actual historical runtime commit/hash is recorded above and in the run manifest. A subsequent independent SR-development commit did not alter any imported frozen method/config or source hash used here. Engineering validity passes; pure External does not improve Never on this cohort. Its higher score than Internal/Full does not establish useful Internal–External complementarity. Phase6 is not started.
