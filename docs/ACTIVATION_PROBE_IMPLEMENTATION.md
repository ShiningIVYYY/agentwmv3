# Supervised Activation Probe v1: implementation and bounded pilot

Status: implementation and real engineering validation complete; training/data collection and independent policy results are reported separately as they become available. This is a new supervised research pilot, not Phase 6 or a completed benchmark.

## Research scope

The user approved moving beyond the previous training-free restriction. Only small activation classifiers are trained. The Qwen3.5-9B Actor and Reviewer remain frozen and share weights within each worker. All work is in agentwmv3; agentwmv2 and prior ScienceWorld/ALFWorld experiment configurations are preserved.

The design borrows activation monitoring and escalation from [OpenAI's activation-classifier article](https://alignment.openai.com/training-activation-classifiers/). Our labels, parameterization, complete-draft BCE objective and environment intervention differ. It is not a reproduction of their streaming safety system, rolling-softmax objective or reported performance. The original approved proposal is [ACTIVATION_PROBE_PLAN.md](ACTIVATION_PROBE_PLAN.md).

This first bounded implementation uses ALFWorld, following the most recent experiments. ScienceWorld is not evaluated by this run. Ground-truth Reset/Replay is an oracle-like environment privilege, not a learned WM. There is no Qwen2.5 WM request.

## Uniform protocol v2

All policies use the same new `activation-pce-v2` Actor and Reviewer protocol, rather than mixing old47-task outcomes with new data:

* Actor: exactly Premise / Expected Consequence / Action, short clauses, no thinking; strict complete-format parser. Truncated or missing Action aborts only that Episode. The actual last native score/success and explicit format-abort termination are retained; no action or reward is invented, no retry is selected by outcomes.
* Reviewer: KEEP/REPLACE, one short reason, exact final command. A nonadmissible replacement is explicitly rejected and the original draft retained. Invalid/truncated output is separately recorded. This is public command grounding, not proof of task utility.
* Actor sampling: BF16, temperature0.4, top-p1, top-k0, repetition1, output512, history2, input2048; model Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; pinned Transformers5.22.2/Torch2.8.0+cu128 environment.
* H50 formal environment turns, one-step independent native lookahead, one Reviewer call. Always budget50, selective policies budget5; budgets are separate from thresholds.
* ALFWorld0.4.2/TextWorld1.7.0 native PDDL backend; success is native `won`. Auxiliary score100*won is not ScienceWorld partial Task Score.

The prior profiles, raw236 completed Episodes and partial matched47 results are untouched and are regression evidence only. New NLL thresholds are recalibrated under v2, not reused from the old prompt.

## Data roles and exposure

`docs/activation_v1/split_manifest.json` is fixed before model collection, with game/metadata hashes, task IDs, seeds, canonical goal groups and source paths. Deterministic hash/family round-robin selection reserves:

| Role | Distinct TRAIN-derived goal groups | Use |
|---|---:|---|
| Probe Train |36| Fit probes and normalization |
| Probe Validation |12| Select architecture/layer across all3 seeds |
| Gate Calibration |12| Quantile thresholds, no outcome-based selection |
| Fresh Independent Dev |12| Complete Episode comparisons only |
| Separate native validation |5| Identity/replay controls, not outcome data |

The entire previous50-task list, its6 environment controls, old3 pilot, Memory/calibration/preliminary tasks, and all140 final valid_seen canonical goals are excluded. Cases and both counterfactual branches inherit their task's role; no random step split. Native initial goals are checked for exact/casefold duplicates against old native identities and all77 current identities. The final Test is never reset.

There were368 unexposed canonical groups before reservation. No remaining goal-disjoint `look_at_obj_in_light` groups were available under these exclusions; this pilot therefore covers five families. Task templates/house layouts can still overlap. Canonical and exact native goal isolation do not prove absence of every semantic synonym or model pretraining contamination.

The existing31-case/two-card ALFWorld Memory is frozen by SHA256. No new cases are synthesized into online Memory. Retrieval retains tri-state Unknown and real evidence checks; no learned Judge or correction module is added.

## Activation capture and provider

The actual model has32 text blocks, hidden dimension4096. Discovery resolves `model.language_model.layers`; zero-based blocks10/15/21 are linear_attention/full_attention/linear_attention. Hooks capture block outputs before final normalization. They slice post-token response positions `[prompt_length:prompt_length+response_length]`, excluding prompt/padding/EOS.

V1 performs one additional frozen, teacher-forced complete Prompt+actual-response forward with `use_cache=False`, verified `logits_to_keep=1`. It does not reuse sampled next-token prediction positions as post-token features. Detached BF16 CPU tensors are cached; Premise/Consequence/Action masks are retained separately. Current trials aggregate all response tokens. Field-wise ablations and generation-time cache hooks are deferred.

The online feature API whitelists only actual prompt IDs/text and response IDs/text. Outcomes, private state and Reviewer output are not accepted. A provider binds Actor revision/config, tokenizer asset hashes, prompt source hash, layer/path/capture point/dtype and checkpoint hashes. Mismatches fail explicitly, not through implicit cross-model fallback. Probe sigmoid scores are supervised uncalibrated scores; they are neither Token NLL nor calibrated error/utility probabilities.

## Real labels and training

Collection executes the frozen Actor for up to12 actual steps on each of60 Train/Validation/Calibration tasks (at most720 valid decision cases). Features are saved before the actual action; feature and outcome timestamps are checked. A failed/incomplete case is not made into a synthetic negative.

The first risk target is **native rejection**: explicit `Nothing happens.` plus a nonadmissible draft. An admissible no-op is Unknown; zero reward alone is not failure. This target may mostly reproduce a public membership rule, so the direct Admissible-Action Rule is mandatory. It does not measure general semantic task correctness.

Utility sampling is fixed in advance: one step-index2 state per task if a valid case exists, up to60 pairs. Each source state is reconstructed by exact full-prefix replay, independently for Skip and Call. Skip executes the original action; Call uses one native trial and the frozen Reviewer. Both continue with the same frozen Never-WM Actor, absolute step seeds, H50 and sampling configuration. Labels are the actual difference in final native success: Help(+1), Neutral(0), Harm(-1). Unknown reconstructions are excluded, not filled with0. Per-step branch receipts and original feature identity are retained. This estimates a single-seed, fixed-policy intervention; it is not universal causal utility or robustness across seeds.

Three layer candidates × mean Linear / mean MLP / EMA-QV × seeds0/1/2 are trained, only on cached detached activations. FP32 CPU probes use fixed AdamW lr3e-4/weight_decay0.01,3 epochs, Train-only positive weighting and normalization. Candidate choice uses mean Validation AP across3 seeds; all3 selected seeds form the deployed ensemble. EMA pooling has a tested constant-state inference update; v1 BCE supervises the final complete-draft score, not every prefix.

Each head requires at least6 Train tasks,5 positive and5 negative Train cases,2 positive and2 negative Validation cases, and8 Calibration cases. Otherwise its artifact states `insufficient_labels` and no classifier is manufactured. Help and Harm must both qualify before any Utility policy is deployed. Report all seeds/trials and sparse label counts.

Thresholds use q0.75 of head scores on separate Gate Calibration; Utility+External uses predeclared q0.65 Help threshold on verified conflict while retaining Harm veto. These are pilot operating points, not optimized thresholds. Raw NLL q0.75 is independently computed under the new protocol, with Pearson/Spearman and quantile call rates. Outcome-based threshold optimization is not performed.

## Evaluation and controls

On the fixed12 untouched Dev tasks, seed0, evaluate Never, Always, NLL Internal, NLL+External, Admissible Rule, Random(p0.1,max5), Periodic(every10,max5), and Reviewer-without-simulation. If training requirements pass, add Activation Risk ± External and Utility ± External. Thus96 to144 Episodes, rather than a full140×8 benchmark. Random/Periodic share maximum budgets; realized calls need not match and cannot be presented as exactly matched realized cost.

Risk calls on high classifier score or verified conflict. Utility calls on sufficiently high Help and low Harm; external conflict may lower only the Help threshold. It never bypasses Harm veto. Memory currently has no confirmed unchanged-action sufficiency rule. Unknown external evidence is not treated as sufficient. Budget exhaustion executes the draft and records the unverified skip. Infrastructure or feature-contract failures stop with explicit error; an Actor format abort affects only its Episode.

Never/Always/public-rule/random/periodic do not extract activations or NLL unnecessarily. NLL and no-feedback-review controls score raw generation logits; Probe policies capture the extra forward and classifier compute. Extra Reviewer inference without simulated feedback is explicitly separated from true WM information.

## Budgets, artifacts and lifecycle

Use available GPU0/6/7, three replicas, never more than the authorized four. Actor and Reviewer share one model per worker. Probe optimization uses CPU, freeing model workers between stages. No model download or package installation is required.

Per worker/stage hard limits:4M tokens,4000 model calls,30000 native transitions,40000 native state receipts. Whole-run:24M tokens/24000 model calls/180000 transitions/240000 state receipts/36h, including CPU training. State/ledger receipts are durable. The controller stops owned workers on an infrastructure failure or cap; it does not silently rerun successful branches or replace tasks. Completed data are retained for an explicit audited resume; automatic resume after arbitrary infrastructure failures is not implemented.

Record offline feature collection, utility branches, training, native preflight and engineering smoke costs separately from online evaluation. Count reset/replay/candidate and passive inspections separately, generation tokens/calls, extra activation forwards/seconds/peak memory, probe time, retrieval, model loading and end-to-end time. Token/seconds/native-query quantities are not monetary costs or interchangeable units. No zero-cost claim is made when WM LLM tokens are absent.

Run artifacts: `outputs/activation_experiment_v1/{manifest,progress,error}.json`, `collect/worker*/`, `training/`, `evaluation/worker*/`; native workers and caches stay outside Git. Controller exports compact real results to `docs/ACTIVATION_V1_REPORT.md` and `docs/activation_v1/results/summary.json`, then attempts commit/push only for its own clean result changes. Failures preserve logs and do not create synthetic metrics.

## Commands

Set `ALFWORLD_DATA` to the existing dataset directory and `ALFWORLD_PYTHON` to the existing isolated CPU environment's Python. Run from the repository; use the existing pinned `.venv-qwen35-clean` environment.

```bash
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_activation_v1.py -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 \
  .venv-qwen35-clean/bin/python scripts/preflight_activation_v1.py --run outputs/activation_preflight_v1
# One-time smoke already executed; do not reuse an existing Run ID.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 \
  .venv-qwen35-clean/bin/python scripts/smoke_activation_v1.py --run outputs/activation_smoke_v1
# Detached, bounded controller; check progress before ever considering a restart.
nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 \
  .venv-qwen35-clean/bin/python -u scripts/run_activation_experiment.py \
  --run outputs/activation_experiment_v1 --auto-publish \
  > outputs/activation_experiment_v1.log 2>&1 < /dev/null &
```

The split generator `prepare_activation_experiment.py` refuses to overwrite the preregistered split. The active configuration is `configs/evaluation/activation_v1.json`; do not regenerate it during a run. The controller freezes code/config/Memory hashes and the source Git commit. Run scripts emit explicit STOP/error receipts and clean owned native environments; no old Qwen or unrelated GPU processes are stopped.

## Observed engineering results and limits

Real Qwen3.5-9B smoke: three selected tensors each[35,4096]; repeated activation max absolute difference0; prefix activation relative RMS difference0 for changed future suffix at all3 layers; Actor token IDs and raw NLL repeat exactly before/after capture; Actor trainable parameters0. A real isolated lookahead produced the same transition as formal KEEP execution, and the shared Reviewer produced a valid KEEP. The smoke consumed2024 tokens,3 model calls,4 transition queries and6 state receipts, separate from the main experiment. These are engineering results, not Probe accuracy or task-success evidence.

Native preflight and expanded CPU test receipts are stored under `docs/activation_v1`; startup/progress is separate from final policy results. No supervised classification score, utility gain or independent task-success claim is made before the real labels, training and fixed evaluation complete. If sparse labels prevent Utility deployment, that is a reported limitation and a reason to reassess the target/Reviewer before scaling, not to manufacture positives or touch Test data.
