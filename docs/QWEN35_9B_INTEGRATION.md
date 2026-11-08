# Qwen3.5-9B Actor/Reviewer integration

This supersedes the initial metadata-only audit. The bounded development run is complete:12 episodes/201 formal actions and21 real reviews, with source/prompt/cost audits passing. General replay fidelity and scientific effectiveness remain unproven; see PHASE5_7_REPORT.md. No4B thresholds are reused.

## Pinned assets and shared frozen model

Official post-trained `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, full BF16 `Qwen3_5ForConditionalGeneration`. All four downloaded safetensor shards match the official revision's LFS SHA256. No Base checkpoint, quantization, tensor deletion, training or strict=False workaround. The strict loading diagnostic has empty missing/unexpected/mismatched/error collections. The full published model is loaded; only text inputs are used.

`Qwen35Actor` initializes one model. Both scored drafts and unscored reviews call that same object with distinct prompts, recording shared ID/revision/config fingerprint and role-specific tokens/latency. New9B backend is `hf_qwen35_generation_raw_bf16_v1`; the old4B backend remains intact. The old hidden-state/SpanUQ code is preserved and is not claimed compatible with9B.

## Environment

Published12-episode records were produced in `.venv-qwen35`:Python3.12,PyTorch2.8.0+cu128,Transformers5.22.2,ScienceWorld1.2.3. It borrowed base framework packages read-only and had inherited unused `verl`/`datasets` resolver conflicts; this is preserved in the provenance, not relabelled as a clean run.

The recommended environment is now **`.venv-qwen35-clean`**, created without system site packages. Torch,Transformers,NumPy andScienceWorld paths are verified inside this venv; pip check returns0. All91 tests pass there. Two existing real Actor/Reviewer prompts reproduce identical response IDs, the Actor token-NLL sequence has max delta0, and a real SDK/reset-replay pair matches. This is numerical/SDK portability evidence, not a claim that the12 episodes were rerun in the clean environment. Old HF/vLLM environments retain Transformers4.57.3/PyTorch2.8.0.

The tested clean installation used:

```bash
python3.12 -m venv .venv-qwen35-clean
.venv-qwen35-clean/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-qwen35-clean/bin/python -m pip install -r environment/qwen35-replay-requirements.txt
.venv-qwen35-clean/bin/python -m pip check
```

Create a new environment rather than overwriting an existing stable one. `environment/qwen35-replay-lock.txt` records all installed distributions; the requirements file lists the minimal observed dependencies. This was tested on lab/Linux/RTX4090, not another physical server.

Official reference PyTorch DeltaNet/causal-convolution kernels are used because optional fast kernels are absent. This is supported reference inference, but slower; no unvalidated kernel workaround is installed. Attention is SDPA. Pin exact package versions and framework code in any later optimized profile and recalibrate/revalidate rather than reuse thresholds across changed computation paths.

## Nonthinking and token boundaries

The actual pinned chat template supports `enable_thinking=False`, rendering an empty closed think prefix in the PROMPT. Do not use soft `/nothink` strings as a substitute. Real ordinary generation was verified; structured ScienceWorld output is a separate required check.

The repository has no `generation_config.json`. Text config EOS is248044 (`<|endoftext|>`), while chat tokenizer EOS is248046 (`<|im_end|>`). Stopping only on the former actually emitted `hello<|im_end|>\n` and caused the strict score extractor to reject a special token. The new profile pins the union of these two official EOS IDs. These are stopping boundaries, not arbitrary deletions of unknown generated tokens. Actual structured/repeated/scored-unscored parity tests passed after the fix;201 evaluation drafts and40 calibration drafts have no format failures.

All four strategies share temperature0.4,top_p1,top_k0,repetition penalty1,512 response tokens,4096 prompt tokens and2 committed history steps under a separate9B development profile. Official model-card recommendations are not silently substituted. No test profile or final task identity file is overwritten.

## NLL semantics and numerical evidence

Reuse the accepted fixed-field parser, exact prompt/response retokenization and offset checks, and selected-token cross entropy. Formal scores use raw generation logits BEFORE sampling temperature/top_p transforms; FP32 reduction on the SAME GPU over BF16 model computation. No additional scoring forward is used online.

- Premise and Consequence: mean NLL of tokens intersecting each field's content.
- Local:max of those means.
- Global:all response text tokens, including labels and Action; excludes prompt, trailing recognized EOS, special tokens and padding.
- Missing/malformed/truncated/misaligned fields remain null with explicit fallback. No zero-as-confidence replacement.

After EOS correction, real ordinary9B validation records raw GPU logits versus formal NLL delta0; CPU FP32 reduction diagnostic delta9.426474571228027e-5; separate teacher-forced forward delta3.203749656677246e-7. The older pre-EOS-fix forward comparison was8.946663001552224e-5 and is retained separately. CPU and re-forward scores are diagnostics ONLY and never enter thresholds. The first attempt's exact CPU/GPU equality assertion failed before calibration; its log remains. Formal same-device comparison is exact in the observed example, with a declared1e-5 validation bound.

These are uncalibrated token surprisals, not action-error probabilities or SpanUQ outputs. The6-task pilot collected40 valid steps:Local q.75=.5749072272810143,Global q.75=.3332634086050418,Pearson.875301/Spearman.918199. Config fingerprint is cad3791bc197e7f95baa16837d072ab26b68b6f76cb12cf164e0ccfcaf71a7a6. No outcome-selected threshold or old4B values were used. It cannot establish stable general calibration or scientific usefulness. Different Actor, EOS/config fingerprint, dtype, kernels/backend or revision require fresh validation/calibration.

## Reviewer

One fixed generic prompt receives public Goal/current Observation/recent committed History/Draft Action and verified trial observation/reward/done/score. The trial is explicitly not committed. Hidden state/RNG/object-tree/private task progress/gold answers are excluded by a whitelist; failed reconstruction cannot construct a review prompt. No candidate search or unlimited retries.

Output is `Decision: KEEP|REPLACE`, one Reason and one Final Action. KEEP must preserve the draft. Format errors keep the draft and record the failure/cost. A replacement is a model proposal, not proof of syntax/goal correctness. No reliable alternative permits KEEP. Public task constraints and exact visible referents are requested without task-specific rules. Real feedback, correction validity and positive/zero/negative score effects must be audited after real runs.

## Commands

```bash
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_replay_integration.py
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_qwen35_tokenizer.py \
  --model-dir checkpoints/qwen35_9b
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv-qwen35-clean/bin/python scripts/run_phase57_replay.py \
  --model-dir checkpoints/qwen35_9b --output outputs/phase57_replay_new_run --device cuda:0
```

Use a new Run ID; keep failures and raw logs. No Qwen2.5 WM launcher or request is part of this entrypoint. Official sources: [model repository](https://huggingface.co/Qwen/Qwen3.5-9B), [Transformers model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5). Hardware peak/costs and actual episode outcomes must come from completed records, not model-card benchmarks.

## Deterministic simulator follow-up

The official1.3.0 native JAR is selected through SDK1.2.3 serverPath, leaving this inference environment/model unchanged. The new v2 development profile collects its own40-step calibration before evaluation; it never assumes engine changes preserve NLL distributions. Observed fresh quantiles coincide numerically with the prior pilot, with separate provenance. The calibrated model/backend/dtype/signature remains frozen. See REAL_ENV_WM_REPLAY_FIX.md.
