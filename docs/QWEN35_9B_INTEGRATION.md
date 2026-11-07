# Qwen3.5-9B integration: verified metadata and deferred execution

Status: **not loaded or integrated**. The user selected official post-trained `Qwen/Qwen3.5-9B` for both Actor and Reviewer. The environment-clone prerequisite failed validation, so no model weights were downloaded, new inference environment installed, generation performed, or threshold calibrated in Phase 5.7. Existing 4B inference and legacy vLLM environments remain intact.

## Official model facts

Metadata was fetched from the [official model repository](https://huggingface.co/Qwen/Qwen3.5-9B), pinned at **`c202236235762e1c871ad0ccb60c8ee5ba337b9a`**. [Metadata evidence](phase57/qwen35_metadata.json) records each small artifact's URL, size and SHA256. This is a metadata revision, not a claim of loaded weights.

The published architecture is `Qwen3_5ForConditionalGeneration`, config type `qwen3_5`, text type `qwen3_5_text`, text hidden size4096,32 layers and vocabulary248320. It is a multimodal post-trained model with a hybrid text backbone; it must not be treated as the old Qwen3 model solely because their hidden dimensions coincide. The requested checkpoint is not a Base variant.

The official template defaults to thinking and supports disabling it through `chat_template_kwargs.enable_thinking=false`. With direct HF tokenization the equivalent template parameter must be verified against the pinned tokenizer/template; `/think` and `/nothink` are not substitutes for a verified configuration. This phase inspected official metadata only: actual nonthinking generation, `<think>` behavior, EOS handling and response-token boundaries are **unverified**.

The model card documents current HF/vLLM serving, including a vLLM text-only `--language-model-only` option. It recommends current framework builds. No guessed minimum version or moving-main dependency has been installed or described as reproducible. A future independent environment must pin exact package versions or commits after real loading/logprob tests. The old Transformers4.57.3 config registry contains neither `qwen3_5` nor `qwen3_5_text`; it cannot be assumed compatible. Legacy vLLM0.11.0 support is not established.

Official general-purpose nonthinking sampling recommendations differ from our paper-aligned evaluation profile. Do not silently substitute those recommendations. Preserve `PAPER_ALIGNED_EVAL.md` and the old profile. If the new architecture retains temperature0.4/top_p1/no extra top_k/512 response tokens/history2/prompt4096, all four new strategies must explicitly share that new profile. None has been run yet.

## Server inventory

Eight NVIDIA RTX4090 cards were observed, each24564MiB total and3MiB used, driver570.172.08. The accepted HF environment has Python3.12.0, PyTorch2.8.0 and Transformers4.57.3; the historical serving environment adds vLLM0.11.0. No package was changed and no GPU process was launched. Package metadata is not a new-model compatibility test or a new-model VRAM estimate.

The searched project model/checkpoint directories have no9B weights. The HF cache has a matching `models--Qwen--Qwen3.5-9B` entry at the official revision, but it contains **only `config.json` (3126bytes)**. A cache directory is not a complete model. This inspection covers the recorded cache roots, not every possible server mount. Metadata HTTP requests downloaded only small config/template/model-card files, never weight shards or tokenizers' large data files.

## Planned runtime and internal scores

After clone acceptance, first build an isolated 9B inference environment and test the official model on ordinary inputs and true development states. Prefer a single loaded model/backend for Actor and Reviewer with separate fixed prompts. Record the same weight revision, tokenizer files, dtype, generation configuration and service identity for both roles. Enforce one review per Call, no Qwen2.5 requests, and count every role separately.

Reuse `internal_signals/decision.py`, `token_nll.py`, provider contracts and accepted boundary methodology, but do not relabel the 4B fingerprint `hf_generation_raw_bf16_v1` as a validated9B provider. A new fingerprint must identify model/tokenizer/template revisions, backend, dtype, exact raw-logit calculation and sampling transformations. Prefer raw next-token logits captured during generation if the backend exposes them; establish that temperature/top_p processors have not changed the reported distribution. If rescoring is required, verify identical full prompt/generated-prefix IDs and next-token offsets and count the extra forward pass. BF16 and FP32 paths remain separate.

Planned statistics retain the accepted meaning: field-content mean Token NLL for Premise and Expected Consequence, their maximum for Local; full generated decision-text mean NLL for Global, with labels and Action included and special/padding tokens excluded under the Phase2 boundary definition. Reconfirm that definition on the new tokenizer before release. Missing/unaligned fields remain null/Unknown, never zero. NLL is uncalibrated generation uncertainty, not an action-error probability or SpanUQ score.

All Qwen3-4B thresholds are **invalid for9B**. Once the provider passes real alignment/numerical tests, collect only approved development tasks, report missing rates and Local/Global distributions/correlations, and calibrate prespecified quantiles independently of outcome evaluation and WM budget. No threshold values or correlation estimates exist for9B in this phase.

## Reviewer design, not an implemented result

Inputs are Goal, current public Observation, bounded past History, Draft Action, and public isolated-trial feedback. Any admissible-action list must come from the unmodified current formal state before execution, with its information permission and query cost recorded. Private snapshots, object-tree audit digests, task answers and future trajectories are excluded.

The proposed fixed prompt asks whether the trial executed successfully, caused damage, and supports goal progress; whether evidence justifies replacing the draft; and whether a proposed replacement is grounded in the **current pre-action** state. A successful trial does not prove goal utility. Failed trial feedback cannot be treated as a committed state change. If no grounded alternative exists, KEEP is permitted. Expected output:

```text
Decision: KEEP / REPLACE
Reason: ...
Final Action: ...
```

A parser must validate one decision and one action, distinguish format failure from model errors, preserve the original draft on an explicitly configured fail-safe, and never perform an unbounded retry loop. This is not a claim that9B can repair the old Reviewer failures. Real tests and their original failures must be retained before approving this policy.

## Reproduce metadata inspection

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_qwen35_metadata.py \
  --output outputs/phase57_setup/qwen35_metadata_new_run \
  --hf-python .venv-spanuq/bin/python \
  --legacy-python /path/to/existing-legacy-environment/bin/python \
  --cache-root /path/to/other/model/cache
```

The optional legacy Python is queried for package versions only; it does not serve a WM. The script never downloads model weights or runs generation. The real lab command and return code are in [verification.json](phase57/verification.json). Metadata evidence cannot satisfy the requested Actor, Reviewer, NLL, latency or GPU-memory acceptance tests.
