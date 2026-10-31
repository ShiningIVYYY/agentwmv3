# Phase 1: backbone verified; SpanUQ checkpoint blocked

## A. Status

**Phase 1 acceptance is NOT passed.** Real Qwen3-4B generation and response hidden-state extraction passed on three English examples. Published SpanUQ weights cannot strictly load in the available official architecture. No valid span/uncertainty result is claimed. No Phase 2, benchmark run, training or new calibration was started.

An independent repository-local .venv-spanuq was created with Python 3.12.0 and include-system-site-packages=false. Versions: torch 2.8.0+cu128, CUDA runtime 12.8, Transformers 4.57.3, Accelerate 1.14.0, tokenizers 0.22.2, HF Hub 0.36.2. All resolved versions are in environment/spanuq-inference.lock.txt. First installation hit a network timeout; retry with --timeout 180 --retries 5 succeeded. pip check passed. No packages were installed into v2 or old environments; no vLLM was installed or modified.

Only pinned Qwen3-4B and Qwen3-4B probe assets were downloaded. Published LFS SHA256 hashes were checked. Weights, cache, venv and bulk outputs are ignored by Git. GPU 1 was idle before the single-process test and returned to 3 MiB afterward. Other services/GPUs were not modified.

## B. Files and purposes

| Files | Purpose |
| --- | --- |
| internal_signals/hidden_states.py | Exact chat/response IDs, offsets, block hooks, finally cleanup, finite/shape checks, optional HF tuple equivalence; reject truncation and merge drift |
| internal_signals/spanuq_adapter.py | Pinned clean official source and weight hash; explicit config aliases; key/shape audit; strict=True loading; fail-closed adapter |
| configs/internal_signals/qwen3_4b.json | Pinned model/source revisions, dimensions and current incompatible status; no benchmark settings |
| scripts/check_spanuq_compatibility.py | Save actual strict-load error; exit 2 on incompatibility |
| scripts/spanuq_smoke.py | Three real samples, token boundaries, layer/repeat checks, timing and peak CUDA memory |
| scripts/download_spanuq_assets.py | Two pinned asset downloads with published LFS hash validation; HF_ENDPOINT/HF_HOME and paths configurable |
| scripts/setup_spanuq_source.sh | Obtain/validate unchanged pinned official source |
| scripts/test_spanuq_integration.py | 11 CPU tests, including actual tokenizer and actual checkpoint rejection |
| environment/spanuq-inference.in, .lock.txt, setup_spanuq.sh | Independent pinned environment; setup refuses an existing destination |
| docs/phase1/, PHASE1_REPORT.md | Small actual traces, state inventory, histories, hashes, tests and failure evidence |
| README, IMPLEMENTATION_PLAN, PROGRESS, SPANUQ_AUDIT, .gitignore | Honest partial status, corrected layer finding, environment/cache exclusions |

Adapter scoring delegates to official predict_spans and native sequence output, with explicit calibration/mapping checks. **This guarded score branch is unverified and unusable for the current release because constructor strict loading fails. It is not a completed inference integration.** Synthetic tensors in unit tests are not experimental results. New operational code uses repository-relative defaults or arguments; no personal absolute paths are hardcoded.

## C. Official compatibility investigation

Pinned [source 778b1c1](https://github.com/damon-demon/SpanUQ/tree/778b1c1e3eeeda66233fe6b426f9303f4c7f76a3), [probe revision 97ca9c8](https://huggingface.co/DamonDemon/SpanUQ/tree/97ca9c820ac3561259956ec21cbdd984c3132d37/Qwen3-4B), and [backbone revision 1cfa9a7](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c).

The complete official Git history has four commits. Model, config, extraction and dataset code have the same blobs in all four; subsequent changes are README-only. No compatible official branch/tag was found. HF original upload 68053f75 and current release have identical probe weights (LFS SHA256 78a203415037cb6645385d27b14d4207d184d522cc2c33af07098f2f2254fb12); the only public HF PR changes model-card metadata. See phase1/source_history.json and hf_history_summary.json.

Unmodified official inference.py fails before generation: unexpected keyword argument mob_k. For deeper diagnosis, our loader explicitly records mob_k=3 -> mob_n_components=3 and seq_agg_mode=importance -> seq_aggregation=importance. Published max_seq_len=2048 becomes an explicit adapter response-input cap, not a positional-buffer resize. Every other unknown field fails. Nine-row uncertainty head tensors support the 3K MoB output dimension. These aliases resolve config construction only, not architecture compatibility.

Strict loading of all 134 checkpoint tensors then fails:

~~~text
Unexpected keys absent from public model:
u_conditioner.cond_mlp.0.weight  [256,3]
u_conditioner.cond_mlp.0.bias    [256]
u_conditioner.cond_mlp.2.weight  [512,256]
u_conditioner.cond_mlp.2.bias    [512]

token_encoder.pos_enc.pe:
  source expected [1,4096,512]
  checkpoint      [1,16384,512]
~~~

The public code never defines u_conditioner. Shapes cannot establish its input features, activation, forward placement or whether it is training-only. We did not attach an unused/guessed module, delete keys, resize/recreate buffers or perform partial loading. All actual adapter state loading uses strict=True. A mechanical PE resize alone would leave the absent module unresolved and is not a solution.

### Corrected layer convention

Phase 0 incorrectly grouped the batch extractor with the on-the-fly evaluator. Full inspection shows that scripts/extract_hidden_states.py hooks zero-based decoder blocks, saves them by that index, and the training dataset consumes those files. The demo uses the same convention. Only scripts/evaluate.py on-the-fly uses outputs.hidden_states[lid], creating the one-layer discrepancy.

Our selected decoder blocks [21,22,23] map to HF tuple indices [22,23,24]. All three samples have elementwise maximum absolute difference 0 between hooks and HF tuple values. Repeated real forward passes also differ by 0. This validates alignment with the published producer. The exact historical backbone/tokenizer revision used to train the released probe remains unpublished and is not proven by current equivalence tests.

## D. Real inference and tests

Qwen3-4B BF16, SDPA, non-thinking chat template; GPU 1 (RTX 4090). Greedy generation, max_new_tokens=96, seed=0, following the minimal-demo style. These are integration samples, not benchmark sampling settings or scientific performance measurements.

Exact prompt/response IDs, offsets, code hashes, revisions, finite values and timing are in phase1/real_smoke.json. Full prompt+response retokenization must exactly reproduce original IDs, with the prompt as a prefix. EOS is explicitly excluded. No prompt token is treated as response. Empty response, special token, overflow and token/offset drift fail rather than silently trimming to a minimum length.

| Example | Prompt tokens | Response tokens | Generation s | Decoder forward s | Full extraction s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 47 | 2.1507 | 0.0384 | 0.0444 | 7.5707 |
| 2 | 28 | 22 | 0.7782 | 0.0343 | 0.0365 | 7.5654 |
| 3 | 32 | 23 | 0.8043 | 0.0376 | 0.0393 | 7.5666 |

Full extraction includes token/offset validation, synchronized decoder forward, hooks and shape/finite checks; this validation run also includes HF layer comparison. One extraction uses one decoder pass; the smoke repeats that extraction for validation, two passes total per sample. The decoder path avoids allocating unused vocabulary logits. No cache is enabled or credited.

Peak memory includes the real backbone and validation tensors, **not a working probe**: SpanUQ failed on CPU before CUDA placement. These numbers are not full Qwen+SpanUQ latency/memory.

### Example 1

Input: In two short sentences, explain why ice melts when heated.

Actual response: Ice melts when heated because heating provides the energy needed to break the bonds holding the water molecules in a solid structure. As the temperature rises, the molecules gain kinetic energy and move more freely, transitioning from a solid to a liquid state.

Three hidden tensors [1,47,2560]. Span and uncertainty: unavailable.

### Example 2

Input: In two short sentences, explain the roles of roots and leaves in a plant.

Actual response: Roots absorb water and nutrients from the soil, while leaves perform photosynthesis to produce food for the plant.

Three hidden tensors [1,22,2560]. Span and uncertainty: unavailable.

### Example 3

Input: Who wrote Pride and Prejudice, and in which year was it first published? Answer briefly.

Actual response: Pride and Prejudice was written by Jane Austen and was first published in 1813.

Three hidden tensors [1,23,2560]. Span and uncertainty: unavailable.

Commands executed from agentwmv3 (PYTHONNOUSERSITE=1 used):

~~~bash
.venv-spanuq/bin/python -m pip check
.venv-spanuq/bin/python -m pip freeze
bash -n environment/setup_spanuq.sh scripts/setup_spanuq_source.sh
.venv-spanuq/bin/python -m py_compile internal_signals/*.py scripts/check_spanuq_compatibility.py scripts/download_spanuq_assets.py scripts/spanuq_smoke.py scripts/test_spanuq_integration.py
CUDA_VISIBLE_DEVICES= .venv-spanuq/bin/python scripts/test_spanuq_integration.py
.venv-spanuq/bin/python scripts/check_spanuq_compatibility.py --source external/SpanUQ --checkpoint-dir checkpoints/spanuq/Qwen3-4B --output outputs/phase1/checkpoint_audit.json
CUDA_VISIBLE_DEVICES=1 .venv-spanuq/bin/python scripts/spanuq_smoke.py --backbone-dir checkpoints/qwen3-4b --source external/SpanUQ --checkpoint-dir checkpoints/spanuq/Qwen3-4B --device cuda:0 --output outputs/phase1/real_smoke.json
~~~

- pip check, shell syntax and Python compile: passed.
- CPU tests: 11/11 new contract tests passed. Unchanged legacy test_wm_reflection.py (33) and test_agentwm_reflection.py (28) also passed in the isolated environment; 72 total passing unit cases. The real checkpoint test passes because it correctly rejects incompatible weights; it does not demonstrate successful loading.
- Strict checkpoint audit: exit 2 (actual architecture failure).
- Unmodified upstream demo: exit 1 (mob_k constructor failure). Run for diagnosis in the existing interpreter without installing/changing packages; phase1/upstream_demo.log preserves it.
- Three real backbone samples: shapes/finite values, IDs/offsets, layer matching and repeat checks passed.
- Overall smoke: exit 2; acceptance_passed=false. detected_spans=null and uncertainty_score=null, not zero confidence.
- No ScienceWorld/environment rollout, task scoring, training or calibration occurred.

## E. Reproduction and blockers

On a fresh compatible Linux/CUDA host, provide Python 3.12 and select data-volume locations for the environment/cache if needed:

~~~bash
PYTHON_BIN=python3.12 bash environment/setup_spanuq.sh
bash scripts/setup_spanuq_source.sh
export HF_HOME="$PWD/.cache/huggingface"
.venv-spanuq/bin/python scripts/download_spanuq_assets.py
.venv-spanuq/bin/python scripts/test_spanuq_integration.py
.venv-spanuq/bin/python scripts/check_spanuq_compatibility.py --source external/SpanUQ --checkpoint-dir checkpoints/spanuq/Qwen3-4B --output outputs/phase1/checkpoint_audit.json
~~~

The final command currently reproduces exit 2, not a working probe. HF_ENDPOINT/HF_HOME and source/env/asset destinations are configurable. Current installation used the direct pins then exported a full lock; the locked fresh-host setup recipe has syntax checks but has NOT been run on a second fresh host. CUDA validation is limited to the inspected driver 570.172.08/CUDA 12.8/4090 setup.

Required upstream evidence:
1. Matching official model implementation/commit including u_conditioner and its actual forward/input-feature contract.
2. Config/constructor for the 16384 positional buffer and exact published field names; a strict-loading example for this checkpoint SHA.
3. Original backbone/tokenizer revision and confirmation of extraction convention.
4. Then re-run strict loading and real probe spans, boundary/score/calibration/sequence validity, full memory and latency tests.

No author issue/message was posted. The saved errors and hashes form a concrete reproducible upstream request. No architecture/layer selection may be based on downstream test results.

## F. Next steps

Resume Phase 1 once matching official architecture/provenance is available. Actual adapter scoring/calibration/native sequence output must pass real checkpoint tests before Phase 2. Do not count backbone-only success as probe success. Keep Qwen3-4B; do not switch Actor to Qwen3.5. PAPER_ALIGNED_EVAL.md and all existing benchmark/WM parameters are unchanged.

## G. Stop condition

Phase 1 remains incomplete, blocked by official source/checkpoint incompatibility. Environment, backbone pipeline, strict guards, evidence and tests are committed for review. Stop before Phase 2.
