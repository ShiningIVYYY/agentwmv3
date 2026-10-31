# SpanUQ compatibility audit (Phase 0)

Phase 1 follow-up: see [PHASE1_REPORT.md](PHASE1_REPORT.md). Real weights expose an absent u_conditioner module and PE shape mismatch. The Phase 0 layer-convention finding is corrected below after full producer inspection and real HF equivalence tests.

## Pinned primary sources

- Source commit: [778b1c1e3eeeda66233fe6b426f9303f4c7f76a3](https://github.com/damon-demon/SpanUQ/tree/778b1c1e3eeeda66233fe6b426f9303f4c7f76a3).
- Checkpoint repository revision: [97ca9c820ac3561259956ec21cbdd984c3132d37](https://huggingface.co/DamonDemon/SpanUQ/tree/97ca9c820ac3561259956ec21cbdd984c3132d37/Qwen3-4B).
- Backbone candidate revision: [1cfa9a7208912126459214e8b04321603b3df60c](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c).
- Paper [arXiv 2607.05721v2](https://arxiv.org/abs/2607.05721v2), checked as context, not proof of Agent-domain performance.

Only official small source/config files were retrieved. No .pt/safetensors or dataset was downloaded. Small metadata/config snapshots are in `docs/audit`. The upstream GitHub commit API initially returned HTTP 403 rate limit; a normal shallow source clone succeeded and `git rev-parse HEAD` supplied the source commit. No package installation occurred.

## Verified configuration and intended tensor path

HF Qwen3-4B config specifies Qwen3ForCausalLM, 36 decoder layers, hidden size 2560, bfloat16, vocabulary 151936. SpanUQ checkpoint model_config specifies d_model=2560, d_proj=512, n_queries=32, multi_layer_ids=[21,22,23], fusion=mean, regression span mode, span enrichment=gate, mob_k=3, importance sequence aggregation and max_seq_len=2048. Backbone family/dimension match on metadata. The exact backbone/tokenizer revision used in probe training is NOT published in the inspected training_config; matching model name is insufficient to prove binary equivalence.

Official extraction performs generation then a teacher-forced HF forward on prompt+response; use ONLY response-token representations. A batch of one supplies three tensors [1,T_response,2560], plus a boolean mask [1,T_response]. CPU-stored per-layer tensors are [T_response,2560]. Probe returns pred_u/valid_logits [B,32], span_coords [B,32,2] normalized boundaries (regression), and seq_uncertainty [B]. Official model.predict_spans decodes normalized coords with T-1, uses inclusive token end, filters sigmoid(valid_logits)>=0.5, and returns list per batch of {start,end,u,nu,valid_prob}. Convert inclusive token end to exclusive character end through offsets.

## Blocking mismatches found in actual source

1. `scripts/inference.py:load_spanuq_probe` calls SpanUQConfig(**config_dict). Checkpoint has unknown fields mob_k, seq_agg_mode and max_seq_len. Actual CPU import and config construction raised `TypeError ... unexpected keyword argument 'mob_k'` (see audit/spanuq_config_load_check.txt). Config.from_json silently drops unknown keys; that is also unsafe because current mob_n_components default is 1 whereas checkpoint mob_k=3. Do not silently filter or invent a mapping without state-dict compatibility checks.
2. Demo `predict_spans` passes token_entropies to probe.forward; actual signature is `(self, hidden_states, attention_mask)`. It then expects pred_start/pred_end, absent from the dataclass; it could report no spans even if the model ran. Actual implementation exposes span_coords and start/end logits. Prefer official model.predict_spans after checkpoint compatibility is resolved, not a new detector.
3. Demo loads weights strict=False without checking missing/unexpected keys. Phase 1 must inspect and validate all keys/shapes, including MoB head and sequence head; partial random heads are unacceptable. No claim of weight compatibility is possible until real checkpoint load.
4. Demo hooks `model.model.layers[lid]` (zero-based decoder block output). Correction: the batch extractor also hooks decoder blocks and saves them by zero-based index; training dataset.py consumes those files. Only the on-the-fly evaluator indexes HF outputs.hidden_states[lid], whose index 0 is embeddings: block lid corresponds to hidden_states[lid+1]. Thus [21,22,23] is ambiguous across official paths. Inspect upstream history/training extraction provenance or obtain author clarification; record the chosen established convention before Agent scores are accepted. Do not select layers using test outcomes.
5. Demo retokenizes prompt, prompt+response and response separately, then truncates to their minimum lengths; cross-boundary merges can misalign offsets. It has no assured Agent chat-template alignment and defaults to 4096 total tokens, while Actor can use 4096 prompt + output. A long prompt may erase response tokens. Reject overflow and alignment mismatch explicitly, never treat an empty span result as confidence.
6. Demo does a second full backbone forward just to compute entropy; current probe.forward does not take token entropy. Record that overhead in the unmodified official baseline; optimize only after equivalence validation. Hooks must be removed even on exceptions.

## Chat template and hidden states proposal (not implemented)

Use exact generation messages + pinned tokenizer apply_chat_template(add_generation_prompt=True, enable_thinking=False). Keep generated response token IDs, assistant prefix, EOS and padding masks. Verify that prompt token IDs are a prefix of the full sequence; derive character offsets from a consistent full tokenization and response-only range. Never tokenize a different simplified goal in place of the true Agent context. Store token/character boundaries for premise/consequence fields and exclude prompt, demo, future observations and EOS from scored spans.

HF forward with output_hidden_states=True or correctly matched decoder hooks can expose the required layers. The corrected official batch-producer convention is blocks [21,22,23], matching hidden_states[22,23,24]; Phase 1 real equivalence tests passed. Do not substitute the on-the-fly evaluator indices. Historical checkpoint-training revision remains unpublished. Use eval/inference mode and an explicit mask; cache by backbone/tokenizer revision, full input IDs, response range, layer convention and dtype. vLLM generation + HF re-forward is a viable proposed fallback, with extra model memory/latency/token-compute recorded. No existing HTTP endpoint provides these states.

## Official sequence score and calibration

The actual model has a native seq_uncertainty [B] importance/validity aggregation, recomputed after span enrichment. A token/sequence dual path exists but is conditional on use_multi_granularity; do not claim that path is in this checkpoint without validating keys/config. Prefer the validated native score in Phase 2; max/mean/length-weighted aggregation must be labelled separate deterministic ablations.

Published Qwen3-4B temperature.json gives T=1.0261235077222721. Official calibration transforms a probability as sigmoid(logit(p)/T), with clamping for numerical safety; do not divide raw uncertainty p by T. The official calibrate.py fits a scalar on development data and also evaluates test data. Do not run it as part of integration, do not retrain the probe, and do not imply the published T is calibrated for Agent decisions or necessarily for the native sequence score. Start with the published T for documented span outputs, retain raw scores, and declare separately any later dev-only calibration.

## Acceptance decision

Metadata and available GPU memory suggest a feasible isolated short test. Current checkout is NOT plug-and-play: config construction is confirmed broken, output signature mismatches are statically confirmed, layer convention and exact trained tokenizer revision are unresolved, and weights have not been loaded. Phase 1 must first resolve these gaps through official compatible source/release/history and minimal explicitly reviewed compatibility adaptation if needed. If that cannot be done without guessing, stop and report the blocker. Agent-domain suitability is a separate Phase 2 empirical go/no-go decision. All claims must say task-specific training-free, never universally untrained.
