# Internal Signals — official Token NLL provider

Status: route B selected by the user; Phase 2 implemented and functionally verified on real ScienceWorld development steps. No Experience Memory or WM invocation Gate is implemented. SpanUQ remains a preserved unavailable optional future provider, not the current scorer. See PHASE2_REPORT.md and phase2/ for actual tests/traces.

## Frozen scoring contract

Actor: Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c. Tokenizer/config/weight files are checked against Phase 1 SHA256 manifest at initialization. Official configuration is configs/internal_signals/scienceworld_token_nll_v1.json:
- backend hf_generation_raw_bf16_v1; HF Transformers 4.57.3, torch 2.8.0+cu128 on the tested server.
- Frozen BF16 parameters, eval mode, SDPA; raw generation logits and NLL reduction in FP32.
- Non-thinking chat template, sampling T=.4/top_p=1/top_k=0 (HF no cutoff), repetition_penalty=1, max_new_tokens=512, seed=0.
- Single unpadded prompt per request; exact original prompt IDs plus generated prefix. No batched/rescore/vLLM substitution is authorized by this provider ID.

HF generate(output_logits=True) exposes the raw next-token logits for each generated token. In the inspected pinned implementation it copies next_token_logits to FP32 before the sampling processors, then records them in raw_logits. No output_scores=True is used in the production provider; no temperature-processed likelihood is accepted. Model generation config assets are hash-pinned and temperature/top-p/top-k/repetition are explicitly overridden.

For response token y_t and raw logits z_t:
~~~text
nll_t = logsumexp(z_t) - z_t[y_t] = -log p_theta(y_t | original_prompt, y_<t)
premise_uncertainty = mean(nll_t over Premise value tokens)
consequence_uncertainty = mean(nll_t over Expected Consequence value tokens)
local_uncertainty = max(premise_uncertainty, consequence_uncertainty)
global_uncertainty = mean(nll_t over ALL scored response tokens)
~~~

Units: unbounded nonnegative nats/token. Computed zero can be a legitimate near-certain token likelihood; missing values are always null, never converted to zero. These are uncalibrated model-generation statistics, not action-error probabilities and not original SpanUQ outputs.

## Prompt, token scope and action grammar

The default legacy Agent prompt/HTTP backend remains unchanged. A caller explicitly supplying internal_signals_config enables the new backend behind the same Agent.count/generate interface. Only the new structured Science prompt skips the legacy one-shot and uses up to two completed observation/action pairs. Goal/current observation are mandatory; whole oldest pairs can be removed to meet the 4096-token cap. Required fields are never silently clipped. This is an independently named diagnostic/research cohort, not evidence of exact legacy/paper prompt equivalence.

The model must emit:
~~~text
Premise: <current condition the action depends on>
Expected Consequence: <predicted immediate effect, not actual future feedback>
Action: <native ScienceWorld command>
~~~

Science.run remains the actual environment action loop. Its native last-Action-tail extraction is preserved; only its opt-in valid-format accounting recognizes this three-field schema instead of requiring Thought:. No fuzzy action repair, hidden alternative command, retry-on-parse or extra environment action is introduced. If malformed output has no Action separator, the existing empty-action behavior remains a real environment rejection; the module does not invent a successful action. Available-action membership is diagnostic and not an exhaustive executability oracle.

Local boundaries:
- Trim whitespace in the two field values; exclude the field labels by character interval.
- Use offsets from full prompt+response tokenization after verifying the entire token sequence is exactly original prompt_ids + generated response_ids.
- Count a whole token if its character interval intersects a field value; log boundary overhang tokens (e.g. leading whitespace shared with a value). Do not invent fractional character likelihood.
- Reject a token shared between the two separated field values.

Global boundaries:
- INCLUDE all response text tokens: both field labels, Premise, Expected Consequence, Action label/value, separators/newlines.
- EXCLUDE prompt (including the template empty thinking prefix), all trailing EOS, padding and other special tokens.
- No generation prefix is dropped; NLL is indexed by the generation step. Teacher-forced causal shift exists only for separate diagnostics, never as an automatic provider fallback.
- Unexpected special token, alignment mismatch or token cap/truncation invalidates the signal. Batch size 1 has no active padding tokens; future batched providers need separate mask validation.

Scoring is completed before the current real environment step. Only committed prior observations/actions enter the prompt. Real next observations/scores are logged afterward as audit evidence, never injected into the same step generation. scripts/audit_internal_diagnostic.py independently rebuilds every saved prompt from its real current/past observations and checks all field/global reductions.

## Missing/error policy

TokenNLLProvider supports missing_policy=global_only (default) or null:
- Bad/empty/missing field: local and that field score are null; valid global response NLL can remain available.
- global_only exposes fallback_uncertainty=global only when token alignment/numerics and full-response completion are valid; it does not rewrite local_uncertainty as global.
- null leaves effective_uncertainty null on missing fields.
- Alignment/numeric failure, unexpected special token, empty response or truncated generation: no numeric fallback is supplied, including no global score.
- Provider, dtype or model-revision mismatch is a hard error, not a silent conversion.

Output records missing_span, missing_reason, missing_fields where parsing succeeds, errors and fallback policy/applied flags. No call_wm or downstream Gate decision is produced.

## Unified interface and provenance

InternalSignals dataclass in internal_signals/base.py contains:
~~~text
premise_uncertainty, consequence_uncertainty
local_uncertainty, global_uncertainty
missing_span, token_count, extraction_latency
backend, dtype, model_revision, status, errors, details
~~~

details retain field spans/offsets, raw-generation scoring semantics, missing/fallback flags, EOS-inclusive generation token count, prompt count and additional-forward count. The generation record preserves original messages/text, prompt/response IDs and selected-token NLL. Metadata retain tokenizer/weight asset hashes, config hash, exact torch/Transformers version, GPU and attention mode.

InternalSignalProvider.extract(record) is provider-neutral. Future SpanUQ must return this structural contract with its own method/range/calibration/span-origin metadata after real strict loading succeeds. The existing strict loader and hidden pipeline remain intact. A working SpanUQProvider is NOT simulated. Switching a future backend requires explicit configuration, validated span-to-field association and separate development thresholds; interface interchangeability does not make [0,1] SpanUQ scores equivalent to token NLL.

## Numeric consistency and cost

Official scoring never mixes BF16/FP32 or generation/rescore paths. Pinned HF generation is the source of scores, not the legacy vLLM HTTP service. The diagnostic repeats one real prompt through this exact generation path; response IDs and all NLL values match exactly (max difference 0). This is a checked sample, not a hardware-independent determinism guarantee.

A separate BF16 full-forward diagnostic on that real step differs from generation by max 0.487322 nats, mean 0.024485. It uses the same model, original prompt/full prefix and causal shift, but is a different execution path; it is NOT accepted as equivalent or used in the reported signals. Earlier independent FP32 evidence is retained in Phase 1B, with no FP32 scores used here. Finite-precision, caching/attention/matmul shape/backend differences are plausible contributors; this test does not isolate each cause or justify widening a tolerance. Mixing backend/dtype metadata raises errors.

Online internal extraction uses zero extra backbone forward passes, retaining only raw generation logits and computing NLL/alignment/aggregation. It still incurs O(T*V) logit retention and FP32 normalization, temporary memory/CPU transfer and latency. No hidden-state extraction pass or probe runs. The first-step validation adds one repeat Actor generation and one teacher-forced forward; those costs are explicitly separate from normal online extraction.

Cost fields record generation/extraction latency, raw retained bytes, CUDA peaks, output tokens including EOS, action-value tokens and non-action response tokens. An action-only prompt with the same actual goal/state/history is counted without model inference to isolate extra instruction tokens. This is not a measured counterfactual baseline latency/token output comparison. All comparable Gate arms in later experiments must share the structured prompt/provider, or use separately labelled prompt ablations.

## Reproduction and current scope

Use the existing Phase 1 isolated environment/assets. Diagnostic entrypoint is scripts/diagnose_scienceworld_internal.py; it instantiates original Agent and Science and calls original Science.run. It deliberately does not expose the new backend as a full-benchmark launcher in this phase.

Inputs are pinned train/test ID JSONs, the pinned official paper task catalog file, an existing inspected ScienceWorld 1.2.3 AgentGym HTTP service and its environment Python executable. No large assets/dependencies were downloaded in Phase 2. The reference catalog is data identity, not an RL-training dependency to execute. Diagnostic selection filters ALL 200 evaluation IDs and reserves the three chosen training instances as development; they must not later be presented as untouched calibration validation.

See PHASE2_REPORT.md for exact executed lab command and portable command variables. Paper-Aligned Evaluation Protocol is unchanged; five diagnostic steps per selected training task do not redefine the later 20-turn benchmark horizon.
