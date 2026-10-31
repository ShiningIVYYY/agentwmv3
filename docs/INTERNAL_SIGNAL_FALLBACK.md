# Training-Free Internal Signal Fallback — Phase 1B design and feasibility

**Proposed route B: fixed-field token surprisal from frozen Qwen/Qwen3-4B.** This is a standalone prototype and design, not complete Phase 2 development. No Agent loop, benchmark configuration, external memory or Gate was integrated/changed. User route selection is still required.

## 1. Signal definition and meaning

For an exact non-thinking chat prompt x and generated response token sequence y, use raw frozen-model logits z_t before sampling temperature, top-p/top-k or other probability processors:

~~~text
nll_t = -log p_theta(y_t | x, y_<t)
      = logsumexp(z_t) - z_t[y_t]        (natural logarithm)
U_premise = mean(nll_t over tokens intersecting Premise value)
U_consequence = mean(nll_t over tokens intersecting Expected Consequence value)
U_local = max(U_premise, U_consequence) (only if both are valid)
U_global = mean(nll_t over all non-special response tokens)
~~~

Units are nats/token, range [0,infinity), higher means lower model likelihood. This is deterministic aggregation of frozen-model statistics: no probe/MLP/router training, labels, gradients, temperature fitting or parameter update.

Fixed three-line format for the independent test:

~~~text
Premise: <current condition>
Expected Consequence: <predicted immediate effect>
Action: <one proposed action>
~~~

Only field values enter local scores; labels are excluded by character interval. Global includes labels, newlines and Action as part of the entire response, with no prompt, EOS or padding. This definition is fixed here; removing formatting/Action tokens would define a separately versioned statistic. Phase 2 must preserve each benchmark native Action grammar if this format is selected.

The Phase 1 exact chat/response token alignment function is reused unchanged. We do not independently tokenize each field and assume its token IDs match the original sequence. A token belongs to a field if its response character interval intersects the trimmed field-value interval; use the whole token probability. Boundary tokens can include preceding spaces/delimiters, and every overhang is explicitly logged. No fractional character NLL is invented. Token overlap between the two separated values is rejected.

Strict format failure, duplicate labels, empty/unscorable field or token drift is never zero uncertainty. Missing fields are listed, U_local=null if either required field is missing, and valid global NLL can still be reported. A future Gate must select a documented missing-signal policy; no automatic WM fallback or fake score is implemented in Phase 1B. No future environment feedback enters the input or scorer.

## 2. Difference from SpanUQ

| Property | Intended official SpanUQ | Proposed NLL fallback |
| --- | --- | --- |
| Span selection | Learned semantic span detection from hidden states | Fixed syntactic field-value intervals; no semantic detector |
| Score source | Frozen pretrained DETR/MoB probe, learned heads | Direct next-token likelihood of the frozen Actor |
| Meaning | Learned uncertainty/consistency-related prediction for detected claims, conditional on its training domain | Lexical/model surprisal, not probability of an incorrect premise/consequence/action |
| Scale | Span score in [0,1], optional published calibration | Nonnegative, unbounded nats/token; uncalibrated |
| Evidence | Current official weights cannot be reliably loaded | Real raw-logit and teacher-forcing feasibility measured here |
| Parameters | Already trained probe is frozen | No additional learned parameters |

NLL can favor common wording, generic claims, copied observations or fluent errors. Long spans, tokenization and phrasing affect averages; a confident wrong premise can still receive very low NLL. Global averages can dilute a difficult short field. Thus the fallback does not solve overconfidence detection by definition and is not equivalent to SpanUQ or a true WM-utility predictor. Complementarity with external evidence and actual error association remain unverified research questions. No paper AUROC, task score or error-detection performance is claimed.

## 3. Two probability acquisition paths

| Path | Available statistic | Extra decoder forward? | Conditions |
| --- | --- | --- | --- |
| HF generation with output_logits=True | Chosen-token raw NLL; full-vocabulary entropy is also possible but is not used here | No additional backbone pass | Preserve original prompt/response IDs and logits; retain raw values before sampling transforms |
| HF output_scores=True | NLL under processed sampling distribution | No extra pass, but this is a different score | At T=.4 it is not base-model NLL; saved only as a diagnostic |
| Existing generation service returning token log-probs | Selected-token NLL potentially available | Potentially none | Must empirically establish raw vs processed semantics, complete selected-token coverage, EOS and chat alignment; not tested for current vLLM services |
| Saved response or service without trustworthy selected log-probs | Exact teacher-forced sequence rescore | One extra causal LM forward | Same pinned model/tokenizer, full original input IDs, correct shift and provider-specific dtype/backend |
| Top-k-only log-prob API | Chosen-token NLL if chosen token is returned; only partial distribution | Depends on API | Do not call truncated-distribution entropy the full entropy |

Teacher-forcing indexes logits[prompt_length-1 : prompt_length+response_length-1], targeting response_ids. The next-token shift has a dedicated analytic unit test. All reductions use FP32 logits/logsumexp even when backbone weights/compute are BF16. Neither path requires intermediate hidden states; the existing hidden extractor stays available for eventual SpanUQ. If HF generation lacks logits but a re-forward is used, final hidden states/logits and selected intermediate hooks can be captured in that same pass in a later validated adapter rather than launching two identical forwards. This optimization is planned, not implemented.

Proposed default if this route is selected: raw-generation NLL from the same fixed HF Actor generation path. Teacher-forced NLL is an explicitly different provider, not a silent recovery path. For future vLLM deployment, verify semantics before selecting generation-logprob or HF-rescore provider. No new vLLM service or integration was launched.

## 4. Numerical cross-check: preserve the failed BF16 result

BF16 cached generation and whole-sequence re-forward differed by maximum token NLL 0.1041–0.1717 nats, exceeding the predeclared 0.1-nat tolerance on all three samples. The real test exits 2 and this failure is preserved. The same repeated BF16 generation produced identical responses/scores, but equality across cached and full passes is not established.

An additional FP32 diagnostic, retaining the same frozen checkpoint weights and doing no training, passed: maximum raw-generation vs full-forward token NLL difference across its three samples was about 1.83e-5 nats. The FP32 test exits 0. This supports the token shift/alignment implementation and suggests a BF16 execution-path contribution; it does not prove the entire drift is caused by precision, because generated text is not identical in every BF16/FP32 case. No tolerance was increased after observing failures.

Therefore provider, dtype, backend and processing semantics must be versioned and fixed within an experiment. Do not interchange BF16 generation NLL, BF16 rescore NLL or FP32 NLL with a shared numeric threshold. If a downstream calibration uses these scores, it must use the exact deployed provider and development split. We do not recommend silently moving the Actor to FP32 in benchmark comparisons; FP32 here is diagnostic only.

## 5. Real standalone feasibility

Three synthetic observation/goal text prompts (door, plant, lamp) were used for real frozen Qwen3-4B inference. These are not collected trajectories or simulator feedback. Action correctness and task outcomes were not evaluated. BF16: T=.4, top_p=1, top_k=0 (HF disabled cutoff), repetition_penalty=1, seed=0, max_new_tokens=128, enable_thinking=false. Three fields were valid and all offsets/scores finite. Raw logits and temperature-processed scores were both saved, and they gave different NLL values as expected.

| BF16 case | Premise NLL | Consequence NLL | Local max | Global mean | Max raw-vs-rescore token drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00299638 | 0.00280412 | 0.00299638 | 0.00847669 | 0.11188765 |
| 2 | 0.00005934 | 0.11694770 | 0.11694770 | 0.03562068 | 0.10413088 |
| 3 | 0.05013156 | 0.03908388 | 0.05013156 | 0.03007612 | 0.17166395 |

All values above are actual model likelihood statistics, not calibrated uncertainty/error probabilities. Raw token NLL, field token indices/offsets, exact prompts/responses, teacher-forced values and processed sampling values are stored in phase1b/real_nll_bf16.json; the independent FP32 diagnostic is phase1b/real_nll_fp32.json.

| BF16 case | Online total s | Reduction/summary s | Extra validation forward s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.9393 | 0.0109 | 0.0414 | 7.701 |
| 2 | 1.3813 | 0.0008 | 0.0356 | 7.706 |
| 3 | 1.4204 | 0.0008 | 0.0355 | 7.707 |

Online total includes prompt preparation, generation, alignment and summary, excluding checkpoint loading/hash verification and the diagnostic teacher pass. Generation here retains both raw logits and processed scores; the processed copy is validation-only. These times are one small run, include cold/warm effects, and are not a measured incremental slowdown relative to generation without logits (that baseline was not timed).

Raw logits retained about 23–24 MiB per BF16 sample; the diagnostic processed-score copy retains a similar amount, plus stack/reduction temporaries. FP32 diagnostic peak was about 15.3 GiB. The raw NLL reduction adds memory, normalization/gather work, possible device transfer, formatting tokens and latency even with zero extra backbone forwards. It is not free inference.

## 6. Total Inference Cost accounting

Record provider, generation/retention overhead, response tokens, extra forward count and prompt+response forward tokens, logit buffer bytes, reduction/alignment latency and peak memory. Generation-only raw NLL adds zero decoder passes but full-logit retention is O(T*V); a later streaming selected-token reduction can reduce retention, but has not been implemented or benchmarked.

HF-rescore adds a full frozen Actor forward with language-model head over the exact sequence. In this short BF16 test it took about .036–.041 seconds; do not extrapolate to long benchmark prompts. A remote Actor plus local HF rescorer can also duplicate model residency and load cost. The existing SpanUQ hidden-state path cannot supply token probabilities without the final language-model head.

Count additional structured-draft tokens in every comparable Actor arm. Total system cost must still include WM prediction/reflection, external retrieval/judging/correction and offline memory synthesis separately. No prices are fabricated. Hypotheses about better success–cost remain untested.

## 7. Proposed unified Internal Signal Interface (design, not Agent integration)

~~~text
schema_version
status: ok | partial | unavailable | error
method: fixed_field_mean_token_nll_v1 | spanuq_official_<revision>
provider: hf_generation_raw | hf_teacher_forced | official_probe
local_uncertainty, global_uncertainty
premise_uncertainty, consequence_uncertainty
spans: field/text/char and token boundaries/score/span_origin
missing_span, missing_fields, missing_reason
score_semantics: unit/range/higher_means/calibrated/aggregation/boundary_policy
model_metadata: backbone + tokenizer + probe revisions, dtype, backend, prompt hash
cost: online/offline latency, calls, forward passes/tokens, retained bytes/cache
~~~

Prototype summarize_fields produces the numeric/missing/field portion; a production wrapper and Gate wiring are intentionally not implemented. The interface must preserve span_origin=fixed_field vs learned_detected and method-specific metadata instead of making NLL look like a SpanUQ output.

Future official SpanUQ can replace the provider behind this structural interface without rewriting the Agent/Memory flow. Numeric thresholds, missing-span behavior, field association, cost and calibration still need method-specific development validation. An unbounded NLL threshold cannot be reused for a [0,1] probe score. Optional dev-percentile normalization would be a separately named, frozen calibration, not automatic comparability or training a router. Configure one named backend per run; never silently fall back after a probe loading error.

## 8. Commands and tests

No dependencies/models were newly installed/downloaded. Use the Phase 1 isolated environment and existing pinned assets:

~~~bash
CUDA_VISIBLE_DEVICES= .venv-spanuq/bin/python scripts/test_token_nll.py
CUDA_VISIBLE_DEVICES=1 .venv-spanuq/bin/python scripts/token_nll_feasibility.py --backbone-dir checkpoints/qwen3-4b --device cuda:0 --dtype bfloat16 --output outputs/phase1b/real_nll_bf16.json
CUDA_VISIBLE_DEVICES=1 .venv-spanuq/bin/python scripts/token_nll_feasibility.py --backbone-dir checkpoints/qwen3-4b --device cuda:0 --dtype float32 --output outputs/phase1b/real_nll_fp32.json
~~~

10 CPU tests pass (analytic uniform/selected-token NLL, causal shift, invalid logits/alignment, local/global aggregation, malformed/empty/duplicate fields, boundary policy and non-probability scale). Unit fixtures are not research results. BF16 feasibility exits 2 because the cache/full equivalence check fails, despite valid field scores; FP32 diagnostic exits 0. Keep both outcomes. All 11 existing hidden-state/strict-loader tests also pass as regression checks (21 passing unit cases total). Python compile, documentation fence/whitespace and isolation checks pass.

## 9. Next-stage recommendation and limits

Choose route B after review, retain Qwen3-4B and the strict unavailable SpanUQ backend. In authorized Phase 2, first productionize the provider/schema and exact native action format, select a fixed dtype/backend, enforce missing/overflow policies and collect a small REAL development decision trace set. Investigate relation to actual errors without using test outcomes or training parameters. Then evaluate internal/external complementarity under the existing research plan. Do not infer usefulness from these three synthetic prompts.

No Phase 2 development, agent-loop change, threshold selection, full benchmark, memory synthesis or Gate training was performed in Phase 1B. User route selection is pending.
