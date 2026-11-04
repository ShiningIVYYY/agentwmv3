# Phase 5 small-scale ScienceWorld results

This is fixed **development functional validation**, not the final200-task benchmark. Five goal-disjoint held-out development tasks, rollout seed0, four Gates, complete episodes up to20 interactions. No across-seed sample SD can be estimated from one seed. Means weight five task instances equally, including early failures; neither batches nor task categories are reweighted.

| Policy | Mean final Task Score | Success (score>=100) | Env steps | Gate calls | WM forecasts | Total model calls | Total tokens | Online seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| never | -9.00 | 0% | 70 | 0 | 0 | 70 | 27242 | 93.797 |
| always | -29.00 | 0% | 65 | 65 | 190 | 423 | 900153 | 543.016 |
| internal | -29.00 | 0% | 72 | 21 | 63 | 190 | 351690 | 285.148 |
| full | -29.00 | 0% | 73 | 21 | 62 | 190 | 349342 | 290.145 |

| Development task | Never | Always | Internal | Full |
| --- | ---: | ---: | ---: | ---: |
| sciworld_742 | -100.0 | -100.0 | -100.0 | -100.0 |
| sciworld_128 | 0.0 | 0.0 | 0.0 | 0.0 |
| sciworld_2417 | 0.0 | 0.0 | 0.0 | 0.0 |
| sciworld_1569 | 55.0 | 55.0 | 55.0 | 55.0 |
| sciworld_3070 | 0.0 | -100.0 | -100.0 | -100.0 |

Native raw negative scores are retained. Success is the auxiliary final score>=100 diagnostic. Native termination and horizon truncation are recorded separately per episode in [results.json](phase5/results.json). Initial state and first Draft tokens match across all four policies for each task. Later state sequences are not assumed equal.

Budgets were frozen before outcomes: Never0, Always20, Internal5, Full5 **invocations per episode**, with4-forecast reservation per invocation and actual forecast settlement. These deliberately different functional budgets do not constitute a matched-cost efficacy comparison. Correction and confirmed-action skip are off for all Gates; unused signal modules do not run.

## Online inference costs

| Cost component (all5 tasks) | Never | Always | Internal | Full |
| --- | ---: | ---: | ---: | ---: |
| draft_calls | 70 | 65 | 72 | 73 |
| reflection_revision_calls | 0 | 168 | 55 | 55 |
| wm_calls | 0 | 190 | 63 | 62 |
| internal_signal_seconds | 0 | 0 | 0.275150 | 0.270384 |
| retrieval_calls | 0 | 0 | 0 | 73 |
| retrieval_seconds | 0 | 0 | 0 | 0.021003 |
| external_judge_seconds | 0 | 0 | 0 | 0.001766 |
| gate_rule_seconds | 0.000617 | 0.000694 | 0.001215 | 0.001050 |
| signals_evidence_gate_seconds | 0.035320 | 0.034319 | 0.042758 | 0.071327 |
| wm_prediction_http_seconds | 0 | 74.815408 | 41.221817 | 44.911377 |
| wm_context_setup_seconds | 0 | 20.813924 | 21.292851 | 21.390968 |
| agent_generation_seconds | 76.262159 | 421.231241 | 195.236351 | 196.684058 |
| reflection_revision_generation_seconds | 0 | 339.301117 | 108.915561 | 109.377258 |
| episode_total_latency_seconds | 93.796951 | 543.016455 | 285.147745 | 290.145186 |
| draft_prompt_tokens | 25014 | 25392 | 33721 | 32040 |
| draft_completion_tokens | 2228 | 2419 | 2520 | 2556 |
| draft_total_tokens | 27242 | 27811 | 36241 | 34596 |
| reflection_revision_prompt_tokens | 0 | 257208 | 100676 | 99443 |
| reflection_revision_completion_tokens | 0 | 9750 | 3070 | 3096 |
| reflection_revision_total_tokens | 0 | 266958 | 103746 | 102539 |
| wm_prompt_tokens | 0 | 601568 | 209527 | 209814 |
| wm_completion_tokens | 0 | 3816 | 2176 | 2393 |
| wm_total_tokens | 0 | 605384 | 211703 | 212207 |
| total_prompt_tokens | 25014 | 884168 | 343924 | 341297 |
| total_completion_tokens | 2228 | 15985 | 7766 | 8045 |
| total_total_tokens | 27242 | 900153 | 351690 | 349342 |

The full20-episode diagnostic used **1,628,427 tokens / 873 model calls / 1212.106s** summed episode wall time. Model load/hash, WM service startup, environment validation, historical training-data collection and distribution calibration are separate. Whole runner wall time: 1225.637s; Actor load/hash: 12.992s. Actor peak torch allocated memory: 8.418GiB; the dedicated vLLM WM service separately reserves its KV cache and model memory. See server GPU observations/provenance; torch allocation is not nvidia-smi total VRAM.

Agent generation time already includes review generation; reflection generation time is a subset and must not be added twice. Internal extraction includes raw-logit reduction/field scoring with zero extra backbone forward passes. Signals/evidence/Gate time overlaps retrieval and Gate rule time; component times are descriptive, not additive wall-time terms. Total tokens include complete structured Drafts, labels and Action, all reviewer prompts/replies and every WM forecast. Additional premise/consequence value tokens by policy: `{"never": 1241, "always": 1484, "internal": 1513, "full": 1521}`; these are included, not added twice. No action-only paired generation was run, so a net incremental reasoning-token saving/cost is not claimed.

Offline old rule synthesis:0 LLM calls/0 tokens,0.005727s. Phase5 train-only third-card extension:0 new cases/0 new model calls/0 tokens,0.022964s. Historical Phase3 collection:48 Actor calls /1,726 output tokens including EOS /59.252s generation /0.1812s extraction; six controlled training probes add0 model calls and0.7163s native step-request time. Historical input-token totals, failed-first-run cost and complete startup overhead were not fully profiled; they are unmeasured, not zero. See PHASE3_REPORT.md. Development distribution calibration:273 actual Actor calls, 116,215 tokens,426.011280s whole run, no WM. Canary/probes are setup diagnostics, excluded from these20-episode metrics. The WM remained resident on its dedicated GPU during this sequential cohort, including Never; no idle reservation is converted into a fictitious per-request inference charge. No reliable monetary tariff or electrical-energy measurement is available.

## Causal and scientific diagnostics

- Replay checks 280 real decisions: 107 Call / 173 Skip, 315 observed WM HTTP forecasts. Native receipts, query timestamps, evidence resolution, exact Gate outputs, budgets, score/cost recomputation and source archives agree. Independent server wire counts are in [server_wire_audit.json](phase5/server_wire_audit.json).
- Internal: Pearson0.893067 / Spearman0.898846 on273 development decisions. After threshold freeze, explicit native failure feedback appears in47/68 above-Local-threshold decisions and113/205 at/below threshold. Other feedback is **not** a correctness label. These dependent in-sample observations are preliminary, not an error-probability calibration or a test-set detector AUROC.
- External: 11 low-Local-NLL development decisions conflict with the already-existing closed-route evidence in retrospective replay. This shadow analysis did not drive those Actor-only trajectories. Live Full: coverage true 10/73, conflict true 3, sufficiency true 0. New-snapshot retrospective natural sufficiency cases: 0. Do not backdate the new card into calibration-time online claims.
- WM: 79/107 invocations changed the action; 40 changed actions received explicit real failure feedback. Final forecasts unavailable/incomplete: 0. An action change or failed revised action alone does not establish causal benefit/harm: its unexecuted alternative outcome is unknown. No same-state Call/Skip counterfactual rollout was performed.
- Reviewer: all278 responses are syntactically valid JSON.34 `invalid_format` labels mean a semantic protocol violation (replace chose the same candidate), not JSON decoding failure;50 `invalid_action` labels reject unsupported replacement commands. Full examples/reasons: [reviewer_quality.json](phase5/reviewer_quality.json). These errors and weak grounding remain visible; no ad hoc critic repair is used.

Representative true decisions: [representative_traces.json](phase5/representative_traces.json). Full model/native trajectories and executed source remain on lab under `outputs/phase5_episodes_v1`; immutable hashes/sizes: [raw_artifact_manifest.json](phase5/raw_artifact_manifest.json). Compact acceptance/scientific diagnostics: [acceptance.json](phase5/acceptance.json). No synthetic fixture is included in these results.
