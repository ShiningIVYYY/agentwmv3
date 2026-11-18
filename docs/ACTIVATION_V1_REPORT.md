# Activation Probe v1 pilot results

This new protocol trains only small probes; Actor/Reviewer remain frozen. Historical47-task outcomes are not merged.

| Policy | Success | SR | Tokens | WM calls | Format aborts |
|---|---:|---:|---:|---:|---:|
| admissible | 3/12 | 25.0% | 283458 | 10 | 0 |
| always | 5/12 | 41.7% | 664080 | 488 | 0 |
| never | 3/12 | 25.0% | 258299 | 0 | 0 |
| nll | 3/12 | 25.0% | 306399 | 54 | 0 |
| nll_external | 3/12 | 25.0% | 306399 | 54 | 0 |
| periodic | 3/12 | 25.0% | 316678 | 54 | 0 |
| random | 2/12 | 16.7% | 333654 | 49 | 0 |
| review_only | 4/12 | 33.3% | 299164 | 0 | 0 |

Utility label counts: {"neutral": 58, "harm": 2}

Utility deployment: False. Insufficient heads are omitted, not fabricated.

Full train/validation/calibration metrics, all3 seeds, selected layer/architecture, costs and limitations: [JSON](activation_v1/results/summary.json).

No significance or causal generalization claim. Native rejection direct-rule baseline is essential; Risk classification alone is not WM Utility. No original benchmark results were overwritten.
