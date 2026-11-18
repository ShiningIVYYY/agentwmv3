# Utility sampling v2: startup status

The bounded experiment is running in the background after real startup validation. No final positive-sample or Probe-effectiveness claim is made.

- Same frozen Qwen3.5-9B Actor/Reviewer, original prompt/dtype/Memory/Gate hashes unchanged.
- 15 new plus14 prior CPU tests pass.
- Three strict model loads on GPU0/3/5; no missing/unexpected/mismatched weights.
- Startup checked185 real source decisions and30 cached feature records: BF16 finite tensors, valid field masks and pre-outcome timestamps.
- Two same-task Never/Always source pairs completed; exact common-state divergence observed.
- Eight initial state labels are Neutral under the final-success definition; one changed-action state has three observed deltas[0,0,0]. These are TRAIN diagnostics, not independent performance.

See [protocol, sampling and hard budgets](UTILITY_SAMPLING_V2.md), [startup audit](utility_sampling_v2/startup_audit.json). Final results will replace this status after collection/audit. No final Test or prior independent-Dev task is used.
