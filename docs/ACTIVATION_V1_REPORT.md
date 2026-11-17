# Activation Probe v1: engineering validation and experiment status

Implementation and engineering smoke passed. The supervised data collection, classifier metrics and independent Episode comparison are not yet complete.

- 14 CPU tests pass.
- 77 native task identities and 5 separate real replay controls pass; no final Test reset.
- Real Qwen3.5-9B: selected post-token tensors [35,4096], repeat difference0, future-suffix prefix difference0, frozen Actor parameters0.
- Full experiment results are pending; no accuracy, utility or success-rate improvement is claimed.

See [implementation, commands, data roles and budgets](ACTIVATION_PROBE_IMPLEMENTATION.md) and [real receipts](activation_v1/real_model_smoke.json).

Startup confirmed: detached supervisor PID1506811, GPUs0/6/7, source commit c21aeb4b2fc19f0241e736dde810c1859524ce51. All three models strictly load with empty missing/unexpected/mismatched/error lists and remain frozen. The startup audit checked45 real cached decisions, field masks, BF16 finite tensors, pre-outcome timestamps, unchanged source contracts and token-ledger prefixes. The controller remains in real-data collection; supervised training and independent Episode results are pending. See [startup audit](activation_v1/startup_audit.json).

First completed paired labels are training-data diagnostics only, not independent task performance. The bounded controller may deploy only heads with sufficient real labels, and will generate the final results here after evaluation. It is left running in the background after startup verification, as requested.
