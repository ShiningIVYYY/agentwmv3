# Activation Probe v1: engineering validation and experiment status

Implementation and engineering smoke passed. The supervised data collection, classifier metrics and independent Episode comparison are not yet complete.

- 14 CPU tests pass.
- 77 native task identities and 5 separate real replay controls pass; no final Test reset.
- Real Qwen3.5-9B: selected post-token tensors [35,4096], repeat difference0, future-suffix prefix difference0, frozen Actor parameters0.
- Full experiment results are pending; no accuracy, utility or success-rate improvement is claimed.

See [implementation, commands, data roles and budgets](ACTIVATION_PROBE_IMPLEMENTATION.md) and [real receipts](activation_v1/real_model_smoke.json).
