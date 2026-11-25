# Utility-v5 resource recovery

Status: connection restored; the prior7 tests and updated9-test adaptive-resource/retained-corpus suite pass. Actual model startup is recorded separately; no new utility outcome is claimed.

The original V5 completed100 densification task scans. It produced294 candidate
records:125 Neutral,22 Uncertain,6 Help,1 Harm and140 whole-state budget-Unknown.
Unknowns remain excluded from hard labels and are not retried in this recovery.
Combined Train has16 Help across13 goals. Validation retains2 Help. No Probe
has trained and no independent task-performance improvement is claimed.

At phase switch, physical GPU6 was externally occupied while V5 released/reloaded
models. Worker1 failed at model.to with CUDA OOM: external PID2793087 used15.81GiB
and the loading worker used7.60GiB. Worker0 was stopped during read-only corpus
loading. New Train issued zero model requests/native queries and produced no
cases. Interrupted loading consumed compute; exact failed-load GPU time is unknown.
There are no new unknown model-output receipts. Earlier historical2 unknown
outputs/1,890-token upper charge remain unchanged.

Preserve outputs/utility_expansion_v5 and old code. The recovery receipt hashes
old logs/data/counters and retains100 completed task markers and all294 labels.
Resume only the SAME80 Train and40 Validation goals. Collect the fixed Validation phase first, then Train, under current scarcity; no model/memory update occurs between them. The retained corpus has9628 cases and1076 labels after importing densification. Model, Reviewer, prompts, BF16 scoring,
thresholds, Memory, source/continuation seeds, label standards and split stay fixed.

Scheduling chooses1-4 actually empty4090 cards, checks again before EACH phase/spawn,
waits at most2h within the ORIGINAL deadline and takes advisory local GPU leases.
Other users do not honor these leases; this reduces phase-switch races and cannot
guarantee exclusive hardware. Do not terminate external jobs or blindly retry a
partial experiment. Preserve receipts if a future race occurs.

Original V5 known cost:11,503,917 tokens/21,556 model calls/55,892 native transitions/
56,200 state receipts. Native preflight165 transitions/175 receipts is charged once.
Remaining original global caps:52,496,083 tokens/98,444 calls/543,943 transitions/
603,625 state receipts. Keep original manifest deadline, not a new72h window.
Phase caps remain32M Train/20M Validation. Combined old+recovery output size must
stay below90GiB, minimum free disk50GiB. No final Test/freshDev/Phase6/deployment.

Run only from the existing isolated Python3.12 environment:

```sh
.venv-qwen35-clean/bin/python scripts/prepare_utility_v5_recovery.py
# Never rerun if the registered receipt/config already exists.
.venv-qwen35-clean/bin/python scripts/test_utility_v5_recovery.py
# Record ACTUAL passed test results before launch.
.venv-qwen35-clean/bin/python scripts/recover_utility_v5.py --run outputs/utility_expansion_v5_recovery
```

The preparation attempt using system Python3.9 failed at type-annotation import
before writes/inference. The isolated-env preparation then succeeded and registered
the receipt/config. The original7 tests completed and passed while SSH was unavailable. Updated adaptive-resource tests:9 pass in outputs/utility_v5_recovery_tests_v2.log. Before launch, freeze source and actual test proof; then audit strict model loading/fresh cases/ledgers before background operation. Root GPU list denotes maximum capacity; each phase writes a hash-checked runtime config differing ONLY in its actual GPU list. Original deadline remains authoritative.
