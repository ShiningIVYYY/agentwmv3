# Legal training expansion v3: startup status

The new bounded collection is running after real startup validation; final additional positive counts and Probe training are pending.

- Reuse original36 TRAIN goals/2651 real cases; collect fixed24 new goal-disjoint TRAIN targets. Original Val/Cal/Dev/Test roles retained.
-44 CPU tests pass.29 new native identities and5 replay controls pass, with zero model calls in preflight.
- Three strict frozen Qwen3.5-9B Actor/Reviewer loads on GPU5/6/7.
- Startup checked117 new real decisions,30 cached features,2 new paired tasks and8 state labels; timestamps/contracts/ledger prefixes pass, no errors. Initial new labels are Neutral, not evidence that actions are correct.
- Old raw corpus is unchanged. Correct5 mixed-seed Neutral labels to Uncertain in derived V3 data; all4 prior Help candidates preserved.
- Complete all24 new targets before checking the unchanged>=5 Help/across>=3 Train-task rule; only then collect fixed Val/Cal and attempt eligible small-Probe training. No automatic Gate deployment or independent/final evaluation.

See [protocol](UTILITY_EXPANSION_V3.md), [startup evidence](utility_expansion_v3/startup_audit.json), and [per-case legacy label correction](utility_expansion_v3/label_corrections.json). Final results will replace this status.
