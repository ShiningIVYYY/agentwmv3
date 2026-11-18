# Utility sampling v2 results

Paired TRAIN-pool Never/Always source trajectories, temporal/public-risk sampling and repeated same-state interventions.

Real decision cases: 2651; selected state labels: 229.

State labels: {"neutral": 214, "uncertain": 11, "help": 4}

Paired task outcomes: {"always_success_never_success": 12, "always_failure_never_failure": 19, "always_success_never_failure": 5}

Validation/Calibration expanded: False. No final Test or independent Dev task was rolled out.

No Actor/Reviewer/Memory/Gate changes. No automatic learned Gate deployment. Full actual examples, seed differences, cost and heads: [results](utility_sampling_v2/results/summary.json).

Insufficient positive labels remain a feasibility limitation; no success or future feedback was invented.
