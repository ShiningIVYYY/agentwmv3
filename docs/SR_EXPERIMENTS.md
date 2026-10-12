# SR baseline, Oracle and Word2World comparison

The experiment uses each environment's `osunlp/early-experience-<env>/sr`
checkpoint for both the actor and reviewer. The three treatments are `none`,
`oracle` and `word2world`, with 140 ALFWorld, 200 ScienceWorld and 500 WebShop
tasks per treatment (2,520 total). All task IDs and agent sampling settings
come from the existing audited EE-IL experiment manifests. Pure SR is run
again; paired outcomes always compare against this new SR baseline.

The copied AgentWM templates and maximum of three review rounds are retained.
ScienceWorld uses separate critique and guarded replanning. Existing JSON
whitespace constraints apply only when structured output is requested.

| GPU | Model | Work |
|---|---|---|
| 0 | ALFWorld SR, port 8201 | Baseline, then Oracle |
| 1 | ALFWorld Word2World, port 8111 | Predicted transitions |
| 2 | ScienceWorld Word2World, port 8112 | Predicted transitions |
| 3 | WebShop Word2World, port 8113 | Predicted transitions |
| 4 | ScienceWorld SR, port 8202 | Baseline, then Oracle |
| 5 | WebShop SR, port 8203 | Baseline, then both WM groups |
| 6 | ALFWorld SR replica, port 8211 | Word2World treatment actor/reviewer |
| 7 | ScienceWorld SR replica, port 8212 | Word2World treatment actor/reviewer |

Each treatment uses four isolated task workers. The three environments run
concurrently; WM treatments become eligible only after their own baseline
completes with clean worker exits. Thus the full WM stage can run 24 task
workers. The SR replicas use identical checkpoint files and decoding settings.

Before full evaluation, all nine groups must pass a four-task, three-step
preflight. The preflight checks configuration identity, real environment
steps, Oracle commit equality, review/replan budgets and usable model outputs.
It rejects systemic review-format/truncation failures. Model success is not a
preflight acceptance criterion. Preflight trajectories are separate from full
trajectories and full tasks start from the initial state.

The coordinator pins input hashes in `experiment.json`, records owned process
identities, and resumes through task locks without replacing completed tasks.
Pure-agent task queues use the same supervisor and clean-exit checks as WM
queues. A GPU service may be adopted after coordinator restart only if its
recorded process identity, checkpoint, endpoint and decoder configuration match.

Run from the repository:

```bash
/home/shn/miniconda3/envs/verl-agent/bin/python scripts/run_sr_experiments.py \
  --output /data1/shn/agentwmv2/outputs/wm_reflection/ee_sr_agentwm_prompts_v1
```

Monitor `RESULTS.md`, `summary.json`, `comparison.json`, and `manager.log` in
that directory. `comparison.json` includes matched completed task counts and
the task IDs for failure-to-success and success-to-failure changes. Pending or
errored tasks are excluded from paired statistics. Infrastructure errors remain
separately visible in the progress report.

Pinned model files are downloaded using manifests read from the official HF
API. Transport can use the mirror, but every file is verified against the
official Git blob or LFS SHA256 object hash before its provenance is published.
