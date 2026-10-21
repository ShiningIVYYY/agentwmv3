# ScienceWorld SR: evidence-gated intervention experiments

This is a new experiment, preserving the completed context-v3 pure SR and
AgentWM-prompt outputs. It does not overwrite the copied AgentWM templates.
The actor checkpoint, actor context, training-only one-shot, 200-task manifest,
100-step limit, 4096/1024 prompt/response limits, temperature 0.4 and seed 0 stay
fixed. The baseline is the frozen 123/200 corrected pure SR run, not the old 37% run.

## Frozen stages

1. `stage1_guard`: Oracle and Word2World, success preservation and evidence-gated
   criticism. No replacement merely for zero reward or a different plausible plan.
2. `stage2_compare`: same, plus explicit original/alternative comparison using
   forecasts from the same current real state. Uncertain comparisons preserve
   the original, except a verified safe fallback avoids known terminal failure.
3. `stage3_memory`: same, plus source-labelled historical excerpts from committed
   observations. They are not asserted to remain current and contain no forecasts.
4. `stage4_observation_only`: Oracle with stage 3 policy but all transition reward,
   done, success and score fields hidden from criticism, comparison and selection.
   This measures the benefit of state prediction separately from privileged labels.

The seven full arms have 200 tasks each. Stages run in this fixed order regardless
of measured test success. No training, prompt selection or threshold tuning uses
the fixed evaluation outcomes. Each stage's six-step interface preflights use six
deterministically selected training task IDs, disjoint from test and the demo.
They cover two conductivity families, temperature measurement, melting point,
animal lifespan and living-object search. Goal hashes and native mappings are
verified by the installed simulator before use.

## Policy details

The reviewer emits a finite JSON schema with a typed defect and verbatim evidence.
Fabricated quotes, contradictory fields and unsupported defects preserve the
original action. Explicit terminal failures are checked for all actions, not only
focus. Available-action syntax is not treated as proof of executability: replacement
forecasts are also checked. Repeated waits, heating, connections and focus are not
automatically declared loops. Information/navigation loops require three identical
committed action/feedback pairs and the same predicted feedback.

At most three review/replan attempts are allowed per environment action. Forecasts
are cached; alternatives do not modify committed history. The chosen action must
be the final forecast, including a cached-original record when reverting. Oracle
heap snapshots verify and commit the chosen transition once. Word2World never
reads true future reward, done, success or score. Its success marker is a forecast,
not a reliable simulator label. No reset/teleport replacement is allowed.
If every model replan is rejected and the original is a trusted terminal failure,
one additional forecast can verify a nonterminal `look around` fallback. This is
not a fourth review round (at most five distinct forecasts in this rare case).

Oracle terminal success protection uses trusted done=True, success=True, score=100.
It is disabled in the observation-only arm. Historical excerpts retain exact text,
source step and action, including whether an excerpt was truncated. They keep four
early and eight later relevant excerpts. Prompt overflow drops whole recent history,
then historical evidence, then object hints; required goal/state/forecast is never
silently truncated. If the required packet cannot fit, preserve the original action.

## Run and monitor

Only GPU 2 (Word2World) and GPU 3--6 (four SR replicas) are used. GPU 0, 1 and 7
remain released. Paired arms have 16 workers each; the final single arm has 32.
Model services and the existing inference router are reused after identity checks.

```sh
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-scienceworld/bin/python \
  scripts/prepare_science_evidence_dev.py --output OUTPUT/dev_manifest.json
/home/shn/miniconda3/envs/verl-agent/bin/python \
  scripts/run_science_evidence_experiments.py --output OUTPUT --workers 16
```

`OUTPUT/RESULTS.md`, `summary.json`, `comparison.json`, `events.jsonl`, and each
arm's `progress.json` expose progress and paired improvements/degradations.
Every preflight and full arm receives configuration, success-accounting, budget,
forecast/commit, success-protection and metadata-isolation checks. Completed task
files survive infrastructure retries; native worker exits are observed. New source
hashes, baseline task hashes and input provenance are frozen in `experiment.json`.

Report both successes-to-failures and failures-to-successes, action replacements,
protected successes, rejected interventions, comparison calls/tokens and total
model cost. A partial run is not a final success-rate estimate. These checks do not
guarantee a better final success rate; they make the proposed corrections testable.

## Native duplicate-label repair

ScienceWorld can return distinct numbered options with identical printed labels.
Unique-label menus retain the existing canonical sorting and number translation.
Duplicate-label menus now preserve the full native observation and native numbers;
objects are not collapsed and distinguishing information is never invented.

The stage-1 Oracle `sciworld_581` hit this adapter exception on three runner
launches. The manager correctly withheld stage 2 while one task was unresolved.
An explicit maintenance transition archives the old source/configs and hashes all
399 completed task results. These results remain byte-identical. One additional
launch is recorded for the failed task; the original retry count is retained.
The Oracle stage-1 aggregate includes the retained original cohort and the one
adapter-repair continuation task, identified in `maintenance_transitions`.
