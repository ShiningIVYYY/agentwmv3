# WM controls on the corrected ScienceWorld SR baseline

## Baseline audit

The completed pure SR context-v3 cohort has 200 completed tasks, zero task
errors, 123 terminal score-100 successes (61.5%), and mean score 50.91.
All 7,056 model generations end with stop, all parse correctly, all prompts
stay within 4096 tokens, and every latest user message contains the real goal.
No step contains WM reflection. The mean score differs from the success rate
because ScienceWorld can return negative scores on terminal failure.

The legacy cohort has 74/200 successes (37.0%). On matched IDs, context-v3
improves 70 tasks and degrades 21. These are different prompt protocols, so
this is a context comparison, not a WM-effect result. The result remains
4.5 percentage points below the public checkpoint report of 66.0%; it is not
claimed as an exact reproduction of the unpublished author evaluation.

## WM experiment

Output root: outputs/wm_reflection/ee_sr_science_context_v3

The existing none/scienceworld_ee_sr directory is read-only reference data.
New arms are oracle/scienceworld_ee_sr and word2world/scienceworld_ee_sr,
each over exactly the same 200 fixed task IDs and native task/variation map.

The v2 method is retained from outputs/wm_reflection/ee_sr_agentwm_prompts_v2:
- Locally copied AgentWM ScienceWorld critique followed by a separate guarded
  replan; maximum three review rounds.
- The same SR checkpoint for actor, critique and replan.
- Oracle uses live JVM snapshot/rollback and verifies the adopted transition.
- Word2World uses the same pinned ScienceWorld checkpoint, config and source
  revision. It does not receive Oracle reward/done/success/score metadata.
- The same action validation, prediction budgets, JSON/regex constraints,
  temperature .4, generation seed 0, response limit 1024, prompt limit 4096
  and task horizon 100.

The corrected task-anchored context and the exact verified short training
one-shot are used by the actors and native replan context in both arms,
matching the completed pure baseline. Critique/replan templates and WM source
hashes are checked against the v2 frozen manifest before any launch.

Four tasks with three steps per task preflight each WM arm. The existing v2
validators check prediction/commit agreement for Oracle, review/replan format,
reflection budgets and accepted replan provenance. Both preflights must pass
before either full arm starts. Full arms use 16 independent environment workers
each, 32 total. Native process exits are checked; task JSON cannot hide a
crash during environment teardown. Each task keeps its exclusive queue lock.
Completed results survive retries.

## Resources and records

Existing ScienceWorld SR replicas on GPUs 3..6 and the WM on GPU2 are reused.
Released GPUs 0, 1 and 7 are outside the layout. The manager does not stop or
allocate services on those cards. The existing router can show its retired
GPU7 URL as unhealthy; only the four active URLs are required.

The new manager is separate from the completed pure-baseline manager:
- wm_experiment.json: frozen inputs, v2 method hashes, baseline result hashes.
- wm_baseline_audit.json: baseline gating and immutable result references.
- wm_runstate.json / wm_manager.json / wm_events.jsonl: process records.
- wm_summary.json / wm_comparison.json / WM_RESULTS.md: live results and paired
  failure-to-success / success-to-failure counts against context-v3.
- wm_preflight/{oracle,word2world}/scienceworld_ee_sr: preflight results.
- oracle/... and word2world/...: full results and isolated worker logs.

Original experiment.json, runstate.json, summary.json and pure task JSON files
are preserved. None of the legacy v2 WM outcomes enter the new paired counts.

Start/resume:

    /home/shn/miniconda3/envs/verl-agent/bin/python scripts/run_scienceworld_sr_wm.py --output outputs/wm_reflection/ee_sr_science_context_v3 --workers 16

Status:

    /home/shn/miniconda3/envs/verl-agent/bin/python scripts/run_scienceworld_sr_wm.py --output outputs/wm_reflection/ee_sr_science_context_v3 --status
