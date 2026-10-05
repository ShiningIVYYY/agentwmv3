# JSON whitespace repair and experiment continuation

Copied AgentWM JSON reviews could generate unlimited spaces/newlines between
fields until the unchanged 1024-token limit. String-length constraints do not
bound JSON structural whitespace. This caused multi-minute reviews and retries.

The EE-IL copied-prompt matrix now starts each agent service with:

```text
--structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}
```

This applies XGrammar's fixed JSON separators. Normal spaces inside action,
assessment and reason strings are preserved. The copied prompt sources, JSON
fields, current-action enums, sampling, checkpoints, token limits, task lists,
three-review bound, and ScienceWorld separate replan/WM guard stay the same.
Unconstrained actor calls and the ScienceWorld XML replan regex are unaffected
by the JSON whitespace option. The original Qwen service is restored without
this EE-only option.

A ready HTTP endpoint is insufficient to validate this fix: the scheduler also
checks that the repository-owned GPU-0 service has the required startup flags.
Switching benchmarks retains the option, so ScienceWorld critiques receive the
same fix. Engine startup logs and service-event commands record the actual flag.

The interrupted v9 outputs remain in their original directory. Only completed
task JSON files are copied unchanged to the continuation; `resume_provenance.json`
records their IDs, byte hashes, original source/configuration and the intentional
maintenance stop. In-flight tasks restart from their initial environment state.
Copied results and provenance are frozen inputs of the new matrix.

Aggregate continuation scores contain two decoder cohorts. `summary.json` and
`comparison.json` report `before_whitespace_fix` and `after_whitespace_fix`
separately, including reflection health and paired success changes. Do not
interpret the aggregate as a homogeneous run of the new decoder. All six real
four-worker preflights are run again before continuing the full task queue.

```bash
/home/shn/miniconda3/envs/ee-qwen38-direct/bin/python scripts/run_wm_experiment_matrix.py \
  --output outputs/wm_reflection/ee_il_agentwm_prompts_v10_no_json_whitespace \
  --models ee_il --ee-workers 4 \
  --ee-reflection-protocol agentwm-copied-v1 --launch
```

Validation includes the unit suite, actual installed XGrammar acceptance/rejection
of looping whitespace with intact string spaces, replay of captured slow review
inputs through the real EE-IL checkpoint, and the six simulator preflights.
