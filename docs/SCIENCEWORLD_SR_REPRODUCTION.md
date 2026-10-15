# ScienceWorld SR baseline audit and context repair

Reported public checkpoint result: 66.0%; original local pure-SR result:
74/200 = 37.0%, mean score 40.895. The local success counter is correct:
74 tasks have terminal score 100. This is not an average-score conversion issue.

The public EarlyExperience repository supplies README-level instructions and
weights/data, but not the exact evaluation runner or concrete inference one-shot.
It specifies one-shot inference; its checkpoint table uses a unified training
setup differing from the paper's per-environment setup. Matching its 66.0 is an
empirical reproduction target, not a guaranteed property of this checkpoint.

Confirmed local failure pattern: many failed tasks repeatedly issue the heating
demo's metal-pot/furnace actions, even in unrelated task families. The legacy
one-shot has 54 messages and costs 2503 model tokens. Its training episode is
valid: replay at boil variation 5 reaches terminal score 100 after 26 actions.
The earlier suspicion that it was an unfinished episode was disproved and must
not be used as the explanation.

The new opt-in protocol uses a compact complete training example, puts an explicit
episode boundary before the real task, and repeats the current goal in the
latest user message. Legacy defaults remain unchanged. The short example costs
863 tokens. On an audited failure snapshot the same 4096-token budget retains
56 history steps instead of 28. This is a context repair based on observed
failures, not a claim that the authors used this exact prompt.

The example was selected before any corrected-policy rollout: sort released
training dialogues with at least four actions by message count, serialized
character count and source row index; match the source training action prefix;
exclude test IDs; accept the first whose goal matches and whose complete action
sequence reaches terminal score 100 on the installed simulator. Selected:
released expert row 1914, training ID sciworld_4509, task
lifespan-longest-lived-then-shortest-lived, variation 19, four actions.
Its original messages are retained. No test trajectory or gold action enters
the one-shot.

Pinned source dataset revision:
7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce.
Selected demo: data/paper_eval/sources/science_short_verified_demo.json.
Selection/replay evidence, source hashes, old failure audit, token measurements
and optional saved-state generation probes are in
outputs/wm_reflection_setup/science_sr_reproduction.

The installed production runtime is ScienceWorld 1.2.3 with JAR SHA256
f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7.
A 1.1.3 wheel was inspected in an isolated directory; production packages and
the live environment server were not changed. Runtime version differences were
not established as a causal explanation.

A separate pure-SR rerun uses exactly the same 200 AgentEval IDs, task mapping,
checkpoint, success criterion, 100-step horizon, 4096 prompt-token limit,
1024 response-token limit, temperature .4, top_p 1, repetition penalty 1 and
generation seed 0. Only context construction/one-shot and scheduling change.
It shares current inference replicas on GPUs 3..6, using eight environment
workers after a four-task, three-step preflight. It allocates no GPU model
services; GPUs 0, 1 and 7 stay released.

Start/resume:

    /home/shn/miniconda3/envs/verl-agent/bin/python scripts/run_scienceworld_sr_context.py --output outputs/wm_reflection/ee_sr_science_context_v3 --workers 8

Outputs: experiment.json, runstate.json, summary.json, and
none/scienceworld_ee_sr/tasks. Old results are preserved. The legacy cohort's
source transition records the opt-in code addition without restarting its
active evaluators.

New baseline numbers must not be paired against old WM trajectories generated
with the legacy context. WM arms need a matching rerun before making a fresh
WM-effect claim. Whether this repair reproduces 66.0 remains to be measured on
the full cohort; a preflight or saved-state probe is not a success-rate result.

Sources:
- https://github.com/OSU-NLP-Group/EarlyExperience
- https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/envs/scienceworld/README.md
- https://huggingface.co/osunlp/early-experience-scienceworld
