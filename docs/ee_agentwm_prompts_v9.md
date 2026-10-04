# EE-IL evaluation with copied AgentWM prompts

Run only EE-IL's ALFWorld, ScienceWorld and WebShop policies against both Oracle
and the official pinned Word2World checkpoints. Use a fresh result directory;
no full or preflight task results are inherited from v8. Sampling, task manifests,
agent checkpoints, original actor prompts, horizons and token limits remain those
of the audited pure-agent baselines. Each WM group has four independent workers.

```bash
/home/shn/miniconda3/envs/ee-qwen38-direct/bin/python scripts/run_wm_experiment_matrix.py \
  --output outputs/wm_reflection/ee_il_agentwm_prompts_v9 \
  --models ee_il --ee-workers 4 \
  --ee-reflection-protocol agentwm-copied-v1 --launch
```

Templates are byte-for-byte copies into `reflection/agentwm_*_prompts.py`.
`reflection/agentwm_prompt_provenance.json` records each origin filename and
SHA-256. Neither evaluator nor scheduler imports or reads the AgentWM repository.
The copied sources and the adapter are frozen in `experiment.json`.

ALFWorld JSON reviews and ScienceWorld JSON critiques use schema-constrained
decoding because the EE-IL checkpoints otherwise terminate incomplete JSON with
their learned action tags. The schema enforces the exact fields requested by the
copied template; it does not choose any decision/action value. Prompt text and
temperature/top-p/seed/token limits are unchanged. Ordinary actor calls,
ScienceWorld replan calls and WebShop reviews retain unconstrained generation.
The failed unconstrained preflight is retained separately; no full tasks ran in
that attempt. Exact response schemas are recorded in each generation log.
Brief JSON assessment/reason strings are bounded to 256 plain-text characters;
action/suggestion strings are bounded to 160. This prevents the fine-tuned policy
from repeatedly emitting native XML tags inside a JSON string until the token
limit. Decision booleans and action content remain model-generated.
For ALFWorld, the action field is constrained to the prompt's actual admissible
commands plus the original proposal, which remains available for a keep
decision. The model chooses among these commands; the adapter never supplies
an answer. ScienceWorld's non-exhaustive SDK combinations are not enumerated
as a decoder whitelist. Each actual schema, including ALFWorld's current command
set, is retained in the review generation metadata.

- ALFWorld uses `build_reflection_messages` with its original four-key JSON
  response: assessment/action/change/reason. Setting change=false keeps the
  original action, matching AgentWM. JSON must actually occur in the completion;
  a native action is never fabricated into a JSON review.
- WebShop uses `WEBSHOP_REFLECTION_TEMPLATE`, with native think/action tags.
- ScienceWorld uses `SCIWORLD_REFLECTION_TEMPLATE_CRITIQUE_REPLAN_V1`. Its
  execute_proposed/needs_replan decision can trigger a separate agent call using
  `SCIWORLD_GUARDED_REPLAN_TEMPLATE_V1`. suggested_action is non-binding. The
  separately generated action must be valid for the real current state and pass
  a same-prefix WM check for explicit invalidity or terminal bad focus. Missing
  or partial repair predictions are rejected. The copied XML replan output and
  the EE checkpoint's native Thought/Action output are both parsed; the executed
  history remains native Thought/Action, and raw completions remain in logs.

Review is bounded by three rounds. ScienceWorld replanning calls are counted
separately and do not create a fourth review. At most four physical WM forecasts
occur per step. Returning to a cached original branch does not advance state or
consume an extra forecast. Oracle commits must match the saved branch exactly.

The v8 evidence gate is not applied to this prompt suite: changing an action is
the reviewer's decision, subject to current-state validity and the ScienceWorld
replan guard. Original actor output is preserved on keep/fallback. Baseline
ScienceWorld actor calls still retain the full 54-message training dialogue;
review/replan calls use the copied AgentWM standalone templates and recent five
real steps, as that project does.
ScienceWorld's action-information field uses the original project's compact SDK
action templates and current object list. The Oracle SDK getters and the HTTP
environment's `/action_hint` endpoint provide the same two fields. Expanded
action/object combinations remain available for action validation but are not
inserted into the reviewer prompt. Current ambiguity choice labels are preserved.

Unlike v8, the copied templates use transition metadata. Oracle supplies its
predicted reward/done/success/score where applicable. Word2World supplies its
own learned `[SUCCESS]` marker only for complete forecasts; reward/done/score
are None (unknown). No real-environment labels are fetched for learned WM
reviews. A source/availability note explains that the copied word "oracle" is
a legacy label in learned-WM calls. Partial forecasts are explicitly identified.
This metadata change must be considered when comparing v9 against v8.

All review/replan inputs obey the original 4096-token agent prompt budget. Old
history pairs are removed before clipping only the hypothetical preview. A
mandatory-context overflow is logged as a budget skip with original-action
fallback, never as a fabricated successful review.

The scheduler validates all six four-worker preflights before full execution,
then runs ALFWorld, ScienceWorld and WebShop in order. GPU 0 serves the appropriate
EE-IL checkpoint; GPUs 1/2/3 serve the three world models. GPUs 4–7 are not used.
The existing Qwen2.5 service is restored after the EE-only matrix finishes, but
no Qwen tasks are evaluated.

Progress: `summary.json`, `RESULTS.md`, `manager.log`, and each WM/benchmark's
`progress.json`. Steps retain exact prompt messages, model output, critique,
replan output, guard decision, WM forecasts and real committed feedback.
