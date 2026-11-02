# Phase 4 — Training-Free Internal–External Invocation Gate

This module makes an invocation decision, not an action correction or a calibrated WM-utility prediction. It uses no parameter training. The accepted Internal Signal configuration, Experience Memory snapshot and PAPER_ALIGNED_EVAL.md remain unchanged. All runtime integration is opt-in; legacy runs without agent.gate_session retain their old path.

## Inputs and scoring identity

The internal provider is hf_generation_raw_bf16_v1: frozen Qwen/Qwen3-4B @1cfa9a7208912126459214e8b04321603b3df60c, BF16 model parameters, FP32 raw-logit reduction BEFORE temperature, Transformers4.57.3/torch2.8.0. Local=max(mean Premise-value token NLL, mean Expected-Consequence-value token NLL). Global averages the entire textual response including labels/newlines/Action, excluding prompt/EOS/padding/special tokens. These are nats/token, not probabilities of action error or SpanUQ outputs. No re-forward/dtype/scoring-path substitution is allowed.

Configuration includes the complete accepted Actor-config fingerprint plus backend, dtype, model revision, scoring path, method and unit. The runtime validates the Actor fingerprint; the Gate validates per-signal identity. A mismatch raises an error rather than silently reusing thresholds. Missing/invalid values remain missing, not zero. Valid numeric zero remains a low score.

The external provider still has two door predicates and54 train Cases, including6 separately labelled real scripted environment-probe steps from approved train1667/1545. No dev/test feedback is inserted into Memory. All source/canonical/goal/time checks are repeated when loading the frozen snapshot. BM25 relevance is never an input to reliability or a substitute for applicability.

External coverage/conflict/sufficiency use the strings true/false/unknown. gates.evidence binds claims to the exact query hash, snapshot, known Card/Case IDs, supported reliability, absence of counterexamples, availability time and explicit current-state/action applicability. Unsupported, malformed, future or inapplicable claims cannot assert confirmation. Unknown is retained in logs and explicitly handled, not converted to False. Missing evidence does not mean no conflict.

## Action status and sufficiency

| Status | Meaning | Memory Correction off |
| --- | --- | --- |
| confirmed_action | Reliable evidence confirms the CURRENT unchanged draft within the provider's bounded claim scope | Optional confirmation skip only; default disabled |
| correctable_conflict | Input declares a conflict with a supported correction proposal | Request WM; do not execute the proposal or treat it as resolved |
| unresolved_conflict | Reliable evidence identifies a problem with the current action | Request WM |
| unknown | Insufficient/unverified/inapplicable/unsupported evidence | Full Gate falls back to internal signals |

The current Cards contain no suggested corrections, so natural reliable conflicts are unresolved. A proposed correction is not an executed correction. Conflict=true always wins over sufficiency=true, even if those claims are inconsistent. No code here changes the action through Memory. A future correction module needs its own action/evidence/validation interface and shared settings across all compared Gates; merely adding a suggestion must never activate it.

The current provider's positive sufficiency concerns a narrowly supported door-opening immediate effect, with exact condition/effect wording. It does not establish task progress, general action safety or WM utility. Phase3 had no natural positive sufficiency examples. Therefore allow_confirmed_action_skip=false in the actual Phase4 configuration. The opt-in branch is CPU-tested with explicit confirmed-action inputs, not validated as a useful real skip policy. Enabling it requires separate real development validation; it cannot mask a reliable conflict or an invalid draft format.

## Rule and priority

With confirmation skipping disabled, Full uses:

~~~text
requested = reliable_action_conflict
            OR U_local > tau_local
            OR U_global > tau_global
            OR missing_required_internal_signal
            OR invalid_draft_format
call_wm = budget_available AND requested
~~~

Missing local uses valid Global against tau_global. Missing/invalid Global for a Global-using Gate, or both scores unavailable, requests verification. Truncation/alignment/numeric failure cannot produce apparently valid low scores. Format failure requests verification in internal-using Gates. External-only and Never are separate baselines, not general validity checkers.

For an explicitly enabled, verified confirmed-action policy: reliable conflict has priority over confirmation; a confirmed unchanged valid draft can suppress internal-only triggers; otherwise use missing/high-uncertainty triggers. Budget constraints mask every requested call. On budget exhaustion, execute the original draft with verification_unavailable=true and reason budget_exhausted_execute_unverified_draft. This is an unverified fallback, never an assertion that the action is safe or sufficient. Config signature failures abort before any action rather than mixing incompatible scores.

## Unified interface and costs

RuleGate.decide(GateInput) is pure, deterministic apart from its measured latency, and cannot read models, environment outcomes or WM predictions. GateInput contains decision identity, internal signals, typed external evidence, remaining invocation/forecast budgets and draft-format validity. Future outcome/reward/score fields are not accepted. Output contains call_wm, request/unavailable flags, reason, triggers, missing/validity diagnostics, copied uncertainty and external tri-states/evidence IDs, budget-before values, thresholds, requirements, latency and zero model/token/forward Gate cost. gates.runtime separately logs actual branch execution, forecasts/reviews, budget settlement and actual commit time.

| Policy | Internal extraction | External retrieval/Judge | Decision |
| --- | --- | --- | --- |
| never | no | no | Never request |
| always | no | no | Always request, subject to configured budget |
| local | yes | no | Local high; missing-local Global fallback |
| global | yes | no | Global high; missing Global requests |
| internal | yes | no | Local OR Global OR missing-required |
| experience | no | yes | Reliable conflict only; Unknown abstains |
| full | yes | yes | Reliable conflict OR internal triggers; optional confirmed-action override |

All share the Actor weights/config/structured draft, WM checkpoint, one-step predictions, review protocol and Memory Correction=false. Local/Global reuse the same accepted Internal Provider, which computes both small field aggregates; this minor CPU overhead is recorded rather than claimed absent. No additional backbone forward is introduced. Internal-only Gates do not retrieve Memory; Experience-only/Never/Always do not retain generation logits or calculate NLL. Reviews also use unscored generation. A real same-prompt parity check confirmed that scored and unscored BF16 paths emitted identical token IDs under the fixed sampling configuration. Unscored output is explicitly labelled hf_unscored_generation_bf16_v1 and never used to invent uncertainty values or calibrate thresholds.

For classical Always WM, configure enough budget for every turn. An Always policy with a smaller budget is a budget-limited Always condition and must be labelled accordingly. The actual one-turn Always smoke has sufficient budget for its only decision.

Random/Periodic matched-budget interfaces are reserved for Phase6. Planned scheduling inputs are explicit turn index, horizon, predeclared budget and separate policy RNG namespace/seed, alongside the same GateInput signals/budgets. Random samples a precommitted set of positions; Periodic uses precommitted evenly spaced positions. They must not inspect test outcomes or future calls from another method. Invocation budgets and actual forecast/token cost remain different quantities because review trajectories differ. No active random/periodic policies or performance experiments are claimed in Phase4.

## Budget semantics and WM preservation

remaining_wm_budget counts verification invocations, NOT individual forecast RPCs. A separate remaining_forecast_budget caps actual WM predictions. The existing JSON controller permits3 reflection rounds and, after a final replacement, one final prediction: reserve4 forecasts before starting a call. Settle actual predictions and release unused reserved slots, never shorten the review protocol for a particular Gate. Failed invocations retain their reservation and fail the run; no infrastructure retry is disguised as another policy choice. Duplicate dispatch/reservation is rejected.

The diagnostic uses one invocation/four forecast slots PER Episode. Later comparable experiments should use predeclared budget grids and report both quantities, not call them interchangeable. Budget decisions never use the number of predictions that a future call will happen to need.

gates.runtime dispatches into the EXISTING worldmodel.runtime.select -> ReflectionController -> Word2WorldModel prediction/review/revision flow. Default review protocol=json, max_rounds=3, WM temperature0/top_p1/response512/context32768, fixed learned ScienceWorld Qwen2.5-7B checkpoint @6196b3531c5634a8229af6b2cd111e02a70c13b5. Simulation remains a one-step forecast from the real prefix;3 reviews do not create a deeper rollout. No Oracle backend is permitted by this Phase4 adapter.

Controller/context initialization is lazy: Never/Skip cannot pay the author context-collector cost. If calls begin after skipped steps, initialize from the original reset then teacher-force every previously committed REAL transition. After a Skip with an existing controller, commit the real action/observation without predict/review. Memory and Gate never see the WM's privileged initialization context. The unchanged author collector runs in the existing separate ScienceWorld SDK Python; the main Actor's isolated environment is not modified. Local WM tokenizer counting uses the checkpoint chat template. Both are default-off adapters to the old Runtime, not replacements for its forecasting/reflection logic.

## Provisional thresholds

scripts/calibrate_gate.py consumes only approved Development IDs, exact goal identity, generation-config identity and Local/Global NLL. It rejects test/train IDs, duplicate decisions and incompatible scoring configurations. It does not read success/reward/outcome fields for selection. The actual input is Phase3's25 steps over5 goal groups, including previously exposed targeted examples: insufficient and selection-biased for stable calibration.

Q=.75, linear interpolation at (n-1)*q, strict > gives tau_local=.13648279807529134 and tau_global=.06599500845849053. Pearson=.9588707683; Spearman=.9484615385. Thus these are highly correlated lexical statistics, not independent evidence streams. The illustrative union would request7/25 before budgets/external evidence; it is not a matched-test-budget result. Preset diagnostic budget1/four forecasts was not optimized for success. Configuration is configs/gates/provisional_full_v1.json, status provisional_functional_only; explicit --allow-provisional is required by the diagnostic entrypoint. No final threshold or budget validation is claimed.

Future calibration should freeze group-separated dev-calibration/dev-validation before collecting more data, reserve already exposed groups appropriately, report distribution stability and held-out coverage/cost, and keep final Test untouched. A proposed volume target300 steps/30 tasks alone is not sufficient scientific validation. Changing model/backend/dtype/prompt/scoring path requires a new calibration configuration.

## Reproducible commands

No dependency/model download or parameter training was performed. Provide the accepted Qwen assets, pinned existing Word2World assets/source, existing ScienceWorld1.2.3 service/SDK Python, and an isolated WM serving process. Scripts accept model/SDK/service paths through CLI; no personal absolute path is required in code. The actual lab run reused old WM WEIGHTS read-only and copied only132KB of pinned helper source into v3's ignored external directory; agentwmv2 code, environments and services were unchanged.

~~~bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python scripts/test_gates.py
.venv-spanuq/bin/python scripts/audit_gate_inputs.py
.venv-spanuq/bin/python scripts/calibrate_gate.py --trace docs/phase3/phase3_dev_v1/steps.jsonl --trace docs/phase3/phase3_targeted_dev_v1/steps.jsonl --manifest docs/phase3/split_manifest.json --actor-config configs/internal_signals/scienceworld_token_nll_v1.json --quantile .75 --allow-provisional --output-config "$NEW_GATE_CONFIG" --output-report "$NEW_CALIBRATION_REPORT"
CUDA_VISIBLE_DEVICES="$FREE_ACTOR_GPU" PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python scripts/diagnose_gate.py --manifest docs/phase3/split_manifest.json --snapshot docs/phase3/memory_snapshot.json --gate-config configs/gates/provisional_full_v1.json --policy full --task-id sciworld_1709 --task-id sciworld_349 --steps 5 --model-dir "$QWEN3_MODEL_DIR" --wm-checkpoint "$SCIENCEWORLD_WM_DIR" --wm-url "$WM_URL" --wm-gpu "$WM_PHYSICAL_GPU" --science-python "$SCIENCEWORLD_PYTHON" --science-url "$SCIENCEWORLD_URL" --allow-provisional --validate-unscored-parity --output "$NEW_FULL_RUN"
# Same entrypoint/config with --policy never or always and --steps 1:
# independent functional branch checks, NOT a performance comparison with Full's 10 turns.
.venv-spanuq/bin/python scripts/audit_gate_diagnostic.py --manifest docs/phase3/split_manifest.json --snapshot docs/phase3/memory_snapshot.json --run-dir "$NEW_FULL_RUN" --output "$NEW_AUDIT_REPORT"
~~~

WM service uses the existing vLLM0.11.0 environment with bf16/enforce-eager/max_model_len32768/max_num_seqs1. Actual service was temporarily launched on free GPU6 at127.0.0.1:18212; Actor used GPU3. The first startup on GPU2 failed because another process occupied it; logs were retained and no other process was stopped. All setup/weight hashes, service commands and real HTTP request counts are in phase4/. The owned service was stopped after the12 real forecast requests; its GPU was released. Full logs/WM-only initialization context remain under outputs on lab. Committed compact traces retain exact generation records, evidence, decisions, model forecasts/reviews and native receipts; SDK admissible-action lists are replaced by count/hash.

Full cost must include Actor draft/reviews, raw-logit extraction, retrieval/evidence checks/Judge, Gate, WM predictions and cold initialization. Online tokens/calls and instrumented components are distinct from whole-driver elapsed time; service/model loading and offline memory collection/synthesis are separately reported. No monetary price or success-cost improvement is invented. See PHASE4_REPORT.md for measured values and negative outcomes.

The diagnostic restricts1–5 turns and approved Development IDs. It does NOT change formal ScienceWorld20-turn/seed0–7 evaluation or any existing benchmark profile. Enabling Gate in a formal cohort, larger dev calibration and performance evaluation are later phases, after user confirmation.
