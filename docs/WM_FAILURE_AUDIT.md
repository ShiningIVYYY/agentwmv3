# WM failure audit — Phase 5

## Finding

The two Phase 4 failures are grounded-object selection / reflection errors under the preserved three-round policy. Both final failures were forecast correctly by the learned WM. There is no evidence of a Qwen3/Qwen2.5 tensor or transport incompatibility, JSON parser corruption, or a wrong environment action introduced by the runtime. Valid review JSON does not imply a valid environmental decision.

Source: `outputs/phase4_full_v1/steps.jsonl`, SHA256 `a13b23836d661ebac22ac9da4b0b2dbfe09c0ad2b348e508459f03acd776c4b3`. Complete original inputs/reviews and fresh probe receipts: `outputs/phase5_setup/wm_failure_audit.json`; compact copy: `docs/phase5/wm_failure_audit.json`.

## sciworld_1709, first step

- Goal: locate and test the electrical conductivity of the glass jar in the kitchen; orange/yellow destination boxes are part of the observed task goal.
- Actual current state: bathroom, visible glass cup, closed door to kitchen. No executed history.
- Draft: `go kitchen`. Premise explicitly says the jar is not in the bathroom. Local NLL 0.0291141323; Global NLL 0.0146814868.
- Gate: reliable closed-route experience conflict, not high NLL. Memory Correction disabled.
- WM gets the exact saved Word2World initialization plus candidate action, then the same committed real prefix for every alternative. No hypothetical action is committed. Initialization hash: `38052f913f41bbe424cbd08d0bc66a526db152f3fd9c2203099f0f3fe1e92fae`.

| Forecasted action, in order | WM observation |
| --- | --- |
| go kitchen | The door is not open. |
| focus on glass jar | No known action matches that input. |
| focus on glass cup | You focus on the glass cup. |
| focus on glass jar | No known action matches that input. |

The reviewer cycles from a blocked route to an absent goal object, then the visible cup, then back to the absent jar. Final real execution: `focus on glass jar` → `No known action matches that input.` The reviewer fails to respect current-room grounding and the negative forecast. The round-limit policy executes its last replacement after one final forecast; it does not implement a final veto.

## sciworld_349, first step

- Goal: first focus on a thermometer, then use it to measure an unknown substance B in the bedroom.
- Actual current state: greenhouse, visible substance called air and bee hive; thermometer absent. No executed history.
- Draft: `focus on thermometer`. Local NLL 0.3032013739; Global NLL 0.1118328393. Gate triggered by high Local/Global uncertainty.

| Forecasted action, in order | WM observation |
| --- | --- |
| focus on thermometer | No known action matches that input. |
| focus on substance called air | No known action matches that input. |
| focus on bee hive | You focus on the bee hive. |
| focus on substance called air | No known action matches that input. |

Final real execution: `focus on substance called air` → `No known action matches that input.` A fresh same-initial-state probe finds `focus on air` → `You focus on the air.` The descriptive observation phrase is not the canonical command object name. The alternate bee hive action parses but does not by itself demonstrate task progress. Reviewer reasoning is inconsistent with its selected final action.

## Independent environment probes

`scripts/audit_wm_failures.py` performed seven true single-action probes, each after a fresh reset to these already-exposed development tasks. The first-state comparison is checked; probe outcomes are not inserted into Memory, calibration, or evaluation scores. All four final failing commands and the relevant alternatives behave as recorded above. These are controlled diagnostic actions, not natural Agent outcomes or Call/Skip counterfactual utility estimates.

## Engineering versus capability

All six original review responses parse as valid JSON. The ScienceWorld last-`Action:` tail is preserved exactly. The SDK action list is not an exhaustive grammar validator; aliases can be accepted outside it. Therefore, enforcing membership or adding specific object repairs would alter the verification policy and could reject legal commands. No answer-specific action correction, extra reflection round, WM model change, or final-forecast veto was added.

The actual engineering defect found in Phase 5 was separate: the new HTTP observer initially accepted positional `address` but requests passed keyword `url`. The first canary failed before generation/environment actions. Its failure/source are retained, the argument binding was fixed, and the observer regression plus a real two-step canary passed. This does not explain Phase 4 failures.

The Actor (Qwen3-4B) and WM (Word2World Qwen2.5-7B) communicate through plain action/observation text; their tokenizers remain independent. Model revisions, sampling and prompt hashes are recorded in the Phase 5 report. WM-only initial world context must not enter the Actor/Internal/External/Gate input. Weak reflection and WM prediction errors remain model/policy limitations, not evidence that selective invocation is effective. Any future critic or veto change requires a separately frozen protocol shared by all comparison Gates.

## Complete Phase5 cohort follow-up

All278 real review responses are valid JSON. The controller reports34 semantic replace-same-action violations (labelled invalid_format) and50 invalid replacement actions; these are not JSON decode errors.315 actual WM requests agree between the Gate ledger, HTTP observer and independent owned server counters.79/107 invocations change an action,40 revised actions receive explicit failure feedback, and all final forecasts are available/complete. No same-state alternative rollout establishes causal benefit or harm. The poor review compliance/grounding is a capability limitation requiring a separately controlled future protocol decision; no evaluated answer is patched. See PHASE5_REPORT.md, SMALL_SCALE_EVAL.md and docs/phase5/reviewer_quality.json.
