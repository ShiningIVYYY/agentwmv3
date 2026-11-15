# Qwen3.5-9B ALFWorld five-policy pilot

Status: COMPLETE for the fixed tiny development pilot; see [ALFWORLD_PILOT_REPORT.md](ALFWORLD_PILOT_REPORT.md) for audited results and limitations. User authorized a small comparison of Never/Always/Internal/External/Full. This is not the140-game final evaluation or an automatic Phase6 sweep. ScienceWorld and SR profiles/results remain unchanged.

## Configuration and comparison scope

Frozen official Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, one shared Actor/Reviewer model object, BF16/HF5.22.2/Torch2.8.0, nonthinking,T=.4/top_p1/top_k0,response512. Reuse existing raw-generation TokenNLL implementation and RuleGate. Local=max(mean Premise,mean Expected Consequence);Global=mean all response text tokens including labels/Action, excluding prompt/EOS/padding. These are uncalibrated nats/token, not error probabilities or SpanUQ outputs.

ALFWorld-specific settings follow the already audited paper evaluation target: H50/history2/prompt2048/response512. Source: More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning, arXiv2610.19789, official source commit35a44f5. We preserve docs/PAPER_ALIGNED_EVAL.md and all old configs. The current diagnostic uses3 TRAIN-pool development games/seed0, not the official140 valid_seen×8 seeds, and a structured decision prompt rather than the paper's action-only prompt. It is not reproduction of paper scores.

Actor and original three-line Reviewer receive ALFWorld-native commands and CURRENT public admissible commands; both use the same prompts across all five policies. Reviewer still has cap512,one round,no retry; malformed output remains INVALID with explicit unchanged-Draft fallback, as in Phase5.9. No compact-JSON repair or SR decoder is silently imported. New prompt/config identity and NEW model/domain-specific q.75 thresholds are necessary; ScienceWorld thresholds are never reused. Six legal calibration games×up to8 steps, frozen before evaluation, same distribution-only rule, no outcome-based selection. Prompt overflow is an error; no silent history/goal truncation.

WM uses independent native ALFWorld0.4.2/TextWorld1.7.0 processes, reset the exact pinned game, replay ALL formally executed actions, verify each public transition/admissible list and private symbolic-fact digest, then ONE candidate action. Original environment is passively checked before/after; no formal preview/rollback, learned WM or Qwen2.5 service. Private fact digest is validation-only, never model/Gate/Memory input. Historical replay is not extra future simulation depth. Full hidden RNG/process serialization is not claimed; native PDDL seed() is a no-op and the pinned compiled engine is deterministic in the tested range. Python/NumPy caller seed is fixed and recorded.

Budgets Never0,Always50,Internal5,External5,Full5 per Episode. Always's budget changes with the ALF50-step horizon to preserve its definition; the selective absolute5-call cap is retained. This is not a matched-budget performance comparison. Memory Correction and confirmed-action skip remain disabled. Unused Internal/External modules are not computed. Main metric is native won Success Rate;100×won is only an auxiliary display, not ScienceWorld partial Task Score. No training invalid-action penalty or reward normalization.

## Data and experience

Metadata catalog contains3553 supported solvable TRAIN games and140 valid_seen games, with actual game/metadata SHA256. Exclude all140 final game IDs AND duplicate canonical goals defined by task type and relevant object/receptacle/lamp parameters; irrelevant scene fields cannot make false goal disjointness.438 eligible TRAIN goal groups remain. Deterministic family-round-robin/SHA256 ordering reserves4 memory,6 calibration,3 evaluation goal groups. No expert trajectory, walkthrough, planner or expert-action wrapper is used. The native instance uses official AlfredDemangler(shuffle=False)/AlfredInfos wrappers only. Existing stable CPU environment is reused without any package modification; model inference stays in the separate9B environment.

Final memory source:31 real feedback cases from4 reserved TRAIN games;2 deterministic supported cards: repeated open of an observably open container, and take through an observably closed container. Require supports from at least2 distinct tasks AND episodes, no counterexamples. BM25 ranks candidates; exact public precondition matching plus source validation determines reliability. Similarity is never evidence of applicability/utility. Unsupported/absent evidence is Unknown; Sufficiency positives are not manufactured. No ScienceWorld card or cross-domain sharing. Frozen before any evaluation; no online writes, no LLM Judge/synthesis calls, no parameter training.

Initial2-game probe yielded7cases/0 supported cards because one game's drawers were non-openable surfaces. Its costs/results remain separate and are not silently discarded or counted as independent support. Two deterministic reserve source slots were activated before any model evaluation; a stricter semantic goal-key representation was fixed before the final4-game collection. The final selected calibration/evaluation task set was frozen before outcome generation. Initial probe data are NOT in the final31-case snapshot. The cohort covers clean/cool/heat tasks only; it is not representative of every ALFWorld family or unseen physical layout.

## Real environment acceptance already passed

-18 controlled Reset/Replay comparisons across3 legal calibration games,2 caller seeds and prefix lengths0/3/6: exact public transitions plus private fact digest match; formal state unchanged.
-150 action transitions match the official ALFWorld Gym wrapper across3 games, including50th-step truncation with won=false. Direct process adapter retains original physics/scoring and counts invalid commands as formal turns.
-Supplement:13 actual native Goal/initial-state identities verified;6 further replay comparisons include actual rejected commands in history. Native rejects the controlled invalid command with Nothing happens; subsequent reconstructed/normal candidate transitions match. These are environment tests, not natural Actor results or benchmark metrics.
-16 CPU contracts cover split/goal isolation, native action format, prompt scope/private-info filtering, parser errors, evidence support/Unknown and state-fingerprint sensitivity.

All checks are bounded-range evidence, not proof of arbitrary replay or future scientific gains. No final valid_seen game was reset; official-wrapper comparison overrides its game list with reserved TRAIN games and disables Expert.

## Reproduction and resource limits

Use actual runtime paths via environment variables, not hardcoded personal paths in new code:

```bash
python scripts/prepare_alfworld_pilot.py --data-root "$ALFWORLD_DATA"
python scripts/test_alfworld_pilot.py
python scripts/collect_alfworld_memory.py --env-python "$ALFWORLD_PYTHON" --data-root "$ALFWORLD_DATA" --run outputs/alfworld_memory_v2
python scripts/validate_alfworld_replay.py --env-python "$ALFWORLD_PYTHON" --data-root "$ALFWORLD_DATA" --run outputs/alfworld_replay_validation_v1
"$ALFWORLD_PYTHON" scripts/alfworld_batch_parity.py --data-root "$ALFWORLD_DATA" --run outputs/alfworld_batch_parity_v1
python scripts/validate_alfworld_inputs.py --env-python "$ALFWORLD_PYTHON" --data-root "$ALFWORLD_DATA" --base-validation outputs/alfworld_replay_validation_v1 --run outputs/alfworld_inputs_validation_v1
CUDA_VISIBLE_DEVICES=1 python scripts/run_alfworld_pilot.py --env-python "$ALFWORLD_PYTHON" --data-root "$ALFWORLD_DATA" --memory outputs/alfworld_memory_v2/snapshot.json --native-validation outputs/alfworld_inputs_validation_v1 --run outputs/alfworld_five_policy_v1
```

`python` above is the isolated9B environment Python; `ALFWORLD_PYTHON` selects the separate pinned CPU environment. Fresh output names are required; old partial runs are never overwritten. Real run environment/path/commit hashes are recorded. Final31-case snapshot and compact validation receipts are under docs/alfworld_pilot; raw trajectories/temp logs are ignored.

One GPU/shared persistent9B.3 evaluation games×5policies×seed0=15 full Episodes, plus6calibration games×up to8actions and2 explicitly charged numerical-repeat generations. Limits2Mtokens/1100model calls/7000native reset+transition queries/20000state captures/2h. Passive state inspections, extra reset/replay/candidate queries and offline collection/validation are separately counted; environment trial steps do not consume formal H50. WM LLM tokens0 is not zero simulation cost. Owned native processes close in finally and temporary native-library copies are removed only after process exit. No other user's jobs/services are stopped.

Expected final delivery: all15 per-game outcomes, five-policy SR/cost table, paired differences, real traces and source/cost/causal/Call-Skip audit, format/missing rates, calibration limits, owned-process cleanup and pushed Git commit. No effectiveness claim is made before data.
