# Real-environment World Model: Phase 5.7 prerequisite audit

Status: **BLOCKED at full-state transfer into an independent environment**. No usable `GroundTruthEnvironmentWM` is claimed or enabled. The requested architecture is now Qwen3.5-9B Actor/Reviewer with isolated real-environment one-step lookahead. The previous learned WM and same-live-JVM rollback oracle are historical implementations, not approved backends for this architecture.

## Required semantics

The formal environment must remain untouched until the final action is executed. A Call must copy its **current complete state** into another isolated simulator, execute exactly the Draft Action once there, then give only public trial feedback to the Reviewer. KEEP still executes the Draft Action normally in the formal environment; it must not adopt a trial transition. REPLACE executes the revised action normally without a second speculative search or recursive review. A Skip creates no trial transition and makes no Reviewer call.

The planned adapter contract is:

```python
simulate_one_step(opaque_environment_snapshot, draft_action, config)
# public result, not a snapshot or private state:
{
    "predicted_observation": str,
    "predicted_reward": float,
    "predicted_done": bool,
    "action_valid": True | False | None,
    "environment_feedback": str,
    "simulation_latency": float,
    "simulation_cost": {"environment_queries": 1},
    "source": "ground_truth_environment",
}
```

This is a **design contract**, not implemented functionality. Validity remains Unknown unless the public API or feedback establishes it; grammar validity does not establish task utility. Errors must fail explicitly, without fabricated predictions, formal-environment execution/rollback, learned-WM fallback or post-action reflection. A trial terminal state must not terminate the formal episode. Native simulator ticks, formal Actor turns, trial queries and model calls require separate counters.

## Installed implementation and direct evidence

See [full preflight evidence](phase57/clone_preflight.json) and [execution/check records](phase57/verification.json). The actual installed SDK is ScienceWorld **1.2.3**. Its Python source SHA256 is `23a37a2e9ed8e43d8c5e27d8660ea344d38a544abcd8c7e6af40071c7832ce68`; JAR SHA256 is `f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7`.

The public Python and native `scienceworld.runtime.pythonapi.PythonInterface` method inventories expose reset/load, queries, steps and run-history export, but no complete state export/import or cross-instance clone API. Direct Java serialization fails:

```text
java.io.NotSerializableException: scienceworld.runtime.pythonapi.PythonInterface
```

Serializing `agentInterface`, `agent`, `actionHandler`, `taskMaker` and `currentHistory` independently also fails, on their contained engine classes. An enclosing Scala `Option` does not make its contents serializable. Object-tree JSON and run history omit engine/task/parser/RNG state and are not complete restorable checkpoints.

`worldmodel/science_snapshot.py` and `worldmodel/HeapSnapshot.java` retain references to objects/fields in the **same JVM**. Their earlier restore/commit diagnostics do not transfer state to a separate process. Phase 5.7 uses only `take()` and read-only `matches()` as an additional pollution witness: **no `restore()`, `predict()` or speculative action is applied to the reference environment**. The witness is not a transferable snapshot and is not asserted to prove all possible hidden-state invariants.

CRIU is not installed and the probing process has zero effective/permitted Linux capabilities. No validated OS/JVM process-copy facility is available in this setup. This is a statement about the inspected installation and tested mechanisms, not a proof that ScienceWorld can never be extended to clone.

Official source context: [ScienceWorld SDK](https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py), [ScienceWorld repository](https://github.com/allenai/ScienceWorld). The installed source/JAR hashes, not a floating upstream branch, identify what was actually tested.

## Real independent-instance diagnostic

The existing frozen split manifest is validated, including normalized goal separation from the final 200 tasks. The probe selects only approved development instances from `docs/phase5/development_plan.json`:

| ID | Native task | Variation | Probe seeds |
| --- | --- | ---: | --- |
| sciworld_128 | use-thermometer | 8 | 0, 1 |
| sciworld_2417 | test-conductivity-of-unknown-substances | 81 | 0, 1 |
| sciworld_3070 | find-living-thing | 134 | 0, 1 |

For each trial two fresh independent JVMs are seeded before load. They receive the same initialization and then `look around`, `wait`, `focus on air` in lockstep. The worker executes first. Before the reference executes normally, its observation/inventory, native score, completion flag, native move counter, SDK last score, executed-action history and object-tree digest must remain unchanged; the read-only heap witness must also match. Public worker/reference transitions are then compared. Observation comparison uses the existing versioned canonicalizer; raw equality is counted separately. No gold action/goal-progress query is used. Internal audit digests never enter an Actor, Gate or Reviewer prompt.

This establishes bounded evidence of **independent-instance isolation and seeded reset/replay consistency**. It does **not** export/import the reference's current state, prove complete hidden/RNG-state equality, cover arbitrary stochastic actions, or establish a general snapshot implementation. Probe seeds are Scala simulator seed inputs, not a validated Qwen rollout configuration. Passing these tests is necessary but insufficient for the requested real-environment WM.

The observed trial feedback includes accepted `focus on air` actions that end some tasks at score -100. These are genuine diagnostic failures, not invented invalid-action penalties or strategy evaluation results. `wait` advances 11 native ticks; it is one API action. Counts include initialization queries separately.

## Reproduce without models

Run from the repository root with the existing isolated ScienceWorld SDK environment; substitute its Python path for `SDK_PYTHON`:

```bash
export SDK_PYTHON=/path/to/scienceworld-environment/bin/python
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
  "$SDK_PYTHON" scripts/audit_phase57_clone.py \
  --output outputs/phase57_clone_new_run
```

Use a new output directory. **Exit 42 means prerequisite blocked**, not a passing clone test. Other exceptions are failures to investigate. The script intentionally cannot promote seeded replay to a clone. `phase57/verification.json` contains the actual lab command/return code and derived counters. Full local logs remain in ignored `outputs/phase57_setup/` and `outputs/phase57_clone_preflight_v2/`; the initial v1 run is retained separately.

Run `python3 scripts/check_phase57_evidence.py --check-worktree` to check the committed real evidence, source/split hashes and accounting. Its exit0 certifies record consistency only; its output explicitly retains `phase_status=blocked` and `full_state_clone_verified=false`.

## What must be implemented before resuming

The preferred engineering option is a versioned ScienceWorld engine fork with an explicit state codec, built under agentwmv3 without modifying the installed stable SDK or agentwmv2. It must cover object identity/aliases, world and agent state, task/goal evaluation state, action-parser/disambiguation state, native and Python counters/history, random generators, UID allocation and process-global mutable state. Import must target an independent JVM. A separately validated process-checkpoint system is an alternative infrastructure project, not an assumed workaround.

Acceptance must demonstrate full round-trip restoration into an independent instance, noninitial states, varied task families/seeds, random transitions, interactive ambiguity, negative/terminal states, public transition equality and unchanged formal state/counters/history after both successful and failing trials. Identical observable projections alone cannot certify complete-state copying. Tests must also prove repeated trial disposal, resource cleanup, error isolation and no hidden snapshot fields in model inputs. Then implement the adapter and the Qwen3.5-9B runtime, and run the four development strategies.

## Information permissions and costs

Real simulator access grants true counterfactual one-step environment feedback unavailable to a learned WM. It must be reported as a simulator-access experiment, not an equivalent deployable learned-world-model method. No test answer, gold path, hidden task progress or future rollout is exposed to the model. Decide and freeze whether publicly available trial reward/done are included in the Reviewer evidence; the planned interface allows them, which must be disclosed explicitly in all comparable variants.

Count trial queries, clone/export/import and environment-compute latency in addition to Actor/Reviewer tokens and latency, NLL extraction, Gate and retrieval. A zero learned-WM token count is not zero simulation cost. Keep formal episode steps and trial steps separate. Memory remains frozen, and old Qwen3-4B thresholds are ineligible for the new Actor.
