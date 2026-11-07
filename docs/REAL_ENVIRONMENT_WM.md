# GroundTruthEnvironmentWM: verified Reset + Replay

The user superseded the initial complete-JVM-clone requirement. The new implementation uses an independent ScienceWorld JVM, reset to the same task/variation/simplifications/transition seed, then replays every committed command. No installed SDK or Java/Scala modification, formal-environment preview/rollback, learned WM, or post-action reflection is used.

Status: bounded adapter/model integration is complete, with12 real9B development episodes and201 formal actions. Dynamic histories are **not universally reproducible**: controlled tests verify27/48 scenarios and reject21/48, with all27 subsequent public transitions matching. In the complete policy run21/75 attempts verify,54 reject;8 retained Draft transitions match formal execution. This is bounded evidence, not complete hidden-state equivalence. See [phase report](PHASE5_7_REPORT.md) and [actual audit](phase57_replay/audit.json).

## Protocol

`ReplayScienceEnvironment` owns one SDK/JVM, exact initialization spec, public current feedback and a journal containing EVERY executed command, including invalid commands, free observation commands and numeric disambiguation selections. Stable disambiguation labels are mapped to each JVM's native choices using the existing `ScienceActionView`. Journals contain actual committed actions, never rejected Reviewer alternatives or prior trial transitions.

`GroundTruthEnvironmentWM.simulate_one_step(formal, draft_action)`:

1. Capture reference verification fields; do not execute a reference action.
2. Create a fresh independent JVM and initialize the identical spec and initial `look around` query.
3. Verify initial projection; replay each committed command and verify its resulting projection, then verify the complete current recorded projection.
4. Only if every check passes, perform ONE candidate command in B. No alternative-action search or extra review round.
5. Return genuine trial observation/reward/done/score and measured costs. `action_valid` stays Unknown unless independently established.
6. Close B; require reference verification fields to remain unchanged. A reference mutation is fatal, not a normal fallback.

Reconstruction errors return `status=reconstruction_failed,prediction_available=false`, an explicit cause and costs, with no predicted/public trial feedback. The runtime consumes the invocation attempt budget, calls no Reviewer and executes the original Draft normally. This protects state/information integrity, not task correctness: the Draft can still fail. It does not call another WM. A failed reconstruction is not successful verification. KEEP or REPLACE always executes once normally in A; no trial state is adopted. Trial done/score cannot terminate or update A. A verified KEEP whose actual transition differs is a fatal diagnostic error.

## State verification and its limits

Compared fields: canonical current observation, reward, score, done, native move count; current room/inventory; SDK last score; native completion; executed native action history; object-tree hash; selected Scala/Java RNG state; journal action count and disambiguation mapping. Raw feedback is retained; existing official flat-list presentation canonicalization is shared across strategies and is the only observation-order normalization. Object-tree property differences are never rounded away.

Snapshots/digests/RNG and private mismatch evidence stay inside validation/audit logs. The Gate sees only accepted Internal/External Signals and budget; the Reviewer receives only whitelisted public trial feedback. No gold path, task answer, private goal progress or future formal trajectory is queried.

The object-tree and selected RNG do not cover every hidden engine/global state. Passing a finite projection and transition test cannot certify complete state equality or all future stochastic behavior. The additional validator itself has privileged simulator inspection access, which must be disclosed alongside the real trial queries. This method is **Ground-Truth Simulator / Oracle-Like Lookahead**, not learned-WM prediction performance or an equivalent deployable learned world model.

## Why the thin native wrapper is necessary

Installed ScienceWorld1.2.3 `SDK.step` calls the public native step/score/completion/move APIs, computes reward as native score delta, then also calls `get_valid_action_object_combinations`. Real two-JVM tracing shows that this extra action-enumeration query consumes the global RNG by different amounts. It is not a read-only query suitable for this replay pipeline. The SDK's `load` additionally reseeds from Variation, overriding a seed set before load.

The new wrapper invokes the SAME public native transition APIs and preserves SDK semantics: `round(100*native_score)`, reward delta, native completion OR moves>21 OR negative score. It omits the unused RNG-consuming enumeration and explicitly sets the transition seed after load in both A and B. The native task construction still uses the identical task/variation/simplifications. Initial public Goal uses `server.getTaskDescription`; SDK `taskdescription` adds a presentation label absent from AgentGym identity metadata.

This is an explicit new query/seed profile shared by all four9B strategies. It is not claimed identical to the old SDK-query schedule or old4B experiments. No validity list is invented for Reviewer grounding; it uses visible state, committed history, native action forms and trial evidence. Its unproven action validity remains Unknown. Do not re-enable the RNG-mutating enumeration in the formal path without investigating and declaring the change.

## Real reproducibility evidence

SDK1.2.3 JAR SHA256: `f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7`.

Approved dev instances128/142 (thermometer variations8/22),2417 (unknown conductivity81),3070 (living thing134); seeds0/1/7; histories of lengths0/2/4/8. Controlled commands include look, inventory, a real invalid command, wait, door opening and navigation. These are actual SDK transitions for infrastructure diagnosis, not Actor benchmark results or fabricated feedback.

- Original SDK query schedule:6/48 reconstructed and matched,42 rejected,0 verified public-transition mismatches.
- Corrected native query schedule:27/48 reconstructed and matched,21 rejected,0 verified public-transition mismatches; all formal-isolation checks pass. Two runs independently reproduce these counts; final cleanup/accounting run113.349s.
- Remaining failures include real material-temperature drift after `wait`, despite matching selected RNG. For example, room air temperature24.20014089209703 versus24.200482233189756. This is physical-state divergence, not merely object ordering. It remains a hard reconstruction rejection; no task-specific tolerance or action exception is introduced.

Raw runs remain under ignored `outputs/phase57_replay_validation_v{1,2,3}`; minimal failure evidence is `outputs/phase57_setup/replay_minimal_failure.json`. The final controlled run includes source hashes; the public Goal-label-only fix afterward does not alter transition or verification logic. The subsequent complete9B run exercises the final code. Compact evidence is versioned under [phase57_replay](phase57_replay/replay_validation.json), including a [minimal failure](phase57_replay/minimal_replay_failure.json).

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
  "$SCIENCE_PYTHON" scripts/validate_reset_replay.py --output outputs/replay_new_run
PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
  "$SCIENCE_PYTHON" scripts/reproduce_replay_rng.py --output outputs/replay_failure.json
```

These tests return an error if accepted prediction and actual subsequent transition disagree. Safe reconstruction failures are reported separately and must not be hidden in an average accuracy denominator. No accepted reconstruction means blocked, not100% predictive accuracy.

## Costs and fairness

Count one fresh JVM, its initialization query, EVERY replayed committed command, the candidate query if reached, validation, disposal and Reviewer generation. Failed reconstruction still has real setup/replay costs. Formal Actor turns, native simulation ticks, initialization queries, replay queries, candidate queries and model calls are separate counters. `wait` can advance11 native ticks while remaining one API command. These extra queries are not free because learned-WM tokens are zero.

Never performs no replay/Reviewer. Always attempts a Call each formal turn while budget permits. Internal/Full use independent9B thresholds and the SAME one-round feedback/review policy and frozen Memory. Reconstruction failure executes the original draft consistently; it does not silently convert the method into post-action reflection. No extra model parameters are trained. Formal paper settings and final200 IDs remain unchanged in their original files; this bounded development profile is separate.

In the actual run,Always adds190 transition queries (54initialization+127replay+9candidate),Internal47,Full61. There are744 read-only audit capture bundles in addition; their latency is included, not equated to zero API cost. No reliable monetary/FLOP conversion is available. The full12-episode native transition count is201 formal actions+12 formal initialization queries+298 extra lookahead queries. Native move clocks are a different unit.
