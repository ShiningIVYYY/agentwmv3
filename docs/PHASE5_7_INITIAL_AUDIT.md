# Historical initial Phase5.7 clone audit

The user later superseded the full-JVM-clone prerequisite with verified Reset + Replay. This file preserves the original audit, not current requirements or architecture status. Source baseline fdbfa664a7beec7976f83bd177d11afff42d0408. See PHASE5_7_REPORT.md for the amended phase.

# Phase5.7 — Qwen3.5-9B + real-environment WM

**Result: BLOCKED at mandatory clone feasibility, not an accepted architecture integration.** The audit deliverables and bounded real SDK tests are complete; the requested independent complete-state clone is not. The user's stop condition is applied before installing9B dependencies, downloading weights, calibrating or running episodes. No Phase6 work, training, fake prediction or learned-WM fallback occurred.

## A. Work performed and evidence

Reviewed the Phase5.5 findings, original snapshot/rollback mechanism, runtime boundary, frozen splits and existing inference compatibility. Inventoried the installed native SDK and Java methods, probed full-state serialization, tested real independent simulator instances, fetched pinned official9B metadata and checked cache/environment resources. [Real-environment audit](REAL_ENVIRONMENT_WM.md) explains the implementation evidence and required state contract; [9B integration record](QWEN35_9B_INTEGRATION.md) separates verified metadata from unrun generation.

ScienceWorld1.2.3 lacks a complete native state export/import API in the inspected installation. `ObjectOutputStream.writeObject` fails on `PythonInterface` and five critical engine fields. Existing `HeapSnapshot` holds same-JVM references; executing and restoring that live formal environment would violate the new requirement. CRIU is absent, with zero effective/permitted process capabilities. No verified cross-JVM state codec or process checkpoint was found or implemented. These observations do not prove that a future engine fork could never implement cloning.

The first audit and the final manifest-guarded audit are retained separately. Each covers3 approved Development Tasks (`128/use-thermometer/v8`, `2417/test-conductivity-of-unknown-substances/v81`, `3070/find-living-thing/v134`) x simulator probe seeds0/1. Each uses12 distinct JVMs and18 worker/reference action pairs. Initialization uses the same task/variation/seed; worker then acts before the untouched reference acts normally. Formal-environment trial/rollback counts are zero. Only approved dev identities are read; no final-test execution, gold action, experience synthesis or threshold selection occurs.

## B. Changed files

| Paths | Purpose |
| --- | --- |
| `scripts/audit_phase57_clone.py` | Real SDK capability/serialization/independent-instance probe; validates frozen dev split; preserves raw responses; fail-closed exit42 for unvalidated full clone |
| `scripts/audit_qwen35_metadata.py` | Fetch pinned small official metadata; inspect cache completeness and existing package/model mappings; no weights or generation |
| `scripts/check_phase57_evidence.py` | Reproducible checks of real recorded pairs, costs, source/split hashes and blocked readiness; no Mock scores or clone certification |
| `docs/REAL_ENVIRONMENT_WM.md` | Isolation semantics, installed failure evidence, prospective adapter contract and clone acceptance |
| `docs/QWEN35_9B_INTEGRATION.md` | Official model/template metadata, compatibility issues, planned shared backend and new NLL fingerprint/calibration |
| `docs/PHASE5_7_REPORT.md`, `docs/phase57/*.json` | Honest phase outcome, actual logs/versions/commands and machine-readable bounded test evidence |
| `docs/IMPLEMENTATION_PLAN.md`, `docs/PROGRESS.md`, `README.md` | New desired architecture, blocked state, historical boundary and resumption order |

No usable `GroundTruthEnvironmentWM`,9B Actor, Reviewer policy or new Gate config is claimed. Existing core modules/configs are preserved; changing them before clone acceptance would bypass the phase ordering. `docs/PAPER_ALIGNED_EVAL.md` remains byte-identical to baseline.

## C. Key design decisions

One Call will eventually mean one isolated Draft Action trial and at most one9B Reviewer call, not the old multi-round learned prediction protocol. The final action must execute normally on the original state. An opaque snapshot cannot reach Actor/Gate/Reviewer; only explicitly permitted public feedback may. A trial must not alter formal score, counters, history, task state or RNG. Errors fail explicitly. The current reset/lockstep experiment does not meet complete-state transfer and is not promoted to an adapter implementation.

Actor/Reviewer must share the same official post-trained weights/backend while using distinct prompts. Model revision metadata is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Published config has `qwen3_5`/`qwen3_5_text`,32 text layers,4096 hidden size and248320 vocabulary. The official nonthinking template flag is documented, but real nonthinking9B output is untested. Old Transformers4.57.3 lacks both config mappings. The cache directory contains config only, not a usable weight set. A new independent pinned environment is needed after unblocking.

Retain field mean NLL, max Local and whole-text Global definitions, but validate a new9B provider fingerprint and token boundaries. Do not reuse4B thresholds or mix BF16/FP32 paths. Development-only distribution calibration follows provider validation. Memory/evidence/Gate semantics stay frozen. Real simulator access is a stronger information permission than a learned WM and must be separately described; it is not equivalent to a deployable learned-world-model result.

## D. Actual commands and test status

Working directory: `/data1/shn/agentwmv3` over SSH lab. The recorded final commands are:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
 /home/shn/miniconda3/envs/ee-scienceworld/bin/python \
 scripts/audit_phase57_clone.py --output outputs/phase57_clone_preflight_v2
# exit 42: required complete independent clone not validated

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_qwen35_metadata.py \
 --output outputs/phase57_setup/qwen35_metadata_v2 \
 --legacy-python /home/shn/miniconda3/envs/verl-agent/bin/python \
 --cache-root /data1/shn/models
# exit 0: metadata audit only
```

Both new Python sources compile successfully using built-in `compile`. Automated checks validate all recorded pairs, raw/public equality, untouched-reference witnesses, expected exit codes and exited owned processes. The frozen manifest's `validate_manifest(require_native=True)` passes, and selected rows must match the existing Development identity/goal group. `git diff` against baseline confirms unchanged core/config/split/paper-aligned paths; agentwmv2 worktree remains clean. See [verification.json](phase57/verification.json) for exact execution records and protected paths.

The recorded-evidence checks are reproducible with `python3 scripts/check_phase57_evidence.py --check-worktree`. Exit0 means the **blocked** evidence is internally consistent, not that a clone has passed. This verifier and both probes pass Python syntax checks. It refuses to relabel reset/replay as full-state copying and verifies the executed sources and frozen split hashes.

| Check | Actual result |
| --- | --- |
| Independent processes and reference-isolation witnesses |18/18 paired steps pass within this bounded probe;12 own JVMs closed |
| Raw observation, reward, done, native score/move comparisons |18/18 match; raw observation equality checked separately from canonical equality |
| Complete-state export/import into independent JVM |**BLOCKED / not implemented**; native inventory and serialization failures retained |
| Hidden/RNG state equivalence after arbitrary cloning |**Not established**; same-seed reset is insufficient |
| New9B generation, Reviewer parsing, NLL/backend/dtype alignment |**Not run**, prerequisite stop |
| New Gate thresholds,4 matched strategy episodes and utility |**Not run**, not zero-score evaluations |
| New full episode, lookahead cost and14-item end-to-end acceptance |**Not satisfied**; existing historical tests cannot substitute |

The `focus on air` probe sometimes genuinely ends the native task at-100. That is recorded task feedback from a deliberately bounded infrastructure diagnostic, not evidence that the unrun new Actor or Reviewer failed or succeeded. No diagnostic case is added to experience or used to optimize a threshold.

Final authoritative probe costs:36 compared transition calls plus12 initialization `step('look around')` calls;31.141s wall time; worker transitions2.273s, reference transitions2.231s. This includes distinct-process setup/serialization/witness overhead in wall time; component times are not double-added. The preserved initial run adds another36 compared transitions and12 initializations over31.140s. Thus the two metered runs total72 compared transitions,24 initialization calls and62.281s wall time. Additional public getters, metadata HTTP requests and preliminary inventory work are not model inference and do not have a complete phase-wide latency meter. Do not describe these numbers as complete new-system episode costs.

Actor calls0, Reviewer calls0, learned-WM requests0, model tokens0 and model downloads0. This is **not** a zero-cost WM result: CPU/JVM simulator queries consumed resources. New9B VRAM/episode latency/cost and correction/validity/success rates are unavailable. All observed GPUs were essentially idle (3MiB/card); this is not9B peak VRAM. No credible money/FLOP conversion is reported. No new offline experience construction occurred.

## E. Final twelve acceptance answers

| Question | Answer |
| --- | --- |
|1.9B Actor correctly replaced? |No. Official metadata verified; actual loading/integration deferred at clone prerequisite. |
|2. Reviewer shares same9B model? |Required design; not run or established. |
|3. Qwen2.5 stopped? |No learned-WM service started or requested in this phase. Historical code/configs retained; no fallback enabled. This does not imply legacy code was deleted. |
|4. Reliable ScienceWorld copy? |Independent fresh instances work; complete current-state copy into them is not validated. Existing same-JVM restore is ineligible. |
|5. Ground-Truth WM running? |No usable compliant adapter; stopped rather than return fake trial feedback. |
|6. Trials do not pollute formal trajectory? |18 worker/reference checks observe isolation, with no reference trial/restore. No compliant snapshot-based lookahead exists, so general clone isolation is not certified. |
|7. New Internal NLL calibrated? |No9B generation/scoring/calibration. Old4B values retained for historical runs only. |
|8. New Reviewer repairs actions? |Not tested; no evidence of its effectiveness. |
|9. Never/Always/Internal/Full scores? |All **not run**, no means or success percentages. |
|10. Positive-utility simulation? |Not assessed. Transition agreement is not utility evidence. |
|11. Full compute cost? |Only the bounded SDK audit costs above are measured; no new-system episode costs exist. Simulator work is not free. |
|12. Enter Phase6? |No. Clone prerequisite and all downstream new-architecture acceptance remain outstanding. |

Engineering evidence supports independent JVMs, repeatable selected transitions and bounded nonpollution checks, not a complete clone. Model capability and scientific effectiveness are unassessed. The earlier five development outcomes cannot be reused as9B results.

## F. Next technical recommendation

Implement and validate complete engine-state transfer as a separate scoped infrastructure task, pinned to a new ScienceWorld source/JAR version under v3. Include all world/task/parser/RNG/UID/Python/global state, independent import and error/resource isolation, rather than an unverified Java graph-copy workaround. A validated process checkpoint facility is an alternative requiring infrastructure work. Do not operate on and roll back the formal environment or silently relax the definition to observable replay.

After clone acceptance, establish an isolated pinned9B backend, real nonthinking/logprob tests, new provider metadata, development-only calibration, one-round generic Reviewer and the four small matched development variants. Keep Memory, final test IDs and paper-aligned settings frozen. Real simulator access and its extra queries/costs must be explicit in every future report.

## G. Handoff

Stop at Phase5.7 blocker and await the user's choice of follow-up infrastructure work. Phase5.7 is not marked accepted, and Phase6 is not started. Commit/push identity is reported in the final handoff; the committed verification records identify the pre-change baseline without embedding a self-referential commit hash.
