# Real-environment WM: deterministic Reset + Replay repair

This follow-up addresses the user's request to make real-environment one-step lookahead reliable. It does not start Phase 6, change the Reviewer/Gate algorithm, retrain any model, edit the installed SDK/JAR, or change the final evaluation protocol.

## Cause and verified repair

ScienceWorld 1.2.3 iterates identity/hash-based object collections in an unstable order across JVMs. Heat transfer updates temperatures in place, so equal task, variation, seed and action history are insufficient: changing traversal order changes actual material temperatures. RNG equality does not solve this. The previous strict tree rejection was detecting genuine physical differences, not merely JSON presentation differences.

The official [1.3.0 determinism PR #88](https://github.com/allenai/ScienceWorld/pull/88) sorts objects/portals by UUID and makes related parser, RNG-consuming enumeration and process iteration deterministic. The later official [door referent PR #87](https://github.com/allenai/ScienceWorld/pull/87) also normalizes generated door references. We use the exact official bundled JAR at commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52`, not a locally invented physics patch.

- Native engine: **1.3.0**, JAR SHA256 `e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465` (7,793,417 bytes).
- Python client: the existing isolated SDK **1.2.3** public/native API wrapper, using its supported `serverPath` argument to launch this pinned official JAR. No environment installation or replacement was needed. Do not confuse the Python package version with the native engine version.
- New explicit profile: `scienceworld130`; historical `scienceworld123` profile and `phase57_replay_dev_v1.json` remain available. All new A/B environments use the SAME profile and binary.
- Developer CLI now defaults to `phase57_replay_dev_v2.json`; historical exact reproduction uses the previous Git commit. Results in `docs/phase57_replay/` are unchanged historical 1.2.3 results.

This is an explicit environment-version change. Upstream warns it can change physics trajectories, ambiguous-action resolution, gold paths and ordering. It is not exact replication of the paper/old 1.2.3 benchmark. `docs/PAPER_ALIGNED_EVAL.md`, final test IDs and original configuration files remain unchanged. Every baseline in the new regression uses the same 1.3.0 environment.

## One-step WM contract

1. A owns the committed environment, initialization specification and every actual executed command, including invalid, observational and numerical-disambiguation commands.
2. On Call, B starts in a different JVM with identical pinned engine/task/variation/simplifications/seed. Replay only A's committed history.
3. Verify EACH reconstructed prefix and the current projection: observation, reward, task score, done, moves, room, inventory, action history, object-tree hash, selected RNG, ambiguity mapping, journal count and engine identity. There is no tolerance, tree omission or guessed restoration.
4. Only after verification, execute the candidate once in B; return its real next observation/reward/done/score, plus costs. B is closed; A must remain byte-for-byte equal in its captured projection.
5. Reviewer receives only permitted public evidence, never private tree/RNG/fingerprint data. A executes the final action exactly once. No B state or simulated step is committed to A.
6. If Reviewer keeps the Draft, compare BOTH public transition fields and the complete checked post-step fingerprint against B; mismatch stops execution. REPLACE is not a second lookahead and is not claimed valid automatically.
7. Any reconstruction error returns unavailable, removes all prediction fields and calls no Reviewer. KEEP fallback protects state/information integrity, not action correctness.

The profile retains the previous explicit native step/query schedule and post-load seed policy. It still does not make RNG-consuming validity enumeration part of every step. All strategies share this schedule. No trial uses A, and no post-action feedback is disguised as a prediction.

State fingerprints cover the same finite projection, not every hidden JVM variable or all future behavior. The tests establish consistency over recorded scenarios, not a mathematical proof of complete simulator equivalence. Privileged tree/RNG auditing and additional real transitions are explicitly Oracle-Like Lookahead permissions.

## Real regression checks

- Controlled suite: four lawful development instances, seeds0/1/7 and history lengths0/2/4/8. **48/48 reconstruction + public next-transition matches**, zero rejection or false acceptance; A unchanged. Previous version on the same controlled scenario design:27/48 accepted,21 rejected.
- Recorded-history suite: three lawful development Full histories replayed as fixed commands, seeds0/1/7, tested history lengths0/1/4/8/12/16/19. **63/63 reconstruction + public transition + full checked post-step fingerprints match**. These are new real 1.3.0 executions of earlier recorded commands, not a new Actor evaluation or equality with the old physics.
- Nine repeated independent workers agree on the post-step fingerprint. Nine deliberately corrupted evidence journals are rejected before any candidate transition; no feedback leaks. These negative checks are infrastructure tests, not benchmark outcomes.
- All formal isolation checks pass. The history suite counts81 JVM initializations,549 replay commands and72 candidate commands, including repeated-worker and negative checks; these queries are real cost.
- 98 combined contract/Internal/Gate/tokenizer tests pass. Real Actor/Reviewer episodes and cost audit are recorded separately below; mocks are not substituted for real transitions.

## Reproduce

Use the existing isolated inference environment documented in [Qwen integration](QWEN35_9B_INTEGRATION.md). Prepare only the small native engine asset; no new model download or training is required:

```bash
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/prepare_scienceworld_replay.py
# Optional: SCIENCEWORLD_REPLAY_JAR=/a/different/location/scienceworld.jar
# Custom paths must still match the pinned official hash.
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/prepare_scienceworld_replay.py --verify-only

CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python \
 scripts/validate_reset_replay.py --simulator-profile scienceworld130 \
 --output outputs/new_controlled_130_run

CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python \
 scripts/validate_replay_histories.py --reference outputs/phase57_replay_9b_v5 \
 --output outputs/new_recorded_history_130_run

PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_replay_profiles.py
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_replay_integration.py
```

The recorded-history command requires the original ignored raw run on lab; it is a lab regression, not an asset promised inside Git. The controlled suite works from the committed development split and official JAR on a new server. Committed compact evidence contains provenance hashes and representative real transitions; large logs and binaries remain ignored.

For new bounded end-to-end regression, use a NEW output directory and do not pass the old calibration source:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-qwen35-clean/bin/python \
 scripts/run_phase57_replay.py --model-dir checkpoints/qwen35_9b \
 --profile configs/evaluation/phase57_replay_dev_v2.json \
 --output outputs/new_e2e_130_run
```

The driver collects fresh lawful dev NLL values, freezes the prespecified .75 quantiles before four-policy evaluation, and keeps the same frozen9B weights/scoring/Reviewer/Memory/Gate rules. Resuming from a different native engine is rejected. These are distribution calibration and functional checks, not outcome-based threshold selection or final Benchmark evaluation.

## Remaining limits

Reliable true next-state feedback does not guarantee a useful revised action. Reviewer can still generate unsupported or invalid replacements. No Reviewer prompt/algorithm adjustment was made in this repair. Costs include fresh JVM, initialization, replay, private validation, candidate transition and Reviewer; WM LLM tokens being zero does not make simulator cost zero. Successful finite replay validation is distinct from proving Task Success improvements or long-horizon counterfactual utility.

Final end-to-end results and source audit: [repair report](REAL_ENV_WM_REPLAY_FIX_REPORT.md).
