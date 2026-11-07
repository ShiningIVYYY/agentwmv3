# Phase5.7 Reset + Replay: resumption checkpoint

Status: IN PROGRESS, SSH connectivity blocked. This is not a completed Phase5.7 report. The user approved verified independent reset/replay instead of complete Java-state serialization. No Phase6 authorization.

## Current server and baseline

- SSH lab:10.20.98.46:35461; project /data1/shn/agentwmv3.
- Last committed/pushed baseline fdbfa664a7beec7976f83bd177d11afff42d0408.
- Current new implementation is UNCOMMITTED on the server. No agentwmv2 edits.
- New code is staged locally for transfer under /private/tmp/agentwmv3-replay. This file records observations, not replacement raw experiment data.
- Preserve docs/PAPER_ALIGNED_EVAL.md, existing Memory snapshot and old threshold/config files unchanged. Do not reuse4B thresholds.

## Implemented and actually tested

- worldmodel/ground_truth_environment.py: independent JVM per Call; same task/variation/simplifications/seed; replay all committed commands (invalid and disambiguation included); compare every prefix/current projection; one candidate step only; explicit reconstruction_failed with no public prediction when checks fail; keep draft without Reviewer; close owned worker; compare formal state before/after.
- Projection: canonical current/public feedback, reward/done/score/native moves, room, inventory, Python last score, native completion, executed action history, object-tree hash, selected Scala/Java RNG and ambiguity mapping. This is NOT complete hidden-state equivalence.
- SDK1.2.3 get_valid_action_object_combinations consumes Scala RNG differently across JVMs; load also resets the seed from variation. Corrected Python wrapper uses public native step and identical SDK scoring/termination, avoids unused RNG-consuming valid-action enumeration, sets transition seed after load. Installed SDK/JAR/Java/Scala are NOT modified.
- Actual thermal state drift remains after wait histories (e.g. temperatures differ in third/fourth decimal or greater). Tree mismatches remain hard rejection. No tolerance was silently introduced.
- Real CPU cohorts:4 approved dev IDs128,142,2417,3070; seeds0/1/7; prefix lengths0/2/4/8,48 scenarios each.
  - v1 SDK.step:6 verified/matched,42 rejected,0 verified-transition mismatches;119.832s.
  - v2 native wrapper:27 verified/matched,21 rejected,0 verified-transition mismatches;108.579s.
  - v3 final cleanup/accounting:27 verified/matched,21 rejected,0 verified-transition mismatches;113.349s. All formal-isolation checks pass. Own JVMs close.
- Full raw logs on server:outputs/phase57_replay_validation_v{1,2,3}; v3 has source hashes/provenance. Minimal real failure script scripts/reproduce_replay_rng.py and outputs/phase57_setup/replay_minimal_failure.{json,txt}.
- CPU tests:8 new replay/review/budget/privacy contracts;63 legacy Gate cases;14 legacy Internal Signal tests pass. First premature test raced an incomplete upload and is preserved separately.
- Gate rule unchanged; constructor additionally recognizes one-forecast budget for the explicitly named new one-round replay protocol, retaining4 for legacy JSON review.
- Provider and old HFActor receive minimal backward-compatible backend selection for new fingerprint hf_qwen35_generation_raw_bf16_v1. New Qwen35Actor reuses accepted generation/scoring method.

## 9B assets, environment and genuine failures

- Downloaded official post-trained Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a under checkpoints/qwen35_9b.4 shard hashes match official LFS metadata; manifests/logs under outputs/phase57_setup. No Qwen2.5 service/request.
- .venv-qwen35 is a separate overlay environment:Transformers5.22.2,ScienceWorld1.2.3,Python3.12,borrowed base PyTorch2.8.0+cu128. Old environment packages unchanged. Inherited unused verl/datasets constraints produce resolver warnings; do NOT claim pip check clean. Final reproducibility docs must disclose this.
- Shared Actor/Reviewer uses ONE Qwen3_5ForConditionalGeneration instance, full BF16 parameters, SDPA and official PyTorch reference DeltaNet kernels (optional fast kernels absent). Strict loading has empty missing/unexpected/mismatched/error sets; no strict=False or tensor edits.
- model process v1 PID3367931 exited at an over-strict CPU/GPU raw-NLL equality assertion.1 ordinary generation occurred but its token receipt was not persisted; report missing cost evidence, not fabricated exact tokens.
- v2 PID3373913 exited after strict loading because Transformers5 loading_info contains sets and JSON serialization failed. No generation in that attempt. Metadata-only conversion of sets to lists fixes it.
- v3 PID3375567 exited BEFORE any ScienceWorld Actor decision at native development Goal identity check. Its finally closes the environment. Last observed process/GPU list after exit was empty.
- v3 real model evidence outputs/phase57_replay_9b_v3/model_smoke.json:
  - strict loading all error collections empty;load19.546538s.
  - official nonthinking template has closed empty think prefix.
  - captured raw GPU NLL and formal GPU reduction exact delta0.
  - CPU FP32 diagnostic max delta9.426474571228027e-5; excluded from thresholds.
  - separate teacher-forced validation forward max delta8.946663001552224e-5; never mixed with generation-path calibration.
  - ordinary output contains hello + im_end + newline because only model EOS was used.
- Actual official metadata:config text EOS248044=endoftext;tokenizer EOS248046=im_end;pad248044. No generation_config.json exists in this pinned repository. Official chat EOS must be included in the new generation stop IDs. Do not strip arbitrary internal special tokens to fake validity.
- Goal issue:SDK.taskdescription adds 'Task Description:\n'; AgentGym/native identity is server.getTaskDescription() without this presentation label. Keep original goal_key/split rules; use raw native Goal rather than weakening identity validation.

## Pending local fixes, NOT yet remotely verified

The latest three SCP uploads failed/timed out during campus-network loss; deployment status must be checked before restarting:

1. GroundTruth adapter self.goal=str(self.env.server.getTaskDescription()).
2. Qwen35Actor pinned EOS union of official model and tokenizer IDs, asserted against config.
3. configs/internal_signals/qwen35_9b_replay_v1.json adds eos_token_ids[248044,248046] and explicit EOS policy, changing its config fingerprint.

Latest local source files contain these fixes. Do NOT claim they passed inference or calibration yet. All earlier runs are preserved; no thresholds or four-policy episodes exist.

## Authorized continuation

1. Restore lab route/SSH. Last failed fresh SSH ConnectTimeout10 and TCP35461 probes; local route sends10.20.98.46 through Wi-Fi/default192.168.1.1, no campus route. User asked asynchronously to restore campus/VPN. Do not alter VPN settings without authorization.
2. Verify no own GPU/JVM process, inspect git status/source hashes. Complete three pending uploads and syntax/unit checks.
3. Run new output outputs/phase57_replay_9b_v4 using same args as v3, GPU0 only if free. Never reuse failed Run IDs. Shared model runner command is in outputs/phase57_setup/qwen35_job_v3.json. Keep failures and all model costs.
4. main performs ordinary raw-logit check, true dev structured generation, scored/unscored parity and exact repeated-NLL checks;6 calibration dev tasks xup to8 actions; freeze prespecifiedq.75 thresholds before3 disjoint dev tasks xNever/Always/Internal/Full,seed0,H20,budgets0/20/5/5. Do not tune using score.
5. Observe actual outputs; fix only verified generic engineering issues. Pipeline fallback is keep draft/no Reviewer on failed reconstruction; never old WM or post-action reflection. Fatal verified-KEEP mismatch stops run.
6. Analyze call attempts vs verified trials vs Reviewer calls, score/success and all model/environment/replay costs, failures, missing spans, parity, Local/Global correlation and limited6-task calibration. Report thermal-drift coverage limitation and oracle-like permissions. Do not claim scientific effectiveness from this smoke.
7. Update REAL_ENVIRONMENT_WM,QWEN35_9B_INTEGRATION,PHASE5_7_REPORT,README,IMPLEMENTATION_PLAN,PROGRESS; archive prior clone audit as historical; commit compact real evidence and scripts, not weights/cache/full logs; git commit/push existing remote and verify. STOP before Phase6.

## Latest handoff after reconnection attempts

The campus connection briefly recovered. A full staging-directory SCP completed with exit0, transmitting the raw native Goal and dual-official-EOS fixes. Subsequent SSH readback/native-Goal validation and v4 launch both timed out. Ten additional bounded, hash-checked/idempotent startup attempts all failed at SSH connection (255). The retry process has finished; v4 never launched and there is no queued retry. No commit/push occurred. Current GPU availability cannot be asserted; do not disturb unrelated jobs. Latest pending code is backed up under /Users/ivy/agentwmv3-phase57-pending. The raw server results remain authoritative; this checkpoint is not a completed acceptance report.

## Final completed continuation

Network restored. v4 strict9B/model/tokenizer/parity/repeated-NLL checks passed,40 real calibration steps over6 legal tasks,0 missing,q.75 frozen. v4's zero-based query caller stopped evaluation before any environment action; fixed to one-based, added eager generation logs and strictly reused calibration in v5.

v5 completed12 episodes/201 formal actions on GPU3,PID3939349 exited. Means Never-61/Always-7.6667/Internal-100/Full41.6667,all0 successes.75 lookahead attempts,21 verified/Reviewer calls,54 rejected.13 real corrections,7 rejected execution,one+8 immediate score delta;no downstream utility proof. Independent trace/source/cost/leakage audit passes.

A separate pure .venv-qwen35-clean now has no inherited RL distributions,pip check0,91 tests passed. Real Actor/Reviewer prompt IDs match original run,Actor NLL delta0;real SDK replay pair matches. PortabilityPID3964254 exited;no new benchmark was substituted. Original environments unchanged. Complete report/protocol/README/plan and compact evidence updated;no Phase6. Final git commit/push verified at handoff.
