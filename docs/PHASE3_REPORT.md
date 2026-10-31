# Phase 3 report — Training-Free Experience Memory

## A. Completed scope

Engineering acceptance is met for a conservative, narrow rule-based memory. Fresh true training interactions produce validated Cases and two Cards; BM25 plus explicit applicability produces traceable tri-state External Signals through the original ScienceWorld decision loop. The Phase2 Actor prompt, hf_generation_raw_bf16_v1 provider/config and PAPER_ALIGNED_EVAL.md are unchanged. No training, WM invocation Gate, action correction, threshold search or full benchmark was implemented/run. Scientific effectiveness and broad experience coverage remain unverified.

Memory snapshot: memory_6af512c385caf54a11ede606. Final split is phase3/split_manifest.json; technical contract/portable commands are EXPERIENCE_MEMORY.md. User confirmation is required before Phase4.

## B. Files

| Files | Purpose |
| --- | --- |
| experience/schema.py | Case validation, frozen causal query, explicit tri-state contract |
| experience/splits.py | Pinned source audit, canonical identity/goal duplicate exclusion, group-preserving train/dev partition |
| experience/collection.py | Actual HTTP receipt capture, native Case extraction, policy provenance and causal timestamps |
| experience/synthesis.py | Evidence validation, conservative independence/counts/counterexamples and two fixed rule types |
| experience/memory.py | BM25, temporal/source eligibility, applicability, independent provider interface and cost-recorded deterministic judgments |
| experience/__init__.py | Module boundary; no correction/Gate |
| scripts/prepare_experience_data.py; audit_experience_goal_identity.py | Reproducible metadata/split audit; test identities only |
| scripts/diagnose_experience_memory.py; collect_environment_rule_probes.py | Original Qwen Agent loop diagnostics and separately labelled real controlled train probes |
| scripts/extract_experience_cases.py; build_experience_memory.py; audit_experience_diagnostic.py | Receipt-backed extraction, frozen synthesis, causal/prompt/signal replay |
| scripts/test_experience_memory.py | 37 CPU unit cases, with synthetic fixtures explicitly limited to unit testing |
| scripts/run_audited_paper_eval.py | 12-line default-off read-only hook after Draft/Internal Signals and BEFORE actual step |
| README.md; IMPLEMENTATION_PLAN.md; PROGRESS.md; EXPERIENCE_MEMORY.md; phase3/ | Status, contracts, reproducibility, provenance and compact real evidence |

No agentwmv2 file/environment/process was changed. Existing Internal Signals, hidden states, strict SpanUQ code, WM/reflection implementations and benchmark configs were not edited.

## C. Data, synthesis and interface

Pinned AgentGym source: 2,120 training IDs and 200 fixed test IDs; 61 intersect. Exclusions are disjoint: test IDs61, unverified initial-goal format178, other normalized test-goal duplicates336, reserved demo-goal instances56. Remaining1,489: train1,292/431 goal groups; development197/43 goal groups. Previously exposed Phase2 sciworld_349/1709 are reserved as development. sciworld_2997 is excluded through test-goal duplication, not repurposed for memory. Whole demo4509 goal group is excluded. Development follows the paper plan's five families and deterministic SHA ranking with group rounding. Templates/layouts can still overlap.

All 200 native test identities/goals match cached audited hashes/mappings. This involved209 resets including nine selected train/dev identities,23.896s, no Actor/policy actions and no saved test state/reward/score/done. Reset internally obtains an initial look-around; it is not an evaluated policy rollout. Only task identity/goal fields are retained and used for exclusion. Old train trajectories provide initial goal metadata only; no old v2 feedback is memory evidence. ID-less expert files are not eligible. Every selected actual native goal is checked before Actor generation. Goal normalization handles the verified old 'degrees'/current 'degrees celsius' wording and otherwise whitespace/case; arbitrary semantic duplicates are not claimed eliminated.

Memory construction sources:

- 48 real Qwen decisions: sciworld_1667,1545,325,326,3035,3044, each at most8 steps, source-approved train, distinct selected goal groups.
- Six real environment actions in two separate probe episodes on1667/1545. These are explicitly scripted training probes, NOT Qwen outputs or natural Agent errors. Their actual feedback supports environment rules only.
- No dev/test cases in memory; no Oracle calls. The failed first collection is completely excluded.

Cases retain current state/draft fields separately from native executed action/feedback/reward/score/done. Success is null for nonterminal partial trajectories; terminal definition is native done AND score100. Source IDs, canonical task/variation, goal group, policy origin and before/after timestamps are mandatory. Every saved case is re-extracted from the true HTTP receipt. A query can never contain current next_observation, feedback, reward or success.

Synthesis is deterministic, no model calls. Single cases cannot establish a trusted generalized rule. Repetition within an episode does not increase independent episode/instance/goal counts. Duplicate IDs/source steps are rejected; contrary feedback under the same narrow antecedent is a counterexample that blocks trust. Reliability='supported' requires at least2 independent episodes, canonical instances and goal groups, with no counterexamples. This is an evidence label, not a calibrated probability. The criterion was fixed before dev inference; no score/threshold selection used dev/test outcomes.

Actual Cards:

| Card ID | Observed predicate/outcome | Support and scope |
| --- | --- | --- |
| exp_b5e8e19f18820e2efed9 | Explicit currently closed route + go destination -> 'The door is not open.' | Four independent episodes/instances/goal groups, no counterexamples; includes two real Actor failures and two scripted probes |
| exp_8db1a36db0a95a649654 | Explicit currently visible closed door + open that door -> 'The door is now open.' | Two independent episodes/instances/goal groups, no counterexamples; controlled probes only |

First-card support IDs: phase3_train_v2/sciworld_325/step-4, phase3_train_v2/sciworld_3044/step-3, phase3_rule_probes_v1/sciworld_1667/step-1, phase3_rule_probes_v1/sciworld_1545/step-1. Opening support IDs are probe1667/step-3 and1545/step-3. Full evidence, conditions, source IDs, null correction, reliability and counterexamples are in phase3/memory_snapshot.json. No WM Utility Card is asserted: wm_utility='unknown' everywhere because no credible Call/Skip contrast exists.

BM25(k1=1.5,b=.75,Top-K5) queries Goal/Subgoal+current observation+draft action. Future/current-task/current-episode evidence is filtered before IDF; explicit applicability is checked after relevance. Scores never serve as reliability. Coverage=true requires supported applicable evidence; invalid closed-door movement gives Conflict=true/Sufficiency=false, with action unchanged. Opening confirmation requires exact bounded premise/consequence claims and current visible door identity; otherwise abstain. No arbitrary semantic conflict judging, hidden state inference, correction or long-term success claim is made.

ExternalSignalProvider.evaluate(ExternalQuery) returns coverage/conflict/sufficiency in {'true','false','unknown'}, reliability, matched/source evidence IDs, reason, snapshot/query/method IDs and full retrieval/judge time/token/call/forward metadata. Truth enum forbids implicit boolean conversion. Future Gate code must explicitly handle Unknown, not treat it as False or sufficient. Internal/External providers remain independent; no scores are combined yet.

## D. Real diagnostic and tests

Frozen-hash dev cohort: sciworld_1466,146,3055, five real steps each. All15 Draft Actions unchanged; all15 internal signals valid/nonmissing. Two relevant-word Cards can be retrieved but do not match the actual action/state; all15 Coverage/Conflict/Sufficiency remain Unknown. This is honest abstention, not zero error or evidence of usefulness.

Separate targeted, previously exposed Phase2 development cohort: sciworld_1709/349, five real steps each. Judge/retrieval/synthesis rules and snapshot were NOT changed after the first dev cohort. Two steps have applicable supported movement evidence, Coverage=true/Conflict=true/Sufficiency=false; actual subsequent feedback confirms the blocked movement. Other eight abstain. This targeted cohort is not an unbiased estimate of conflict prevalence or detector performance. No natural Sufficiency=true case was observed; positive confirmation and contradictory evidence paths are CPU-unit-tested only.

Five actual examples (full exact state, decisions, Internal/External signals and post-step feedback: phase3/representative_examples.json):

| Task / step / cohort | Current state and Draft Action | Coverage / Conflict / Sufficiency | Audit feedback only after action |
| --- | --- | --- | --- |
|1709/1 targeted | bathroom, door to kitchen explicitly closed; go kitchen | true / true / false; movement Card + four source IDs | The door is not open. |
|349/5 targeted | greenhouse, door to hallway explicitly closed; go hallway | true / true / false; same movement Card | The door is not open. |
|1709/2 targeted | current text only 'The door is not open.'; open door to the kitchen | unknown / unknown / unknown; explicit current door identity absent | The door is now open. |
|1466/1 fixed hash | kitchen lists aluminum foil; focus on aluminum foil. | unknown / unknown / unknown; door Cards inapplicable | No known action matches that input. |
|146/2 fixed hash | current text is failed-action feedback; look around | unknown / unknown / unknown | foundry room description |

Local/Global NLL for1709/1=.029114/.014681,349/5=.029543/.021542 nats/token. These low NLL examples are not calibrated error probabilities and do not validate Hypothesis1 statistically. A real-recorded-input offline empty-memory check also returns all Unknown/empty hits; it is explicitly an input ablation, not another environment rollout.

Cost measured on lab (RTX4090 GPU1; unrelated GPU0 service unchanged):

| Work | Actual measured cost |
| --- | --- |
| Final offline Actor collection |48 calls,1,726 generated tokens including EOS,59.252s generation,.1812s internal extraction |
| Controlled environment probes |0 model calls/tokens;6 native steps,.7163s step-request time |
| Rule synthesis |54 cases ->2 Cards,5.727ms CPU wall;0 model calls/tokens/extra forwards |
| Fixed dev15 |15 Actor calls,511 tokens,17.892s generation;5.066ms retrieval total,.557ms deterministic Judge total |
| Targeted dev10 |10 Actor calls,347 tokens,12.441s generation;3.139ms retrieval total,.345ms Judge total |
| Mean instrumented online retrieval/Judge (25 steps) |.3282ms retrieval +.0361ms Judge wall;0 added model calls/tokens/forward passes; no GPU computation |
| Snapshot |101,212 serialized compact bytes; validation/load9.85/9.75ms per dev process; not measured Python heap/RSS |
| Total Actor process peak |max7.731GiB allocated; includes accepted Actor/Internal pipeline, not attributable to memory |

Retrieval/Judge CPU-time fields are also saved. Peak GPU memory returns to3MiB after exit. Model loading, whole-process startup, query/hash/result serialization wall time, extraction script timing and Python heap delta are not fully profiled; these are not claimed zero. The failed first run's exact total latency/tokens were not recovered and are not folded into successful-run measurements. No formal Total Inference Cost or success-cost comparison is claimed here.

Commands actually executed on lab (all paths relative to agentwmv3 unless explicit external source/env argument):

~~~bash
# Pinned source args are retained in EXPERIENCE_MEMORY.md and phase3 manifests.
.venv-spanuq/bin/python scripts/prepare_experience_data.py --train-ids /data1/shn/tmp/paper-aligned-audit/train_ids.json --test-ids /data1/shn/tmp/paper-aligned-audit/test_ids.json --catalog /data1/shn/tmp/paper-aligned-audit/official/agent_system/scienceworld_protocol.py --goal-metadata /data1/shn/agentwmv2/data/paper_eval/sources/science_agenttraj_train.json --test-metadata /data1/shn/agentwmv2/data/paper_eval/audited/scienceworld_context_v2.json --native-goal-metadata outputs/phase3_audit/native_goals.json --output outputs/phase3_audit/split_manifest_v3.json
.venv-spanuq/bin/python scripts/audit_experience_goal_identity.py --manifest outputs/phase3_audit/split_manifest_v2.json --output outputs/phase3_audit/native_goals.json
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 .venv-spanuq/bin/python -u scripts/diagnose_experience_memory.py --manifest outputs/phase3_audit/split_manifest_v3.json --split train --steps 8 --model-dir checkpoints/qwen3-4b --science-python /home/shn/miniconda3/envs/ee-scienceworld/bin/python --output outputs/phase3_train_v2
.venv-spanuq/bin/python scripts/collect_environment_rule_probes.py --manifest outputs/phase3_audit/split_manifest_v3.json --output outputs/phase3_rule_probes_v1
.venv-spanuq/bin/python scripts/extract_experience_cases.py --manifest outputs/phase3_audit/split_manifest_v3.json --run-dir outputs/phase3_train_v2 --output outputs/phase3_audit/extracted_actor_cases.jsonl
.venv-spanuq/bin/python scripts/extract_experience_cases.py --manifest outputs/phase3_audit/split_manifest_v3.json --run-dir outputs/phase3_rule_probes_v1 --output outputs/phase3_audit/extracted_probe_cases.jsonl
.venv-spanuq/bin/python scripts/build_experience_memory.py --cases outputs/phase3_audit/combined_cases.jsonl --manifest outputs/phase3_audit/split_manifest_v3.json --output outputs/phase3_audit/memory_snapshot_v1.json
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 .venv-spanuq/bin/python -u scripts/diagnose_experience_memory.py --manifest outputs/phase3_audit/split_manifest_v3.json --split development --snapshot outputs/phase3_audit/memory_snapshot_v1.json --steps 5 --top-k 5 --model-dir checkpoints/qwen3-4b --science-python /home/shn/miniconda3/envs/ee-scienceworld/bin/python --output outputs/phase3_dev_v1
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 .venv-spanuq/bin/python -u scripts/diagnose_experience_memory.py --manifest outputs/phase3_audit/split_manifest_v3.json --split development --snapshot outputs/phase3_audit/memory_snapshot_v1.json --task-id sciworld_1709 --task-id sciworld_349 --steps 5 --top-k 5 --model-dir checkpoints/qwen3-4b --science-python /home/shn/miniconda3/envs/ee-scienceworld/bin/python --output outputs/phase3_targeted_dev_v1
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python scripts/test_experience_memory.py
.venv-spanuq/bin/python scripts/audit_experience_diagnostic.py --manifest docs/phase3/split_manifest.json --snapshot docs/phase3/memory_snapshot.json --run-dir docs/phase3/phase3_dev_v1 --output outputs/phase3_audit/archived_dev_audit.json
.venv-spanuq/bin/python scripts/audit_experience_diagnostic.py --manifest docs/phase3/split_manifest.json --snapshot docs/phase3/memory_snapshot.json --run-dir docs/phase3/phase3_targeted_dev_v1 --output outputs/phase3_audit/archived_targeted_audit.json
~~~

Run order was bootstrap split -> native identity audit -> final split -> train/probes -> extraction -> snapshot -> fixed dev -> targeted dev. combined_cases.jsonl is the concatenation of the two validated training extraction files. All successful-run commands passed. Both committed trace archives pass exact external judgment replay, query hash/timestamp checks, native receipt alignment and reconstruction of the unchanged Phase2 Actor prompt. All25 dev decisions precede their own real outcome; training memory is fixed beforehand.

Unit/regression results:37 new tests +117 accepted existing cases =154 passing cases across eight suites. Coverage: schema/missing fields, splits/goal alias/canonical IDs, source independence/duplicate steps, synthesis/evidence/contradictions, BM25/applicability/Top-K, Unknown, causal/future query rejection and snapshot eligibility, bounded sufficiency and cost. Existing suites: Internal14, TokenNLL10, strict SpanUQ11, WM reflection33, AgentWM28, Science evidence18, Science context3. Science evidence initially failed import because PYTHONPATH was absent, passed with PYTHONPATH=.; both logs retained. No core code change was needed. Complete benchmark, threshold search, fresh-server install and SpanUQ recovery were not run.

Failure disclosure: the first train attempt produced16 archived steps then failed goal validation at325/step1 because of degrees/celsius wording; its last completed HTTP result was not persisted by the original downstream-only sink. The entire attempt is excluded, and that final feedback/cost is unverified. Final construction/loading also rejects bootstrap manifests lacking native-normalized test goal identities; this post-run provenance guard is CPU-tested and leaves judgment rules/scores unchanged. A post-diagnostic logging-only fix now persists native reset/step receipts before downstream validation; a CPU unit test verifies receipt survival/restoration on failure. No GPU rollout was repeated solely for this logging change. Executed original source versions/differences are preserved in the compact archives. A first launch before file synchronization failed before any model inference and was simply rerun after synchronization.

## E. Scientific limits

Only two door predicates are implemented. General semantic Premise/Consequence conflicts, task-specific tool rules, reliable corrections and WM causal utility remain unsupported/Unknown. Current observation can be sparse/aliased; no persistent belief state is reconstructed. Strict consequence confirmation is intentionally limited and did not occur naturally here. Positive sufficiency is not empirically validated in an independent real Agent case. The15-step fixed cohort had zero applicable Cards. Two targeted confirmed conflicts do not establish precision, recall, success benefit or cost benefit. The tiny corpus, correlated layouts/templates, old prompt engineering history and prior-exposed targeted tasks must be disclosed in later experiments.

The evidence-count criterion is fixed and heuristic, not statistical calibration. Unknown must not be converted to no-conflict/sufficient. The current provider is a replaceable baseline for the external module, not the final broad experience-reasoning solution. Additional rule/Judge designs would need versioning and independent development validation, with real cost accounting. No threshold or rule was chosen from final test outcomes.

## F. Phase4 prerequisites and plan

Engineering prerequisites are satisfied: independent typed Internal/External providers, unchanged drafts/prompts, frozen training memory/splits, explicit missing/Unknown semantics, real supported conflict examples and traceable costs. This supports implementing/testing a conservative Phase4 Gate AFTER confirmation. It does not justify enabling an empirically unvalidated broad sufficiency skip policy. Define Unknown/missing/conflicting evidence fallbacks, budget semantics and per-step explanations; CPU-test combinations; calibrate only on approved independent development data. Keep Actor/WM/verification/memory-correction conditions common across baselines. Maintain current narrow coverage limitations rather than silently retuning on these examples.

## G. Stop and Git

Phase3 complete; Phase4 NOT authorized. Await user confirmation. Baseline Git was cb5a28dcdd2b7359a6dded40e8c5e59cb8c3a158; actual working-tree source hashes and archived versions are in each run manifest. Final commit and verified GitHub push are reported at handoff. Models, caches, sensitive configs and full large logs remain ignored. agentwmv2 remains clean and unchanged.
