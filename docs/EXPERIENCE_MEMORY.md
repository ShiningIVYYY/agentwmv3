# Training-Free Experience Memory (Phase 3)

This is a narrow evidence module, not a learned utility predictor. It reads true training interactions, synthesizes two explicit observational predicates, retrieves with BM25, and returns tri-state external signals. It never rewrites the Actor prompt/action or invokes a WM. SpanUQ remains shelved; the accepted Phase2 provider and paper-aligned evaluation protocol are unchanged.

The design borrows the separation between training experiences and recalled insights from [ExpeL](https://arxiv.org/abs/2308.10144v3) and structured evidence organization from [ACE](https://arxiv.org/abs/2510.04618v3). This implementation uses deterministic rules, not their complete LLM reflection/playbook algorithms, and reproduces none of their performance claims.

## Data identity and isolation

Source ID revision: AgentGym/AgentGym-RL-Data-ID @ 99d0b7923bf126d9c6cdea5f362d2d41ccb5d493. Source/catalog/file hashes and all final partition IDs are in phase3/split_manifest.json. The catalog is pinned to the audited reference file at Self-Distillation-AgentRL @ 35a44f5c5647b1785ad23ddf0ffbd196b695bf86. Neither RL training code nor training reward shaping is used.

From 2,120 training IDs, exclude 61 overlapping test IDs, 178 other instances with unverified initial-goal format, 336 other normalized test-goal duplicates, and 56 demo-goal instances. Remaining 1,489 instances form 1,292 memory-training and 197 development instances. The 200 evaluation IDs remain fixed. Whole normalized goal groups are assigned together: 431 train groups and 43 dev groups. Development uses the five paper-aligned families, SHA256 canonical-ID order and category instance quotas, with whole-group rounding and reserved prior diagnostic groups. Additional source-train families remain train-only. The materialized dev size is 197, not the old provisional 195.

Initial goal metadata comes from the old released train-trajectory copy; ONLY the initial goal/ID is used. No old trajectory action/feedback is an experience source. ID-less expert/demo files are rejected for evidence construction. All 200 current native test task names/variations/goal hashes match the old audited identity record. The identity-only audit saved goal fields only, discarded initial observation/reward/score/done, issued no policy actions and made no model calls. It is not test evaluation or an experience source.

Native goals say 'degrees celsius' where some old thermometer metadata says 'degrees'. This was verified in three selected thermometer instances. A versioned whitespace/casefold/unit-wording identity normalizer handles that specific alias; numbers, objects, rooms and task requirements remain significant. All 200 native test goals are normalized for duplicate exclusion. Every collected train/dev goal is checked again before Actor generation and during extraction. Raw and normalized hashes are distinct and retained. Synthesis/loading reject bootstrap manifests without native-normalized test goal identities. The unused pre-audit manifests and failed collection remain under outputs/phase3_audit for diagnosis; only final v3 is accepted.

Task templates and physical layouts overlap by design. Exact ID, canonical task/variation and normalized goal isolation are checked; semantic near-duplicate independence is NOT established. Goal metadata cannot certify every uncollected training instance against simulator changes. New collections must repeat the native goal guard. Prior v2 test-informed engineering choices and previously exposed Phase2 development cases prevent a claim of a completely pristine unseen research process.

## Contracts and evidence

experience/schema.py defines Experience Case validation and frozen ExternalQuery. Cases contain benchmark/task/canonical instance, goal group, episode, goal/subgoal, current observation, Premise/Expected Consequence, draft/executed action, native feedback/reward/score/done/nullable success, WM flag, source trajectory and step, decision/outcome timestamps and provenance. Partial-episode success is null; terminal success is done AND native score==100, retained only as audit metadata. Phase3 does not aggregate benchmark success.

Pre-action fields are separated from post-action outcomes. A saved HTTP step receipt must match the executed action, feedback and score exactly. Query input permits ONLY goal/subgoal, current observation, draft fields and current identity/time. It has no reward, done, next observation, oracle state or feedback fields; extra keys are rejected. Query hashes and wall-clock ordering prove retrieval/judgment preceded the real step request. The Actor receives the unchanged Phase2 visible-history prompt. The observation is the latest actual native text, not a hidden full simulator state or inferred state persistence.

Memory uses 48 fresh Qwen steps from six train tasks, plus six actual environment actions from two separately labelled controlled train probes. Probe policy: attempt an observed closed route, actually look around, then open the observed door. These are genuine simulator outcomes, NOT Qwen-generated actions, natural error examples, WM counterfactuals or benchmark performance. Cases record which policy produced them. All 54 cases have approved training IDs; no development/test episode enters the snapshot. The separate failed first run is entirely excluded.

Cards record applicability_conditions, observed_rule, failure_pattern, nullable suggested_correction, supporting_cases, counterexamples, reliability, evidence_type, independent episode/instance/goal counts, source task/trajectory IDs, availability time and synthesis version. WM utility is always unknown: no credible Call/Skip causal comparison exists. No model training, embedding model or LLM synthesis is used.

Frozen predicates in synthesis.py:

1. closed_route_go: current observation literally names a closed door to the drafted destination; actual feedback is exactly 'The door is not open.'.
2. closed_door_open: current observation literally names the closed door the draft opens; actual feedback is exactly 'The door is now open.'.

Any matching antecedent with different feedback is retained as a counterexample and blocks trusted judgments. Nonmatching cases remain archived without inventing a generalized rule. Repeated steps within one episode do not create independent support. Duplicate case IDs/source steps are rejected. A card is labelled supported only with at least two distinct episodes, canonical instances AND goal groups, and no counterexamples. This is a fixed conservative evidence criterion, not calibrated probability or a fitted threshold. A Qwen failure-pattern tag additionally requires observations from two distinct Actor task instances; scripted probes alone cannot establish Agent error frequency. Supporting cases and their native outcomes are revalidated on snapshot load.

No correction is asserted from these cards. The opening observation establishes a narrow immediate effect, not completion of the task or causal WM benefit. Hypothetical mechanisms and speculative failure explanations are not synthesized.

## Retrieval and tri-state judgments

ExperienceMemory implements ExternalSignalProvider.evaluate(ExternalQuery). Snapshot is immutable during rollouts. Before BM25/IDF calculation, filter out future snapshots/cards, the current task's evidence and current-episode evidence. Train/dev goal-group isolation is enforced at construction. BM25 uses lowercase alphanumeric terms, k1=1.5, b=.75, positive scores, deterministic card-ID ties and configurable Top-K (diagnostic5). Returned scores are relevance ONLY; they are not reliability or WM utility. Applicability is then checked with the explicit current-observation/action predicate. No positive score means an empty retrieval result; irrelevant hits give no applicable evidence.

coverage/conflict/sufficiency serialize as 'true', 'false' or 'unknown'. Truth enum prohibits implicit bool conversion. Consumers must compare these values explicitly; bool('unknown') is incorrect. No evidence, insufficient independent evidence or counterexamples yield Unknown, never automatic False/no-conflict or sufficient.

- Coverage=true only for supported applicable evidence without applicable uncertain cards.
- Conflict=true for an unresolved attempt through a currently observed closed route. Sufficiency=false because the unchanged action remains unresolved.
- Opening confirmation is deliberately stricter: exact supported immediate-effect and condition wording, the visibly closed named door, no conflicting evidence. Only that bounded claim can yield conflict=false/sufficiency=true. Other Premise/Consequence claims remain Unknown. This is not general semantic NLI; arbitrary premise/consequence conflicts are not covered.
- Missing current door identity, sparse action feedback, field parse failure or unsupported actions abstain. Previously visible state is not silently reconstructed from history. The natural opening step in this diagnostic therefore abstains.

Output includes matched card IDs, supporting case IDs, retrieved relevance/applicability/source IDs, reasons, method/snapshot/query hashes, retrieval/Judge wall and CPU time, additional tokens/calls/forwards and serialized snapshot size. Judge is deterministic CPU logic, not a hidden LLM call. Unknown WM utility is a separate field. No proposed action or correction is executed.

Phase4 must explicitly define Unknown handling, budget behavior and missing internal-signal handling; this module makes no invocation decision. Internal and external providers remain independent and can be replaced without altering the Actor. Broader predicates or a frozen LLM Judge need a new version and independent development validation, not silent adaptation to these results.

## Reproduce

Existing isolated .venv-spanuq is sufficient; no new dependency was installed. Supply pinned source files, a ScienceWorld1.2.3 HTTP service with the recorded JAR, Qwen3-4B pinned weights and its existing accepted config. Paths below are variables, not code defaults. Use a new output directory for every call. ScienceWorld identity metadata collection creates/closes its own session and does not modify server packages.

~~~bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python scripts/test_experience_memory.py
# Bootstrap metadata manifest (never use it for synthesis).
.venv-spanuq/bin/python scripts/prepare_experience_data.py --train-ids "$TRAIN_IDS" --test-ids "$TEST_IDS" --catalog "$CATALOG" --goal-metadata "$TRAIN_GOAL_METADATA" --test-metadata "$TEST_IDENTITY_METADATA" --output "$BOOTSTRAP_MANIFEST"
.venv-spanuq/bin/python scripts/audit_experience_goal_identity.py --manifest "$BOOTSTRAP_MANIFEST" --science-url "$SCIENCEWORLD_URL" --output "$NATIVE_GOALS"
.venv-spanuq/bin/python scripts/prepare_experience_data.py --train-ids "$TRAIN_IDS" --test-ids "$TEST_IDS" --catalog "$CATALOG" --goal-metadata "$TRAIN_GOAL_METADATA" --test-metadata "$TEST_IDENTITY_METADATA" --native-goal-metadata "$NATIVE_GOALS" --output "$FINAL_MANIFEST"
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 .venv-spanuq/bin/python scripts/diagnose_experience_memory.py --manifest "$FINAL_MANIFEST" --split train --steps 8 --model-dir "$QWEN3_MODEL_DIR" --science-python "$SCIENCEWORLD_PYTHON" --science-url "$SCIENCEWORLD_URL" --output "$TRAIN_RUN"
.venv-spanuq/bin/python scripts/collect_environment_rule_probes.py --manifest "$FINAL_MANIFEST" --science-url "$SCIENCEWORLD_URL" --output "$PROBE_RUN"
.venv-spanuq/bin/python scripts/extract_experience_cases.py --manifest "$FINAL_MANIFEST" --run-dir "$TRAIN_RUN" --output "$ACTOR_CASES"
.venv-spanuq/bin/python scripts/extract_experience_cases.py --manifest "$FINAL_MANIFEST" --run-dir "$PROBE_RUN" --output "$PROBE_CASES"
cat "$ACTOR_CASES" "$PROBE_CASES" > "$ALL_CASES"
.venv-spanuq/bin/python scripts/build_experience_memory.py --manifest "$FINAL_MANIFEST" --cases "$ALL_CASES" --output "$SNAPSHOT"
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 .venv-spanuq/bin/python scripts/diagnose_experience_memory.py --manifest "$FINAL_MANIFEST" --split development --snapshot "$SNAPSHOT" --steps 5 --model-dir "$QWEN3_MODEL_DIR" --science-python "$SCIENCEWORLD_PYTHON" --science-url "$SCIENCEWORLD_URL" --output "$DEV_RUN"
.venv-spanuq/bin/python scripts/audit_experience_diagnostic.py --manifest "$FINAL_MANIFEST" --snapshot "$SNAPSHOT" --run-dir "$DEV_RUN" --output "$AUDIT_RESULT"
~~~

Use --task-id only for an explicitly labelled targeted development diagnostic; it cannot collect memory from development. The final collector also persists reset/step receipts before downstream validation, a CPU-tested post-diagnostic logging fix. Actual executed versions are preserved in phase3 source archives. Original full logs are under outputs on lab; compact committed archives exclude enormous SDK admissible-action lists, which never entered the Actor or memory. Native receipts/cases, exact generation records and causal query checks remain available. Fresh-server provisioning is still Phase7 work, not claimed by this diagnostic.
