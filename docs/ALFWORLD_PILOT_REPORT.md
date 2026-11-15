# ALFWorld five-policy development pilot report

Completed and audited:3 goal-disjoint TRAIN-pool development games × seed0 ×5 policies=15 complete Episodes. No final140-game rollout, model training, SR promotion or Phase6 evaluation. These are exploratory functional results, not an estimate of official ALFWorld benchmark performance.

## Results

| Policy | Success | SR | Formal steps | Lookahead | LLM tokens | Native API queries | Episode seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Never | 0/3 | 0.0% | 150 | 0 | 109,356 | 153 | 389.2 |
| Always | 2/3 | 66.7% | 113 | 113 | 208,239 | 2826 | 1840.3 |
| Internal | 2/3 | 66.7% | 96 | 8 | 82,780 | 173 | 329.9 |
| External | 0/3 | 0.0% | 150 | 0 | 109,356 | 153 | 388.5 |
| Full | 2/3 | 66.7% | 96 | 8 | 82,780 | 173 | 327.5 |

Counts/tokens/seconds are totals across3 games per policy. Native `won` defines success; auxiliary100×won is not ScienceWorld partial Task Score. Each task contributes one instance equally; one seed has no across-seed sample SD. No significance claim or method/threshold selection is made from these outcomes. Always has50 allowed calls/episode; selective policies have5, so this is not matched-budget evaluation. Token differences reflect both extra verification AND different trajectory lengths; do not attribute them entirely to Gate overhead.

| Task ID (seed0) | Family | Never | Always | Internal | External | Full |
|---|---|---:|---:|---:|---:|---:|
| alf-30fc098bac52af9a | pick_cool_then_place_in_recep | not won (50 steps) | won (23 steps) | won (37 steps) | not won (50 steps) | won (37 steps) |
| alf-bc4a562ca75cc95a | pick_clean_then_place_in_recep | not won (50 steps) | won (40 steps) | won (9 steps) | not won (50 steps) | won (9 steps) |
| alf-e15a00b84f261aa6 | pick_heat_then_place_in_recep | not won (50 steps) | not won (50 steps) | not won (50 steps) | not won (50 steps) | not won (50 steps) |

Machine-readable per-instance outcomes, timing components and paired differences: [episodes.csv](alfworld_pilot/results/episodes.csv), [summary.json](alfworld_pilot/results/summary.json). Real public decision/review/feedback examples: [examples.json](alfworld_pilot/results/examples.json). Raw files remain under the recorded server run paths; [raw-file hashes](alfworld_pilot/results/raw_file_hashes.json) pin them without committing large trajectories.

## Model, protocols and source identity

Actor/Reviewer use one shared frozen Qwen/Qwen3.5-9B model object per worker, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16/SDPA, Transformers5.22.2/Torch2.8.0+cu128, official nonthinking chat template; temperature0.4/top_p1/top_k0/repetition1,response512. H50/history2/input2048 and one-round original three-line Reviewer remain fixed for all policies. No compact-JSON repair or Qwen2.5 service was introduced. Exact configuration: [profile](../configs/evaluation/alfworld_pilot_v1.json), [model profile](../configs/internal_signals/qwen35_9b_alfworld_v1.json), [model runtime](alfworld_pilot/results/alfworld_model_runtime_versions.json), [CPU runtime and source hashes](alfworld_pilot/results/alfworld_cpu_runtime_versions.json).

Original source commit `9b06876c673f5c9a69808f679e8840aa92b02d47`; repaired continuation source `4029c6472723bdeb26a87b702fe128444d8ea5ab`. Original calibration and already-complete Never Episode were reused by checked hashes. Continuation copies the original per-Episode AST exactly. Model load receipts contain no missing/unexpected/mismatched tensors. The same per-task/turn/role seeds are used across policies; this tiny pilot uses rollout seed0, not all eight paper evaluation seeds.

The already-audited paper supplies the ALFWorld evaluation horizon/history/token/sampling target; [PAPER_ALIGNED_EVAL.md](PAPER_ALIGNED_EVAL.md) remains unchanged. This pilot uses native ALFWorld0.4.2/TextWorld1.7.0 PDDL processes and TRAIN-derived development games, a structured draft and public current admissible commands. Native adapter matched150 real transitions against the official ALFWorld Gym wrapper, including H50 truncation. That does not establish parity with every historical AgentGym build or paper score.

One frozen shared draft template has a legacy “ScienceWorld” word in its Action placeholder; surrounding instructions, current admissible commands and all executed environments are ALFWorld. This defect/limitation is disclosed, not silently edited during calibration/evaluation. The model config retains a legacy unused `action_format` label; actual parsing uses the verified generic three-field parser and native commands. Any later prompt cleanup requires a new version/calibration and fresh fair comparison.

## Internal and external signals

Local=max(mean Premise token NLL,mean Expected Consequence token NLL); Global=mean all response-text token NLL, including field labels/Action and excluding prompt/EOS/padding. Whole intersecting tokens are included at field boundaries. Use raw generation logits before temperature, BF16 only, `hf_qwen35_generation_raw_bf16_v1`, no additional backbone forward.48/48 legal reserved calibration steps valid; scored/unscored/repeated generation token IDs equal, repeated raw NLL exact. New ALF-specific q.75 thresholds Local0.5448528666853629/Global0.32670785406992164 frozen BEFORE evaluation, not ScienceWorld thresholds. Pearson0.89687/Spearman0.89796; calibration q.75 union call fraction1/3 before Episode budgets. Six tasks×8 decisions do not establish stable calibration or an error-probability interpretation.

Frozen ALF-only Memory: 31 true feedback cases/2 supported cards, four reserved TRAIN games, deterministic synthesis/BM25/current public preconditions, no LLM judge/synthesis/training. Collection uses scripted public-observation probes with genuine native feedback; these are not natural LLM Actor failure trajectories. Supports require independent tasks AND episodes; Unknown is preserved. Two narrow cards concern already-open container/open repetition and taking through a visibly closed container. No online writes, ScienceWorld cards, future feedback or direct action correction. No natural positive sufficiency is asserted. Native goal hashes and task/goal-group exclusion separate the final frozen Memory/calibration/evaluation/final140. One discarded preliminary-probe task later appears in calibration, but no preliminary case enters final Memory; neither preliminary task overlaps the final3 evaluation or valid_seen goal groups. [Explicit overlap audit](alfworld_pilot/results/alfworld_preliminary_overlap_audit.json). Calibration is not claimed untouched by earlier probing. Task templates/layouts and model pretraining exposure cannot be proven disjoint. The audit requires exact retrieval IDs/order/preconditions/evidence, with1e-12 tolerance only for BM25 scores because the legacy summation traverses a Python set; this does not alter cards or Gate rules.

Full-vs-Internal same-current-Full-state shadow changes: 0. Exact whole-trajectory comparisons and their scope are in summary.json. Zero/rare coverage limits conclusions about the proposed external mechanism; lack of triggers is not evidence that external experience is universally ineffective. A changed action or paired policy score alone is not per-call causal utility. No strict same-state continuation experiment was performed.

An extra Reviewer inference may itself improve planning; there is no no-feedback Reviewer control here to isolate the information value of environment feedback. Random/Periodic matched-budget controls also remain unrun. Policy gains in this pilot therefore do not prove that NLL ranks WM utility correctly or that true lookahead, rather than extra inference, causes the improvement.

## Reviewer and simulator audit

| Policy | Reviewer calls | Format invalid | Replacements | Admissible replacements | Nonadmissible replacements | External coverage |
|---|---:|---:|---:|---:|---:|---:|
| Never | 0 | 0 | 0 | 0 | 0 | 0 |
| Always | 113 | 7 | 52 | 35 | 17 | 0 |
| Internal | 8 | 0 | 5 | 5 | 0 | 0 |
| External | 0 | 0 | 0 | 0 | 0 | 0 |
| Full | 8 | 0 | 5 | 5 | 0 | 0 |

“Admissible” concerns the current native command list, not task correctness; nonadmissible is not automatically a particular syntax/precondition failure. Format failures retain the draft with an explicit INVALID receipt. No trial feedback was fabricated or treated as a formal transition. 129 verified lookaheads and 67 actual KEEP transitions match their candidate receipts. Reset/replay verifies the complete committed history, including invalid actions, independent OS PIDs, public states/admissible commands and private symbolic-fact digests. Private digests are audit-only, absent from Actor/Gate/Reviewer inputs. Passive `inspect` reads a cached last native receipt, not a fresh JVM/hidden RNG snapshot; isolation is supported within tested native deterministic PDDL behavior, not arbitrary process serialization. ALFWorld uses native processes, not ScienceWorld JVMs.

## Complete costs and bounded retries

| Cost scope | Model calls | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|
| calibration | 48 | 23,833 | 2,311 | 26,144 |
| numerical_validation | 2 | 1,472 | 92 | 1,564 |
| completed_evaluation | 734 | 546,557 | 45,954 | 592,511 |
| interrupted_partial_episode | 18 | 15,689 | 1,567 | 17,256 |
| failed_scheduler_overhead | 4 | 3,684 | 172 | 3,856 |

Completed evaluation online tokens=592,511; all recorded model attempts including calibration/numeric checks/interrupted and failed scheduling=641,331. Model loading, generation, NLL extraction, retrieval, Gate, simulator initialization/replay/candidate/cleanup times and peak allocated GPU memory are separately retained in summary.json. The Episode timer starts after formal reset and stops before formal cleanup; total active-run receipts include that extra setup/reset/cleanup time. Model setup attempts include the failed four-worker loads; they are not free. Latency was measured under shared-server load/serialized policy order, not a dedicated hardware performance benchmark.

Offline final/preliminary experience construction uses no model calls/tokens: final35 native queries/11.806s, discarded preliminary9 queries/3.124s, with both retained in [construction receipts](alfworld_pilot/results/offline_construction_cost.json). Validation includes18 controlled replays/122.393s,150 official-wrapper transition matches/46.782s and13 native identities plus6 invalid-prefix controls/76.043s; their detailed receipts remain separate from online query counts. No reliable dollar/FLOP/energy conversion is claimed; tokens, measured latency and environment queries are complementary costs.

An initial two-process native canary predates those formal suites; only native stderr artifacts remain, without a standalone structured query/latency receipt. Its offline environment cost is unverified, not asserted zero or silently folded into exact totals. Exact online evaluation and model-token totals above do not depend on that missing offline receipt.

The initial source run was safely interrupted for scheduling; one completed Never Episode is preserved, its partial Always attempt is excluded from outcome means but charged. Its8 executed actions/transitions and18 Actor/Reviewer prompt/response token-ID sequences reproduce exactly in the repaired continuation; [prefix check](alfworld_pilot/results/alfworld_reschedule_prefix_audit.json). First four-GPU scheduler failed from missing `dataclasses.asdict` at trace serialization; three workers reported NameError and the supervisor stopped the fourth. No completed failed-run Episode was assigned a zero score. Fixed only the missing global binding, statically checked all global names, tested complete trace serialization, and persisted state receipts after STOP.21 CPU contracts pass. Original Episode AST, Actor/Gate/threshold/Memory unchanged.

During that failed run, recorded model cost is exact (4 calls/3856 tokens); one interrupted native branch has bounded receipt uncertainty:10–11 API query attempts and9–11 state replies, not falsely exact counts. Explicit conservative charges are included in the resource cap, not relabelled as measured queries. See [engineering repair](alfworld_pilot/parallel_engineering_repair.json). On reconnection only GPU0 was sufficiently free; repaired continuation used one GPU under the user-authorized maximum4. The cumulative2h ACTIVE server-runtime cap includes prior415.225s and failed40.738s, excluding network downtime after all owned model processes exited. No other job was stopped.

Final result export initially used the CSV module's default CRLF line endings, which Git's whitespace check rejected. Export now uses explicit LF and the full strict audit is rerun; no trajectory or numeric result changes. The original finalizer error log remains under outputs/alfworld_finalizer.log.

## Reproduction and remaining scope

Run the native validation/collection commands in [ALFWORLD_PILOT.md](ALFWORLD_PILOT.md) with isolated `$MODEL_PYTHON` and `$ALFWORLD_PYTHON`; preserve pinned data/assets and independent split. Exact continuation performed:

```bash
"$MODEL_PYTHON" scripts/test_alfworld_pilot.py
"$MODEL_PYTHON" scripts/test_alfworld_parallel.py
"$MODEL_PYTHON" scripts/run_alfworld_parallel.py --source outputs/alfworld_five_policy_v1 --run outputs/alfworld_parallel_v2 --gpus 0 --failed-prior outputs/alfworld_parallel_v1
"$MODEL_PYTHON" scripts/analyze_alfworld_pilot.py --run outputs/alfworld_parallel_v2 --output outputs/alfworld_final_audit_v1
"$MODEL_PYTHON" scripts/check_alfworld_cleanup.py --run outputs/alfworld_parallel_v2
"$MODEL_PYTHON" scripts/export_alfworld_pilot.py --run outputs/alfworld_parallel_v2
```

Continuation requires the original checked run/receipts and a fresh continuation Run ID; a fresh-server run starts with the documented original driver and collection, not a fabricated source run. Full fresh-server reproduction has not been performed. All owned model/native processes must be checked after completion; the final supervisor receipt is retained. Engineering verification and observed policy results are separate from scientific effectiveness. Stop here; no automatic Phase6/final140×8 evaluation.

Final [process cleanup receipt](alfworld_pilot/results/cleanup.json) reports no owned model/native processes. No other user's process was terminated. This ALFWorld diagnostic does not revise or replace earlier ScienceWorld findings.
