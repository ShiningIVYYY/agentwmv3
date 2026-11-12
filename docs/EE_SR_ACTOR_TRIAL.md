# Early Experience SR Actor: bounded real trial

Status: COMPLETE; checkpoint is usable as an opt-in Actor, but is not approved as a drop-in main-system replacement. No Phase6, training, Gate/Memory/threshold modification or final-test rollout. Existing Qwen3.5-9B default remains preserved.

## Outcome

The released SR policy strictly loads and runs real ScienceWorld episodes. Native-H2 produces scores83,−100,5 (mean−4;0/3 success). The current structured decision prompt fails in all3 episode attempts, and all7 structured generations lack Premise/Expected Consequence spans. Numerical Token NLL works, but the required semantic Local signal is unavailable. We do not fabricate fields, use missing=0, or apply old9B thresholds.

This is a task-trained Qwen2.5-7B policy, not the retired Qwen2.5 learned WM. It was tested only as Actor; no old WM service was started. SR Reviewer effectiveness and full selective invocation integration remain UNTESTED.

## Sources, versions and training boundary

- [Released model](https://huggingface.co/osunlp/early-experience-scienceworld/tree/4b8f0835a3f613e40541619462ae906db8dba95c/sr): revision `4b8f0835a3f613e40541619462ae906db8dba95c`, subfolder `sr`; full Qwen2ForCausalLM/Qwen2.5-7B-Instruct policy,28layers/3584hidden/BF16. It is not a LoRA-only checkpoint.
- [Official ScienceWorld reproduction notes](https://github.com/OSU-NLP-Group/EarlyExperience/blob/3810f05ef7d98465e3abefd38d1c0e7441d4dee0/envs/scienceworld/README.md), source commit `3810f05ef7d98465e3abefd38d1c0e7441d4dee0`. The upstream reported66% is not our measurement. Its one-shot/history/training-admissible-action protocol is not reproduced here. A complete matching author inference implementation was not verified.
- [Published experience corpus](https://huggingface.co/datasets/osunlp/early-experience/tree/7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce), revision `7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce`. Recipe includes task-specific expert/reflection training. We perform no additional parameter updates. If adopted, describe the contribution as training-free invocation/Gate over a frozen task-trained Actor; do not claim the entire system has no task-specific parameter training.
- Existing isolated `.venv-qwen35-clean`: Transformers5.22.2/Torch2.8.0+cu128/Python3.12, RTX4090, BF16. No dependency installation or old-environment change. Strict loading returns empty missing/unexpected/mismatched/error lists. All parameters frozen/eval.
- ScienceWorld native1.3.0 remains unchanged: native source `e8216d6044e8e39be9fcb185e3b2dfb602584b52`, JAR SHA256 `e77b0fee7d68abe3ca5b12e57d86e2bfe7603f20c5200da0b0f085939faeb465`, SDK1.2.3. No Java/Scala changes or formal-environment previews.

Only14 SR files were downloaded:15,247,179,567bytes; every file verified against upstream LFS SHA256 or Git blob hash. No IL/IWM weights downloaded. Initial metadata-only downloader assertion expected13files; it stopped before weights, was corrected to14, and original logs retained.

## Data isolation audit

The expert file has2038 dialogues/18,808,399bytes, SHA256 `096895094fe44005b03037b44414388bc4935dcc0224fcc678f9811545cc2708`. Only a64KiB reflection-file prefix was inspected. Published expert goals overlap1226/1292 training IDs,186/197 development IDs (40/43 goal groups), and106/200 final-test IDs. These counts refer to normalized Goal text, not proven native Task×Variation membership.

Across the pinned2120-record AgentTraj reference corpus, full user-observation sequences match0 records. Goal+complete expert action-sequence matching identifies261 unique reference records (including14dev/2test);14 matches are ambiguous. These sequence matches do not establish hidden Variation equality or actual checkpoint optimizer-input membership. That training-row manifest is unavailable/unverified. Therefore neither clean goal-level isolation nor independent generalization may be claimed; it is also incorrect to assert106 exact test instances definitely trained this checkpoint.

The reference corpus was read only for identity auditing; no expert actions, answers or solved demonstrations were supplied to Actor/Memory/Gate. No final200-test environment queries occurred. Full IDs, methods and caveats: [data_overlap.json](ee_sr_trial/data_overlap.json).

## Fixed design and fair controls

Three already-exposed development instances, not independent held-out evidence: `sciworld_2948`/variation12, `sciworld_715`/55, `sciworld_1619`/183, seed0. Same public initial-state fingerprints match across all four conditions, including historical baseline. This verifies the observed initial-state contract, not equality of every hidden JVM bit.

Two prompts: (1) `structured`, exactly the existing Premise/Expected Consequence/Action prompt content; (2) `native_h2`, the published generic instruction/acknowledgement and Thought/Action dialogue style adapted to the same2-observation history. Native-H2 inserts no expert action, solved one-shot or future feedback. It is an adaptation, not an official-protocol replication.

Horizon20/history2/prompt4096/response512/T=.4/top_p=1/top_k=0/repetition=1, single unpadded sequence, native-cap21. Official saved generation defaults are T=.7/top_p=.8/top_k=20/repetition=1.05. Inference explicitly enables cache; saved training config has cache=false. Existing single-sequence generation helper uses EOS-as-PAD151645 (official pad151643); EOS list151645/151643 preserved. These deviations are explicit and fixed. Qwen2.5 has no invented Qwen3-style thinking switch; ordinary Thought output counts toward tokens.

After the first six SR attempts, a further three Qwen3.5-9B native-H2 episodes were preregistered and frozen to assess prompt confounding, without task replacement or prompt tuning. This preserved9B revision is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Historical structured9B results are references, not reruns or new costs. Actor-only trials have0 Gate, Memory, Reviewer and WM calls.

## Every task result

| Task/variation | SR structured | SR native-H2 | 9B native-H2 | Historical9B structured |
|---|---:|---:|---:|---:|
|2948/12 find living thing|NA:format failure before action|83|−100|−100|
|715/55 melting point|NA:format failure after4steps|−100|−100|−100|
|1619/183 conductivity|NA:format failure before action|5|NA:format failure after19steps|0|
|Complete episodes|0/3|3/3|2/3|3/3|
|Mean over complete cohort of3|NA|−4.00|NA|−66.67|
|Successes|NA|0/3|0/2 completed; third incomplete|0/3|

Incomplete attempts retain partial native scores8/6/0 (SR structured) and60 (9B native1619) in raw audit files; these are NOT completed-episode scores. The9B native mean−100 among only its2 completed episodes is not a comparable3-task mean.

SR2948 makes substantial partial progress, but all3 SR native episodes still fail to succeed within termination/H20. Task715 reaches native−100; this is a real task failure, not assigned as a parser fallback. Differences from historical9B structured are confounded by prompt format; the additional matched-native control does not yield a complete cohort comparison. No general model ranking, Gate benefit, causal WM utility or independent generalization conclusion follows.

SR structured failures:2948 emits bare `open door to hallway`;715 eventually emits multiple Action fields and imagined Observation text;1619 emits3 Action fields with repeated prose. Outputs are retained and explicitly rejected. No imagined Observation is accepted as environment feedback. The parser accepts exactly one anchored Action field with one command and no trailing prose. The9B native1619 failure emits512tokens of Thought without a final Action, ending by length; its partial score60 cannot be reported as task completion. No hidden retry or fallback action is used.

## NLL and compatibility

New configuration identity `hf_ee_scienceworld_sr_raw_bf16_v1`, fingerprint `99a3c64d41b49912edc1e6ffde1af11a35ab60e4d98e977dc0e4ea9a121b738a`. Existing alignment/reduction methods are reused through an explicit model-specific provider; old9B provider/config remains unchanged.

First scored/unscored/repeated generations have identical token IDs and repeated raw NLL max delta0.0. Its4 non-EOS response tokens yield GlobalNLL0.06773191690402314 nats/token;5 generated tokens include EOS. Prompt/EOS/padding excluded from Global, labels/Action included when present. No extra rescoring forward is used;2 extra diagnostic GENERATION calls are charged. The numerical test verifies this first response only, not all possible sequences/backend equivalence.

All7 primary structured generations have missing Premise/Consequence; Local remains null. Provider reports partial/global-only diagnostic fallback explicitly, never valid Local evidence. Global-only fallback is not an approved replacement for our Local+Global research definition. No SR thresholds calibrated and no old thresholds reused. Native mode does not claim Premise/Consequence scores. Full-Gate/Reviewer integration is therefore not accepted.

## Actual cost and budgets

|New condition|Actor calls|Input+output tokens|Formal steps|Native transition queries*|Episode wall seconds|
|---|---:|---:|---:|---:|---:|
|SR structured, including failed generations|7|3533|4|7|16.382|
|SR native-H2|50|36305|50|53|32.851|
|9B native-H2|33|31193|32|35|212.482|
|Two extra SR numerical generations|2|670|0|0|included in SR job total|
|All NEW measurements|92|71701|86|95|see job totals below|

*Queries include one native initialization transition per attempt plus formal action transitions; there are no lookahead queries. State-audit capture bundles are separately183, not free operations and not action transitions. Reviewer/WM/Gate/Memory calls0; no offline synthesis/training occurred. Monetary cost and GPU energy were not measured.

SR job wall76.001s includes strict load26.742s, all6attempts and numerical calls. Control job wall246.797s includes9B load34.308s. Jobs ran sequentially on GPU1; sum322.798s excludes download/CPU audit. Observed peak CUDA allocated memory over executed decisions: SR14.408GiB,9B17.868GiB; these are allocated memory, not device-wide residency/energy. Historical9B structured cost12842tokens/29steps/32queries/71.300s is separate and not counted as new usage.

SR budget130calls/250ktokens/140transitions/700captures/2h and control60calls/150ktokens/63transitions/200captures/1h were respected. No outcome-based retry/task substitution. Both owned model processes completed; environment instances close in finally. Cleanup/immutable-profile checks are included in the final receipt.

## Implementation and reproducibility

- `internal_signals/ee_sr_actor.py`: strict frozen Actor and explicit SR NLL provider; original Actor preserved.
- `internal_signals/ee_sr_prompt.py`: unchanged structured content, native-H2 adapter, strict single-action parser.
- `configs/internal_signals/ee_scienceworld_sr_v1.json`, `configs/evaluation/ee_sr_actor_trial_v1.json`: independent opt-in model/settings/cohort/budgets/protected hashes.
- `scripts/download_ee_sr.py`, `scripts/audit_ee_sr_data.py`: pinned assets and corpus isolation audit.
- `scripts/run_ee_sr_trial.py`, `scripts/run_ee_native_control.py`: bounded real attempts and registered prompt control.
- `scripts/test_ee_sr.py`:15 CPU contract tests passed before real generation.
- `scripts/analyze_ee_sr_trial.py`, `scripts/export_ee_sr_evidence.py`: strict full/partial aggregation, causal/initial-state/cost/hash audit and compact export.
- `docs/ee_sr_trial/results/`: actual decisions, failed raw outputs, revisions, usage, state fingerprints, manifests/ledgers, numerical results,12-row per-task CSV (includes3 historical references) and raw-log SHA256 indexes. Large source trajectories/checkpoints stay ignored.

Commands executed from repository root (run scripts refuse existing output directories; use fresh run names for an intentional reproduction):

```bash
.venv-qwen35-clean/bin/python scripts/download_ee_sr.py
.venv-qwen35-clean/bin/python scripts/test_ee_sr.py
CUDA_VISIBLE_DEVICES=1 .venv-qwen35-clean/bin/python scripts/run_ee_sr_trial.py --run outputs/ee_sr_actor_trial_v1
CUDA_VISIBLE_DEVICES=1 .venv-qwen35-clean/bin/python scripts/run_ee_native_control.py --run outputs/ee_qwen35_native_control_v1
.venv-qwen35-clean/bin/python scripts/analyze_ee_sr_trial.py --output outputs/ee_sr_analysis_v1
.venv-qwen35-clean/bin/python scripts/export_ee_sr_evidence.py
```

Trial source commits: SR `7e2869e353cf0676135d5c840ed92a83841212c2`; matched native prompt control `7551bd7def5dbe82081f0a5b5d4070745f026077`. Final documentation/export commit follows these; runtime manifests preserve exact experimental source. Download/run/parser failure logs remain in ignored outputs. No original method, Paper-Aligned protocol or agentwmv2 change.

## Recommendation and stopping point

Keep this as an opt-in task-trained Actor candidate; do not silently promote it as the complete main model. It demonstrates partial ScienceWorld competence and loads correctly, but does not satisfy structured span signals, clean validation, reliable end-task success or Reviewer/Gate integration.

If the user selects this route next, first resolve the claim/data-isolation boundary and decision-format/internal-signal design, then independently verify a Reviewer and newly calibrate thresholds. Those are new scoped experiments, not completed work. No fine-tuning, new Gate rules, Memory changes, final-test tuning or Phase6 was performed. Stop for user review.
