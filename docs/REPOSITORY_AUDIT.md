# Phase 0 Repository and environment audit

Status: Phase 0 only. No training, model-weight download, GPU model launch or benchmark rollout was performed. Audit session: 2026-10-31 UTC (server `nvidia-smi` reported 2026-11-01; preserve this clock discrepancy in run metadata).

## Scope and provenance

Source: `/data1/shn/agentwmv2`, clean tracked worktree at the commit recorded in `audit/baseline_git.txt` (short ID b55bf32). Created `/data1/shn/agentwmv3` using `git clone --no-hardlinks`, retaining history but removing the local clone origin. v2 has no configured remote. Only tracked source was inherited: 87 files; no weights, external environments or experiment outputs copied. v2 has 208 GB checkpoints, 17 GB external dependencies and 65 GB outputs. This is intentionally not a runnable deployment copy yet. GitHub target supplied by user: https://github.com/ShiningIVYYY/agentwmv3.git .

The inherited README is historical v2 deployment documentation, not evidence of v3 runtime validation. `audit/source_files.txt` lists the baseline tree. No applicable AGENTS.md was found in v2 or its parent; the sibling AgentWM repository's instructions do not apply here.

## Actual reusable structure and execution

| Path | Observed role | Reuse / caveat |
| --- | --- | --- |
| `scripts/run_audited_paper_eval.py` | Main resumable audited runner: Agent, Alf, Science, Web | Preferred integration loop; do not replace with the older short runner |
| `scripts/run_paper_agent_eval.py` | Older compact evaluation implementation | Historical reference; avoid mixing protocols |
| `worldmodel/runtime.py` | make_controller, select, commit, metrics, audit_settings | Central integration seam; no pre-invocation generic Gate exists |
| `worldmodel/backends.py` | OracleWorldModel / Word2WorldModel predict and commit | Retain prediction and real-history semantics |
| `worldmodel/contexts.py` | Benchmark WM initialization, action/observation conversion, ScienceActionView | Preserve lossless duplicate ambiguity menus |
| `worldmodel/science_snapshot.py`, `HeapSnapshot.java` | ScienceWorld live JVM snapshot/rollback | Optional oracle control, not learned WM; build artifacts absent from new checkout |
| `reflection/controller.py` | ReflectionController.select / commit | Fixed 1–3 review rounds; final replacement prediction; JSON keep/replace/action/reason |
| `reflection/native.py`, `conservative.py`, `agentwm.py`, `science_evidence.py` | Alternative verification protocols | Must freeze ONE protocol across Gate comparisons |
| `experiments/{matrix,parallel,scheduler,task_queue}.py` | Scheduling, task locks, retries | Reuse when new cohort configuration is explicit |
| `scripts/test_*.py`, `validate_*.py` | CPU regression and real-environment validators | Distinguish mock unit tests from real inference |
| `configs/worldmodel/*.json` | Three WM configs | Preserve; relative weights absent in v3 |
| `docs/SCIENCEWORLD_*.md` | Prior context fixes / evidence ablations | Prior experiments are not new method results |

Flow: construct/reset environment -> bounded Agent messages -> Agent.generate -> parse draft action -> wm_runtime.select -> real environment step -> wm_runtime.commit -> trajectory/metrics. ScienceWorld takes the whole tail of the last `Action:` and requires `Thought:` for format validity; ALFWorld/WebShop reuse external verl-agent projection functions and native action tags. Current draft is free-form reasoning/action; there are no Premise/Expected Consequence fields and no SpanUQ pipeline. Proposed structured draft must retain native action grammar.

Agent API is OpenAI-compatible HTTP. `generate` currently fixes temperature=.4, top_p=1, seed=0, 4096 prompt-token limit; output limit is CLI-configured. `--model`, `--url`, `--tokenizer-path`, `--tokenizer-url`, `--disable-thinking` already exist. disable-thinking passes `chat_template_kwargs.enable_thinking=False`, but also sends a JSON-string `generation_config` intended for Transformers serve. Qwen3/vLLM payload compatibility and sampling parameters must be checked separately, not assumed. Token counting is a separate HTTP service (`scripts/paper_eval_tokenizers.py`). Qwen3 model registration in launchers/matrices is still needed.

## WM and selective invocation

ScienceWorld learned WM: `X1AOX1A/WorldModel-Sciworld-Qwen2.5-7B`, revision `6196b3531c5634a8229af6b2cd111e02a70c13b5`. See copied provenance JSON for published-file hashes (weights themselves were not rehashed). Config: temperature=0, top_p=1, max_tokens=512, context=32768, port=8112. Other configs cover Alfworld/Webshop. Official initialization source revision is e8dc240269fb222de3c242b09c70bfbf0b4ac1c9. Predictions branch from committed real history; only actual observations advance WM context. Actor and WM communicate through action/observation strings, so Qwen3 Actor does not require replacing this Qwen2.5 WM. Actual Qwen3 behavior remains unverified.

Never is currently backend=none; Always is a configured controller invoked at every parsed step. Searches of tracked reflection/worldmodel/experiments/run scripts did not find a reusable pre-WM Risk/Uncertainty/Random/Periodic Gate. Evidence-gated interventions occur AFTER prediction: they are action-revision filters, not invocation gates. Prior frameworks elsewhere on the server are outside this baseline.

Important integration hazard: runtime.commit assumes every controller-enabled step has `metadata['reflection']` and a final prediction. Oracle commit additionally requires a predicted action. Simply returning early on Skip will fail or silently omit history. A future gated runtime must distinguish observed real transitions from forecast verification; on every Skip it must update WM real history without predicting, and preserve final prediction verification on Call. Oracle Skip requires an explicitly tested execute/observe path. Do not fake a forecast or count an observation update as a WM call.

## Benchmarks, splits, reset and logs

All three benchmark wrappers exist, but v3's ignored runtime assets are absent. ScienceWorld currently uses AgentGym HTTP at 36001 for none/learned WM, a dedicated SDK JVM for oracle, ScienceWorld 1.2.3, 200 official ordered AgentEval test IDs, and historical 100-step cohorts. Current checkout does NOT establish RLVMR / L0-L1-L2 / maximum 30 steps. The short verified one-shot is training ID sciworld_4509, expert row 1914; its source revision is recorded in inherited docs. Do not silently replace the current protocol with the proposed 30-step protocol. Resolve the desired new cohort before Phase 5.

Science tasks() verifies goal hash, native task/variation mapping, initial score, test and demo hashes, and JAR hash. ALFWorld owns separate native engines for formal execution and replay. WebShop checks index/catalog identity and isolates terminal-reset behavior. Oracle replay compares observation/reward/done/success/score; Science snapshots support rollback of stochastic state and ambiguity mappings. These are useful mechanisms, not revalidated here.

Prior cohorts commonly use ALFWorld 140 tasks/50 steps/1024 output tokens, ScienceWorld 200/100/1024, WebShop 500/15/512. These are historical audited settings, not universal defaults. Exact selected cohort must come from manifests/configuration, never inferred from README alone. `data/` is generally ignored, except one already tracked verified demo. Many essential manifests, original demo and task lists are therefore absent from v3. Existing outputs organize per-run config/audit, tasks JSON, traces, progress.json, reflection_events.jsonl and summaries. Scheduler errors, retries and partial results are distinguished. No full cost collector for hidden-state passes, retrieval/judge/correction exists yet.

Dependencies are split across conda environments and setup/download scripts; there is no root lockfile/pyproject/requirements pinning the full application. External AgentGym commit 3ef9235d23e68e7c2920c5422ad957dc8ced5c6c has LOCAL MODIFICATIONS to ScienceWorld environment.py and server.py; external verl-agent commit 20bd331bdbc9026a5668e11362178e10ab7400c8 is clean. See `audit/external_sources.json` for dirty-file hashes. Git commit alone cannot reproduce that environment. No symlinks to writable v2 runtime trees were created. Later reconstruction must copy/pin reviewed patches into v3, not edit shared v2 installations.

## Hardware and software evidence

8 x RTX 4090, 24564 MiB each. Initial GPU 0: 18141 MiB allocated by another user's llm_train service; GPUs 1–7: 3 MiB each. Occupancy is a snapshot, not a reservation. Driver 570.172.08 advertises CUDA 12.8; system nvcc is CUDA 10.1 (these are different facts). /data1: 1.3 TB free; root/home/tmp volume: only 39 GB free. Put future model caches and environments on the data volume, with configurable paths. Host has many old service/JVM processes; process existence does not establish service health. No service was stopped or launched.

| Environment | Python | torch | transformers | vLLM | Notes |
| --- | --- | --- | --- | --- | --- |
| vllm | 3.10.19 | 2.5.1 | 4.57.3 | 0.6.6.post1 | Old vLLM; do not assume Qwen3 support |
| verl-agent | 3.12.0 | 2.8.0 | 4.57.3 | 0.11.0 | Used only for CPU tests/static checks |
| vllm-cu128 | 3.12.13 | 2.11.0+cu128 | 5.13.0 | 0.24.0 | Installed metadata, not compatibility-tested |
| vllm-qwen3 | 3.12.13 | absent | absent | absent | Name does not imply a working environment |
| ee-qwen38-direct | 3.12.0 | 2.8.0 | 5.17.0 | 0.11.0 | Mixed versions, not a validated template |
| ee-scienceworld | 3.8.20 | absent | absent | absent | scienceworld=1.2.3; below SpanUQ Python minimum |

HF cache inspection found Qwen2.5-7B, Qwen3-8B, Qwen3.5-9B, Qwen3.8-27B and EE checkpoints. No Qwen3-4B or SpanUQ in the inspected default caches; this is not an exhaustive all-disk search. Existing WM checkpoints are under v2/checkpoints/word2world. A 4B half-precision backbone needs roughly 8 GB weight storage in VRAM plus activations/logits/probe; one free 24 GB GPU looks plausible for a short sequential inference test, not proven. Prefer separate generation and HF extraction processes/devices if necessary; no memory allocation claim is made.

## Data leakage and fairness risks

- Existing test trajectories and past test-informed context/policy repairs must not become development memory, thresholds or judge examples. Historical policies are inherited choices, not a fresh test-independent design.
- Explicitly version train/dev/test task, variation, episode and source IDs; hash overlap at task/variation/goal level. Future feedback may only enter memory after its environment step, and cross-episode online memory is a separate predeclared experiment.
- Oracle hidden initialization/reward/success and future predictions must never enter Internal/External/Gate input. Some inherited reflection protocols expose privileged oracle metadata; do not compare them against observation-only learned WM as equivalent.
- Cross-benchmark sharing is opt-in; label evidence provenance and distinguish hypothesis from observed facts. Similarity, action changes and published SpanUQ AUROC are not utility evidence.
- Freeze Actor, prompts, sampling, WM revision/depth/review protocol, task manifests, seeds, memory/correction settings. Actor-backbone changes define a new cohort. Count all online calls/forward passes and offline memory synthesis separately.

## Validation executed

Commands run in the v3 checkout: `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' /home/shn/miniconda3/envs/verl-agent/bin/python scripts/TEST.py` for the eight scripts listed in `audit/baseline_tests.json`. 6 scripts passed: wm_reflection (33), agentwm_reflection (28), native_reflection (11), conservative_reflection (21), wm_experiment_matrix (33), scienceworld_sr_context (3): 129 unit tests. Scheduler mock error events in test output are intentional fixtures, not real benchmark failures.

Initial science_evidence invocation failed to resolve the root package. Rerun `PYTHONNOUSERSITE=1 PYTHONPATH=. /home/shn/miniconda3/envs/verl-agent/bin/python scripts/test_science_evidence.py` passed 18 tests. Protocol suite ran 25 tests and had 10 errors because ignored external projection/prompt files were not cloned; this is an unresolved deployment dependency, not a reason to alter tests. Total passing suites: 7, 147 test cases; do not describe the entire repository as green. `audit/baseline_tests.json` preserves initial failures. No model load, hidden-state numeric test, live-service health check or benchmark success rate was run. Unit mocks are not research results.
