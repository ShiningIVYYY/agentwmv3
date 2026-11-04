# Phase 5 report — end-to-end ScienceWorld integration

Phase5 development engineering validation completed; Phase6 is not started. **Engineering correctness, model capability and scientific effectiveness are distinct.** The existing system executes real complete episodes with real learned-WM Call/Skip. Reflection remains weak, experience coverage is narrow, and these data do not establish either research hypothesis.

## A. Completed work and configuration

Actor: frozen `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`. BF16 HF SDPA, torch2.8.0+cu128 / Transformers4.57.3, nonthinking, T0.4/top_p1/top_k0/repetition_penalty1, max input4096/response512, history2. Actor GPU0 and WM GPU1 on RTX4090; old environments remain unchanged.

Independent WM: `X1AOX1A/WorldModel-Sciworld-Qwen2.5-7B`, revision `6196b3531c5634a8229af6b2cd111e02a70c13b5`, existing weights read-only. vLLM0.11.0 BF16, greedy T0/top_p1/max512, context32768, max-seqs1, eager. Existing helper source commit `e8dc240269fb222de3c242b09c70bfbf0b4ac1c9`; existing JSON reflection up to3 rounds, final replacement forecast, max4 one-step forecasts of the same real prefix per invocation. Simulation alternatives do not increment real interaction steps. WM-only initialization includes existing all-room context; it never enters Actor Draft/Internal/External/Gate input. Review receives predicted observations, not the private world initialization.

Internal backend `hf_generation_raw_bf16_v1` and base Actor configuration SHA256 `980d1f3f3f07e0df1061a842e80d42926e573e1d1810a2663c13fbefe001504b` are preserved. The frozen BF16 model supplies raw pre-temperature generation logits; NLL reduction uses FP32: Local=max(mean Premise-value NLL, mean Expected-Consequence-value NLL); Global=mean all textual response tokens incl field labels/Action, excl prompt/EOS/padding. No extra forward. Opt-in profile adds verified official per-turn RNG and exact suffix input truncation; the profile hash binds new thresholds. Fifteen old real Phase2 records replay identically. Legacy behavior stays default-off.

Pipeline: observation → structured Draft → requested Internal/External signals → rule Gate → either learned WM prediction/unchanged reflection or genuine Skip → exact final Action → real SDK feedback → commit. Reflection/WM tokenizers are independent of Actor tokenizer; plain-text transport is compatible. Score/accounting and source-bound real replay are verified.

## B. Files and purpose

| Files | Purpose |
| --- | --- |
| evaluation/profile.py, seeds.py, observer.py, metrics.py | Explicit paper numeric/presentation profile, official seed derivation, actual WM request observation, raw instance metrics and full costs |
| configs/evaluation/paper_aligned_v1/scienceworld.json | Independent opt-in setting; legacy configs retained |
| configs/gates/dev_distribution_phase5_v1.json | Frozen dev-only quantiles; separate invocation/forecast budgets |
| scripts/serve_scienceworld_profile.py | Owned H20/SDK21 thin environment service; reuse installed wrapper, do not edit v2 |
| scripts/prepare_phase5_plan.py, calibrate_phase5.py | Goal-disjoint frozen development cohort; distribution-only calibration |
| scripts/run_scienceworld_validation.py | Bounded runner using existing Science.run, source archiving, role RNG, native receipts and per-episode resume |
| scripts/audit_wm_failures.py, audit_scienceworld_profile.py, audit_experience_coverage.py | Original failures, pinned200-task metadata and train evidence audit |
| scripts/prepare_phase5_memory.py; experience/memory.py, synthesis.py; gates/evidence.py | Third narrow same-room observation card from existing approved TRAIN cases; preserve old card behavior and reliability requirements |
| internal_signals/hf_actor.py, hidden_states.py, provider.py; gates/generation.py | Opt-in seeded generation/exact left truncation with verified response alignment; unchanged raw NLL formula and old config |
| scripts/run_audited_paper_eval.py | Small profile hooks for fixed last2 history/native menu IDs, preserving original loop and other profiles |
| scripts/test_scienceworld_validation.py, audit_phase5.py; analysis/phase5_audit.py |37 CPU integration contracts and source/causal/metric/cost real replay; read-only scientific diagnostics |
| scripts/report_phase5.py, docs/phase5/*, WM_FAILURE_AUDIT.md, SMALL_SCALE_EVAL.md | Compact true results, reproducible reports and provenance; full large logs remain ignored |
| README, IMPLEMENTATION_PLAN.md, PAPER_ALIGNED_EVAL.md, PROGRESS.md, PHASE5_WORKLOG.md | Current state, intentional paper differences, risks and next-phase boundary |

## C. WM failure investigation / actual fix

See [WM_FAILURE_AUDIT.md](WM_FAILURE_AUDIT.md). Phase4 `focus on glass jar`: jar absent from bathroom; reviewer ignored negative forecast and selected the absent goal object. `focus on substance called air`: descriptive phrase does not resolve as the canonical command; fresh probe `focus on air` works. Both final failures were predicted correctly; valid JSON and last-Action parsing preserved. The final-round execution policy is an existing capability/design limitation, not a transport/parser defect. Seven independently reset real development probes retained; no answer-specific action patch or evaluation-driven Gate change.

Actual deterministic engineering bug: new observer keyword binding initially failed on requests `url=`. It was fixed with regression coverage before the successful two-step canary. Initial failure and executed source preserved. No model inference or environment action from that failed canary is counted as research data. In the complete cohort all278 reviews are valid JSON;34 same-action replace violations and50 unsupported replacements are model/protocol failures, not corrupted transport. See [reviewer_quality.json](phase5/reviewer_quality.json).

## D. Paper profile / data / thresholds / memory

Exact ordered200 test task IDs/template/variation mapping and prior native goal hashes checked against pinned official source/current SDK catalog; no test rollout. SDK1.2.3/JAR hash pinned, H20 validated. Package identity does not prove identical historical authors' binary. Structured Qwen3-4B HF Actor, native Action parser and WM addition are intentional deviations from paper Qwen3.5/action-only/backend. [Protocol](PAPER_ALIGNED_EVAL.md) preserves earlier records and details these differences. No RL loss, training invalid penalty or reward normalization enters evaluation.

Train1292 / dev197 / test200 instances, with conservative normalized-goal and instance isolation, frozen source manifests.30 dev goal groups for calibration;5 **different** dev goal groups for full validation. Chemistry/lifespan each have only one available dev goal group, allocated to calibration; complete-episode validation covers Measure, Conductivity and Find only. Template families can be shared across partitions; goal/instance overlap is excluded, not a claimed template-held-out split.

273 real dev decisions,244 unique prompt-response states,0 missing; repeated steps remain real workload but not independent evidence. Predeclared q0.75 linear quantiles: Local **0.12081397919163489**, Global **0.053253251821409955**, strict>. Expected Internal requests on Actor-only calibration states85/273=31.1355% before budgets/external feedback. Sensitivity and all points in [calibration_distribution.json](phase5/calibration_distribution.json); leave-one-episode-out ranges Local0.119247–0.121801 / Global0.052727–0.055246. No outlier removal, task-outcome selection or test tuning. This is distribution calibration, **not proof of multi-seed stability, useful thresholds or optimal WM budget**.

Memory v2 freezes3 cards, same54 verified TRAIN cases; old2-card snapshot preserved. Only added same-room `look around` fact with2 independent episodes/instances/goal groups and no counterexamples. Train inventory: navigation16/object interaction16/preconditions20/failure29 cases, but broad rules/corrections/subgoal progress lack enough independent supporting evidence. WM utility remains Unknown. Sufficiency is not inferred from BM25, and no reliability criterion was relaxed. Natural sufficient confirmations: live Full 0, retrospective new-snapshot dev 0. Correction and confirmation Skip remain off. Frozen snapshot shared by all policies; unused modules cost0 online.

## E. Tests, actual results and costs

| Policy | Mean final Task Score | Success (score>=100) | Env steps | Gate calls | WM forecasts | Total model calls | Total tokens | Online seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| never | -9.00 | 0% | 70 | 0 | 0 | 70 | 27242 | 93.797 |
| always | -29.00 | 0% | 65 | 65 | 190 | 423 | 900153 | 543.016 |
| internal | -29.00 | 0% | 72 | 21 | 63 | 190 | 351690 | 285.148 |
| full | -29.00 | 0% | 73 | 21 | 62 | 190 | 349342 | 290.145 |

Actual counts: 280 real decisions; 107 Call/173 Skip; 315 WM forecasts. Budget-blocked 31; missing Internal 0; Draft format failures 0; reviewer parse/action validation errors 84. Invalid native actions are retained rather than repaired. All5 initial-state/first-Draft parity checks pass. Native termination, horizon, actual server requests, independent evidence, NLL recomputation, role seeds, budget settlement, exact prompt history and costs pass replay.

37 new CPU contracts +217 prior regression tests =254 passed. Unit fixtures are not research results. Real complete20-episode cohort and2-step canary passed; prior7 single-action probes andH20 horizon probe retained. Old Phase4 byte-identity source audit is historical; optional Phase5 RNG/truncation hooks are explicitly versioned and validated by golden-score and real replay rather than misreported as byte-identical source.

All online runs: **1,628,427 actual input/output tokens, 873 model calls, 1212.106s episode wall time**, Actor peak allocated8.418GiB plus dedicated WM reservation. Full per-policy costs, service/setup/calibration/offline separation and scientific audits: [SMALL_SCALE_EVAL.md](SMALL_SCALE_EVAL.md). Judge/router learned-model calls0; raw NLL uses no extra forward. No reliable dollar conversion.

Core actual commands (from repo root):

```bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python scripts/test_scienceworld_validation.py
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. .venv-spanuq/bin/python -u scripts/run_scienceworld_validation.py --mode episodes --plan outputs/phase5_setup/development_plan.json --model-dir checkpoints/qwen3-4b --wm-checkpoint /data1/shn/agentwmv2/checkpoints/word2world/scienceworld --science-python /home/shn/miniconda3/envs/ee-scienceworld/bin/python --threshold-config configs/gates/dev_distribution_phase5_v1.json --snapshot outputs/phase5_setup/memory_v2.json --output outputs/phase5_episodes_v1
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-spanuq/bin/python scripts/audit_phase5.py
PYTHONPATH=. .venv-spanuq/bin/python scripts/report_phase5.py
```

These are actual server provenance paths, not portable installation defaults. The new profile/SDK service and separate WM must be running. Commands to reconstruct service/model/development inputs are recorded in docs/phase5 and PHASE5_WORKLOG; no paths imply permission to modify v2. Use a **new output Run ID** for a rerun. `--resume` only skips completed episodes with identical configuration/source signatures and permits one explicit infrastructure retry; it never merges partial steps or replaces unfavorable valid outcomes.

## F. Limits / Phase6 recommendation

Engineering preparation for Phase6 is available: fixed IDs/profile, complete episodes, true Call/Skip, strict accounting and frozen train-only memory/dev-only thresholds. This is **not** scientific acceptance of Gate benefit or a reason to launch200x8 automatically. Reviewer can ignore accurate negative forecasts; two NLL features are strongly correlated; many low-NLL actions fail; memory coverage remains narrow and sufficiency lacks a natural positive. Five tasks/one seed/three families cannot rank methods reliably. If changing the reviewer/final-veto/memory, freeze a new shared protocol and repeat appropriate development validation before formal comparisons; do not choose it on final-test results.

Phase6 should first agree on critic-policy stability and shared memory version, predeclare matched-budget grids and correction groups, then obtain explicit user authorization for resources/sweeps. Across8 rollout seeds, aggregate fixed task instances within each seed and report mean±sample SD across seed means. Actor remains Qwen3-4B. Cross-benchmark/fresh-server portability work remains later. No new training, final200x8 evaluation or Phase6 work was performed. **Stop here and await user confirmation.**

## G. Version control and stop

Baseline commit: `7a039fa78688e1caf36451dd2d3e599c215d6dfe`. Final Phase5 commit and verified push are reported in the delivery message. Only agentwmv3 source/docs/compact evidence are committed; model weights, environments, private WM initialization and complete large trajectories remain excluded. The original paper document is preserved verbatim as a prefix plus the explicit Phase5 addendum. Phase5 is complete; no Phase6 work is authorized by this report.
