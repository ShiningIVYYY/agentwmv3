# Paper-Aligned Evaluation Protocol (planning only)

Status: configuration/source audit completed; protocol implementation and experiments NOT started. This follow-up authorizes documentation only. No Actor switch, inference package install, training, environment start, existing configuration override or SpanUQ edit is included. Actor remains **Qwen/Qwen3-4B**, with the compatible frozen SpanUQ checkpoint. The paper uses Qwen3.5 models; our results must be labelled **paper-aligned evaluation settings, Qwen3-4B**, not a same-backbone reproduction.

## 1. Primary sources and evidence boundaries

Paper: [arXiv:2610.19789v1](https://arxiv.org/abs/2610.19789v1), submitted 14 October 2026. Read Section 4 (PDF page 4), Appendix C.1 (pages 14–15), Table S2 (page 16), Table S3 and Appendix C.4 (page 17). PDF SHA256: `2038e0601b8cab431720f40b3542991236e8a47f25d5a213d6c3716c1e445b80`.

Official source pinned to [35a44f5c5647b1785ad23ddf0ffbd196b695bf86](https://github.com/aim-uofa/Self-Distillation-AgentRL/tree/35a44f5c5647b1785ad23ddf0ffbd196b695bf86). All paths below are relative to that commit unless explicitly marked local. This is the public release inspected now, not proof that every byte exactly matches the paper's original execution environment.

Source index:

- `launchers/test_pretrained_{scienceworld,alfworld,webshop}.sh`, `test_pretrained_1gpu.sh`, `run_skill_method_1gpu.sh`: effective evaluation sizes, horizons, prompt/response limits, sampling overrides and PAPER_CONFIG behavior.
- `verl/trainer/config/ppo_trainer.yaml`: history_length=2; val top_p=1.0, top_k=-1; WebShop fixed_eval_tasks=true/fixed_eval_seed=1000/use_small=true/human_goals=false. Generic val defaults are greedy, but benchmark launchers override to sampling T=.4; do not read the YAML in isolation.
- `agent_system/scienceworld_protocol.py`, `launchers/prepare_scienceworld.py`: pinned IDs, catalog/version, overlap exclusion and manifests.
- `agent_system/environments/env_package/scienceworld/{envs,manager}.py`: initial look, step limit, raw score, observation normalization, action grammar and history prompt.
- `agent_system/environments/env_package/alfworld/envs.py` and bundled `alfworld/agents/environment/alfred_tw_env.py`: strict fixed valid_seen assignment and game filtering.
- `agent_system/environments/env_package/webshop/envs.py`, bundled `webshop/web_agent_site/engine/goal.py`: fixed goals, native reward vs binary success.
- `agent_system/evaluation/{seeding,rollout_writer,scienceworld}.py`, `agent_system/multi_turn_rollout/rollout_loop.py`: task/turn seeding, serialized identity and final-score extraction.
- `verl/trainer/ppo/ray_trainer.py`, `launchers/compute_pass_at_k.py`, `eval_pass_at_k.sh`: retain final batches, aggregate native instance metrics, eight seed sweeps and sample SD.

Local baseline inspected: agentwmv3 HEAD `672c21b4bd3b09c8e492381172b3e20186454952`, with clean worktree before edits. v3 lacks ignored runtime data/external trees; existing deployment manifests/data under v2 were read only. No model inference or task reset was performed. Official source/PDF and small pinned ID JSONs were fetched into a separate temporary audit directory, not into runtime assets.

## 2. Verified target matrix vs current implementation

| Item | Paper and official evaluation target | Local state | Required later work |
| --- | --- | --- | --- |
| ScienceWorld tasks | 200 fixed AgentGym IDs | Exact file hash, order and existing task/variation manifest match verified | Materialize independent v3 manifest and revalidate installed SDK catalog/goal/JAR identity |
| ScienceWorld turns/history/prompt | 20 / 2 / 4096 tokens | Historical 100 turns; one-shot + token-bounded full history; limit 4096 | New paper profile, recent-2 prompt builder and explicit 20 actor-action horizon |
| ALFWorld tasks | All 140 solvable valid_seen games, exactly once | Existing runner sorts 140 eval_in_distribution game files | Verify normalized game IDs and asset hashes; retain fixed assignment, no resampling |
| ALFWorld turns/history/prompt | 50 / 2 / 2048 | 50 and up-to-2 history available; global prompt bound is 4096 with 2→1→0 fallback | Per-benchmark bound 2048, explicit overflow policy |
| WebShop tasks | 128 fixed goals from test pool 0..499 | Existing manifest evaluates all 500 | Use exact fixed permutation and goal fingerprints, not first 128 |
| WebShop turns/history/prompt | 15 / 2 / 4096 | 15 / up-to-2 / 4096; history may be dropped on size | Separate fixed-goal profile; record fallback/truncation explicitly |
| Draft Actor sampling | do_sample=true, T=.4, top_p=1, top_k disabled | T=.4/top_p=1 but seed hardcoded 0; disable-thinking generation_config has backend-specific behavior | Resolve effective payload precedence; disable inherited model generation defaults explicitly |
| Response cap | 512 tokens per interaction turn | CLI configurable; old ALF/Science cohorts use 1024 | New profile response=512; never inherit old 1024 |
| Repeated evaluation | Rollout seeds 0–7; same tasks each sweep | Single seed=0 request payload | Task/turn-derived request seeds, separate task selection seed, eight complete sweeps |
| Thinking/dtype | Thinking chat formatting disabled; final inference BF16 | Flag exists; Qwen3-4B real integration pending | Keep Qwen3-4B, validate enable_thinking=false and BF16 if supported |
| Main metrics | ALF SR; Science raw Task Score; Web SR + graded score | Science historically emphasizes SR; Web stores reward and score*100 | Add explicit names/units and instance-level aggregation |

The paper's batch sizes (ALF 140, Science 32, Web 128) are execution settings, not metric denominators. Science requires 6 full batches of 32 plus 8 final tasks, or equivalent complete serial execution. No task dropping or average-of-batch-means is allowed.

## 3. Task identity and environment audit

### ScienceWorld: identity match confirmed, full behavioral equivalence pending

Official source dataset: `AgentGym/AgentGym-RL-Data-ID`, revision `99d0b7923bf126d9c6cdea5f362d2d41ccb5d493`.

- `eval/sciworld_test.json`: 200 distinct IDs, SHA256 `78fbe5ed7ba545f1a2ee08397a5e740f622872b6a3ee70315d349c4287675c79`.
- `train/sciworld_train.json`: 2120 IDs, SHA256 `9b7ce7f214c738de96d0048919e08e8241c5c2bca48843c53dae5098b938e28d`.
- Read-only local comparison: v2/data/paper_eval/sciworld_test.json has the identical test-file hash; all ordered IDs equal; audited/scienceworld.json has the same 200 ordered IDs and zero task_name/variation_id mismatches against official resolve_item.
- Categories and exact weights: Measure 53, Test-Cond. 83, Find 44, Chem-Mix 5, Lifespan 15. Overall is weighted by these counts, not one fifth per category.
- Source train/test intersection is exactly 61 IDs; 2059 disjoint train instances remain. Filtering must always reference ALL 200 test IDs, including when a smoke run selects fewer test tasks.

Official mapping pins ScienceWorld 1.2.3 and enumerates the unsorted task catalog in AgentGym order, omitting families 5-1/5-2/9-1/9-2/9-3/10-1/10-2. The official comment references AgentGym commit `640f8bca6901a6a6d540ff61522b813988da47c4`. Local external AgentGym is a different commit with local changes (Phase 0 audit). Do not identify a task solely by its integer suffix without the catalog. Both inspected local package metadata and official constraints specify ScienceWorld 1.2.3, but no official JAR checksum was supplied in the inspected protocol; exact JAR and live reset equivalence remain unverified. Later pin local JAR, SDK/source hashes, task goal, variation, simplification string and initial score/observation per manifest.

Official worker uses simplificationStr='', initializes ScienceWorldEnv(envStepLimit=H+1), executes one initial `look around`, resets agent step counter to zero and allows H further actor actions. Terminal state or H actions ends the episode. Invalid commands still consume a turn. Local HTTP service/default SDK limits must be checked for early termination; changing only max_steps in the outer loop is insufficient.

Official presentation sorts only contiguous flat single-tab `a/an ...` object rows, preserving numbered ambiguity choices and nested lists. Local ScienceActionView additionally sorts unique ambiguity labels and retains duplicate-label native menus. These are different prompt presentations even when simulator task identity matches. The paper profile should reproduce the official observable representation and native choice mapping where possible, with shared adapters for WM Call/Skip; if retaining the existing lossless presentation is necessary, record a named deviation and do not assert prompt equivalence. Never rewrite native action numbers without a reversible mapping.

Official Science prompt is action-only `<action>...</action>` with two previous observation/action pairs and no one-shot. Projection also accepts a single `Action:` line. Local parser expects the last `Action:` tail and checks Thought:, and the prelude injects a training dialogue. A paper profile therefore needs an opt-in prompt/projection builder; no silent reuse of the old one-shot/full-history interface.

### ALFWorld

Official PAPER_CONFIG=1 selects 140 and ALFWORLD_STRICT_EVAL=1: sorted solvable valid_seen games, one fixed game per worker, group_n=1. Without this mode, a 128-game default or repeated random resets is possible. The bundled collector excludes missing/unsolvable games, movable/Sliced tasks and unsupported task types. Local runner likewise has a sorted 140-game manifest, but numeric IDs alfworld_000 etc. alone do not prove game equality. Future audit must compare paths normalized after `json_2.1.1/`, game.tw-pddl hashes, task type and goal text against the official filtering applied to the same asset snapshot.

Official make_envs initializes validation at task_seed+1000 and worker i at task_seed+1000+i. Local ALF seeds each task with 1000; a serial implementation must preserve the official stable manifest-position seed, not allocate a new seed by scheduler order. Official and inspected local metadata agree on alfworld=0.4.2, textworld=1.7.0. The official downloader is not pinned to dataset file hashes, so release/package version equality does not establish asset equality. Preserve task/logic assets and native engine fingerprints. No environment launch or official 140-game asset regeneration occurred in this audit.

### WebShop

Effective official defaults: small 1000-product catalog, synthetic goals (human_goals=false), fixed_eval_tasks=true, fixed_eval_seed=1000. Test pool is integer indices 0..499; 128 are chosen using `np.random.RandomState(1000).permutation(np.arange(500))[:128]`. Selection seed 1000 is distinct from task seed 0 and rollout seeds 0–7. Ordered indices are recorded in Appendix A; goal text/content hashes still need verification against pinned data, generation code and seed. Selecting goals 0..127 or changing the selection for each rollout seed is incorrect.

Critical identity detail: make_envs seeds validation at task_seed+1000, then worker position i uses task_seed+1000+i. WebAgentTextEnv seeds its RNGs; SimServer generates prices/goals and shuffles goals with that worker seed. Thus the selected index at slot i refers to the goal catalog generated with seed 1000+i when task_seed=0. The 128-task identity is the ordered tuple (slot, environment_seed, selected_index, goal_content_hash), not selected_index alone. Local WebShop uses one seed-1000 catalog for all 500 tasks. Simply taking the 128 published indices from that local catalog will generally NOT reproduce official goal identities. Future serial/multibatch scheduling must preserve original slot seeds, or freeze the exact per-slot generated goals and equivalent simulator state before changing execution layout. The official request task_key still uses webshop:test:N; retain richer content identity in manifests.

Existing local WebShop evaluates all 500 and uses a 1000-item catalog with an audited index. The inspected local and official `engine/goal.py` and `utils.py` are byte-identical, a useful scoring/source check, but not proof of the same generated goals or search results. Official prepare_webshop.sh downloads items_shuffle_1000.json, items_ins_v2_1000.json and items_human_ins.json via a mutable HF main revision/fallback, then builds a Lucene index. Future profile must freeze file hashes/HF revisions, ordered goals and price generation, index ASIN contents, pyserini/Java/normalizer versions. Batch allocation must not cause selection to restart and duplicate the first goals. Official small mode passes num_products=None with the small file paths, so init_search_engine selects search_engine/indexes; the setup converter builds this from its default small product file. Local runner passes num_products=1000 and selects indexes_1k. These directory names do not by themselves establish a content mismatch or match: validate indexed ASIN sets/content and actual source files before claiming equivalence.

## 4. Scoring and termination contract

| Benchmark | Per-instance evaluation value | Completion/success | Do not substitute |
| --- | --- | --- | --- |
| ALFWorld | Binary info.won from the final active transition | Won=true is success; env done or 50 agent actions terminates | Native score, summed step rewards or shaped training return |
| ScienceWorld | Raw simulator info.score at last active step or horizon, including negative values | Full-success diagnostic is score>=100; score>0 is a distinct positive-score diagnostic. Stop at simulator completion or 20 agent actions | Success rate, maximum score over history, sum of cumulative scores, clipped score, score/100 training reward |
| WebShop | Native graded reward r in [0,1] at final active step; separately binary success | Official worker sets won=(done and native_reward==1.0); terminate at done or 15 agent actions | 10/0 worker training reward, partial score as binary success, post-terminal automatic reset output |

ScienceWorld local success uses done and score==100; official diagnostics use score>=100. Keep native simulator_done and horizon_truncated separate, then compute official success diagnostic from raw score; never replace the main Task Score by this diagnostic. Preserve negative failures, which are visible in the paper. If a service clips them, it is incompatible until corrected in the independent profile.

WebShop get_reward computes `(attribute_matches + option_matches + price_match) / (number_of_goal_attributes + number_of_goal_options + 1)`, multiplied by the existing type-match factor. Category/query/title matching, option normalization and price generation must remain the official implementation; do not approximate this formula with custom string similarity. The worker captures native r into info.task_score before converting its training reward to 10 or 0; the manager reads info.task_score from the last active step. Our raw graded metric stays 0..1 and may additionally be displayed as 100*r with an explicit label. Current v3 stores WebShop score=100*r, so a second multiplication must be prevented.

Training invalid-action penalty=.1 and reward transforms in C.1/Table S3 belong to optimization, not evaluation scores. Invalid actions still receive actual feedback, consume turns, and have validity logged, but no synthetic penalty is subtracted from raw task metrics. No GRPO/PPO/teacher loss/reward normalization or optimization runner will be adopted for this training-free project.

## 5. Decoding, history and request seeds

New profile contract: Qwen/Qwen3-4B; enable_thinking=false; BF16 where validated; do_sample=true; temperature=.4; top_p=1.0; vLLM top_k=-1 (HF equivalent top_k=0); max_new_tokens=512; one candidate per draft generation. Explicit payload/config must override model generation_config defaults. Existing disable-thinking branch sends a generation_config string; inspect effective service behavior before treating its nested/top-level parameter precedence as correct. No extra top-k cutoff is permitted for Actor evaluation.

History length 2 means two previous committed observation/action pairs plus the current observation/goal, not two total chat messages and not the full episode. Keep all history separately for audit and WM state; logging a full trace does not mean it enters the Actor prompt. The official Science input is one user message after flattening recent history; ALF/Web use their released prompt templates. WebShop additionally falls back to a no-history template when the formatted text exceeds 13000 characters. In rollout preprocessing Science truncation=left; ALF/Web truncation=error. The local runner instead drops entire history pairs to fit the token budget. These are material behavior differences: implement named profile-specific policies and log the exact effective input token IDs, truncation/fallback, action parser and prompts. Do not silently change the old default. Any goal lost through exact paper left-truncation must be visible in the trace, not repaired invisibly.

The research structured draft (premise/consequence/action), external evidence and WM revision are intentional method additions. A pure paper-prompt Never baseline and a unified structured-draft Never baseline should be separate labelled arms if both are used. All Gate comparisons must share the structured format, memory/correction setting and verification mechanism; differences against the published paper may not be attributed solely to Gate quality. Paper history=2 applies to the base Actor; additional Internal/External/WM context limits must be explicit and identical across comparable arms.

Eight rollout seeds do not mean eight environments per task in one pass. Keep task_seed=0 fixed and run the identical full manifest once for each rollout_seed in [0,1,2,3,4,5,6,7]. Official eval_pass_at_k defaults to per_task_turn, derivation version 1. It hashes the compact UTF-8 JSON array:

```text
["eval_request_seed_v1", benchmark, task_seed, rollout_seed, task_key, turn_index]
request_seed = int.from_bytes(SHA256(payload)[:8], "big") & ((1 << 63) - 1)
```

turn_index is zero-based. task_key: Science `sciworld_N`; ALF `alfworld:` plus normalized path after json_2.1.1/; Web `webshop:test:N`. This makes draft sampling independent of worker/batch scheduling. Preserve this exact namespace for paper-aligned draft requests. Additional review/replan/judge calls need separately versioned role/subcall namespaces so extra WM calls do not shift the next draft's seed; record them as a project extension. Random Gate RNG is separate from Actor sampling/task selection. Same seed does not imply identical trajectories across Gates or across GPU/backend versions.

WM backbone, temperature, output limit and review depth remain the existing fixed WM protocol; do not import Actor sampling into the WM or increase simulation depth. A WM review does not consume an extra interaction turn; only committed environment actions count toward H. Every online model call and SpanUQ extraction still counts toward total inference cost. Exact bitwise matching to paper H800/Qwen3.5 execution is not claimed on 4090/Qwen3-4B.

## 6. Instance-weighted statistics

For a benchmark with N fixed instances, compute each complete rollout seed s first:

```text
m_s = sum(x_s,i for i in fixed_manifest) / N
mean = sum(m_s for s in 0..7) / 8
sample_SD = sqrt(sum((m_s - mean)^2 for s in 0..7) / 7)
```

N=140/200/128 for ALF/Science/Web. For SR, x is binary; display 100*mean and 100*SD together if using percent. For Science, x is raw final Task Score. For WebShop report both SR and mean graded score, with scale stated. SD is across eight full-seed aggregate metrics, not the standard error, a population SD (divide by 8), variation across tasks or across training runs. Pooled eight-seed means equal mean-of-seed means only if all sweeps have the same complete task set; they do not define the requested SD.

Category results are diagnostic subsets with their own counts. Reconstruct overall by sum(category_mean*category_count)/N. Do not average the five Science categories equally; do not average six full Science batches and the short batch equally. Record each (protocol, gate, memory snapshot, rollout_seed, task identity) exactly once. Infrastructure failures remain incomplete and retain logs; do not exclude them from a smaller denominator or fabricate zero/model feedback. Agent failures and horizon terminations are real observations and remain in the denominator. Retry policy and seeds are fixed, and only one finalized attempt per task enters aggregation.

Official trainer retains validation drop_last=false; Science uses each trajectory's last active native score. Some legacy ALF/Web trainer summaries average batch-level success values; with a single complete 140/128 batch this is harmless, but it is unsafe after changing batch size. Our aggregator must read per-instance records for all metrics. Official repeated Web graded score can fall back to a console summary; our profile must store per-instance native reward to support denominator/fingerprint checks. pass@k is optional, separately labelled, and not a replacement for mean±sample SD or Science raw score.

## 7. Independent development data and memory isolation

The paper uses the fixed evaluation pool for training validation too. We adopt its final evaluation identity, **not** that development practice. No test success, raw trajectory, one-shot, card, judge example or future outcome may guide experience synthesis, thresholds, budgets or prompt selection. Previous v2 test results and test-informed policy repairs are historical contamination risks even though current test IDs match; disclose inherited choices, freeze them before new evaluation, and do not claim a pristine unseen test set.

Proposed split procedure, to be implemented and frozen before Phase 2/3 data use:

1. Load the pinned 2120 train and 200 test IDs with source hash checks, resolve IDs to the canonical simulator task/variation, remove all 61 overlapping train IDs. This was audited, not materialized as a new runtime split here.
2. Reserve any demo/prompt fixtures and near-duplicate goal/variation groups before selection. Existing sciworld_4509 is in source train and not in test (verified); the paper action-only profile does not use this one-shot. If the research arm retains it, it must not become a hidden calibration example.
3. For the five evaluation families only, select 10% of eligible disjoint training instances in each family, floor with minimum 1, by ascending SHA256 of `agentwmv3-dev-v1|canonical_instance_id`; tie-break by numeric canonical ID. Reserve the whole normalized goal/variation duplicate group in the same split; any group overlapping test is excluded from both dev and memory construction. Record canonical goal hash/version without looking at test outcomes.
4. Before goal-duplicate checks, available disjoint train counts are Measure=453, Test-Cond.=707, Find=596, Chem-Mix=45, Lifespan=176. Reserving demo4509 gives provisional dev quotas 45/70/59/4/17 = 195. This is a proposed independent dev size, not an already verified 195-task manifest; semantic/goal de-duplication can change it and must be recorded before any tuning.
5. Memory construction uses only the remaining approved train pool and true feedback. Development rollouts calibrate/evaluate Gate with a frozen train-derived memory snapshot; do not synthesize cards from a dev episode and then score that episode as held-out. If dev-driven card/prompt selection is later used, declare it and use a separate held-out development validation partition.
6. Freeze manifests (IDs, task/variation, category, source revision, goal hashes, split algorithm) and memory/card provenance hash. Test rollouts may be logged for analysis but must not mutate the shared offline memory during primary evaluation. If within-episode online memory is enabled, only already committed feedback is eligible; reset at episode start and label it as a separate shared condition across Gates. No cross-seed/cross-test-episode carryover.

Develop ALF from train only and Web from an explicitly reserved non-test goal pool; keep all 0..499 Web test-pool goals out of memory/dev for conservative isolation, even though only 128 are evaluated. Because per-worker Web catalogs are shuffled/generated with different seeds, numeric train/test ranges alone are insufficient: check canonical goal/product/options/price-constraint fingerprints against the union of all reserved per-slot test catalogs and prevent semantic duplicates from entering memory/dev. Cross-benchmark sharing is opt-in and must retain domain/source IDs. No new training algorithm is involved in these partitions.

## 8. Independent profile proposal and Phase 5–6 implementation

No runtime config was added or changed in this documentation task. Proposed future files:

| Proposed file | Work |
| --- | --- |
| `configs/evaluation/paper_aligned_v1/{common,scienceworld,alfworld,webshop}.json` | Immutable opt-in limits, history, sampling, task source, seed scheme, prompt/projection and scoring IDs; keep configs/worldmodel and legacy cohorts untouched |
| `scripts/prepare_paper_aligned_eval.py` | Validate pinned source files; materialize task/goal/game manifests and independent dev exclusion, no implicit downloads during run |
| `scripts/run_paper_aligned_eval.py` | Thin configuration/scheduling adapter to existing audited benchmark loops; require complete manifest, per-task seeds, new Run ID/output namespace |
| `scripts/run_audited_paper_eval.py` | Minimal opt-in configurable prompt bound/history/parser/horizon/seed hooks; retain all legacy defaults |
| `scripts/paper_eval_tokenizers.py` | Exact Qwen3 template, non-thinking, truncation/token boundary agreement with generation and SpanUQ |
| `analysis/paper_aligned_metrics.py` | Per-instance metrics, last native score, both Web metrics, complete eight-seed mean/sample SD |
| `scripts/test_paper_aligned_eval.py` | CPU manifest/schema/hash, task mapping, seed, scoring, weighted aggregation and resume checks; mocks only for unit tests |

Example design values (NOT a runnable command or an installed config): common Actor=Qwen/Qwen3-4B, task_seed=0, rollout_seeds=0..7, temperature=.4, top_p=1, top_k=-1 for vLLM, response_tokens=512, history_pairs=2; Science N=200/H=20/prompt=4096; ALF N=140/H=50/prompt=2048; Web N=128/H=15/prompt=4096/fixed_goal_seed=1000. Separate legacy and paper-aligned output roots and source/protocol hashes; refuse resume into a run with a different manifest, score scale, seed derivation, model or memory snapshot.

Phase 5: after SpanUQ and prior phases are accepted, implement Science profile first. Validate exact source IDs/catalog, reset look vs 20 actions, prompt/action grammar and two-pair history, raw negative/partial/terminal scores, truncation semantics, seed derivation and real-feedback flow on WM Skip. Run a small real dev-only smoke with Qwen3-4B and no WM, then Call/Skip traces with fixed WM. Do not load eight Actor replicas or run 1600 test episodes as a smoke test. Test-pool metadata checks are allowed; development rollouts must remain outside test.

Phase 6: add explicit paper-aligned baseline and unified structured-draft Gate arms, freeze configurations/memory and thresholds on dev. Validate aggregation on real trace schema and hand-computable unit fixtures: unequal categories; batch counts 32*6+8; score=-100/0/partial/100; Web partial vs full purchase; duplicate/missing task; seed/resume changes. Run eight complete Science sweeps only after authorization/resource planning. That is 200*8=1600 episodes per arm, not eight training seeds. Each ALF arm later has 140*8=1120; Web has 128*8=1024, after their environments pass independent validation. No cross-benchmark experimental launch is authorized by this plan.

Report score–cost curves for Science as the paper primary outcome and retain success–cost as an auxiliary research view; ALF/Web use success–cost and Web graded-score–cost. Count all Actor/WM/reflection/judge/router/retrieval calls, input/output tokens, hidden-state forward passes, online/offline memory cost and latency. Seed pairing improves analysis but does not ensure identical visited states. Do not copy the paper's 12-setting macro-average into our three-benchmark/one-Actor results.

## 9. Reproduction risks and acceptance gates

Confirmed alignment: numeric settings in Sections 4/C.1/C.4/S2/S3 and effective official launchers; exact Science test file/order/mapping; source overlap count; installed Science version; ALF package versions; official Web subset rule; two Web scoring source files equal.

Not yet validated: live Science reset/catalog/goal/JAR and observation equivalence; full ALF game asset equality; generated Web goal and product/index equality; Qwen3-4B effective sampling/truncation; actual eight-seed scheduling; complete grading aggregation; structured-draft domain validity. Official dependency constraints (e.g. Transformers 5.13.0/vLLM 0.19.1 and H800 execution) must not be installed over the isolated SpanUQ stack; use interface-level evaluation alignment and disclose backend/hardware differences.

Intentional differences: Qwen3-4B, frozen task-specific training-free Gate, WM prediction/revision and memory costs; no RL training. Prompt/draft additions or retained ambiguity representation are separately labelled deviations. Existing old experiments remain valid historical cohorts but cannot be merged into the new means or directly attributed to the Gate under changed settings.

Before declaring the profile runnable: pin every manifest/source/environment; resolve all ID/content checks; verify turn accounting and prompt/parsing; actual model payload checks; real dev smoke traces; meaningful CPU scoring/aggregation tests; complete-cost schema; independent old-default regression. Until then, describe this file as an audited plan, not completed paper reproduction.

## 10. Audit commands and outcomes

Executed read-only/source-audit commands: `git clone --depth 1 https://github.com/aim-uofa/Self-Distillation-AgentRL.git` into a separate temp directory; `git rev-parse HEAD`; `curl -L --fail https://arxiv.org/pdf/2610.19789`; SHA256 checks; source reads of the paths above; pinned HF source-ID JSON fetches; CPU ID/catalog-map intersection/order comparison; installed-package metadata queries; Web scoring-source byte comparison. No preparation launcher was run (some launchers download large assets/build indexes); no trainer was invoked.

The first CPU comparison using system Python 3.8 failed because official resolve_item uses str.removeprefix (Python >=3.9). Rerunning with existing verl-agent Python 3.12 succeeded without installing anything. This is an audit-interpreter issue, not a failed benchmark. Full simulator/model/evaluation tests were not run. Git diff/whitespace/core isolation checks accompany the documentation commit; no new runtime test claim is made.

## Appendix A. Audit-derived identity evidence

The JSON below is audit evidence, not experimental results or a substitute for runtime manifests. Web indices identify positions only; generated goal hashes still need validation. The overlap list must be excluded from experience/dev.

```json
{
  "official_commit": "35a44f5c5647b1785ad23ddf0ffbd196b695bf86",
  "source_revision": "99d0b7923bf126d9c6cdea5f362d2d41ccb5d493",
  "source_hashes": {
    "train": "9b7ce7f214c738de96d0048919e08e8241c5c2bca48843c53dae5098b938e28d",
    "test": "78fbe5ed7ba545f1a2ee08397a5e740f622872b6a3ee70315d349c4287675c79"
  },
  "ordered_ids_equal": true,
  "local_test_sha256": "78fbe5ed7ba545f1a2ee08397a5e740f622872b6a3ee70315d349c4287675c79",
  "manifest_order_equal": true,
  "mapping_mismatches": [],
  "category_counts": {
    "find": 44,
    "lifespan": 15,
    "measure": 53,
    "test_conductivity": 83,
    "chemistry_mix": 5
  },
  "train_count": 2120,
  "eval_count": 200,
  "overlap_count": 61,
  "disjoint_train_count": 2059,
  "overlap_ids": [
    "sciworld_1816",
    "sciworld_1821",
    "sciworld_1822",
    "sciworld_1824",
    "sciworld_1831",
    "sciworld_1835",
    "sciworld_1838",
    "sciworld_1844",
    "sciworld_1873",
    "sciworld_1879",
    "sciworld_1882",
    "sciworld_2487",
    "sciworld_2488",
    "sciworld_2496",
    "sciworld_2497",
    "sciworld_2498",
    "sciworld_2500",
    "sciworld_2503",
    "sciworld_2504",
    "sciworld_2506",
    "sciworld_2511",
    "sciworld_2512",
    "sciworld_2513",
    "sciworld_2521",
    "sciworld_2527",
    "sciworld_2531",
    "sciworld_2546",
    "sciworld_2548",
    "sciworld_2552",
    "sciworld_2555",
    "sciworld_2559",
    "sciworld_2563",
    "sciworld_2569",
    "sciworld_2573",
    "sciworld_2577",
    "sciworld_2581",
    "sciworld_2582",
    "sciworld_2589",
    "sciworld_2619",
    "sciworld_2621",
    "sciworld_2625",
    "sciworld_2631",
    "sciworld_2633",
    "sciworld_3861",
    "sciworld_3867",
    "sciworld_3895",
    "sciworld_3931",
    "sciworld_4170",
    "sciworld_4175",
    "sciworld_4176",
    "sciworld_4178",
    "sciworld_4283",
    "sciworld_4284",
    "sciworld_4287",
    "sciworld_4288",
    "sciworld_4292",
    "sciworld_4294",
    "sciworld_4295",
    "sciworld_4296",
    "sciworld_4297",
    "sciworld_4299"
  ],
  "demo_id_is_train": true,
  "demo_id_in_eval": false,
  "webshop_ordered_128_indices": [
    319,
    207,
    22,
    420,
    352,
    491,
    60,
    264,
    454,
    21,
    427,
    43,
    444,
    423,
    295,
    258,
    164,
    89,
    178,
    25,
    57,
    347,
    266,
    203,
    488,
    65,
    42,
    167,
    338,
    61,
    189,
    477,
    180,
    328,
    96,
    208,
    93,
    26,
    114,
    489,
    367,
    116,
    6,
    229,
    4,
    364,
    354,
    396,
    263,
    222,
    388,
    346,
    98,
    37,
    363,
    481,
    235,
    299,
    320,
    494,
    358,
    245,
    480,
    426,
    52,
    451,
    280,
    24,
    95,
    431,
    450,
    300,
    28,
    165,
    118,
    153,
    303,
    399,
    466,
    55,
    101,
    401,
    255,
    248,
    490,
    133,
    395,
    479,
    236,
    318,
    285,
    246,
    310,
    428,
    46,
    459,
    438,
    272,
    136,
    32,
    177,
    197,
    418,
    132,
    33,
    313,
    150,
    142,
    380,
    120,
    453,
    448,
    66,
    75,
    461,
    70,
    467,
    312,
    108,
    102,
    360,
    92,
    322,
    122,
    265,
    226,
    64,
    117
  ]
}
```

## Phase 5 verification and independent ScienceWorld profile

This section supersedes earlier *planned/unverified* ScienceWorld runtime settings only where explicitly stated below. Earlier audit entries and legacy experiment configurations remain historical records. ALFWorld/WebShop plans are not changed.

Sources checked: [paper Section 4 and Appendix C, Tables S2/S3](https://arxiv.org/pdf/2610.19789), official repository commit [35a44f5](https://github.com/aim-uofa/Self-Distillation-AgentRL/tree/35a44f5c5647b1785ad23ddf0ffbd196b695bf86). Specific implementations: [ScienceWorld worker](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/agent_system/environments/env_package/scienceworld/envs.py), [task catalog](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/agent_system/scienceworld_protocol.py), [request seeds](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/agent_system/evaluation/seeding.py).

Implemented separately in `configs/evaluation/paper_aligned_v1/scienceworld.json`:

| Item | Phase 5 implementation / verification |
| --- | --- |
| Evaluation set | Exact ordered 200 IDs, template names and variation indices compared against pinned official resolver and current SDK catalog. Native initial-goal hashes match previous audit. Metadata-only checks; no final-test rollout. |
| Task revision | AgentGym-RL-Data-ID `99d0b7923bf126d9c6cdea5f362d2d41ccb5d493`; test JSON SHA256 `78fbe5ed7ba545f1a2ee08397a5e740f622872b6a3ee70315d349c4287675c79` |
| Interaction horizon | 20 Actor actions. SDK envStepLimit=21 because reset performs one initial look. Verified real H20 probe and blocked21st action. Separate native completion and horizon truncation. |
| History / input / response | Last2 executed observation-action entries;4096 actual input tokens,512 maximum response tokens. Explicit left-token truncation; original text and actual suffix IDs retained, goal visibility audited. |
| Sampling | Nonthinking BF16 Actor, temperature0.4, top_p1.0, HF top_k0, repetition_penalty1.0. All online Gates share these settings. |
| Seeds | Official task-turn request seed derivation, rollout0–7. Review subcalls use a separate recorded namespace and cannot advance the next Draft RNG. Phase5 only seed0. |
| Observation | Official flat single-tab a/an object-list canonicalizer; nested lists unchanged. Preserve native numbered disambiguation IDs rather than reordering menus. |
| Primary metric | Final raw SDK Task Score, including negative scores. No training invalid-action penalty, reward normalization, max-so-far score or silent clipping. |
| Auxiliary success | Final score>=100, as official won flag; native termination separately recorded. H20 is not automatically success. |
| Aggregation | Equal task-instance weight within each seed; complete fixed set per seed. Formal8-seed mean and sample SD across seed-level aggregates, denominator7. No batch/category reweighting. One seed in this diagnostic has no across-seed SD. |

The legacy SDK wrapper used envStepLimit100 and the former diagnostic runner allowed configurable short horizons. The new **owned** profile service sets H20/SDK21 and does not modify that wrapper, the old service, or old configs. It reuses the installed SDK1.2.3; JAR SHA256 `f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7`. Exact historical binary identity to the paper's authors is not proven by matching a package version; this remains a reproducibility limitation.

Intentional differences: our Actor remains **Qwen3-4B**, not the paper's Qwen3.5; structured Premise/Expected Consequence/Action and native last-Action-tail parsing differ from the paper's action-only projection; Actor backend/hardware is HF SDPA/RTX4090; learned WM plus unchanged JSON reflection is our method addition. Therefore this is alignment of evaluation tasks, limits and native scoring, **not a literal reproduction of paper numbers** or a fair direct model comparison.

Calibration:30 legal dev goal groups,273 actual decisions, q0.75 distribution-only thresholds frozen before validation. Separate5 held-out development goal groups, no overlap with final200 or train memory;1 rollout seed and4 policies only. Experience snapshot v2 uses the same54 approved TRAIN cases and3 conservatively scoped cards; every comparison policy shares the frozen snapshot identity, with unused retrieval/extraction omitted. Memory Correction and confirmed-action skip remain off. Full results and costs are in [Phase5 report](PHASE5_REPORT.md) and [small-scale results](SMALL_SCALE_EVAL.md). These development runs do not replace the final200x8 evaluation.

Phase6 requires separately authorized fixed task/seed sweeps, predeclared matched-budget grids and per-seed instance aggregation. No final test outcome may influence thresholds, card construction, critic prompts or Gate rules. No RL, GRPO, PPO, self-distillation or teacher-loss code is involved.


## Phase 5.5 clarification: clocks, initialization and utility

This is an explicit correction/addendum; earlier audit entries and original configurations remain unchanged.

- SDK simulator moves are not Actor interaction turns. A reset/look can leave simulator moves at0; wait can advance11. The old explanation that the SDK21 setting follows from every initial look consuming one simulator move was inaccurate. SDK21 is retained to match the official worker configuration, while the wrapper independently enforces20 Actor actions. SDK done includes server completion, clock>21, or negative native score. A historical native_terminal label is not proof of simulator task completion or success. Raw Task Score/success extraction was not changed.
- Fixed Actor/WM model parameters do not imply byte-identical WM initialization. The released helper enumerates non-room ObjectTree nodes, ignores failed/ambiguous teleports, and can mislabel current-room observations. Traversal/parser state creates variable WM system inputs and token counts; this was missing from the Phase5 input-parity audit. Original recorded outcomes remain valid data, but the old original/guard pilot is not a fully isolated causal comparison of guard design.
- The opt-in validated-room-initializer-v1 uses current official teleport commands, same-helper-heap isolation and explicit room-label checks. Five development-only frozen descriptions are preserved with source hashes; independent collections match room observation facts while SDK alias strings can vary. First collection was frozen before any corrected model outcome. This repairs observed serialization/collection defects; hidden helper-world equality to the Actor world is not established. No default protocol or test set was silently replaced.
- Four fixed-development diagnostic variants and five strict live-heap intervention pairs are complete; a separate five-episode original-Reviewer run checks the corrected initializer. This is not a200x8 final evaluation. All original/corrective cohorts have0 successes; strict pairs found0 positive/1 negative/4 zero final-score utilities. Feedback-text accuracy is not task utility.
- Before any Phase6 sweep, agree on a common frozen initialization/Reviewer protocol and budget grid. Existing Qwen3-4B, threshold/Memory hashes and ScienceWorld numeric setting remain unchanged; ALFWorld/WebShop plans are unchanged. See PHASE5_5_REPORT.md, REVIEWER_AUDIT.md and WM_UTILITY_DIAGNOSTICS.md.
