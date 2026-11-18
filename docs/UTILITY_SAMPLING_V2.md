# Paired-task and repeated-state WM utility sampling v2

Status: implementation of the next bounded sampling experiment. No new performance claim. Old activation-v1 data/results remain unchanged.

## Why v1 did not train a Probe

The v1 collector sampled only step-index2. Its60 paired states yielded0 Help/58 Neutral/2 Harm. Among40 unchanged interventions,38 drafts were publicly admissible,2 were nonadmissible with native `Nothing happens.`;12 continuations succeeded and28 failed. Review decisions were31 KEEP,5 rejected replacements and4 REPLACE outputs with the original action. None of these counts establishes semantic action correctness. Format validity, admissibility, unchanged intervention, task outcome and causal utility are separate quantities.

The user suggested paired Never/Always comparisons. This is adopted to locate candidate interventions, while retaining exact-state experiments to establish whether an individual call helps. Whole-trajectory success is not assigned to every action or WM call.

## Frozen method and data boundary

Only agentwmv3 is changed. Qwen/Qwen3.5-9B Actor and Reviewer remain frozen, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, the same shared weights/BF16/backend/protocol `activation-pce-v2`. H50/history2/input2048/output512, temperature0.4/top-p1/top-k0 and the original admissibility guard are preserved. Real WM is ALFWorld independent native-process Reset/Replay, not Qwen2.5 and not ScienceWorld JVM simulation.

The existing31-case/two-card ALFWorld Memory, original Gate rules and NLL thresholds are unchanged. Old q0.75 NLL values are used only as public sampling indicators, not recalibrated or optimized by this collector. No Memory synthesis, Actor/Reviewer training, RL or new online Gate is performed. Small probes may train conditionally on adequate labels.

The original goal-group roles remain fixed:36 Probe Train,12 Probe Validation,12 Gate Calibration. The previously evaluated12 independent-Dev goals are excluded from this new collector, as are the prior exposed50 and final140 Test goals. No task is moved from Dev/Test into training. Native identities/replay proof from the unchanged original77-task preflight are reused; each actual counterfactual additionally verifies its whole replay prefix and exact current-state fingerprint. Task templates/layouts may overlap; only five families remain under the original goal exclusions.

## Source task pairs

For each authorized source task, run Never and Always with the identical initialization and Actor rollout seed1. Both retain actual trajectories, including failures/format aborts. Always has50 one-step lookahead opportunities, one fixed seed0 Reviewer per eligible step. Both source policies collect pre-action activations/NLL solely as offline training data; their source costs therefore include feature collection and are not presented as ordinary online baseline costs.

Report all paired task outcomes: both-success, both-failure, Always-success/Never-failure and the reverse. Before the first different executed action, compare exact public/private-digest states and actual transitions. The first shared-state action difference is an intervention candidate, not a Help label. Later Always states cannot be equated with states from a diverged Never trajectory.

## Candidate selection

Instead of one early step, cover three windows:0–9,10–29,30–49. Within each available window choose one uniform state and one probabilistically enriched state using only public/pre-action indicators: nonadmissible draft, repeated action with unchanged public context, verified frozen-Memory conflict and high raw NLL. Each unique public Actor-context/action has a nonzero probability. Record exact within-pool inclusion probability, sampling kind, window and weight. Deduplication is Actor-context deduplication, not a proof of complete hidden-state equality.

Probe Train additionally includes the first verified common-state action divergence when not already selected. On Always-success/Never-failure TRAIN pairs, up to two further actual Always revisions are mined as candidates. Deduplicate by exact state fingerprint plus draft action. This future-outcome enrichment is an offline Train-only selection rule; no source result or future feedback enters the activation feature path. Val/Calibration receive only the preregistered temporal/public-indicator sampler, without outcome mining.

All task outcomes and sampled neutral/negative cases remain recorded; no task is selected because it favors Full. Outcome-enriched Train proportions do not estimate natural positive prevalence. Val/Calibration are also stratified, not a full natural online traffic sample; no deployment or calibrated-probability claim follows from their unweighted metrics. Future deployment requires an explicit weighting/natural-traffic calibration audit.

## What counts as evidence of utility

For each selected state, independent Reset/Replay must reproduce its entire actual committed prefix and current fingerprint. One native trial of the fixed draft and the unchanged seed0 Reviewer determine a fixed intervention action. Reviewer sees only public current evidence and allowed trial feedback.

Branch A executes the original draft; Branch B executes that verified post-review action. Both then use the same frozen Never-WM continuation, absolute-step random seed and remaining H50 budget. The Reviewer output is held fixed across continuation repetitions; repeated branches do not resample a different treatment. Labels are conditional on that Reviewer seed and continuation, not arbitrary future policies or repeated-WM synergies.

When the source is a Never trajectory, its actual seed1 suffix can be reused as Branch A. This is a hashed/identified real trajectory, not invented feedback. Its seed, fixed draft, prefix, actual transitions and final native result are retained. For an Always-source state, future Always actions cannot be used as a Never continuation: Branch A is rebuilt and genuinely rolled out. Every Call branch is independently rebuilt and actually executed.

Changed actions are checked with continuation seeds1/2/3. If any nonzero return difference appears, automatically add seeds4/5, including apparent harm and inconsistent cases, rather than only confirming favorable observations. A repeated Help candidate requires at least3 of5 deltas positive and none negative; Harm is symmetric. All-zero is Neutral. Mixed/weak/incomplete evidence is Uncertain/Unknown and excluded from hard-label training. These finite-seed rules are diagnostics, not statistical proof or exact estimates of expected utility.

For an unchanged action, the actual seed1 Call trajectory must exactly equal the observed/reconstructed Skip trajectory. Such a case is a verified zero-intervention example; additional unrun seeds are NOT fabricated. This says nothing about whether the action itself is correct. If equal actions yield unequal fixed-seed continuations, fail the consistency audit rather than create a false neutral.

No action correction or admissibility alone is sufficient for Help. A genuine positive label requires improved actual final native success under the controlled comparison. Record source episode, case, role, feature timestamp, label timestamp, every seed's native results, and whether Skip was reused from observed data.

## Stages and stopping conditions

1. Collect36 TRAIN task pairs and up to7–9 candidates per task (fewer if early termination/missing windows). Save per-task and per-state checkpoints.
2. Only if at least5 five-seed repeated-Help candidates occur across at least3 TRAIN tasks, collect the fixed12 Validation and12 Calibration task pairs. Their selection rules cannot be changed after inspecting outcomes.
3. With expanded pools, train only eligible small Linear/MLP/EMA probes using the unchanged strict training code and3 seeds. Heads lacking positive/negative labels remain unavailable. Training artifacts explicitly forbid automatic Gate deployment pending the sampling/calibration audit.
4. If Train feasibility fails, stop with all actual results and a limitation report. Do not manufacture positives, move Test/Dev tasks, or automatically modify the Reviewer. No complete independent evaluation or Phase6 is launched by this script.

If Always helps only through multiple interventions but no individual call helps under Never continuation, report that interaction explicitly. This collector cannot label those cases as single-call positives; a separately defined fixed-continuation or intervention-sequence study would be needed.

## Resources, artifacts and commands

Use GPU0/3/5 (three available cards, maximum4 authorized). Whole-run hard limits:36h,36M tokens,60000 model calls,300000 native transitions and360000 state receipts. The TRAIN stage is allotted at most half the whole budget; heldout stages share only the recorded remainder. Reserve the worst-case entire state comparison/confirmation before beginning it. Budget-skipped states remain Unknown. Source/reset/replay/trial/review/continuation costs, extra activation forwards and raw receipts are charged; no zero-cost simulator claim.

Existing model/cache/environments are reused; no downloads or new package installation. Raw features, traces, native logs and trained probe weights remain under `outputs/utility_sampling_v2`; Git stores scripts/configuration, tests, source freeze, compact examples/results and documentation. No model weights or large raw trajectories are committed.

```bash
PYTHONNOUSERSITE=1 .venv-qwen35-clean/bin/python scripts/test_utility_sampling.py -v
# Set ALFWORLD_DATA / ALFWORLD_PYTHON to the existing local data/CPU env first.
nohup env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-qwen35-clean/bin/python -u scripts/run_utility_sampling.py \
  --run outputs/utility_sampling_v2 --auto-publish \
  > outputs/utility_sampling_v2.log 2>&1 < /dev/null &
```

The controller freezes code/config/Memory/source Git hashes, writes progress and owned-process receipts, cleans only its own workers, and preserves error evidence. Run IDs must be fresh; partial runs are not blindly resumed. On completion it exports `docs/UTILITY_SAMPLING_V2_REPORT.md` and compact JSON, then attempts own Git publication only with no unrelated worktree edits. Startup is checked before leaving it running; no continuous assistant monitoring is required.
