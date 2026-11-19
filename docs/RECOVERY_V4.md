# Actual generated-token alignment repair, recovery and larger training pool

Status: repair validated on the server; collection launch/startup are recorded separately. CPU audit preserves4,294 prior valid signal values exactly, including all144 original calibration records. The actual failed-state GPU regression reproduces all35 actual IDs, all four NLL scalar deltas0 and tensors[35,4096] at blocks10/15/21.45 new native identities and5 independent replay controls pass.14 new regression tests pass. No new utility gain or trained-Probe effectiveness is claimed.

## Confirmed failure and honest costs

The failed V3 task has431 prompt IDs and35 actual response IDs. Canonical re-encoding has34 response IDs: actual `hand`+`t`+`ow`+`els` becomes canonical `hand`+`tow`+`els`. Decoded response text is identical and the three-line action is complete. Original raw-generation logprobs are defined on the35 actual IDs, so replacing IDs, trimming arrays or using the34-token denominator would be wrong.

V3 completed11 whole training tasks and12 source Never/Always pairs. Retain all11 completed tasks irrespective of outcomes. The one new Help case, with deltas[1,1,1,1,1], belongs to a fully completed task and remains usable. Restart all13 unfinished original tasks at unchanged seeds; preserve and exclude their partial-task rows from the final merged corpus to avoid duplicates/outcome-based selection. Preserve all raw partial logs and charge their resource costs, including discarded data.

Confirmed stopped-run cost:4,503,471 tokens/7,898 completed model calls/23,847 native transitions/24,835 state receipts. Two pending draft requests have no final receipts; actual output tokens cannot be recovered. Their maximum reserved totals are907+983=1,890 tokens. Record them as unknown, retire them from research data, charge their full upper bound for budgeting and never overwrite the old pending ledgers. Counts/outputs for the interrupted attempts are not invented.

## Minimal repair and verification

Keep original files/experiments unchanged. New `actual_token_alignment.py` first takes the existing official offset-mapping path when actual IDs equal canonical encoding, preserving previous numbers exactly. For a noncanonical ASCII response, validate exact prompt IDs/suffix, full prompt+response decode equality, strict monotonic prefix decoding, one nonempty offset per actual token, contiguous complete coverage and absence of special response tokens. Use the actual IDs and their original NLL throughout. Noncanonical non-ASCII/ambiguous decoder cases remain unsupported/Unknown until separately validated.

`ActualTokenNLLProvider` copies the original raw NLL extraction formula and configuration guards; only the imported alignment implementation changes. `AlignedRuntime` replaces the provider instance after strict loading; Actor weights, tokenizer/chat template, BF16 backend, sampling and post-token feature capture are unchanged. Alignment version is explicit metadata. Canonical calibration scores must match exactly before reusing thresholds; score units/uncalibrated meaning do not change.

The new collector records full true source steps, but isolates unalignable feature samples. It retains Missing/error status, never substitutes zero, never creates an unverified feature cache, and excludes such states from training/candidate mining. Original actions and source Never/Always policy still execute normally. Full actual histories, including quarantined steps, remain available for prefix reconstruction/observed Skip suffixes. Backbone shape/nonfinite failures remain fatal, not silently quarantined as tokenizer issues.

Required checks before background launch:

1. Reproduce the legacy exception on the original saved real generation; recover all35 actual offsets/NLL without changing IDs or values.
2. Compare every formerly valid committed signal in V1/V2/V3 against the new provider; require exact Local/Global/Premise/Consequence NLL equality, including all144 original Gate-Calibration scores.
3. Regression tests: tampered prompt, special tokens, truncated/empty/invalid NLL, unsupported noncanonical Unicode, full offset coverage, and quarantine exclusion.
4. In a real native instance, replay the original31 committed actions to the failed state; reload frozen9B, reproduce the original actual response IDs, check raw NLL/field masks and hidden shapes, repeat before/after feature forward, run an isolated real lookahead and review. Mock outputs are not effectiveness evidence.
5. Check no old owned workers remain; freeze raw artifacts, completed-task identities and retained feature SHA256 hashes.

## Larger fixed training pool

Proposed total: **100 canonical training goals**. Reuse original36 V2 goals and11 completed V3 goals, finish13 incomplete original goals, then collect40 additional untouched TRAIN goals. Add5 separate native-validation controls. Preserve original12 Probe Validation,12 Gate Calibration,12 exposed independent-Dev and140 final Test role assignments exactly.

Choose all40 new training targets deterministically before outcomes, excluding every prior reserved/exposed canonical goal and Test goal. Verify actual native goals against all prior identity receipts, game/metadata hashes and initial prompt limit. Run5 independent replay controls on separate training-pool control goals. Availability/family coverage must be reported; do not relax exclusions to favor positive outcomes.

Keep the source/label protocol unchanged: paired same-task Never/Always seed1, early/mid/late uniform+public-difficulty sampling, Train-only outcome mining, fixed Reviewer seed0 and same-state Never-WM continuation seeds1/2/3 with4/5 confirmation whenever any difference occurs. Help still requires at least3/5 positive deltas with no negative; no action change is not an action-correctness label. Complete all100 fixed training goals before testing the unchanged>=5 Help/across>=3 task feasibility rule. Only then collect fixed Val/Cal and train eligible small probes. No automatic Gate deployment, independent-Dev/Test rollout or Phase6.

## Budgets and safe background lifecycle

Use at most4 free GPUs, planned0/1/6/7; do not displace other users. Recovery13 is charged only the original remaining TRAIN caps and obeys the original V3 deadline. Adding40 goals has a new explicit budget, not a silent reset: V3+recovery+new expansion together have at most72M tokens/120k request reservations/600k transitions/720k state receipts, including the conservative1,890-token interruption charge. Old V2 sunk corpus cost is separate. New extension wall cap48h; original recovery deadline remains binding on the recovery phase. If it expires during network downtime, stop and report it rather than silently changing it.

Publish known costs separately from unknown output ranges and upper-bound budget charges. Same frozen Qwen3.5-9B Actor/Reviewer, Experience Memory, Gate, NLL thresholds and paper benchmark profile; no Qwen2.5, Actor training or dependency installation. Only eligible small probes may train on CPU after data audits. Raw weights/caches/trajectories stay outside Git.

Intended run: `outputs/recovery_v4`; immutable stopped source: `outputs/utility_expansion_v3`. Per-phase receipts and `progress/error/publication.json` control fresh-run-only startup, owned-process cleanup and compact result export. Auto-publication may only touch its own report/results and records Git failures separately. Do not start a second job if any recovery receipt/run exists. Verify the first actual recovered task, then leave it bounded in the background.

## Prepared implementation files

* `internal_signals/actual_token_alignment.py`, `actual_token_nll.py`: actual-ID offset repair and unchanged scoring formula.
* `evaluation/aligned_runtime.py`, `recovery_collection.py`: shared frozen Actor/Reviewer, explicit sample quarantine and real-history preservation.
* `scripts/prepare_recovery_v4.py`: completed-task import, two-request cost audit, deterministic100-goal manifest and remaining budgets.
* `scripts/audit_alignment_v4.py`, `test_recovery_v4.py`, `smoke_recovery_v4.py`: real saved-output numerical compatibility and real GPU/native-state regression.
* `scripts/preflight_recovery_v4.py`:45 new native identities plus5 independent controls, zero Test resets.
* `scripts/run_recovery_v4.py`: recover13, expand40, conditional heldout pools/probe training, no deployment.

Connection was restored and preparation receipts were checked before execution. Confirmed known stopped cost is4,503,471 tokens and7,898 completed model calls; two final outputs remain unknown and consume a1,890-token conservative reservation charge. Real engineering smoke additionally uses1,655 tokens/3 completed calls/66 transitions/68 receipts; new CPU preflight85 transitions/95 receipts. These are subtracted once from available caps. Raw old ledgers/partial trajectories remain unchanged.

Data adequacy: the5-Train-positive/2-Validation-positive eligibility rule is only an engineering floor. Any automatically trained head is marked engineering_exploratory/scientific_effectiveness_validated=false/deployment_allowed=false. Practical future study targets (heuristic, not guarantees) are about50 Help states across10–15 training goals for a first regularized linear experiment, with independently grouped validation positives. Five repeats of one state are one labelled state, not five independent samples. This fixed100-goal batch is a feasibility/data-yield study and may not reach that target. No test set is used to fill missing labels.

## Approved plan update before launch

The user now approved the complete simple training workflow. The new versioned
linear-only trainer replaces the optional3-architecture exploratory training
within this fresh recovery controller only. Minimum data targets and heldout
checks are in configs/evaluation/linear_utility_v1.json; the legacy5-positive
rule permits heldout collection only, never formal fitting. Train-only grouped
learning curves use deterministic25%/50% task subsets. All old source files,
scores, raw journals and previous experiments remain unchanged.
