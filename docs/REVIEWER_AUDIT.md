# Reviewer audit — Phase5.5

All278 Phase5 reviews are actual frozen Qwen3-4B generations, not fixtures. Classification axes must not be added together as disjoint categories.

| Per-review protocol category | Count |
| --- | ---: |
| Keep current candidate (may already be revised) | 39 |
| Accepted template-level replacement | 155 |
| Of accepted replacements: current API enumeration present | 51 |
| Of accepted replacements: API enumeration absent, native validity Unknown | 104 |
| Replacement rejected by existing grammar validator | 50 |
| Semantic contract violation: replace returned same candidate | 34 |
| JSON decoding errors | 0 |

39 keep decisions are not necessarily39 keeps of the original Draft. Separate keep-original/keep-revised counters for the rerun are in [results.json](phase55/results.json). “Template accepted”, “API listed”, “environment accepts”, and “advances the task” are different properties. Absence from the incomplete SDK list remains Unknown. No future feedback is used by the diagnostic guard.

Real examples, with Goal/Observation/Draft, WM input hashes/actions, every relevant prediction, exact Reviewer input/output, final action/feedback and following observed scores: [reviewer_examples.json](phase55/reviewer_examples.json). Categories include keep_original_draft, semantic_replace_same_action_violation, accepted_replacement_api_enumerated, correct_text_forecast_task_harmful_revision, executed_replacement_task_score_decrease, keep_revised_candidate, accepted_replacement_validity_unknown, replacement_rejected_by_grammar. A rejected unexecuted proposal is not assigned invented environment feedback; the displayed feedback is explicitly for the final executed action.

The original Reviewer sees generic ScienceWorld action templates, not a grounded finite enumeration of all valid current object-action combinations. It receives predicted observation text, not native future rewards/scores or an Oracle. It can select an absent object, misuse a descriptive referent, violate a task focus constraint, repeat the same action under replace, or generate unsupported commands. Current state versus predicted state is explicitly distinguished by the prompt, but model compliance is unreliable. These are model/protocol failures; no JSON transport corruption is found.

The default three-round policy can cycle candidates, repeat a failed review, and execute a final replacement even when its final forecast is negative. The last forecast has no final veto. We preserve this original policy rather than introducing answer-specific repairs.

## Original two failures and missing task-utility channel

Fresh shared-heap probes of dev1709 and349 reproduce all six distinct stored candidate feedback texts. The absent jar and descriptive “substance called air” fail resolution. Crucially, `focus on glass cup` and `focus on bee hive` return the correctly predicted **You focus…** text but yield native-100. Successful parsing/interaction is not task success. See [old_failure_branch_probes.json](phase55/old_failure_branch_probes.json) and [Phase5 WM failure audit](WM_FAILURE_AUDIT.md). The old hidden heap was not saved; these are new controlled states with matching observable context, not asserted historical hidden-state identity.

All107 selected final-action forecasts in the old cohort match raw real feedback text; this is selection-biased and often dominated by easy failure strings. New Prediction-Only provides an action-unmodified denominator: strict text 69/70, existing profile-canonical equivalence 70/70. The strict mismatch is object-list order, not an established fact error. Validity-Checked has the same kind of order-only mismatch. Raw forecasts are preserved; normalization is analysis-only using the pre-existing official observation canonicalizer. General WM accuracy and the causal effect of an incorrect prediction on Reviewer decisions remain unverified.

## Independent diagnostic guard

The original Reviewer runs unchanged. Only its final replacement is checked against the already available official current action list. An unlisted replacement is Unknown and reverts to the original Draft, whose safety is also not invented. Raw model outputs/reviews are not rewritten. A previously generated draft forecast is logged as cached when reversion needs its commit comparison; no synthetic model response or duplicate inference is inserted. The guard blocked 28/65 proposed final replacements, but cannot reject an API-listed goal-damaging focus action. It does not guarantee fewer raw invalid model proposals.

| Variant | Mean final Task Score | Success | Env actions | WM requests | Model calls | Total tokens | Episode seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| never | -9.0 | 0% | 70 | 0 | 70 | 27,242 | 95.407 |
| original | -29.0 | 0% | 65 | 190 | 423 | 914,019 | 542.843 |
| prediction_only | -9.0 | 0% | 70 | 70 | 140 | 237,009 | 149.952 |
| validity_checked | -28.0 | 0% | 65 | 164 | 381 | 972,306 | 569.777 |
| original + validated initializer (separate corrective run) | -29.0 | 0% | 65 | 190 | 423 | 804,134 | 522.052 |

The guard's small partial-score difference is not proof of benefit, and independently collected WM initialization strings are a confound described in the main report. The variant is not silently enabled for Always, Internal or Full.
