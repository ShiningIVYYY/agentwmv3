# WM utility diagnostics — Phase5.5

## Matched states, not similar states

Five points were frozen before these experiments: first Full WM invocation in each original development task, independent of reward selection. A fresh SDK world replays the real prefix and verifies observations/reward/score/done. One live-JVM checkpoint then restores **the same** captured heap (world, parser, RNG/object IDs) plus Python reward state and wrapper Actor-step counter for both branches. Original historical hidden-state identity is not claimed. Scope is these new reconstructed shared states.

Snapshot probe captures826,857 fields/34,598 arrays and reproduces look/wait/focus feedback, rewards and clocks. A stochastic greenhouse wait also repeats after restore. Both branches reproduce identical first Draft token IDs, including the original stored Draft. Model weights/configuration and remaining Actor budgets match. Branch A executes the Draft; Branch B invokes the original WM/Reviewer once. All subsequent decisions follow the same frozen **Never WM** Actor policy with common task-turn seeds. No future branch result is visible when choosing the first action; no branch output trains/tunes a Gate.

| Task / intervention step | Skip final score | Call final score | Call−Skip | Skip tokens | Call tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| sciworld_742 / 4 | -100 | -100 | 0 | 595 | 25,585 |
| sciworld_128 / 1 | 0 | 0 | 0 | 7,502 | 14,489 |
| sciworld_2417 / 3 | 0 | 0 | 0 | 3,335 | 34,292 |
| sciworld_1569 / 2 | 55 | 55 | 0 | 6,412 | 13,344 |
| sciworld_3070 / 1 | 0 | -100 | -100 | 5,975 | 7,426 |

Observed utility: **0 positive, 1 negative, 4 zero**. In3070 the Call branch focuses on a finger painting and ends at-100, whereas Skip ends at0. In742 verification delays a fatal focus but both branches still end at-100; delaying failure is not positive final-score utility. These are one-seed sample-path effects of one intervention followed by Never, not the effect of Always or an expected utility estimate over all states. No task success or positively useful final-score WM intervention was established in these five pairs.

## Prediction versus useful revision

The model can predict literal feedback accurately in this small distribution. That does not establish its ability to predict task reward or termination. Reviewer receives only the hypothetical observation field, not Oracle score/done labels. A truthful “You focus…” forecast may accompany a catastrophic task failure. Predicted feedback accuracy and Reviewer goal reasoning must be evaluated separately. Additional unexecuted alternatives remain Unknown unless tested in a real isolated branch.

Prediction-Only exactly reproduces Never's full actions/observations/scores in all five tasks. It is a negative control: model serving itself does not mutate the real environment or consume extra Actor steps. It also cannot improve actions when action modification is disabled.

## Initialization engineering defect and corrective run

The released initializer treats every root ObjectTree child as a room; terminals and sewer nodes appear among them. It ignores failed/ambiguous teleports and labels the unchanged current observation with the requested target. A terminal ambiguity poisons later room queries; the isolated failure/restore reproducer confirms this. JVM traversal order makes the wrong room sections and token counts vary. Local Python RandomIsolation does not make the entire JVM/parser/alias pipeline deterministic.

The independent repair uses official admissible teleport commands, verifies actual room identity, restores the same helper heap before every room view, excludes unsupported root nodes and freezes a source-hashed real description. It preserves the WM backbone, original review prompt and rounds. All five tasks have10 verified rooms; two independent collections reproduce room observation facts. SDK object aliases can still vary; both samples are retained and the first is frozen **before** model evaluation, without reward selection. This is replay of actual helper observations, not invented room text or a claim of universal cold-JVM byte equality.

Five corrective Episodes use the same original tasks/seed/settings, with new initializer only. Mean score **-29.0**, success **0%**, tokens **804,134**. Original results remain unchanged. Exact initializer/provider/model-request/score/cost audit: [fixed_initializer_audit.json](phase55/fixed_initializer_audit.json). The strict five pairs used the original initializer; corrected initialization was tested in full episodes, not new matched pairs, so the old pair utilities are not silently transferred to that new input version. The repair is explicit opt-in; legacy Always/Full definitions are preserved. Separately created helper-world hidden facts are not proven identical to every actual Actor world; this remains an initialization limitation requiring controlled future protocol design.

Full large logs, checkpoints' verification metadata, all real model requests and executed sources remain under `outputs/phase55_*`. Compact hashes: [raw_artifact_manifest.json](phase55/raw_artifact_manifest.json). No final200-task inference, training, threshold calibration or experience expansion occurred.
