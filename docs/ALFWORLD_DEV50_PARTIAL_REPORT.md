# ALFWorld50-task evaluation: stopped partial report

Status:236/250 Episodes complete and individually audited; four owned workers exited.47 tasks have all five policies complete, plus one additional Never result.14 Episodes remain incomplete/unrun. Do not report these as50-task final results or assign zeros to missing Episodes.

## Same47-task complete-case comparison

| Policy | Success | SR | Tokens | Lookahead | Native API queries |
|---|---:|---:|---:|---:|---:|
| never | 7/47 | 14.9% | 1,115,199 | 0 | 2204 |
| always | 18/47 | 38.3% | 2,445,423 | 1785 | 48610 |
| internal | 12/47 | 25.5% | 1,209,779 | 187 | 5420 |
| experience | 7/47 | 14.9% | 1,117,661 | 2 | 2250 |
| full | 11/47 | 23.4% | 1,208,996 | 186 | 5365 |

Incomplete-case selection may be informative: the run stopped at an Actor formatting failure rather than random missingness. These are descriptive interim figures, not a completed independent50-task estimate. All task instances are equally weighted; no category/batch equal averaging.

## Failure and reliability

Worker0 stopped at task alf-54e65619afcc0470/Never: the Actor emitted512 tokens of a long Premise, finish_reason=length, without a complete Expected Consequence/Action. The unchanged Episode runtime correctly refused to invent an action (Invalid single-action draft); the fail-fast scheduler stopped the other workers. This was not GPU exhaustion, Replay failure or a budget limit. Other workers stopped safely and all ledgers have no unknown pending receipts. The original long response and missing task-policy list are retained in the JSON summary.

All236 completed Episode hashes and individual audits pass. On the common47 tasks, all2160 completed lookaheads were verified. Always has163/1785 Reviewer format failures (158 length-limited),841 actual replacements including192 nonadmissible replacements. Internal12/187 and Full12/186 review format failures persist. No prompt, threshold, Memory or model change was made.

## Internal/external interpretation

Full has11 successes vs Internal12. The only success disagreement is alf-cbca1237088cbd17 (Full failed/Internal succeeded). Two reliable external triggers changed Skip to Call at low NLL on current Full states; revisions were admissible but did not complete the goal immediately. One other external conflict overlaps an internal trigger. Full coverage3 steps, External-only coverage2 steps, no sufficiency positives. This is limited experience coverage, not proof that experience in general is useless.

Full uses about50.6% fewer tokens than Always but has23.4% vs38.3% observed SR. Policy trajectory differences/action revision do not prove single-call causal utility; no controlled same-state future continuation was run.

## Costs and next boundary

All recorded new-run attempts, including partial work, total 7225988 tokens/12457 model calls. Native transition queries=60330 and state receipts=64720, plus the preflight104 transitions/116 receipts recorded separately. Do not confuse common-cohort online table costs with all-attempt totals.

Raw run: outputs/alfworld_dev50_v1. No jobs restarted; frozen source remains unchanged. A blind same-seed resume is expected to reproduce the malformed draft. Completing the cohort first requires an explicit, uniformly applied Actor-format failure policy and episode-level fault isolation, versioned separately from any Gate/Memory/threshold improvement. No hidden fallback, strict=False, model training or Test rollout.

See [JSON evidence](alfworld_dev50_v1/partial_236/summary.json) and [all236 completed rows](alfworld_dev50_v1/partial_236/completed_episodes.csv).
