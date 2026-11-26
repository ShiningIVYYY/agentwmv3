# V5 bounded two-pass continuation

The first resource recovery completed38/40 new Validation tasks. One worker
hit its18000-call local allocation during the final Always source; the other
stopped with17515 calls. All pending receipts are null. Retain38 complete tasks
and both complete Never episodes from the2 incomplete tasks; exclude their
unfinished Always source cases. Retain all cost, including excluded work.

Validation has6 new Help over5 new goals,8 cumulative Help over7 goals. Train
remains16 Help/13 goals; no Probe trained. Original140 densification Unknowns
and23 new Validation Unknowns remain excluded from hard labels.

Shared Validation residual:485 calls/975632 tokens/34768 transitions/45996
receipts. Two complete Never episodes are reused with explicit provenance and
zero ADDITIONAL generation/native cost. Restart only both unfinished Always
sources at fixed seeds; worst case200 calls/512000 tokens. Collect both sources
before labels so later tasks cannot lose their source budget. Selection rules
and actual3/5-seed utility definitions do not change.

New Train likewise collects all assigned Never/Always source trajectories
first, then applies the original candidate sampler and true counterfactual
labels. Per-request RNG depends on task/step/role/seed, not task scheduling;
Actor/Reviewer/Memory stay frozen and no cross-task update occurs. Unknown is
recorded when a whole-state label cannot fit; never convert it to negative.

The remaining original global caps are33471715 tokens/62929 calls/428711
transitions/484621 receipts. The fixed original absolute72h deadline stays
binding; no new budget or deadline reset. Phase caps are original residual
Validation and32M/60k-call Train. Original+first-recovery+continuation directories
share the same90GiB cap/free-disk50GiB floor. Use1 empty GPU for the short shared
Validation residual, then1-4 empty cards for Train, never evict external jobs.

Only two fixed Val tasks and80 fixed Train tasks remain. All completed38 Val,
100 old-task scans/294 labels and both complete Never trajectories are imported,
not regenerated. Preserve source hashes, feature contracts, scoring path,
thresholds, role boundaries and all historical costs. No final Test/fresh Dev,
automatic Gate deployment or Phase6. Fit linear Help heads only after original
50TrainHelp/20ValHelp and other readiness conditions are met.

Commands using the existing isolated environment:

- .venv-qwen35-clean/bin/python scripts/test_v5_remaining.py
- .venv-qwen35-clean/bin/python scripts/resume_v5_remaining.py --run outputs/utility_expansion_v5_remaining

Six tests pass, including actual retained positive counts, role/time/source
audits, original remaining caps, whole-source funding bounds and a synthetic
schedule-only regression that requires every source before any label. Actual
model startup is recorded separately. This is an engineering scheduling repair,
not a change chosen to improve task score.
