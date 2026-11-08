# PHASE5_10_REPORT

Phase5.10 IN PROGRESS: Original Reviewer retained; compact-json-conservative-v2 proposed. Gate/Memory/thresholds/Actor unchanged. Bounded protocol:20 old-debug unique requests +44 unique Phase5.9 format-failure prompts (53 original events) +12 Full quality cases;32 strict one-step Draft/Original/Fixed branch comparisons;3 new-ID known-goal dev variants ×5 conditions =15 Episodes at seed0. No independent unseen goal groups remain. No Phase6 or final test.

Pre-generation plan, not results. Source base327b463. See configs/evaluation/phase510_reviewer_repair_v1.json and docs/phase510/{original_format_audit,task_isolation,gate_audit}.json.

Format audit:50 requests finish length/truncated at512 without Final Action;3 stop normally with extra blank line before Final Action. Closed-empty think template present; no generated think markers. Original is a three-line protocol, not JSON. Every simulation has at most one Review, no output retry. New version uses strict JSON guidance and validation, not grammar-constrained decoding;192 output tokens, same BF16 weights/sampling/history/public information. Actor cap512 unchanged. No Task Score tuning.

Global caps:650k Tokens,600 model calls,6500 native transitions,25000 audit bundles,3h;one GPU worker/persistent shared Actor-Reviewer. Batch limits200k/150calls/1800queries/8000bundles. Two bounded512-to768 length diagnostics only, not an online768-token variant. Do not expose new-ID Episode outcomes before protocol locks after offline regression. All regression outputs retained even if bad. Utility from controlled branches is one-step only, not downstream return.
