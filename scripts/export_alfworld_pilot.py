"""Publish compact receipts from a completed, strictly audited ALFWorld run."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.independent_dev import atomic
from experience.splits import file_hash
from scripts.analyze_alfworld_pilot import analyze, read, rows, require

NAMES = {'never': 'Never', 'always': 'Always', 'internal': 'Internal', 'experience': 'External', 'full': 'Full'}


def export(run):
    require((run / 'complete.json').exists(), 'Do not publish incomplete experiment as final')
    s = analyze(run, ROOT / 'outputs/alfworld_final_audit_v1')
    require(s['status'] == 'complete', 'Final audit not complete')
    destination = ROOT / 'docs/alfworld_pilot/results'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['summary.json', 'examples.json', 'episodes.csv']:
        shutil.copy2(ROOT / 'outputs/alfworld_final_audit_v1' / name, destination / name)
    source = Path(read(run / 'manifest.json')['source_run'])
    for name in ['calibration.json', 'gate_config.json', 'threshold_freeze.json']:
        shutil.copy2(source / name, destination / name)
    shutil.copy2(ROOT / 'outputs/alfworld_model_runtime_versions.json', destination / 'model_runtime_versions.json')
    inventory = []
    for receipt in s['episodes']:
        ep = ROOT / receipt['raw_episode_path']
        for name in ['complete.json', 'initial.json', 'steps.jsonl']:
            p = ep / name
            inventory.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=file_hash(p)))
    for base in [source, run] + [p for p in run.glob('worker*') if p.is_dir()]:
        for name in ['manifest.json', 'model.json', 'ledger.json', 'generation_calls.jsonl', 'complete.json', 'supervisor_complete.json']:
            p = base / name
            if p.exists():
                inventory.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=file_hash(p)))
    atomic(destination / 'raw_file_hashes.json', inventory)
    table = ['| Policy | Success | SR | Formal steps | Lookahead | LLM tokens | Native API queries | Episode seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['policies']:
        table.append(f"| {NAMES[r['policy']]} | {r['successes']}/{r['episodes']} | {100*r['success_rate']:.1f}% | {r['formal_steps']} | {r['wm_calls']} | {r['total_tokens']:,} | {r['costs']['api_queries']} | {r['seconds']:.1f} |")
    quality = ['| Policy | Reviewer calls | Format invalid | Replacements | Admissible replacements | Nonadmissible replacements | External coverage |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for r in s['policies']:
        c = r['audit_counts']
        quality.append(f"| {NAMES[r['policy']]} | {c.get('reviewer_calls',0)} | {c.get('review_format_invalid',0)} | {c.get('revisions',0)} | {c.get('revision_admissible',0)} | {c.get('revision_nonadmissible',0)} | {c.get('coverage',0)} |")
    outcomes = ['| Task ID (seed0) | Family | Never | Always | Internal | External | Full |', '|---|---|---:|---:|---:|---:|---:|']
    for task in sorted({r['task_id'] for r in s['episodes']}):
        values = {r['policy']: r for r in s['episodes'] if r['task_id'] == task}
        row = [task, values['never']['family']]
        row += [('won' if values[p]['success'] else 'not won') + f" ({values[p]['formal_steps']} steps)" for p in NAMES]
        outcomes.append('| ' + ' | '.join(row) + ' |')
    costs = ['| Cost scope | Model calls | Input tokens | Output tokens | Total tokens |', '|---|---:|---:|---:|---:|']
    for category, c in s['all_attempt_generation_costs'].items():
        costs.append(f"| {category} | {c['model_calls']} | {c['prompt_tokens']:,} | {c['completion_tokens']:,} | {c['total_tokens']:,} |")
    memory = read(ROOT / 'docs/alfworld_pilot/memory_snapshot.json')
    total_tokens = sum(c['total_tokens'] for c in s['all_attempt_generation_costs'].values())
    eval_tokens = sum(r['total_tokens'] for r in s['policies'])
    verified = sum(r['audit_counts'].get('verified_lookaheads',0) for r in s['policies'])
    keeps = sum(r['audit_counts'].get('keep_transition_matches',0) for r in s['policies'])
    report = f'''# ALFWorld five-policy development pilot report

Completed and audited:3 goal-disjoint TRAIN-pool development games × seed0 ×5 policies=15 complete Episodes. No final140-game rollout, model training, SR promotion or Phase6 evaluation. These are exploratory functional results, not an estimate of official ALFWorld benchmark performance.

## Results

{chr(10).join(table)}

Counts/tokens/seconds are totals across3 games per policy. Native `won` defines success; auxiliary100×won is not ScienceWorld partial Task Score. Each task contributes one instance equally; one seed has no across-seed sample SD. No significance claim or method/threshold selection is made from these outcomes. Always has50 allowed calls/episode; selective policies have5, so this is not matched-budget evaluation.

{chr(10).join(outcomes)}

Machine-readable per-instance outcomes, timing components and paired differences: [episodes.csv](alfworld_pilot/results/episodes.csv), [summary.json](alfworld_pilot/results/summary.json). Real public decision/review/feedback examples: [examples.json](alfworld_pilot/results/examples.json). Raw files remain under the recorded server run paths; [raw-file hashes](alfworld_pilot/results/raw_file_hashes.json) pin them without committing large trajectories.

## Model, protocols and source identity

Actor/Reviewer use one shared frozen Qwen/Qwen3.5-9B model object per worker, revision `{s['config']['model_revision']}`, BF16/SDPA, Transformers5.22.2/Torch2.8.0+cu128, official nonthinking chat template; temperature0.4/top_p1/top_k0/repetition1,response512. H50/history2/input2048 and one-round original three-line Reviewer remain fixed for all policies. No compact-JSON repair or Qwen2.5 service was introduced. Exact configuration: [profile](../configs/evaluation/alfworld_pilot_v1.json), [model profile](../configs/internal_signals/qwen35_9b_alfworld_v1.json), [runtime versions](alfworld_pilot/results/model_runtime_versions.json).

Original source commit `{s['source_commit']}`; repaired continuation source `{s['continuation_commit']}`. Original calibration and already-complete Never Episode were reused by checked hashes. Continuation copies the original per-Episode AST exactly. Model load receipts contain no missing/unexpected/mismatched tensors. The same per-task/turn/role seeds are used across policies; this tiny pilot uses rollout seed0, not all eight paper evaluation seeds.

The already-audited paper supplies the ALFWorld evaluation horizon/history/token/sampling target; [PAPER_ALIGNED_EVAL.md](PAPER_ALIGNED_EVAL.md) remains unchanged. This pilot uses native ALFWorld0.4.2/TextWorld1.7.0 PDDL processes and TRAIN-derived development games, a structured draft and public current admissible commands. Native adapter matched150 real transitions against the official ALFWorld Gym wrapper, including H50 truncation. That does not establish parity with every historical AgentGym build or paper score.

One frozen shared draft template has a legacy “ScienceWorld” word in its Action placeholder; surrounding instructions, current admissible commands and all executed environments are ALFWorld. This defect/limitation is disclosed, not silently edited during calibration/evaluation. The model config retains a legacy unused `action_format` label; actual parsing uses the verified generic three-field parser and native commands. Any later prompt cleanup requires a new version/calibration and fresh fair comparison.

## Internal and external signals

Local=max(mean Premise token NLL,mean Expected Consequence token NLL); Global=mean all response-text token NLL, including field labels/Action and excluding prompt/EOS/padding. Whole intersecting tokens are included at field boundaries. Use raw generation logits before temperature, BF16 only, `hf_qwen35_generation_raw_bf16_v1`, no additional backbone forward.48/48 legal reserved calibration steps valid; scored/unscored/repeated generation token IDs equal, repeated raw NLL exact. New ALF-specific q.75 thresholds Local0.5448528666853629/Global0.32670785406992164 frozen BEFORE evaluation, not ScienceWorld thresholds. Pearson0.89687/Spearman0.89796; calibration q.75 union call fraction1/3 before Episode budgets. Six tasks×8 decisions do not establish stable calibration or an error-probability interpretation.

Frozen ALF-only Memory: {len(memory['cases'])} true feedback cases/{len(memory['cards'])} supported cards, four reserved TRAIN games, deterministic synthesis/BM25/current public preconditions, no LLM judge/synthesis/training. Supports require independent tasks AND episodes; Unknown is preserved. Two narrow cards concern already-open container/open repetition and taking through a visibly closed container. No online writes, ScienceWorld cards, future feedback or direct action correction. No natural positive sufficiency is asserted. Native goal hashes and task/goal-group exclusion separate Memory/calibration/evaluation/final140. Task templates/layouts and model pretraining exposure cannot be proven disjoint.

Full-vs-Internal same-current-Full-state shadow changes: {len(s['external_changes_on_same_full_states'])}. Exact whole-trajectory comparisons and their scope are in summary.json. Zero/rare coverage limits conclusions about the proposed external mechanism; lack of triggers is not evidence that external experience is universally ineffective. A changed action or paired policy score alone is not per-call causal utility. No strict same-state continuation experiment was performed.

## Reviewer and simulator audit

{chr(10).join(quality)}

“Admissible” concerns the current native command list, not task correctness; nonadmissible is not automatically a particular syntax/precondition failure. Format failures retain the draft with an explicit INVALID receipt. No trial feedback was fabricated or treated as a formal transition. {verified} verified lookaheads and {keeps} actual KEEP transitions match their candidate receipts. Reset/replay verifies the complete committed history, including invalid actions, independent OS PIDs, public states/admissible commands and private symbolic-fact digests. Private digests are audit-only, absent from Actor/Gate/Reviewer inputs. Passive `inspect` reads a cached last native receipt, not a fresh JVM/hidden RNG snapshot; isolation is supported within tested native deterministic PDDL behavior, not arbitrary process serialization. ALFWorld uses native processes, not ScienceWorld JVMs.

## Complete costs and bounded retries

{chr(10).join(costs)}

Completed evaluation online tokens={eval_tokens:,}; all recorded model attempts including calibration/numeric checks/interrupted and failed scheduling={total_tokens:,}. Model loading, generation, NLL extraction, retrieval, Gate, simulator initialization/replay/candidate/cleanup times and peak allocated GPU memory are separately retained in summary.json. Offline final/preliminary experience construction uses no model calls/tokens. Native setup/validation queries remain separate from online queries. No reliable dollar/FLOP/energy conversion is claimed; tokens, measured latency and environment queries are complementary costs.

The initial source run was safely interrupted for scheduling; one completed Never Episode is preserved, its partial Always attempt is excluded from outcome means but charged. First four-GPU scheduler failed from missing `dataclasses.asdict` at trace serialization; three workers reported NameError and the supervisor stopped the fourth. No completed failed-run Episode was assigned a zero score. Fixed only the missing global binding, statically checked all global names, tested complete trace serialization, and persisted state receipts after STOP.21 CPU contracts pass. Original Episode AST, Actor/Gate/threshold/Memory unchanged.

During that failed run, recorded model cost is exact (4 calls/3856 tokens); one interrupted native branch has bounded receipt uncertainty:10–11 API query attempts and9–11 state replies, not falsely exact counts. Explicit conservative charges are included in the resource cap, not relabelled as measured queries. See [engineering repair](alfworld_pilot/parallel_engineering_repair.json). On reconnection only GPU0 was sufficiently free; repaired continuation used one GPU under the user-authorized maximum4. The cumulative2h ACTIVE server-runtime cap includes prior415.225s and failed40.738s, excluding network downtime after all owned model processes exited. No other job was stopped.

## Reproduction and remaining scope

Run the native validation/collection commands in [ALFWORLD_PILOT.md](ALFWORLD_PILOT.md) with isolated `$MODEL_PYTHON` and `$ALFWORLD_PYTHON`; preserve pinned data/assets and independent split. Exact continuation performed:

```bash
"$MODEL_PYTHON" scripts/test_alfworld_pilot.py
"$MODEL_PYTHON" scripts/test_alfworld_parallel.py
"$MODEL_PYTHON" scripts/run_alfworld_parallel.py --source outputs/alfworld_five_policy_v1 --run outputs/alfworld_parallel_v2 --gpus 0 --failed-prior outputs/alfworld_parallel_v1
"$MODEL_PYTHON" scripts/analyze_alfworld_pilot.py --run outputs/alfworld_parallel_v2 --output outputs/alfworld_final_audit_v1
"$MODEL_PYTHON" scripts/export_alfworld_pilot.py --run outputs/alfworld_parallel_v2
```

Continuation requires the original checked run/receipts and a fresh continuation Run ID; a fresh-server run starts with the documented original driver and collection, not a fabricated source run. Full fresh-server reproduction has not been performed. All owned model/native processes must be checked after completion; the final supervisor receipt is retained. Engineering verification and observed policy results are separate from scientific effectiveness. Stop here; no automatic Phase6/final140×8 evaluation.
'''
    (ROOT / 'docs/ALFWORLD_PILOT_REPORT.md').write_text(report)
    print('Exported final compact results and report; no raw weights/cache copied')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    export(Path(args.run).absolute())
