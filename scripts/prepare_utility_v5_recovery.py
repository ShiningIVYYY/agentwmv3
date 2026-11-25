"""Seal GPU-load failure, retain every completed task, keep original caps/window."""
import json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.splits import file_hash
from scripts.run_utility_sampling import rows,phase_cost


def main():
    r=ROOT/'outputs/utility_expansion_v5';out=ROOT/'docs/utility_expansion_v5/recovery_receipt.json'
    if out.exists():raise ValueError('Recovery already registered; do not overwrite')
    cfg=json.loads((ROOT/'configs/evaluation/utility_expansion_v5.json').read_text());m=json.loads((r/'manifest.json').read_text())
    done=[x for p in (r/'densify_train').glob('worker*/completed_tasks.jsonl') for x in rows(p)]
    if len(done)!=100 or len({x['task_id'] for x in done})!=100:raise ValueError('Incomplete retained cohort')
    for phase in ['densify_train','new_train']:
        if not all(json.loads((r/(phase+'_supervisor.json')).read_text())['workers_exited']):raise ValueError('Old workers active')
    for p in (r/'densify_train').glob('worker*/ledger.json'):
        x=json.loads(p.read_text())
        if x['pending_model'] or x['pending_transition']:raise ValueError('Unknown prior receipt')
    for p in (r/'new_train').glob('worker*/ledger.json'):
        x=json.loads(p.read_text())
        if any(x.get(k,0) for k in cfg['global_limits']) or x['pending_model'] or x['pending_transition']:raise ValueError('Reconcile partial newTrain costs')
    for name in ['generation_calls.jsonl','cases.jsonl']:
        if any(rows(p) for p in (r/'new_train').glob('worker*/'+name)):raise ValueError('Unexpected partial requests/data')
    cost=phase_cost(r);pre=json.loads((ROOT/'outputs/utility_expansion_v5_preflight/complete.json').read_text())
    paths=[p for p in r.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log'] and '/native/' not in str(p)]
    receipt=dict(schema='utility-v5-resource-recovery',prior_run=str(r.relative_to(ROOT)),
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in paths},completed_densification=done,prior_actual_cost=cost,
        preflight_cost={k:pre.get(k,0) for k in cfg['global_limits']},old_deadline_unix=m['deadline_unix'],
        remaining_limits={k:cfg['global_limits'][k]-cost.get(k,0)-pre.get(k,0) for k in cfg['global_limits']},
        new_train_requests=0,new_train_native_queries=0,unknown_new_outputs=0,
        root_cause='Physical GPU6 reused externally after phase release; worker1 OOM at model.to before generation/native transition.',
        new_train_failed_load_compute='Partial GPU loading consumed compute; precise failed-load time unavailable, do not claim zero compute.',
        retained_unknown_labels='140 whole-state budget-reservation Unknowns remain excluded from hard labels; not retried.',
        no_task_role_sampling_prompt_threshold_model_changes=True,registered_unix=time.time())
    if time.time()>=m['deadline_unix']:raise ValueError('Original72h expired; no reset')
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    cfg.update(schema='utility-v5-resource-recovery',gpus=[0,1,3,4],global_limits=receipt['remaining_limits'],
        original_deadline_unix=m['deadline_unix'],recovery_receipt=str(out.relative_to(ROOT)),original_run=str(r.relative_to(ROOT)),
        gpu_wait_seconds=7200,wall_seconds=m['deadline_unix']-time.time())
    (ROOT/'configs/evaluation/utility_expansion_v5_recovery.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(retained_task_operations=100,retained_cost=cost,remaining=receipt['remaining_limits'],hours_left=(m['deadline_unix']-time.time())/3600)))


if __name__=='__main__':main()
