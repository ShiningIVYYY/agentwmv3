"""Freeze the real V2 corpus and correct derived labels without changing raw logs."""
import argparse,json,sys,time
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic
from experience.splits import file_hash
from evaluation.utility_labels_v3 import relabel,SCHEMA
from scripts.run_utility_sampling import check


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def main():
    start=time.perf_counter();dest=ROOT/'docs/utility_expansion_v3';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'prior_corpus.json').exists():raise ValueError('Never overwrite imported-corpus receipt')
    prior=ROOT/'outputs/utility_sampling_v2';m=read(prior/'manifest.json');check(m)
    if read(prior/'progress.json')['stage']!='complete':raise ValueError('Prior must be complete')
    split=read(ROOT/'docs/activation_v1/split_manifest.json');allowed={r['task_id']:r for r in split['partitions']['probe_train']}
    cases=[c for p in sorted(prior.glob('train_pool/worker*/cases.jsonl')) for c in rows(p)]
    labels=[c for p in sorted(prior.glob('train_pool/worker*/state_labels.jsonl')) for c in rows(p)]
    ids={c['case_id']:c for c in cases}
    if len(ids)!=len(cases):raise ValueError('Duplicate old case identity')
    for c in cases:
        if c['role']!='probe_train' or c['task_id'] not in allowed or c['goal_group']!=allowed[c['task_id']]['goal_group']:raise ValueError('Prior role/goal leakage')
        if not c['feature_available_at_ns']<c['label_observed_at_ns']:raise ValueError('Prior future leakage')
    corrected=[];changes=[]
    for l in labels:
        source=ids[l['case_id']]
        if any(source[k]!=l[k] for k in ['role','goal_group','feature_file','state_sha256']):raise ValueError('Prior source mismatch')
        c=relabel(l);corrected.append(c)
        if c['label']!=l['label']:changes.append(dict(case_id=l['case_id'],deltas=l['deltas'],old_label=l['label'],corrected_label=c['label']))
    files={}
    for p in sorted(prior.glob('train_pool/worker*/*')):
        if p.is_file() and p.suffix in ['.json','.jsonl']:files[str(p.relative_to(ROOT))]=file_hash(p)
    for name in ['manifest.json','summary.json','progress.json','feasibility.json']:files[str((prior/name).relative_to(ROOT))]=file_hash(prior/name)
    feature_hashes={c['feature_file']:file_hash(ROOT/c['feature_file']) for c in cases}
    for p in prior.glob('train_pool/worker*/ledger.json'):
        ledger=read(p);calls=rows(p.parent/'generation_calls.jsonl')
        if ledger['pending_model'] or ledger['pending_transition']:raise ValueError('Pending prior cost')
        if len(calls)!=ledger['model_calls'] or sum(c['metadata']['usage']['total_tokens'] for c in calls)!=ledger['tokens']:raise ValueError('Prior journal mismatch')
    data=dict(schema='utility-v3-prior-corpus',prior_run=str(prior.relative_to(ROOT)),prior_source_commit=m['source_commit'],
        source_hashes=files,feature_hashes=feature_hashes,cases=len(cases),state_labels=len(labels),training_tasks=len(allowed),
        original_label_counts=dict(Counter(l['label'] for l in labels)),corrected_label_counts=dict(Counter(l['label'] for l in corrected)),
        label_schema=SCHEMA,corrections=changes,old_help_preserved=sum(l['label']=='help' for l in labels)==sum(l['label']=='help' for l in corrected),
        prior_cost=read(prior/'summary.json')['cost'],seconds=time.perf_counter()-start,raw_prior_files_modified=False)
    atomic(dest/'prior_corpus.json',data);atomic(dest/'label_corrections.json',dict(schema=SCHEMA,changes=changes,original_counts=data['original_label_counts'],corrected_counts=data['corrected_label_counts']))
    print(json.dumps({k:data[k] for k in ['cases','state_labels','original_label_counts','corrected_label_counts','old_help_preserved','seconds']},indent=2))


if __name__=='__main__':main()
