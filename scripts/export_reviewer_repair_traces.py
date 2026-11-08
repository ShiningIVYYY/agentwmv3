#!/usr/bin/env python3
"""Export all156 genuine bounded Episode decisions, excluding activations/caches."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import read_rows,atomic
from experience.splits import file_hash
from scripts.export_phase57_evidence import compact


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output)
    assert json.loads((run/'complete.json').read_text())['episodes']==15
    if out.exists():raise ValueError('New trace file required')
    total=0
    with out.open('w') as f:
        for path in sorted((run/'episodes').rglob('steps.jsonl')):
            for t in read_rows(path):
                f.write(json.dumps(dict(reviewer_version=t['reviewer_version'],rollout_seed=t['rollout_seed'],**compact(t)),ensure_ascii=False)+'\n');total+=1
    assert total==156 and out.stat().st_size<1048576
    atomic(out.with_suffix('.provenance.json'),dict(script_sha256=file_hash(Path(__file__)),source_manifest_sha256=file_hash(run/'manifest.json'),decisions=total,output_sha256=file_hash(out)))
    print(total,out.stat().st_size)


if __name__=='__main__':main()
