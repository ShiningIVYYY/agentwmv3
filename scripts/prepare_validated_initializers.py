#!/usr/bin/env python3
"""Two independent real SDK collections per frozen development task, CPU only."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.initialization import collect
from experience.splits import file_hash


def main():
    out=ROOT/'outputs/phase55_setup/validated_initializers'
    if out.exists():raise ValueError('Frozen initializer directory already exists')
    out.mkdir();plan=json.loads((ROOT/'outputs/phase55_setup/plan.json').read_text())
    config=json.loads((ROOT/'configs/worldmodel/scienceworld.json').read_text());config['source_dir']=str(ROOT/'external/Word2World')
    results=[];began=time.perf_counter()
    for row in plan['tasks']:
        start=time.perf_counter();a=collect(config,row,ROOT);b=collect(config,row,ROOT)
        (out/(row['task_id']+'.attempt1.json')).write_text(json.dumps(a,indent=2)+'\n')
        (out/(row['task_id']+'.attempt2.json')).write_text(json.dumps(b,indent=2)+'\n')
        if a['verified_rooms']!=b['verified_rooms'] or [v['look'] for v in a['views']]!=[v['look'] for v in b['views']]:
            raise ValueError('Independent room observations differ; do not freeze unverified view facts')
        (out/(row['task_id']+'.json')).write_text(json.dumps(a,indent=2)+'\n')
        results.append(dict(task_id=row['task_id'],verified_rooms=a['verified_rooms'],excluded=a['excluded'],
            repeat_description_sha256_matches=a['description']==b['description'],room_observations_repeat_exact=True,
            possible_object_aliases_can_vary=True,frozen_description_choice='first actual verified collection, chosen before model evaluation; second retained without outcome selection',
            description_sha256=a['description_sha256'],seconds=time.perf_counter()-start,
            description_characters=len(a['description'])))
    (out/'summary.json').write_text(json.dumps(dict(tasks=results,seconds=time.perf_counter()-began,model_calls=0,
        source_hashes={p:file_hash(ROOT/p) for p in ('diagnostics/initialization.py','worldmodel/HeapSnapshot.java')},scope='development only; frozen actual WM-only helper observations'),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
