"""Create deterministic train-only native tasks for interface preflights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.matrix import read,save,sha

TYPES=('test-conductivity','test-conductivity-of-unknown-substances','use-thermometer',
       'measure-melting-point-known-substance','lifespan-longest-lived','find-living-thing')


def choose(training, games, excluded):
    ids=sorted({r['item_id'] for r in training},key=lambda s:int(s.rsplit('_',1)[1]))
    tasks=[]
    for name in TYPES:
        candidates=[i for i in ids if i not in excluded and int(i.rsplit('_',1)[1])<len(games)
                    and games[int(i.rsplit('_',1)[1])]['taskName']==name]
        if not candidates:
            raise RuntimeError('No train-only development task for '+name)
        task=candidates[0];index=int(task.rsplit('_',1)[1])
        tasks.append(dict(task_id=task,index=index,**games[index]))
    return tasks


def prepare(output):
    from scienceworld import ScienceWorldEnv
    source=ROOT/'data/paper_eval/audited/scienceworld_context_v2.json'
    train=ROOT/'data/paper_eval/sources/science_agenttraj_train.json'
    manifest=read(source)
    env=ScienceWorldEnv()
    try:
        excluded_categories={'5-1','5-2','9-1','9-2','9-3','10-1','10-2'}
        games=[dict(taskName=name,variationIdx=i) for key,name in env.tasks.items()
               if key not in excluded_categories for i in range(env.getMaxVariations(name))]
        # Establish mapping compatibility with all fixed evaluation IDs first.
        for row in manifest['tasks']:
            if games[row['index']] != {k:row[k] for k in ('taskName','variationIdx')}:
                raise RuntimeError('Native flattened task mapping changed')
        excluded={r['task_id'] for r in manifest['tasks']} | {'sciworld_4509'}
        tasks=choose(read(train),games,excluded)
        for row in tasks:
            env.load(row['taskName'],row['variationIdx'])
            row['task_description_sha256']=hashlib.sha256(env.getTaskDescription().encode()).hexdigest()
            _,_,done,info=env.step('look around')
            if done:raise RuntimeError('Development task already terminal')
            row['initial_score']=info['score']
        result=copy.deepcopy(manifest);result['tasks']=tasks
        save(output,result)
        save(output.with_name('dev_manifest_audit.json'),dict(
            passed=True,selection='first-numeric-training-ID-per-preregistered-family',
            training_source=str(train),training_sha256=sha(train),
            evaluation_manifest=str(source),evaluation_sha256=sha(source),
            task_ids=[r['task_id'] for r in tasks],test_overlap=[],
            demo_task_excluded='sciworld_4509',development_manifest_sha256=sha(output)))
        print(json.dumps(dict(passed=True,tasks=tasks)),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
