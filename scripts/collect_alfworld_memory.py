"""Small public-observation probes on reserved TRAIN games, never Oracle actions."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from worldmodel.alfworld_environment import AlfEnvironment
from experience.alfworld_memory import synthesize
from experience.splits import file_hash


def collect(row,python,data_root,work,out):
    env=AlfEnvironment(row,python,data_root,work);cases=[];queries=1;episode_id=out.name+'/'+row['task_id'];goal=env.goal
    try:
        def step(action,kind=None):
            nonlocal queries
            before=env.current.copy();decision=time.time_ns();actual=env.step(action);queries+=1
            c=dict(case_id=episode_id+'/'+str(env.steps),benchmark='alfworld',task_id=row['task_id'],goal_group=row['goal_group'],episode_id=episode_id,
                source_split='train',goal=goal,observation=before['observation'],draft_action=action,executed_action=action,
                environment_feedback=actual['observation'],reward=actual['reward'],task_score=actual['score'],done=actual['done'],success=actual['success'],
                wm_called=False,source_trajectory_id=episode_id,step_index=env.steps,decision_at_ns=decision,outcome_available_at_ns=time.time_ns(),probe_kind=kind)
            cases.append(c);append(out/'cases.jsonl',c);return actual
        places=[s for s in env.current['admissible_commands'] if s.startswith('go to ') and any(k in s for k in ['fridge','microwave','cabinet','drawer'])]
        places=sorted(places,key=lambda s:([k for k in ['fridge','microwave','cabinet','drawer'] if k in s][0],s))
        repeat_done=closed_done=False
        for go in places[:8]:
            if env.steps>=25 or env.current['done'] or (repeat_done and closed_done):break
            step(go)
            opens=[c for c in env.current['admissible_commands'] if c.startswith('open ')]
            if not opens:continue
            op=opens[0];container=op[5:];step(op)
            if not repeat_done:step(op,'repeat_open');repeat_done=True
            takes=[c for c in env.current['admissible_commands'] if c.startswith('take ') and c.endswith(' from '+container)]
            close='close '+container
            if not closed_done and takes and close in env.current['admissible_commands']:
                take=takes[0];step(close);step(take,'closed_take');closed_done=True
        return cases,dict(task_id=row['task_id'],episode_id=episode_id,steps=env.steps,reset_queries=1,step_queries=env.steps,state_captures=env.capture_count,
            repeat_open_case=repeat_done,closed_take_case=closed_done,native_pid=env.pid)
    finally:env.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env-python',required=True);ap.add_argument('--data-root',required=True);ap.add_argument('--run',required=True);ap.add_argument('--reuse');a=ap.parse_args()
    out=Path(a.run)
    if out.exists():raise ValueError('New training collection Run ID required')
    out.mkdir(parents=True);manifest=json.loads((ROOT/'docs/alfworld_pilot/split_manifest.json').read_text());cases=[];cost=[];started=time.perf_counter();reused=set()
    if a.reuse:
        source=Path(a.reuse);old=json.loads((source/'snapshot.json').read_text());allowed={r['task_id']:r for r in manifest['memory_tasks']}
        for c in old['cases']:
            if c['task_id'] not in allowed or c['goal_group']!=allowed[c['task_id']]['goal_group']:raise ValueError('Unapproved prior training case')
            cases.append(c);append(out/'cases.jsonl',c);reused.add(c['task_id'])
        cost=json.loads((source/'complete.json').read_text())['cost']
        atomic(out/'reuse_provenance.json',dict(source=str(source),snapshot_sha256=file_hash(source/'snapshot.json'),cases=len(cases),source_task_ids=sorted(reused)))
    for row in manifest['memory_tasks']:
        if row['task_id'] in reused:continue
        cs,receipt=collect(row,a.env_python,a.data_root,out/'native',out);cases+=cs;cost.append(receipt)
    snapshot=synthesize(cases,manifest['memory_tasks']);atomic(out/'snapshot.json',snapshot)
    atomic(out/'complete.json',dict(cases=len(cases),cards=len(snapshot['cards']),cost=cost,total_seconds=time.perf_counter()-started,model_calls=0,tokens=0))
    print(json.dumps(dict(cases=len(cases),cards=len(snapshot['cards']),cost=cost),indent=2))


if __name__=='__main__':main()
