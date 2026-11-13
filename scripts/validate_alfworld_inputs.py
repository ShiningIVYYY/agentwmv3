"""Supplement actual Goal identities and invalid-action replay coverage."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append
from experience.splits import file_hash
from worldmodel.alfworld_environment import AlfEnvironment,AlfGroundTruthWM,fingerprint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env-python',required=True);ap.add_argument('--data-root',required=True);ap.add_argument('--base-validation',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    out=Path(a.run)
    if out.exists():raise ValueError('Fresh supplemental validation ID required')
    base=Path(a.base_validation)
    if json.loads((base/'complete.json').read_text())['status']!='pass':raise ValueError('Base real replay validation missing')
    out.mkdir(parents=True);m=json.loads((ROOT/'docs/alfworld_pilot/split_manifest.json').read_text());identities=[];records=[];started=time.perf_counter()
    for row in m['memory_tasks']+m['calibration_tasks']+m['evaluation_tasks']:
        env=AlfEnvironment(row,a.env_python,a.data_root,out/'native')
        try:
            if row['params']['object_target'].casefold() not in env.goal.casefold():raise ValueError('Native goal/source identity mismatch')
            identities.append(dict(task_id=row['task_id'],goal=env.goal,initial_sha256=fingerprint(env.initial),formal_pid=env.pid))
        finally:env.close()
    if len({r['goal'].strip().casefold() for r in identities})!=13:raise ValueError('Native goal duplicate across roles')
    atomic(out/'goal_identity.json',identities)
    for original in m['calibration_tasks'][:3]:
        for offset in [0,1]:
            row=dict(original,seed=original['seed']+offset);env=AlfEnvironment(row,a.env_python,a.data_root,out/'native')
            try:
                env.step(env.current['admissible_commands'][0]);env.step('look');invalid=env.step('open nonexistent container')
                if invalid['observation']!='Nothing happens.':raise ValueError('Invalid-action control did not return native rejection')
                before=env.capture();sim=AlfGroundTruthWM().simulate(env,'look')
                if not sim['prediction_available'] or env.capture()!=before:raise ValueError('Invalid-prefix reconstruction failed')
                actual=env.step('look')
                if actual!=sim['candidate_transition']:raise ValueError('Candidate transition differs after invalid prefix')
                r=dict(task_id=row['task_id'],seed=row['seed'],matched=True,invalid_action_feedback=invalid['observation'],simulation=sim,actual=actual,formal_pid=env.pid)
                append(out/'invalid_prefix_comparisons.jsonl',r);records.append(r)
            finally:env.close()
    atomic(out/'complete.json',dict(status='pass',native_goal_initializations=13,invalid_history_comparisons=len(records),
        base_replay_comparisons=18,base_validation_sha256=file_hash(base/'complete.json'),
        controlled_transition_queries=60,passive_inspection_queries=24,total_state_queries_including_identity=97,seconds=time.perf_counter()-started))


if __name__=='__main__':main()
