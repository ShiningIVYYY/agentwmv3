"""Real independent-process comparisons; no model or benchmark-result claims."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worldmodel.alfworld_environment import AlfEnvironment,AlfGroundTruthWM
from evaluation.independent_dev import atomic,append


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env-python',required=True);ap.add_argument('--data-root',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    out=Path(a.run)
    if out.exists():raise ValueError('Fresh validation ID required')
    out.mkdir(parents=True);m=json.loads((ROOT/'docs/alfworld_pilot/split_manifest.json').read_text());records=[];started=time.perf_counter()
    for original in m['calibration_tasks'][:3]:
        for offset in [0,1]:
            for length in [0,3,6]:
                row=dict(original,seed=original['seed']+offset);env=AlfEnvironment(row,a.env_python,a.data_root,out/'native')
                try:
                    prior=None
                    for i in range(length):
                        action=env.current['admissible_commands'][0] if i%3==0 else 'look' if i%3==1 else prior
                        env.step(action);prior=action
                    action=env.current['admissible_commands'][0];before=env.capture();sim=AlfGroundTruthWM().simulate(env,action)
                    assert sim['prediction_available'] and sim['formal_unchanged'] and env.capture()==before
                    actual=env.step(action);assert actual==sim['candidate_transition']
                    record=dict(task_id=row['task_id'],seed=row['seed'],prefix_length=length,action=action,
                        formal_pid=env.pid,branch_pid=sim['worker_pid'],matched=True,simulation=sim,actual=actual)
                    append(out/'comparisons.jsonl',record);records.append(record)
                finally:env.close()
    atomic(out/'complete.json',dict(status='pass',comparisons=len(records),tasks=3,seeds_per_task=2,prefix_lengths=[0,3,6],
        independent_pids=True,public_transition_and_private_fact_digest_equal=True,full_rng_serialization_claim=False,
        candidate_actions_executed_in_formal_for_controlled_validation_only=True,seconds=time.perf_counter()-started))


if __name__=='__main__':main()
