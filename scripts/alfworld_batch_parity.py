"""Compare standalone isolated engine to official ALFWorld Gym wrapper through H50."""
import argparse,contextlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worldmodel.alfworld_environment import AlfEnvironment
from evaluation.independent_dev import atomic


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();out=Path(a.run)
    if out.exists():raise ValueError('Fresh parity output required')
    out.mkdir(parents=True);os.environ['ALFWORLD_DATA']=a.data_root
    from worldmodel.resources import configure_temporary_directory
    owned_temp=configure_temporary_directory(out)
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    m=json.loads((ROOT/'docs/alfworld_pilot/split_manifest.json').read_text());receipts=[];started=time.perf_counter()
    for row in m['calibration_tasks'][:3]:
        cfg=dict(dataset=dict(data_path=a.data_root+'/json_2.1.1/train',eval_id_data_path=a.data_root+'/json_2.1.1/valid_seen',num_train_games=-1,num_eval_games=-1),
            env=dict(domain_randomization=False,goal_desc_human_anns_prob=0.,task_types=[1,2,3,4,5,6],expert_type='handcoded'),
            general=dict(training_method='dagger'),dagger=dict(training=dict(max_nb_steps_per_episode=50)))
        with contextlib.redirect_stdout(sys.stderr):
            base=AlfredTWEnv(cfg,train_eval='eval_in_distribution');base.game_files=[str(Path(a.data_root)/row['game'])];base.num_games=1
            batch=base.init_env(1);batch.seed(row['seed']);obs,info=batch.reset()
        direct=AlfEnvironment(row,sys.executable,a.data_root,out/'native')
        try:
            assert obs[0]==direct.current['observation'] and info['admissible_commands'][0]==direct.current['admissible_commands']
            count=0
            for i in range(50):
                obs,reward,done,info=batch.step(['look']);actual=direct.step('look');count+=1
                assert actual['observation']==obs[0] and actual['reward']==float(reward[0]) and actual['done']==bool(done[0])
                assert actual['success']==bool(info['won'][0]) and actual['admissible_commands']==info['admissible_commands'][0]
            assert done[0] and not info['won'][0] and actual['truncated']
            receipts.append(dict(task_id=row['task_id'],comparisons=count,all_equal=True,horizon50_checked=True,expert_disabled=True))
        finally:batch.close();direct.close()
    atomic(out/'complete.json',dict(status='pass',cases=receipts,matched_transitions=sum(r['comparisons'] for r in receipts),seconds=time.perf_counter()-started))


if __name__=='__main__':main()
