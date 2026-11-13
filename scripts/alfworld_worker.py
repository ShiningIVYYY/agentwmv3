"""One owned native PDDL engine, no Expert wrapper; stdio public-state RPC."""
import argparse,contextlib,hashlib,importlib.metadata,json,os,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def emit(x):print(json.dumps(x,allow_nan=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--row-json',required=True);ap.add_argument('--temporary',required=True);a=ap.parse_args()
    env=None;steps=0;row=json.loads(a.row_json);root=Path(a.data_root);game=root/row['game']
    try:
        os.environ['TMPDIR']=a.temporary;Path(a.temporary).mkdir(parents=True,exist_ok=True)
        import tempfile;tempfile.tempdir=a.temporary
        import numpy as np,textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler,AlfredInfos
        from worldmodel.resources import install_pddl_cleanup
        install_pddl_cleanup()
        if sha(game)!=row['game_sha256']:raise ValueError('Game asset changed')
        versions={k:importlib.metadata.version(k) for k in ['alfworld','textworld']}
        if versions!={'alfworld':'0.4.2','textworld':'1.7.0'}:raise ValueError('Unpinned native environment')
        random.seed(row['seed']);np.random.seed(row['seed'])
        with contextlib.redirect_stdout(sys.stderr):
            env=textworld.start(str(game),request_infos=textworld.EnvInfos(won=True,admissible_commands=True,moves=True,facts=True),
                wrappers=[AlfredDemangler(shuffle=False),AlfredInfos])
            env.seed(row['seed']);state=env.reset()
        initial=str(state['feedback']);goal=initial.split('Your task is to: ',1)[1].strip()
        def transition(state,reward=0.,native_done=False,action=None):
            # Private symbolic facts are hashed ONLY for reconstruction checks.
            # Neither facts nor plans/goal predicates enter any model prompt.
            facts=sorted(str(f) for f in state['_facts'])
            private=hashlib.sha256(json.dumps(facts).encode()).hexdigest()
            won=bool(state['won']);native_moves=state.get('moves')
            return dict(observation=str(state['feedback']),admissible_commands=list(state['admissible_commands']),
                reward=float(reward),done=bool(native_done or steps>=50),native_done=bool(native_done),success=won,
                score=100.*won,moves=native_moves,formal_steps=steps,truncated=steps>=50 and not native_done,
                action=action,private_state_sha256=private)
        last=transition(state)
        emit(dict(ok=True,initial=last,goal=goal,pid=os.getpid(),versions=versions,expert_enabled=False))
        for line in sys.stdin:
            q=json.loads(line)
            if q['op']=='close':break
            if q['op']=='inspect':emit(dict(ok=True,transition=last,pid=os.getpid()));continue
            if q['op']!='step' or steps>=50:raise ValueError('Invalid operation/episode horizon')
            before=time.perf_counter();state,reward,done=env.step(q['action']);steps+=1
            last=transition(state,reward,done,q['action'])
            emit(dict(ok=True,transition=last,seconds=time.perf_counter()-before,pid=os.getpid()))
            if done: # Native requests after terminal state are prohibited.
                for remaining in sys.stdin:
                    op=json.loads(remaining)['op']
                    if op=='close':break
                    if op=='inspect':emit(dict(ok=True,transition=last,pid=os.getpid()));continue
                    raise ValueError('Step after native termination')
                break
    except BaseException as e:
        emit(dict(ok=False,error=type(e).__name__,message=str(e)));traceback.print_exc(file=sys.stderr)
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
