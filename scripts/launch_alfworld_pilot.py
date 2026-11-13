"""Owned-process supervisor and wall cap. Never stop another experiment."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic


def native_children(run):
    result=[]
    for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
        parts=line.strip().split(None,1)
        if len(parts)==2 and str(ROOT/'scripts/alfworld_worker.py') in parts[1] and str(run/'native') in parts[1]:result.append(int(parts[0]))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',default='1');ap.add_argument('--run',required=True);ap.add_argument('--env-python',required=True);ap.add_argument('--data-root',required=True);a=ap.parse_args()
    run=Path(a.run).absolute()
    if run.exists():raise ValueError('Never overwrite a previous experiment')
    free=int(subprocess.check_output(['nvidia-smi','--id='+a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<22000:raise ValueError('Insufficient free GPU memory; no other job is displaced')
    p=json.loads((ROOT/'configs/evaluation/alfworld_pilot_v1.json').read_text());deadline=time.time()+p['limits']['wall_seconds']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    args=[sys.executable,'-u',str(ROOT/'scripts/run_alfworld_pilot.py'),'--run',str(run),'--env-python',a.env_python,'--data-root',a.data_root,
        '--memory',str(ROOT/'outputs/alfworld_memory_v2/snapshot.json'),'--native-validation',str(ROOT/'outputs/alfworld_inputs_validation_v1')]
    log=Path(str(run)+'.log').open('ab');child=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=Path(str(run)+'-supervisor.json');started=time.perf_counter();error=None
    try:
        while child.poll() is None:
            if time.time()>deadline:raise TimeoutError('Whole-run2h wall limit')
            atomic(receipt,dict(stage='running',pid=child.pid,gpu=a.gpu,deadline=deadline,seconds=time.perf_counter()-started));time.sleep(5)
        if child.returncode!=0:raise RuntimeError('Model worker failed; inspect raw error/receipts before retry')
        if not (run/'complete.json').exists():raise RuntimeError('Missing complete receipt')
    except BaseException as e:error=str(e);raise
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        residual=native_children(run)
        for pid in residual:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        log.close();atomic(receipt,dict(stage='error' if error else 'complete',error=error,model_pid=child.pid,gpu=a.gpu,seconds=time.perf_counter()-started,
            model_exited=child.poll() is not None,owned_native_cleanup_pids=residual))


if __name__=='__main__':main()
