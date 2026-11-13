"""Separate native PDDL processes; replay verified before one candidate step."""
import copy,hashlib,json,os,selectors,shutil,signal,subprocess,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def fingerprint(state):return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()


class AlfEnvironment:
    def __init__(self,row,python,data_root,work):
        self.row=row;self.python=str(python);self.data_root=str(data_root);self.work=Path(work)
        self.owned=self.work/('native-'+uuid.uuid4().hex);self.owned.mkdir(parents=True)
        self.capture_count=0;self.history=[];self.steps=0;self.p=None
        self.stderr=(self.owned/'stderr.log').open('ab')
        try:
            env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT))
            self.p=subprocess.Popen([self.python,'-u',str(ROOT/'scripts/alfworld_worker.py'),'--data-root',self.data_root,
                '--row-json',json.dumps(row),'--temporary',str(self.owned/'tmp')],env=env,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.stderr,text=True,bufsize=1,start_new_session=True)
            initial=self.receive();self.pid=initial['pid'];self.goal=initial['goal'];self.versions=initial['versions']
            if initial['expert_enabled'] is not False:raise ValueError('Expert wrapper forbidden')
            self.initial=initial['initial'];self.current=copy.deepcopy(self.initial)
        except BaseException:self.close();raise
    def receive(self):
        selector=selectors.DefaultSelector();selector.register(self.p.stdout,selectors.EVENT_READ)
        try:
            if not selector.select(90):raise TimeoutError('Owned ALFWorld worker reply timeout')
            line=self.p.stdout.readline()
        finally:selector.close()
        if not line:raise RuntimeError('Owned ALFWorld worker exited without receipt')
        row=json.loads(line)
        if row.get('ok') is not True:raise RuntimeError('ALFWorld native error: '+json.dumps(row))
        self.capture_count+=1;return row
    def request(self,value):
        self.p.stdin.write(json.dumps(value)+'\n');self.p.stdin.flush();return self.receive()
    def capture(self):return self.request(dict(op='inspect'))['transition']
    def step(self,action):
        if self.current['done'] or self.steps>=50:raise ValueError('Formal step beyond native/horizon termination')
        if not isinstance(action,str) or not action:raise ValueError('Empty native action')
        result=self.request(dict(op='step',action=action))['transition'];self.steps+=1
        self.history.append((action,copy.deepcopy(result)));self.current=result;return copy.deepcopy(result)
    def close(self):
        if self.p is not None:
            if self.p.poll() is None:
                try:self.p.stdin.write('{"op":"close"}\n');self.p.stdin.flush();self.p.wait(timeout=10)
                except (BrokenPipeError,subprocess.TimeoutExpired):
                    os.killpg(self.p.pid,signal.SIGTERM)
                    try:self.p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait(timeout=5)
            if self.p.stdin:self.p.stdin.close()
            if self.p.stdout:self.p.stdout.close()
        if getattr(self,'stderr',None):self.stderr.close()
        # The DSO is unmapped by process exit before its uniquely owned temp tree is removed.
        if self.owned.exists():
            for child in [self.owned/'tmp']:
                if child.exists():shutil.rmtree(child)


class AlfGroundTruthWM:
    def simulate(self,formal,action):
        start=time.perf_counter();branch=None
        cost=dict(initializations=0,replay_steps=0,candidate_steps=0,state_capture_bundles=0,formal_inspections=0)
        before=formal.capture();cost['formal_inspections']+=1
        try:
            init=time.perf_counter();cost['initializations']=1
            branch=AlfEnvironment(formal.row,formal.python,formal.data_root,formal.work)
            cost.update(initialization_seconds=time.perf_counter()-init,initialization_receipt_known=True)
            if branch.pid==formal.pid or branch.initial!=formal.initial:raise ValueError('Independent reset mismatch')
            replay_start=time.perf_counter()
            for old_action,expected in formal.history:
                actual=branch.step(old_action);cost['replay_steps']+=1
                if actual!=expected:raise ValueError('Replay public/fact-state mismatch')
            cost['replay_seconds']=time.perf_counter()-replay_start
            if branch.current!=before:raise ValueError('Rebuilt current state differs')
            candidate_start=time.perf_counter();actual=branch.step(action);cost['candidate_steps']=1
            cost['candidate_seconds']=time.perf_counter()-candidate_start
            answer=dict(status='ok',prediction_available=True,source='ground_truth_environment',benchmark='alfworld',
                predicted_observation=actual['observation'],predicted_reward=actual['reward'],predicted_done=actual['done'],
                predicted_score=actual['score'],predicted_success=actual['success'],candidate_transition=actual,
                candidate_state_sha256=fingerprint(actual),verified_prefix_steps=len(formal.history),formal_pid=formal.pid,worker_pid=branch.pid)
        except Exception as e:
            answer=dict(status='reconstruction_failed',prediction_available=False,source='ground_truth_environment',benchmark='alfworld',error=str(e),formal_pid=formal.pid)
        finally:
            if branch is not None:
                cost['state_capture_bundles']=branch.capture_count;cleanup=time.perf_counter();branch.close();cost['cleanup_seconds']=time.perf_counter()-cleanup
        after=formal.capture();cost['formal_inspections']+=1
        if after!=before:raise RuntimeError('Formal ALFWorld state polluted by isolated trial')
        answer.update(formal_unchanged=True,available_at_ns=time.time_ns(),simulation_latency=time.perf_counter()-start,simulation_cost=cost)
        return answer
