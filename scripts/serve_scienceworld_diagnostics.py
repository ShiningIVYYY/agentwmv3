#!/usr/bin/env python3
"""Development-only native service: same scoring, additional causal/snapshot receipts."""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import importlib.metadata
import json
from pathlib import Path
import sys
from urllib.parse import urlparse,parse_qs
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.serve_scienceworld_profile import ProfileEnvironment
from worldmodel.science_snapshot import load_bridge
from scienceworld.constants import JAR_PATH


class DiagnosticEnvironment(ProfileEnvironment):
    def __init__(self):super().__init__(20);self.snapshots={};self.bridges={};self.counter=0
    def _present(self,result,idx,reset=False):
        value=super()._present(result,idx,reset)
        if 'error' not in value:
            sdk=self.env[idx];moves=sdk.get_num_moves();completed=bool(sdk.server.getCompleted())
            value.update(simulator_moves=moves,simulator_completed=completed,
                sdk_clock_limit_exceeded=moves>sdk.envStepLimit,negative_score_termination=value['score']<0,
                sdk_done_definition='server_completed OR simulator_moves>envStepLimit OR native_score<0')
            self.info[idx].update(value)
        return value
    def checkpoint(self,idx):
        if idx not in self.bridges:self.bridges[idx]=load_bridge(self.env[idx],ROOT)
        bridge,_=self.bridges[idx];sdk=self.env[idx]
        before=dict(moves=sdk.get_num_moves(),score=float(sdk.server.getScore()),completed=bool(sdk.server.getCompleted()))
        python={k:copy.deepcopy(v) for k,v in sdk.__dict__.items() if k not in ('_gateway','server','_obj_tree_tempdir')}
        heap=bridge.take(sdk.server);key=str(self.counter);self.counter+=1
        self.snapshots[key]=(idx,heap,python,copy.deepcopy(self.info[idx]),self.counts[idx],before)
        return dict(snapshot_id=key,heap_fields=heap.objectFields(),heap_arrays=heap.arrayCount(),
            heap_matches=heap.matches(),actor_steps=self.counts[idx],native=before)
    def restore(self,idx,key):
        owner,heap,python,info,count,before=self.snapshots[key]
        if owner!=idx:raise ValueError('Snapshot belongs to another live JVM')
        sdk=self.env[idx];heap.restore()
        for k in list(sdk.__dict__):
            if k not in ('_gateway','server','_obj_tree_tempdir') and k not in python:del sdk.__dict__[k]
        sdk.__dict__.update(copy.deepcopy(python));self.info[idx]=copy.deepcopy(info);self.counts[idx]=count
        restored=dict(moves=sdk.get_num_moves(),score=float(sdk.server.getScore()),completed=bool(sdk.server.getCompleted()))
        if not heap.matches() or before!=restored:raise RuntimeError('Snapshot state verification failed')
        return dict(self.info[idx],restore_provenance=dict(snapshot_id=key,heap_matches=True,
            actor_steps=count,native=restored,python_reward_state_restored=True))
    def close(self,idx):
        for key in list(self.snapshots):
            if self.snapshots[key][0]==idx:del self.snapshots[key]
        if idx in self.bridges:self.bridges.pop(idx)[1].close()
        return super().close(idx)


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=36055);a=p.parse_args();env=DiagnosticEnvironment()
    metadata=dict(profile='paper-aligned-scienceworld-v1',horizon=20,envStepLimit=21,
        scienceworld_version=importlib.metadata.version('scienceworld'),jar_sha256=hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest(),
        diagnostic='phase55-receipts-and-live-heap-snapshots',snapshot_for_posthoc_branches_only=True)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):self.dispatch()
        def do_POST(self):self.dispatch()
        def dispatch(self):
            try:
                path=urlparse(self.path);data=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}') if self.command=='POST' else {k:v[0] for k,v in parse_qs(path.query).items()}
                idx=int(data['id']) if 'id' in data else None
                if path.path=='/profile':v=dict(metadata,open_sessions=len(env.env))
                elif path.path=='/create':v=env.create()
                elif path.path=='/reset':v=env.reset(idx,int(data['data_idx']))
                elif path.path=='/step':v=env.step(idx,data['action'])
                elif path.path=='/close':v=env.close(idx)
                elif path.path=='/admissible_actions':v=env.get_admissible_actions(idx)
                elif path.path=='/action_hint':v=env.get_action_hint(idx)
                elif path.path=='/checkpoint':v=env.checkpoint(idx)
                elif path.path=='/restore':v=env.restore(idx,data['snapshot_id'])
                else:raise ValueError('Unsupported endpoint')
                encoded=json.dumps(v,allow_nan=False).encode();self.send_response(200)
            except Exception as error:encoded=json.dumps({'error':repr(error)}).encode();self.send_response(500)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    server=HTTPServer(('127.0.0.1',a.port),Handler)
    try:server.serve_forever()
    finally:
        for idx in list(env.env):env.close(idx)
        server.server_close()


if __name__=='__main__':main()
