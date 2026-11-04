#!/usr/bin/env python3
"""Owned HTTP adapter over EXISTING AgentGym wrapper with paper horizon/presentation.

Run under the existing ScienceWorld SDK Python. Native reward/score retained;
the paper's normalized training reward is never substituted for evaluation.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import importlib.metadata
import json
from pathlib import Path
import sys
from urllib.parse import urlparse,parse_qs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.profile import canonicalize_observation
from agentenv_sciworld.environment import SciWorldEnv
from scienceworld.constants import JAR_PATH


class ProfileEnvironment(SciWorldEnv):
    def __init__(self,horizon=20):
        super().__init__();self.horizon=horizon;self.counts={}
    def create(self):
        result=super().create()
        if "error" not in result:
            # SDK step() reads this public attribute; no action occurred yet.
            self.env[result["id"]].envStepLimit=self.horizon+1
            self.counts[result["id"]]=0
        return result
    def _present(self,result,idx,reset=False):
        if "error" in result:return result
        raw=result["observation"];native=bool(result["done"])
        n=self.counts[idx]
        value=dict(result,native_observation=raw,observation=canonicalize_observation(raw),
            native_done=native,horizon_truncated=(not native and n>=self.horizon),
            actor_interaction_steps=n,environment_step_limit=self.env[idx].envStepLimit,
            done=native or n>=self.horizon)
        self.info[idx].update(value)
        return value
    def reset(self,idx,data_idx):
        self.counts[idx]=0
        return self._present(super().reset(idx,data_idx),idx,True)
    def step(self,idx,action):
        if self.counts[idx]>=self.horizon:return {"error":"Episode horizon exhausted; reset required"}
        result=super().step(idx,action)
        if "error" not in result:self.counts[idx]+=1
        return self._present(result,idx)
    def close(self,idx):
        result=super().close(idx)
        self.env.pop(idx,None);self.info.pop(idx,None);self.counts.pop(idx,None)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--port",type=int,default=36051);p.add_argument("--horizon",type=int,default=20);a=p.parse_args()
    if a.horizon!=20:raise ValueError("The independent paper profile uses H20")
    env=ProfileEnvironment(a.horizon)
    metadata={"profile":"paper-aligned-scienceworld-v1","horizon":20,"envStepLimit":21,
        "scienceworld_version":importlib.metadata.version("scienceworld"),
        "jar_sha256":hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest(),
        "base_wrapper_file":sys.modules[SciWorldEnv.__module__].__file__,"observation_format":"canonical-flat-object-lists-v1"}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):self.dispatch()
        def do_POST(self):self.dispatch()
        def dispatch(self):
            try:
                path=urlparse(self.path);data=json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))) or b"{}") if self.command=="POST" else {k:v[0] for k,v in parse_qs(path.query).items()}
                idx=int(data["id"]) if "id" in data else None
                if path.path=="/profile":value=dict(metadata,open_sessions=len(env.env))
                elif path.path=="/create":value=env.create()
                elif path.path=="/reset":value=env.reset(idx,int(data["data_idx"]))
                elif path.path=="/step":value=env.step(idx,data["action"])
                elif path.path=="/close":value=env.close(idx)
                elif path.path=="/admissible_actions":value=env.get_admissible_actions(idx)
                elif path.path=="/action_hint":value=env.get_action_hint(idx)
                else:raise ValueError("Unsupported endpoint")
                encoded=json.dumps(value,allow_nan=False).encode();self.send_response(200)
            except Exception as error:
                encoded=json.dumps({"error":str(error)}).encode();self.send_response(500)
            self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    server=HTTPServer(("127.0.0.1",a.port),Handler)
    try:server.serve_forever()
    finally:
        for idx in list(env.env):env.close(idx)
        server.server_close()


if __name__=="__main__":main()
