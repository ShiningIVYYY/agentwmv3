"""Actual WM HTTP observer. No mocks, retries, or predicted feedback relabeling."""
from contextlib import contextmanager
import json
import time
import requests


@contextmanager
def observe_wm(url, sink, decision_id):
    original=requests.sessions.Session.request;records=[];endpoint=url.rstrip("/")+"/v1/chat/completions"
    def observed(self,method,url,*args,**kwargs):
        if url.rstrip("/")!=endpoint:
            return original(self,method,url,*args,**kwargs)
        start=time.time_ns();wall=time.perf_counter()
        response=original(self,method,url,*args,**kwargs)
        data=response.json()
        record={"decision_id":decision_id(),"started_at_ns":start,"finished_at_ns":time.time_ns(),
            "seconds":time.perf_counter()-wall,"http_status":response.status_code,"usage":data.get("usage"),
            "request":kwargs.get("json"),"response":data}
        records.append(record)
        with sink.open("a") as f:f.write(json.dumps(record)+"\n")
        return response
    requests.sessions.Session.request=observed
    try:yield records
    finally:requests.sessions.Session.request=original
