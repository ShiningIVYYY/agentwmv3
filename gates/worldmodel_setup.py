"""Exact existing WM context collector in its own SDK environment; local token count."""
import json
from pathlib import Path
import subprocess
import time
from transformers import AutoTokenizer


class ExistingWorldModelSetup:
    def __init__(self, science_python, root, output):
        self.python,self.root,self.output=science_python,Path(root),Path(output)
        self.tokenizers={}
        self.costs=[]

    def count(self, config, messages):
        checkpoint=config["checkpoint"]
        if checkpoint not in self.tokenizers:
            self.tokenizers[checkpoint]=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        ids=self.tokenizers[checkpoint].apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        return len(ids)

    def context(self, config, row, task, initial):
        start=time.perf_counter()
        directory=self.output/("wm_context_"+row["task_id"]);directory.mkdir()
        source=directory/"input.json";dest=directory/"context.json"
        source.write_text(json.dumps({"config":config,"row":row,"task":task,"initial":initial}))
        code=("import sys,json;sys.path.insert(0,sys.argv[1]);"
              "from worldmodel.contexts import initial_context;"
              "x=json.load(open(sys.argv[2]));"
              "v=initial_context(x['config'],x['row'],x['task'],x['initial']);"
              "json.dump(v,open(sys.argv[3],'w'))")
        result=subprocess.run([self.python,"-c",code,str(self.root),str(source),str(dest)],
            capture_output=True,text=True,cwd=str(self.root),timeout=240)
        (directory/"collector.log").write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError("Original WM initialization collector failed; see "+str(directory))
        value=json.loads(dest.read_text())
        self.costs.append({"task_id":row["task_id"],"seconds":time.perf_counter()-start,
            "characters":len(value),"only_visible_to_wm":True,"method":"unchanged_worldmodel.contexts.initial_context"})
        return value
