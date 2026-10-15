import json,time
from types import SimpleNamespace as NS
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch, xgrammar as xgr
from transformers import AutoTokenizer
from vllm.v1.structured_output import StructuredOutputManager
from reflection.agentwm import review_schema, SCIENCE_REPLAN_REGEX
root=Path(__file__).resolve().parents[1]
tok=AutoTokenizer.from_pretrained(root/'checkpoints/early-experience-alfworld/sr',local_files_only=True)
info=xgr.TokenizerInfo.from_huggingface(tok,vocab_size=152064)
compiler=xgr.GrammarCompiler(info)
rows=[]
cases=[
('alfworld',compiler.compile_json_schema(json.dumps(review_schema('alfworld','look',['look','inventory'])),any_whitespace=False),'{"assessment":"The proposed action gathers useful information.","action":"look","change":false,"reason":"Check the current room first."}'),
('scienceworld',compiler.compile_json_schema(json.dumps(review_schema('scienceworld','look around',[])),any_whitespace=False),'{"assessment":"Inspect the current room.","execute_proposed":true,"needs_replan":false,"suggested_action":"look around","reason":"Observe available objects."}'),
('science_replan',compiler.compile_regex(SCIENCE_REPLAN_REGEX),'<think>Inspect the room to find objects.</think>\n<action>look around</action>'.replace('\\n','\n'))]
class Grammar:
 def __init__(self,g): self.matcher=xgr.GrammarMatcher(g)
 def is_terminated(self): return self.matcher.is_terminated()
 def fill_bitmask(self,mask,idx): self.matcher.fill_next_token_bitmask(mask,idx)
for name,g,text in cases:
 if name != 'science_replan': text=json.dumps(json.loads(text))
 tokens=tok.encode(text,add_special_tokens=False)
 # Full generated paths, real StructuredOutputManager branch and mixed row masking.
 managers=[]
 for parallel in (False,True):
  m=StructuredOutputManager.__new__(StructuredOutputManager)
  m.vllm_config=NS(speculative_config=None)
  m.reasoner=None
  m.fill_bitmask_parallel_threshold=1 if parallel else 128
  m.fill_bitmask_parallel_batch_size=1
  m.executor_for_fillmask=ThreadPoolExecutor(max_workers=4)
  m._grammar_bitmask=xgr.allocate_token_bitmask(4,152064)
  m._full_mask=torch.tensor(-1,dtype=torch.int32)
  req={str(i):NS(structured_output_request=NS(grammar=Grammar(g))) for i in range(4)}
  managers.append((m,req))
 times=[0.,0.]
 for token in tokens:
  masks=[]
  for idx,(m,req) in enumerate(managers):
   start=time.perf_counter()
   mask=m.grammar_bitmask(req,{str(i):i for i in range(4)},{})
   times[idx]+=time.perf_counter()-start
   masks.append(mask.copy())
  assert (masks[0]==masks[1]).all(),(name,token)
  for m,req in managers:
   for r in req.values(): assert r.structured_output_request.grammar.matcher.accept_token(token),(name,token)
 for m,req in managers: m.executor_for_fillmask.shutdown()
 rows.append(dict(case=name,tokens=len(tokens),requests=4,serial_seconds=times[0],parallel_seconds=times[1],speedup=times[0]/times[1],masks_identical=True))
 print(json.dumps(rows[-1]),flush=True)
(root/'outputs/wm_reflection_setup/sr_decode_parallel/validation.json').write_text(json.dumps(rows,indent=2))
