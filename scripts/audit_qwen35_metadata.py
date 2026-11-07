#!/usr/bin/env python3
"""Read official small metadata only. Never download weights or run a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[1]


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'agentwmv3-phase57-metadata-audit'}),timeout=25) as response:
        data=response.read(5_000_001)
    if len(data)>5_000_000:raise ValueError('Metadata size limit exceeded')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/phase57_setup/qwen35_metadata')
    parser.add_argument('--hf-python',default=str(ROOT/'.venv-spanuq/bin/python'))
    parser.add_argument('--legacy-python',help='Optional existing legacy environment, metadata inspection only')
    parser.add_argument('--cache-root',action='append',default=[])
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    info=json.loads(fetch('https://huggingface.co/api/models/Qwen/Qwen3.5-9B'))
    revision=info['sha'];files={s['rfilename'] for s in info.get('siblings',[])};artifacts={}
    for name in ('config.json','generation_config.json','tokenizer_config.json','chat_template.jinja','README.md'):
        if name not in files:continue
        url='https://huggingface.co/Qwen/Qwen3.5-9B/resolve/'+revision+'/'+quote(name)
        data=fetch(url);(out/name).write_bytes(data);artifacts[name]=dict(url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    config=json.loads((out/'config.json').read_text());text=config.get('text_config',{})
    cache_roots={str(Path.home()/'.cache/huggingface/hub'),str(ROOT/'checkpoints'),*args.cache_root}
    for variable in ('HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE'):
        value=os.environ.get(variable)
        if value:cache_roots.add(str(Path(value)/'hub') if variable=='HF_HOME' else value)
    caches=[]
    for name in sorted(cache_roots):
        path=Path(name)
        matches=[str(p) for p in path.iterdir() if 'qwen3.5-9b' in p.name.casefold()] if path.is_dir() else []
        snapshots=[]
        for match in matches:
            for snapshot in (Path(match)/'snapshots').glob('*'):
                entries=list(snapshot.iterdir())
                snapshots.append(dict(revision=snapshot.name,files=sorted(f.name for f in entries),
                    present_bytes=sum(f.stat().st_size for f in entries if f.is_file()),
                    weight_files=sorted(f.name for f in entries if f.suffix in ('.safetensors','.bin'))))
        caches.append(dict(root=name,exists=path.exists(),matching_entries=matches,snapshots=snapshots))
    packages={}
    # Distribution names may not equal import names; use explicit exception handling.
    code='''import json,sys,importlib.metadata as m
x={}
for p in ("torch","transformers","vllm"):
 try:x[p]=m.version(p)
 except m.PackageNotFoundError:x[p]=None
print(json.dumps({"python":sys.version,"packages":x}))'''
    environments=[('accepted_hf',args.hf_python)]
    if args.legacy_python:environments.append(('legacy_vllm',args.legacy_python))
    for label,python in environments:
        p=subprocess.run([python,'-c',code],capture_output=True,text=True,timeout=30)
        packages[label]=json.loads(p.stdout) if p.returncode==0 else dict(error=p.stderr[-700:])
    q=subprocess.run([args.hf_python,'-c','from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES;import json;print(json.dumps({k:CONFIG_MAPPING_NAMES.get(k) for k in ("qwen3_5","qwen3_5_text")}))'],capture_output=True,text=True,timeout=30)
    report=dict(model_id='Qwen/Qwen3.5-9B',metadata_revision=revision,metadata_only=True,weights_downloaded=False,
        real_generation_verified=False,token_logprobs_verified=False,thresholds_calibrated=False,
        architectures=config.get('architectures'),model_type=config.get('model_type'),text_model_type=text.get('model_type'),
        official_non_thinking_parameter='chat_template_kwargs.enable_thinking=false',
        hidden_size=text.get('hidden_size'),num_hidden_layers=text.get('num_hidden_layers'),vocab_size=text.get('vocab_size'),
        artifacts=artifacts,cache_search=caches,existing_environment_versions=packages,
        accepted_hf_model_type_mapping=json.loads(q.stdout) if q.returncode==0 else dict(error=q.stderr[-500:]),
        isolated_environment_created=False,stop_reason='Required independent complete-state clone not validated; model installation/inference deferred')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts'},indent=2))


if __name__=='__main__':main()
