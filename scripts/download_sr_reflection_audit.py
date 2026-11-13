"""Pinned public SR training-source audit; not inference or model training."""
import hashlib,json,time,urllib.request
from pathlib import Path
p=Path('outputs/ee_sr_audit/reflection_sft.jsonl')
sha='8020d0c6fe9dd86105870fc4bfe7c9ec45fcedb555318c9979c0e485016eb4ba'
if not p.exists():
    u='https://huggingface.co/datasets/osunlp/early-experience/resolve/7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce/scienceworld/reflection_sft.jsonl'
    tmp=p.with_suffix('.download')
    with urllib.request.urlopen(u,timeout=60) as src,tmp.open('wb') as dst:
        for block in iter(lambda:src.read(4*1024*1024),b''):dst.write(block)
    assert tmp.stat().st_size==271777648
    assert hashlib.file_digest(tmp.open('rb'),'sha256').hexdigest()==sha
    tmp.replace(p)
assert p.stat().st_size==271777648
assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==sha
print(json.dumps(dict(verified=True,path=str(p),sha256=sha,bytes=p.stat().st_size)))
