#!/usr/bin/env python3
"""Only the pinned official9B inference assets; no other model or training data."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import time
from huggingface_hub import HfApi,snapshot_download

REPO='Qwen/Qwen3.5-9B'
REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
PATTERNS=['*.safetensors','*.json','*.jinja','merges.txt','vocab.*']


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',default=os.environ.get('AGENTWM_QWEN35_DIR','checkpoints/qwen35_9b'))
    p.add_argument('--cache-dir',default=None,help='Otherwise Hugging Face environment/cache defaults apply')
    p.add_argument('--verify-only',action='store_true',help='No downloads; reject missing/changed files')
    a=p.parse_args();start=time.perf_counter();dest=Path(a.model_dir)
    info=HfApi().model_info(REPO,revision=REVISION,files_metadata=True)
    if info.sha!=REVISION:raise ValueError('Official revision changed')
    selected=[s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename,pattern) for pattern in PATTERNS)]
    if not a.verify_only:
        snapshot_download(REPO,revision=REVISION,local_dir=str(dest),cache_dir=a.cache_dir,
                          allow_patterns=PATTERNS,max_workers=4)
    files={}
    for item in selected:
        path=dest/item.rfilename
        if not path.is_file():raise ValueError('Missing asset: '+item.rfilename)
        digest=sha256(path)
        if item.lfs and item.lfs.sha256!=digest:raise ValueError('Official LFS checksum mismatch: '+item.rfilename)
        files[item.rfilename]=dict(bytes=path.stat().st_size,sha256=digest)
    manifest=dict(model_id=REPO,revision=REVISION,files=files,seconds=time.perf_counter()-start)
    if not a.verify_only:
        (dest/'assets_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        existing=json.loads((dest/'assets_manifest.json').read_text())
        if existing['revision']!=REVISION or existing['files']!=files:raise ValueError('Local manifest differs')
    print(json.dumps(dict(repo=REPO,revision=REVISION,files=len(files),verified=True,downloaded=not a.verify_only,
                         seconds=manifest['seconds']),indent=2))


if __name__=='__main__':main()
