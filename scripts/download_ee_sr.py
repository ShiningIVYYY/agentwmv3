#!/usr/bin/env python3
"""Only the user-selected SR subfolder, pinned and byte-verified; never train."""
import argparse,hashlib,json,os
from pathlib import Path
import urllib.request
REPO='osunlp/early-experience-scienceworld'
REVISION='4b8f0835a3f613e40541619462ae906db8dba95c'


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',default='checkpoints/ee_scienceworld_sr');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    root=Path(a.model_dir);root.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}/sr?recursive=false&expand=false'
    metadata=root/'upstream_files.json'
    if not metadata.exists():metadata.write_bytes(urllib.request.urlopen(url,timeout=30).read())
    entries=json.loads(metadata.read_text());assert len(entries)==14 and all(x['path'].startswith('sr/') for x in entries)
    if not a.verify_only:
        os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_OFFLINE']='0'
        from huggingface_hub import snapshot_download
        snapshot_download(REPO,revision=REVISION,allow_patterns=['sr/*'],local_dir=root,max_workers=2,token=False)
    files={}
    for row in entries:
        path=root/row['path'];assert path.stat().st_size==row['size'],row['path']
        h=hashlib.sha256();git=hashlib.sha1();git.update(b'blob '+str(path.stat().st_size).encode()+b'\0')
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk);git.update(chunk)
        sha=h.hexdigest()
        if 'lfs' in row:assert sha==row['lfs']['oid'],row['path']
        else:assert git.hexdigest()==row['oid'],row['path']
        files[row['path']]=dict(bytes=row['size'],sha256=sha,upstream_oid=row.get('lfs',{}).get('oid',row['oid']))
    (root/'assets_manifest.json').write_text(json.dumps(dict(repo_id=REPO,revision=REVISION,subfolder='sr',files=files),indent=2)+'\n')
    print(json.dumps(dict(status='all14_verified',revision=REVISION,bytes=sum(x['bytes'] for x in files.values())),indent=2))


if __name__=='__main__':main()
