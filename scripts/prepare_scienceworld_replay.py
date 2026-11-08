#!/usr/bin/env python3
"""Fetch the exact official deterministic ScienceWorld engine, without SDK edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worldmodel.ground_truth_environment import SIMULATOR_PROFILES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',default=str(ROOT/'external/ScienceWorld-replay-audit/scienceworld/scienceworld.jar'))
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();target=Path(a.target)
    info=SIMULATOR_PROFILES['scienceworld130']
    url='https://raw.githubusercontent.com/allenai/ScienceWorld/'+info['upstream_commit']+'/scienceworld/scienceworld.jar'
    def valid(path):return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==info['jar_sha256']
    downloaded=False
    if not valid(target):
        if a.verify_only:raise ValueError('Missing or incorrect official JAR')
        if target.exists():raise ValueError('Refuse to overwrite a different JAR; choose a new --target')
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as f:
            temp=Path(f.name)
        try:
            with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as f:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    f.write(chunk)
            if not valid(temp):raise ValueError('Official JAR SHA256 mismatch')
            temp.replace(target);downloaded=True
        finally:temp.unlink(missing_ok=True)
    with zipfile.ZipFile(target) as z:manifest=z.read('META-INF/MANIFEST.MF').decode()
    if 'Specification-Version: 1.3.0' not in manifest:raise ValueError('Unexpected native engine version')
    print(json.dumps(dict(profile='scienceworld130',**info,url=url,bytes=target.stat().st_size,
        downloaded=downloaded,sha256_verified=True,manifest_verified=True),indent=2))


if __name__=='__main__':main()
