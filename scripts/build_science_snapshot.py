#!/usr/bin/env python3
"""Build the Java 8 heap snapshot helper without modifying the simulator jar."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    source = root / 'worldmodel/HeapSnapshot.java'
    out = root / 'outputs/wm_reflection_setup/science_snapshot'
    out.mkdir(parents=True, exist_ok=True)
    compiler = Path('/home/shn/miniconda3/pkgs/openjdk-8.0.472-h4d9bdce_0/bin/javac')
    if not compiler.is_file():
        found = shutil.which('javac')
        if not found:
            raise RuntimeError('A Java compiler is required to build the snapshot helper')
        compiler = Path(found)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = out / 'provenance.json'
    cls = out / 'classes/agentwm/HeapSnapshot.class'
    if manifest.exists() and cls.exists():
        previous = json.loads(manifest.read_text())
        if previous.get('source_sha256') == source_sha and all(
                hashlib.sha256((out / p).read_bytes()).hexdigest() == sha
                for p, sha in previous['classes'].items()):
            return out / 'classes'
    classes = out / 'classes'
    classes.mkdir(exist_ok=True)
    subprocess.run([str(compiler), '-source', '8', '-target', '8', '-d', str(classes), str(source)], check=True)
    files = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(classes.rglob('*.class'))}
    manifest.write_text(json.dumps(dict(source_sha256=source_sha, compiler=str(compiler), classes=files), indent=2))
    return classes


if __name__ == '__main__':
    print(build())
