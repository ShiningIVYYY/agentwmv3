#!/usr/bin/env python3
"""Install the immutable official collectors without running upstream downloads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

REVISION = 'e8dc240269fb222de3c242b09c70bfbf0b4ac1c9'
ROOT = Path(__file__).resolve().parents[1]


def install(source, target):
    archive = subprocess.check_output(['git', '-C', str(source), 'archive', REVISION,
        'scripts/collect_init_context', 'scripts/interact_with_world_model/run.py',
        'scripts/single_step_accuracy/run.py', 'README.md'])
    target.mkdir(parents=True, exist_ok=True)
    marker = target / 'REVISION'
    if marker.exists() and marker.read_text().strip() != REVISION:
        raise RuntimeError('Existing Word2World source revision differs')
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        for member in tf:
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not (member.isdir() or member.isfile()):
                raise RuntimeError('Unsafe source archive member')
            path = target / relative
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                content = tf.extractfile(member).read()
                if path.exists() and path.read_bytes() != content:
                    raise RuntimeError('Existing source file has local changes: ' + str(path))
                path.write_bytes(content)
    marker.write_text(REVISION + '\n')
    (target / 'SHA256.json').write_text(json.dumps({str(p.relative_to(target)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*.py')},indent=2))
    print('Installed official Word2World source', REVISION, 'at', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-repo', type=Path)
    parser.add_argument('--target', type=Path, default=ROOT / 'external/Word2World')
    args = parser.parse_args()
    if args.source_repo:
        install(args.source_repo, args.target)
    else:
        with tempfile.TemporaryDirectory(prefix='word2world-source-') as tmp:
            subprocess.run(['git', 'clone', 'https://github.com/X1AOX1A/Word2World', tmp], check=True)
            install(Path(tmp), args.target)
