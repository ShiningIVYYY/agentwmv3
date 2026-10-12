#!/usr/bin/env python3
"""Download pinned SR files and verify their official Git/LFS object hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path, lfs):
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(('blob %d\0' % path.stat().st_size).encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(manifest_path, root, endpoint):
    manifest = json.loads(manifest_path.read_text())
    repo, revision = manifest['repo_id'], manifest['revision']
    target = root / 'checkpoints' / repo.split('/')[1]
    verified = {}

    def one(row):
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'sr':
            raise ValueError('Unexpected SR path')
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = row.get('lfs', {}).get('oid') or row['oid']
        expected = expected.removeprefix('sha256:')
        lfs = bool(row.get('lfs'))
        if not path.exists():
            partial = path.with_name(path.name + '.part')
            url = endpoint + '/' + repo + '/resolve/' + revision + '/' + str(relative)
            for attempt in range(8):
                print('download', repo, relative, 'attempt', attempt + 1, flush=True)
                result = subprocess.run(['curl', '-fsSL', '--http1.1', '--connect-timeout', '20',
                    '--max-time', '3600', '--speed-limit', '1024', '--speed-time', '90',
                    '-C', '-', '-o', str(partial), url])
                if result.returncode == 0 and partial.stat().st_size == row['size']:
                    break
                time.sleep(3)
            else:
                raise RuntimeError('Failed download: ' + str(path))
            if digest(partial, lfs) != expected:
                raise RuntimeError('Official object hash mismatch: ' + str(partial))
            partial.replace(path)
        if path.stat().st_size != row['size'] or digest(path, lfs) != expected:
            raise RuntimeError('Existing checkpoint differs from pinned SR: ' + str(path))
        verified[str(relative)] = dict(size=row['size'], official_oid=expected,
            hash_kind='sha256' if lfs else 'git-blob-sha1')
        print('verified', repo, relative, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, manifest['files']))
    value = dict(repo_id=repo, revision=revision, subfolder='sr', files=verified,
                 verified_at=time.time())
    (target / 'sr' / 'sr_provenance.json').write_text(json.dumps(value, indent=2))
    print('COMPLETE', repo, revision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    args = parser.parse_args()
    paths = sorted(args.manifests.glob('*_sr_manifest.json'))
    if len(paths) != 3:
        raise RuntimeError('Expected three pinned SR manifests')
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda p: download(p, args.root, args.endpoint), paths))
