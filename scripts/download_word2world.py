#!/usr/bin/env python3
"""Download immutable official Word2World snapshots; verify LFS SHA256s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

MODELS = {'alfworld': 'Alfworld', 'scienceworld': 'Sciworld', 'webshop': 'Webshop'}


def metadata(url):
    # lab's Python HTTPS stack occasionally sees mirror SSL EOFs. curl is also
    # used for model files and retries the same immutable metadata endpoint.
    for attempt in range(10):
        try:
            return json.loads(subprocess.check_output(['curl', '-fsSL', '--http1.1', '--retry', '3',
                '--retry-delay', '3', '--connect-timeout', '30', '--max-time', '120', url]))
        except subprocess.CalledProcessError:
            if attempt == 9:
                raise
            time.sleep(3)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(env, root, endpoint, workers):
    repo = 'X1AOX1A/WorldModel-%s-Qwen2.5-7B' % MODELS[env]
    target = root / env
    target.mkdir(parents=True, exist_ok=True)
    pin = target / 'download_manifest.json'
    if pin.exists():
        manifest = json.loads(pin.read_text())
        assert manifest['repo_id'] == repo
    else:
        revision = metadata(endpoint + '/api/models/' + repo)['sha']
        rows = metadata(endpoint + '/api/models/' + repo + '/tree/' + revision + '?recursive=true&expand=true')
        # All model/tokenizer/config files; no training artifacts.
        rows = [r for r in rows if r['type'] == 'file' and
                (r['path'].endswith(('.json', '.safetensors', '.model', '.txt', '.jinja')))]
        if not any(r['path'].endswith('.safetensors') for r in rows):
            raise RuntimeError('No published model weights in ' + repo)
        manifest = dict(repo_id=repo, revision=revision, files=rows)
        pin.write_text(json.dumps(manifest, indent=2))
    verified = {}
    def download_one(row):
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe repository file path')
        path = target / relative
        expected = row.get('lfs', {}).get('oid')
        if expected and expected.startswith('sha256:'):
            expected = expected.split(':', 1)[1]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = str(path) + '.part'
            print('Downloading', env, relative, flush=True)
            if not Path(partial).exists() or Path(partial).stat().st_size != row['size']:
                for attempt in range(10):
                    try:
                        subprocess.run(['curl', '-fsSL', '--http1.1',
                            '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                            '-C', '-', '-o', partial,
                            endpoint + '/' + repo + '/resolve/' + manifest['revision'] + '/' + str(relative)], check=True)
                        break
                    except subprocess.CalledProcessError:
                        if attempt == 9:
                            raise
                        time.sleep(3)
            if Path(partial).stat().st_size != row['size']:
                raise RuntimeError('Incomplete download: ' + partial)
            if expected and sha256(Path(partial)) != expected:
                raise RuntimeError('Weight SHA256 mismatch: ' + partial)
            Path(partial).replace(path)
        if path.stat().st_size != row['size']:
            raise RuntimeError('Existing file size mismatch: ' + str(path))
        actual = sha256(path)
        if expected and actual != expected:
            raise RuntimeError('Existing weight SHA256 mismatch: ' + str(path))
        verified[str(relative)] = actual
        print('Verified file', env, relative, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(download_one, manifest['files']))
    (target / 'provenance.json').write_text(json.dumps(dict(repo_id=repo,
        revision=manifest['revision'], sha256=verified), indent=2))
    print('Verified', repo, manifest['revision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'checkpoints/word2world')
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'))
    parser.add_argument('--workers', type=int, default=4, help='Parallel files per environment')
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=len(args.envs)) as pool:
        list(pool.map(lambda env: download(env, args.root, args.endpoint.rstrip('/'), args.workers), args.envs))
