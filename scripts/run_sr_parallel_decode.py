#!/usr/bin/env python3
"""Start/resume SR with an isolated, version-checked small-batch mask overlay."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'outputs/wm_reflection_setup/sr_decode_parallel'

def prepare():
    for name, expected in [('vllm', '0.11.0'), ('xgrammar', '0.1.25')]:
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, got {actual}; revalidate patch')
    distribution = importlib.metadata.distribution('vllm')
    source = Path(distribution.locate_file('vllm/v1/structured_output/__init__.py'))
    old = source.read_text()
    replacements = {
        'self.fill_bitmask_parallel_threshold = 128': 'self.fill_bitmask_parallel_threshold = 1',
        'self.fill_bitmask_parallel_batch_size = 16': 'self.fill_bitmask_parallel_batch_size = 1',
    }
    new = old
    for before, after in replacements.items():
        if new.count(before) != 1:
            raise RuntimeError('Unexpected vLLM implementation: ' + before)
        new = new.replace(before, after)
    OVERLAY.mkdir(parents=True, exist_ok=True)
    provenance = dict(original=str(source),
        original_sha256=hashlib.sha256(old.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(new.encode()).hexdigest(),
        versions={n: importlib.metadata.version(n) for n in ('vllm', 'xgrammar')},
        replacements=replacements)
    (OVERLAY / 'structured_output.py').write_text(new)
    (OVERLAY / 'original_structured_output.py').write_text(old)
    bootstrap = """import importlib.abc
import importlib.util
from pathlib import Path
import sys
class ParallelMaskFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm.v1.structured_output':
            return importlib.util.spec_from_file_location(
                fullname, Path(__file__).with_name('structured_output.py'),
                submodule_search_locations=[ORIGINAL])
sys.meta_path.insert(0, ParallelMaskFinder())
""".replace('ORIGINAL', repr(str(source.parent)))
    (OVERLAY / 'sitecustomize.py').write_text(bootstrap)
    (OVERLAY / 'patch.json').write_text(json.dumps(provenance, indent=2))
    return provenance

if __name__ == '__main__':
    prepare()
    env = dict(os.environ, PYTHONPATH=str(OVERLAY) + os.pathsep + str(ROOT),
               PYTHONNOUSERSITE='1')
    os.execve(sys.executable, [sys.executable, '-u',
        str(ROOT / 'scripts/run_sr_experiments.py'), *sys.argv[1:]], env)
