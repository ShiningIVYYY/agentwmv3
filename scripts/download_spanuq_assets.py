#!/usr/bin/env python3
"""Download only the two pinned inference assets, with published hash checks."""
import argparse
import hashlib
import fnmatch
import json
import os
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/internal_signals/qwen3_4b.json"))
    parser.add_argument("--backbone-dir", default=str(ROOT / "checkpoints/qwen3-4b"))
    parser.add_argument("--probe-dir", default=str(ROOT / "checkpoints/spanuq/Qwen3-4B"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    records = {}
    for repo, rev, prefix, destination in [
        (config["backbone_id"], config["backbone_revision"], "", args.backbone_dir),
        (config["probe_id"], config["probe_revision"], config["probe_subfolder"] + "/", args.probe_dir),
    ]:
        tree_url = endpoint + "/api/models/" + repo + "/tree/" + rev + ("/" + prefix.rstrip("/") if prefix else "")
        with urllib.request.urlopen(tree_url, timeout=60) as stream:
            tree = json.load(stream)
        snapshot_root = Path(destination).resolve()
        if prefix:
            snapshot_root = snapshot_root.parent / ".download"
        path = snapshot_download(repo, revision=rev, local_dir=str(snapshot_root),
                                 allow_patterns=[prefix + pattern for pattern in
                                     (["*.json", "*.pt"] if prefix else ["*.json", "*.safetensors", "*.txt", "LICENSE"])],
                                 max_workers=3)
        results = {}
        selected_patterns = [prefix + pattern for pattern in
                             (["*.json", "*.pt"] if prefix else ["*.json", "*.safetensors", "*.txt", "LICENSE"])]
        for item in tree:
            name = item.get("path", "")
            source = Path(path) / name
            if item["type"] != "file" or not any(fnmatch.fnmatch(name, pattern) for pattern in selected_patterns):
                continue
            if not source.is_file():
                raise FileNotFoundError("Required selected asset missing: " + name)
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            expected = item.get("lfs", {}).get("oid")
            if expected and actual != expected:
                raise RuntimeError("Published LFS SHA256 mismatch: " + name)
            if prefix:
                import shutil
                target = Path(destination) / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            results[name] = {"sha256": actual, "published_lfs_sha256": expected,
                             "bytes": source.stat().st_size}
        records[repo] = {"revision": rev, "files": results}
    out = ROOT / "outputs/phase1/assets_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
