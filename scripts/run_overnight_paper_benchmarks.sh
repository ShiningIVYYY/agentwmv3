#!/usr/bin/env bash
# Replaces the obsolete checkpoint-switching/Word2World-list orchestration.
set -euo pipefail
cd /data1/shn/agentwmv2
exec env PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-qwen38-direct/bin/python \
    scripts/supervise_paper_eval.py "$@"
