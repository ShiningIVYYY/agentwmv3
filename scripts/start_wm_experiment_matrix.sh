#!/usr/bin/env bash
set -euo pipefail
cd /data1/shn/agentwmv2
OUTPUT_DIR=${1:-/data1/shn/agentwmv2/outputs/wm_reflection/full_7b_v8}
PYTHONNOUSERSITE=1 /home/shn/miniconda3/envs/ee-qwen38-direct/bin/python \
  scripts/run_wm_experiment_matrix.py --output "$OUTPUT_DIR" --models qwen25 ee_il --ee-workers 4 --launch
