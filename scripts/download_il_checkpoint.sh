#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/early-experience-alfworld}"
HF_REPO_ID="${HF_REPO_ID:-osunlp/early-experience-alfworld}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-early-exp-alfworld}"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli)
elif command -v conda >/dev/null 2>&1; then
  HF_CLI=(conda run -n "$CONDA_ENV_NAME" huggingface-cli)
else
  echo "huggingface-cli is required. Install huggingface_hub or run setup first." >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"
"${HF_CLI[@]}" download "$HF_REPO_ID" \
  --include "il/*" \
  --local-dir "$CHECKPOINT_DIR"

echo "Checkpoint downloaded to $CHECKPOINT_DIR/il"
