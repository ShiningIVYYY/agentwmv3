#!/usr/bin/env bash
set -euo pipefail

VERL_AGENT_DIR="${VERL_AGENT_DIR:-external/verl-agent}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-early-exp-alfworld}"
VERL_AGENT_REPO="${VERL_AGENT_REPO:-https://github.com/langfengQ/verl-agent.git}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

if [ ! -d "$VERL_AGENT_DIR/.git" ]; then
  mkdir -p "$(dirname "$VERL_AGENT_DIR")"
  git clone "$VERL_AGENT_REPO" "$VERL_AGENT_DIR"
else
  echo "Using existing verl-agent checkout at $VERL_AGENT_DIR"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  conda create -n "$CONDA_ENV_NAME" python==3.12 -y
else
  echo "Using existing conda env $CONDA_ENV_NAME"
fi

conda run -n "$CONDA_ENV_NAME" pip install "vllm==$VLLM_VERSION"
conda run -n "$CONDA_ENV_NAME" pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation --no-cache-dir
conda run -n "$CONDA_ENV_NAME" pip install -e "$VERL_AGENT_DIR"
conda run -n "$CONDA_ENV_NAME" pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
conda run -n "$CONDA_ENV_NAME" alfworld-download -f

echo "Environment ready: $CONDA_ENV_NAME"
echo "verl-agent path: $VERL_AGENT_DIR"
