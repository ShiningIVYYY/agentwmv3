#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${PYTHON_BIN:?Set PYTHON_BIN to a Python 3.12 executable}"
ENV_DIR="${SPANUQ_ENV_DIR:-$ROOT/.venv-spanuq}"
if [[ -e "$ENV_DIR" ]]; then
  echo "Refusing to modify an existing environment: $ENV_DIR" >&2
  exit 1
fi
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple -r "$ROOT/environment/spanuq-inference.lock.txt"
"$ENV_DIR/bin/python" -m pip check
