#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${SPANUQ_SOURCE:-$ROOT/external/SpanUQ}"
COMMIT=778b1c1e3eeeda66233fe6b426f9303f4c7f76a3
if [[ ! -e "$SOURCE" ]]; then
  git clone https://github.com/damon-demon/SpanUQ.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$COMMIT"
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$COMMIT" ]] || { echo "Source revision mismatch" >&2; exit 1; }
[[ -z "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]] || { echo "Source modified" >&2; exit 1; }
echo "$SOURCE"
