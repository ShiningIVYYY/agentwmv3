#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-/home/shn/miniconda3/envs/verl-agent/bin/python}"
MAX_FILE_ATTEMPTS="${MAX_FILE_ATTEMPTS:-50}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-10}"

if (( $# == 0 || $# % 2 != 0 )); then
  echo "Usage: $0 <repo-id> <target-directory-name> [<repo-id> <target-directory-name> ...]" >&2
  exit 2
fi

download_file() {
  local repo="$1"
  local target_name="$2"
  local relative_path="$3"
  local target_dir="${CHECKPOINT_ROOT}/${target_name}"
  local target_file="${target_dir}/${relative_path}"
  local partial_file="${target_file}.part"
  local url="${HF_ENDPOINT}/${repo}/resolve/main/${relative_path}"
  local attempt=1

  mkdir -p "$(dirname "${target_file}")"

  if [[ -s "${target_file}" ]]; then
    echo "[$(date '+%F %T')] Skip existing ${target_file}"
    return 0
  fi

  while (( attempt <= MAX_FILE_ATTEMPTS )); do
    echo "[$(date '+%F %T')] Download ${repo}/${relative_path} (attempt ${attempt}/${MAX_FILE_ATTEMPTS})"

    if curl --location --fail --http1.1 \
      --continue-at - \
      --retry 3 \
      --retry-connrefused \
      --retry-delay "${RETRY_DELAY_SECONDS}" \
      --connect-timeout 30 \
      --speed-limit 1024 \
      --speed-time 120 \
      --output "${partial_file}" \
      "${url}"; then
      mv "${partial_file}" "${target_file}"
      echo "[$(date '+%F %T')] Completed ${target_file}"
      return 0
    fi

    if (( attempt == MAX_FILE_ATTEMPTS )); then
      echo "[$(date '+%F %T')] Failed ${repo}/${relative_path}" >&2
      return 1
    fi

    echo "[$(date '+%F %T')] Retry ${repo}/${relative_path} in ${RETRY_DELAY_SECONDS}s"
    sleep "${RETRY_DELAY_SECONDS}"
    ((attempt += 1))
  done
}

download_repo() {
  local repo="$1"
  local target_name="$2"
  local manifest_file
  local relative_path

  manifest_file="$(mktemp)"
  trap 'rm -f "${manifest_file}"' RETURN

  curl --fail --location --retry 10 --retry-connrefused --retry-delay 10 \
    "${HF_ENDPOINT}/api/models/${repo}/tree/main?recursive=true" \
    | "${PYTHON_BIN}" -c '
import json
import sys

for item in json.load(sys.stdin):
    if item.get("type") == "file":
        print(item["path"])
' > "${manifest_file}"

  while IFS= read -r relative_path; do
    download_file "${repo}" "${target_name}" "${relative_path}"
  done < "${manifest_file}"

  rm -f "${manifest_file}"
  trap - RETURN
}

while (( $# > 0 )); do
  download_repo "$1" "$2"
  shift 2
done
