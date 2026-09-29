#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX_FILE_ATTEMPTS="${MAX_FILE_ATTEMPTS:-50}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-10}"

FILES=(
  added_tokens.json
  chat_template.jinja
  config.json
  generation_config.json
  merges.txt
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
)

download_file() {
  local repo="$1"
  local target_name="$2"
  local filename="$3"
  local target_dir="${CHECKPOINT_ROOT}/${target_name}/il"
  local target_file="${target_dir}/${filename}"
  local partial_file="${target_file}.part"
  local url="${HF_ENDPOINT}/${repo}/resolve/main/il/${filename}"
  local attempt=1

  mkdir -p "${target_dir}"

  if [[ -s "${target_file}" ]]; then
    echo "[$(date '+%F %T')] Skip existing ${target_file}"
    return 0
  fi

  while (( attempt <= MAX_FILE_ATTEMPTS )); do
    echo "[$(date '+%F %T')] Download ${repo}/il/${filename} (attempt ${attempt}/${MAX_FILE_ATTEMPTS})"

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
      break
    fi

    if (( attempt == MAX_FILE_ATTEMPTS )); then
      echo "[$(date '+%F %T')] Failed ${repo}/il/${filename}" >&2
      return 1
    fi

    echo "[$(date '+%F %T')] Retry ${repo}/il/${filename} in ${RETRY_DELAY_SECONDS}s"
    sleep "${RETRY_DELAY_SECONDS}"
    ((attempt += 1))
  done

  mv "${partial_file}" "${target_file}"
  echo "[$(date '+%F %T')] Completed ${target_file}"
}

download_repo() {
  local repo="$1"
  local target_name="$2"
  local filename

  for filename in "${FILES[@]}"; do
    download_file "${repo}" "${target_name}" "${filename}"
  done
}

download_repo \
  osunlp/early-experience-scienceworld \
  early-experience-scienceworld

download_repo \
  osunlp/early-experience-webshop \
  early-experience-webshop
