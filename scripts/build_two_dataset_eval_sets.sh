#!/usr/bin/env bash
# Build D_special and inference-attack subsets for PopQA and HotpotQA eval data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAIN_PYTHON:-python}"
DATA_RUN_NAME="${DATA_RUN_NAME:-two_dataset_v1_timeout120}"
TRIPLES_ROOT="${TRIPLES_ROOT:-${REPO_ROOT}/dataset/retrieved_data/${DATA_RUN_NAME}_relik}"
RL_ROOT="${RL_ROOT:-${REPO_ROOT}/dataset/sample_privacy/${DATA_RUN_NAME}/rl}"
SPECIAL_ROOT="${SPECIAL_ROOT:-${REPO_ROOT}/dataset/special_data/${DATA_RUN_NAME}}"
ATTACK_ROOT="${ATTACK_ROOT:-${REPO_ROOT}/dataset/inference_attack/${DATA_RUN_NAME}}"

mkdir -p "${SPECIAL_ROOT}" "${ATTACK_ROOT}"
cd "${REPO_ROOT}"

for dataset in popqa hotpotqa; do
  triples="${TRIPLES_ROOT}/${dataset}_eval_with_triplets.jsonl"
  rl_data="${RL_ROOT}/${dataset}_eval_rl.jsonl"
  [[ -s "${triples}" ]] || { echo "Missing ReLiK triples: ${triples}" >&2; exit 1; }
  [[ -s "${rl_data}" ]] || { echo "Missing RL eval data: ${rl_data}" >&2; exit 1; }

  "${PYTHON_BIN}" get_special_data.py \
    --input "${rl_data}" \
    --output "${SPECIAL_ROOT}/${dataset}_eval_special.jsonl" \
    --min_special_contexts 2

  "${PYTHON_BIN}" get_inference_attack_data.py \
    --triples_input "${triples}" \
    --sampled_input "${rl_data}" \
    --output "${ATTACK_ROOT}/${dataset}_eval_inferattack.jsonl"

  [[ -s "${SPECIAL_ROOT}/${dataset}_eval_special.jsonl" ]] || {
    echo "Empty D_special output for ${dataset}" >&2
    exit 2
  }
  [[ -s "${ATTACK_ROOT}/${dataset}_eval_inferattack.jsonl" ]] || {
    echo "Empty inference-attack output for ${dataset}" >&2
    exit 2
  }
done

wc -l "${SPECIAL_ROOT}"/*.jsonl "${ATTACK_ROOT}"/*.jsonl
