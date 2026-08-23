#!/usr/bin/env bash
# Evaluate the existing SFT checkpoint on PopQA without retraining or PPO.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAIN_PYTHON:-python}"
DATA_RUN_NAME="${DATA_RUN_NAME:-two_dataset_v1_timeout120}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${REPO_ROOT}/output_checkpoint/SFT}"
RELIK_MODEL="${RELIK_MODEL:-${REPO_ROOT}/models/relik}"
RL_ROOT="${RL_ROOT:-${REPO_ROOT}/dataset/sample_privacy/${DATA_RUN_NAME}/rl}"
TRIPLES_ROOT="${TRIPLES_ROOT:-${REPO_ROOT}/dataset/retrieved_data/${DATA_RUN_NAME}_relik}"
SPECIAL_ROOT="${SPECIAL_ROOT:-${REPO_ROOT}/dataset/special_data/${DATA_RUN_NAME}}"
ATTACK_ROOT="${ATTACK_ROOT:-${REPO_ROOT}/dataset/inference_attack/${DATA_RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/evaluation/popqa_sft_only}"
REWRITE_ROOT="${OUTPUT_ROOT}/rewritten"
METRIC_ROOT="${OUTPUT_ROOT}/metrics"
LOG_ROOT="${REPO_ROOT}/logs/evaluation/popqa_sft_only"
LOG_PATH="${LOG_ROOT}/run.log"

if [[ -e "${METRIC_ROOT}/complete.txt" && "${FORCE:-0}" != "1" ]]; then
  echo "SFT-only evaluation is already complete: ${METRIC_ROOT}/complete.txt" >&2
  exit 1
fi

[[ -x "${PYTHON_BIN}" || "${PYTHON_BIN}" == "python" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
[[ -s "${SFT_CHECKPOINT}/config.json" && -s "${SFT_CHECKPOINT}/model.safetensors" ]] || {
  echo "Incomplete SFT checkpoint: ${SFT_CHECKPOINT}" >&2
  exit 2
}
[[ -d "${RELIK_MODEL}" ]] || { echo "Missing ReLiK model: ${RELIK_MODEL}" >&2; exit 2; }
[[ -s "${RL_ROOT}/popqa_eval_rl.jsonl" ]] || { echo "Missing PopQA RL eval input" >&2; exit 2; }
[[ -s "${TRIPLES_ROOT}/popqa_eval_with_triplets.jsonl" ]] || { echo "Missing PopQA ReLiK triples" >&2; exit 2; }

mkdir -p "${REWRITE_ROOT}" "${METRIC_ROOT}" "${LOG_ROOT}"
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -f "${REWRITE_ROOT}"/*.jsonl "${METRIC_ROOT}"/*.json "${METRIC_ROOT}/complete.txt"
fi
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
cd "${REPO_ROOT}"

echo "Starting PopQA SFT-only evaluation"
echo "checkpoint=${SFT_CHECKPOINT}"
echo "data_run=${DATA_RUN_NAME}"
echo "git_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"

count_lines() {
  wc -l < "$1" | tr -d '[:space:]'
}

run_rewrite_if_needed() {
  local output="$1" expected="$2"
  shift 2
  if [[ -s "$output" ]] && [[ "$(count_lines "$output")" == "$expected" ]]; then
    echo "Reusing complete rewrite: $output ($expected records)"
    return 0
  fi
  rm -f "$output"
  "$@"
}

if [[ ! -s "${SPECIAL_ROOT}/popqa_eval_special.jsonl" ]]; then
  "${PYTHON_BIN}" get_special_data.py \
    --input "${RL_ROOT}/popqa_eval_rl.jsonl" \
    --output "${SPECIAL_ROOT}/popqa_eval_special.jsonl" \
    --min_special_contexts 2
fi
if [[ ! -s "${ATTACK_ROOT}/popqa_eval_inferattack.jsonl" ]]; then
  "${PYTHON_BIN}" get_inference_attack_data.py \
    --triples_input "${TRIPLES_ROOT}/popqa_eval_with_triplets.jsonl" \
    --sampled_input "${RL_ROOT}/popqa_eval_rl.jsonl" \
    --output "${ATTACK_ROOT}/popqa_eval_inferattack.jsonl"
fi

run_rewrite_if_needed "${REWRITE_ROOT}/popqa_eval.jsonl" 1000 \
  "${PYTHON_BIN}" rewrite_docs_special.py \
    --data-dir "${RL_ROOT}/popqa_eval_rl.jsonl" \
    --model-dir "${SFT_CHECKPOINT}" --tokenizer-dir "${SFT_CHECKPOINT}" \
    --output "${REWRITE_ROOT}/popqa_eval.jsonl" \
    --batch-size 1 --max-passage-length 128 --max-concat-length 1300 \
    --seed 42 --triple-scope global

run_rewrite_if_needed "${REWRITE_ROOT}/popqa_special.jsonl" 954 \
  "${PYTHON_BIN}" rewrite_docs_special.py \
    --data-dir "${SPECIAL_ROOT}/popqa_eval_special.jsonl" \
    --model-dir "${SFT_CHECKPOINT}" --tokenizer-dir "${SFT_CHECKPOINT}" \
    --output "${REWRITE_ROOT}/popqa_special.jsonl" \
    --batch-size 1 --max-passage-length 128 --max-concat-length 1300 \
    --seed 42 --triple-scope global

run_rewrite_if_needed "${REWRITE_ROOT}/popqa_inferattack.jsonl" 167 \
  "${PYTHON_BIN}" rewrite_docs_inferattack_inference.py \
    --data-dir "${ATTACK_ROOT}/popqa_eval_inferattack.jsonl" \
    --model-dir "${SFT_CHECKPOINT}" --tokenizer-dir "${SFT_CHECKPOINT}" \
    --output "${REWRITE_ROOT}/popqa_inferattack.jsonl" \
    --batch-size 1 --max-passage-length 128 --max-concat-length 1300 \
    --seed 42 --triple-scope global

"${PYTHON_BIN}" test_special.py \
  --data "${REWRITE_ROOT}/popqa_eval.jsonl" --relik_model "${RELIK_MODEL}" \
  --device cuda --batch_size 8 --seed 42 --output-json "${METRIC_ROOT}/popqa_eval.json"
"${PYTHON_BIN}" test_special.py \
  --data "${REWRITE_ROOT}/popqa_special.jsonl" --relik_model "${RELIK_MODEL}" \
  --device cuda --batch_size 8 --seed 42 --output-json "${METRIC_ROOT}/popqa_special.json"
"${PYTHON_BIN}" test_inferattack.py \
  --data "${REWRITE_ROOT}/popqa_inferattack.jsonl" --relik_model "${RELIK_MODEL}" \
  --device cuda --batch_size 8 --seed 42 --output-json "${METRIC_ROOT}/popqa_inferattack.json"

"${PYTHON_BIN}" scripts/validate_popqa_sft_only_eval.py \
  --rewrite-root "${REWRITE_ROOT}" --metric-root "${METRIC_ROOT}" \
  --output "${METRIC_ROOT}/validation.json" \
  --manifest "${OUTPUT_ROOT}/run_manifest.json" \
  --checkpoint "${SFT_CHECKPOINT}" --repo-root "${REPO_ROOT}"

sha256sum "${REWRITE_ROOT}"/*.jsonl "${METRIC_ROOT}"/*.json >"${METRIC_ROOT}/outputs.sha256"
date '+%F %T %Z' >"${METRIC_ROOT}/complete.txt"
echo "PopQA SFT-only evaluation completed: ${METRIC_ROOT}"
