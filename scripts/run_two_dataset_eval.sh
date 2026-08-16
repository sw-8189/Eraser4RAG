#!/usr/bin/env bash
# Generate final-policy evaluation rewrites and extract all six metrics.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAIN_PYTHON:-python}"
RUN_NAME="${RUN_NAME:-two_dataset_b8_3000}"
DATA_RUN_NAME="${DATA_RUN_NAME:-two_dataset_v1_timeout120}"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-${REPO_ROOT}/output_checkpoint/RL-${RUN_NAME}/step_final}"
RELIK_MODEL="${RELIK_MODEL:-${REPO_ROOT}/models/relik}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset/sample_privacy/${DATA_RUN_NAME}}"
SPECIAL_ROOT="${SPECIAL_ROOT:-${REPO_ROOT}/dataset/special_data/${DATA_RUN_NAME}}"
ATTACK_ROOT="${ATTACK_ROOT:-${REPO_ROOT}/dataset/inference_attack/${DATA_RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/evaluation/${RUN_NAME}}"
REWRITE_ROOT="${OUTPUT_ROOT}/rewritten"
METRIC_ROOT="${OUTPUT_ROOT}/metrics"
LOG_ROOT="${REPO_ROOT}/logs/evaluation/${RUN_NAME}"

if [[ -e "${METRIC_ROOT}/complete.txt" && "${FORCE:-0}" != "1" ]]; then
  echo "Evaluation is already complete: ${METRIC_ROOT}/complete.txt" >&2
  echo "Set FORCE=1 only when the existing generated outputs may be replaced." >&2
  exit 1
fi

[[ -s "${FINAL_CHECKPOINT}/config.json" ]] || { echo "Missing final checkpoint" >&2; exit 1; }
[[ -d "${RELIK_MODEL}" ]] || { echo "Missing ReLiK model" >&2; exit 1; }
for path in \
  "${DATA_ROOT}/rl/popqa_eval_rl.jsonl" \
  "${DATA_ROOT}/rl/hotpotqa_eval_rl.jsonl" \
  "${SPECIAL_ROOT}/popqa_eval_special.jsonl" \
  "${SPECIAL_ROOT}/hotpotqa_eval_special.jsonl" \
  "${ATTACK_ROOT}/popqa_eval_inferattack.jsonl" \
  "${ATTACK_ROOT}/hotpotqa_eval_inferattack.jsonl"; do
  [[ -s "${path}" ]] || { echo "Missing evaluation input: ${path}" >&2; exit 1; }
done

mkdir -p "${REWRITE_ROOT}" "${METRIC_ROOT}" "${LOG_ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

run_special_rewrite() {
  local label="$1" input="$2"
  "${PYTHON_BIN}" "${REPO_ROOT}/rewrite_docs_special.py" \
    --data-dir "${input}" --model-dir "${FINAL_CHECKPOINT}" \
    --tokenizer-dir "${FINAL_CHECKPOINT}" --output "${REWRITE_ROOT}/${label}.jsonl" \
    --batch-size 1 --max-passage-length 128 --max-concat-length 1300 \
    --seed 42 --triple-scope global \
    >"${LOG_ROOT}/${label}_rewrite.log" 2>&1
}

run_attack_rewrite() {
  local label="$1" input="$2"
  "${PYTHON_BIN}" "${REPO_ROOT}/rewrite_docs_inferattack_inference.py" \
    --data-dir "${input}" --model-dir "${FINAL_CHECKPOINT}" \
    --tokenizer-dir "${FINAL_CHECKPOINT}" --output "${REWRITE_ROOT}/${label}.jsonl" \
    --batch-size 1 --max-passage-length 128 --max-concat-length 1300 \
    --seed 42 --triple-scope global \
    >"${LOG_ROOT}/${label}_rewrite.log" 2>&1
}

run_retention_eval() {
  local label="$1"
  "${PYTHON_BIN}" "${REPO_ROOT}/test_special.py" \
    --data "${REWRITE_ROOT}/${label}.jsonl" --relik_model "${RELIK_MODEL}" \
    --device cuda --batch_size 8 --seed 42 \
    --output-json "${METRIC_ROOT}/${label}.json" \
    >"${LOG_ROOT}/${label}_metrics.log" 2>&1
}

run_connectivity_eval() {
  local label="$1"
  "${PYTHON_BIN}" "${REPO_ROOT}/test_inferattack.py" \
    --data "${REWRITE_ROOT}/${label}.jsonl" --relik_model "${RELIK_MODEL}" \
    --device cuda --batch_size 8 --seed 42 \
    --output-json "${METRIC_ROOT}/${label}.json" \
    >"${LOG_ROOT}/${label}_metrics.log" 2>&1
}

cd "${REPO_ROOT}"
run_special_rewrite popqa_eval "${DATA_ROOT}/rl/popqa_eval_rl.jsonl"
run_special_rewrite hotpotqa_eval "${DATA_ROOT}/rl/hotpotqa_eval_rl.jsonl"
run_special_rewrite popqa_special "${SPECIAL_ROOT}/popqa_eval_special.jsonl"
run_special_rewrite hotpotqa_special "${SPECIAL_ROOT}/hotpotqa_eval_special.jsonl"
run_attack_rewrite popqa_inferattack "${ATTACK_ROOT}/popqa_eval_inferattack.jsonl"
run_attack_rewrite hotpotqa_inferattack "${ATTACK_ROOT}/hotpotqa_eval_inferattack.jsonl"

run_retention_eval popqa_eval
run_retention_eval hotpotqa_eval
run_retention_eval popqa_special
run_retention_eval hotpotqa_special
run_connectivity_eval popqa_inferattack
run_connectivity_eval hotpotqa_inferattack

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/validate_two_dataset_eval.py" \
  --rewrite-root "${REWRITE_ROOT}" \
  --metric-root "${METRIC_ROOT}" \
  --output "${METRIC_ROOT}/validation.json"
sha256sum "${REWRITE_ROOT}"/*.jsonl "${METRIC_ROOT}"/*.json >"${METRIC_ROOT}/evaluation_outputs.sha256"
date '+%F %T %Z' >"${METRIC_ROOT}/complete.txt"
echo "Evaluation completed: ${METRIC_ROOT}"
