#!/usr/bin/env bash
# Launch the reproducible two-dataset PPO run used for the public result summary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAIN_PYTHON:-python}"
RUN_NAME="${RUN_NAME:-two_dataset_b8_3000}"
DATA_RUN_NAME="${DATA_RUN_NAME:-two_dataset_v1_timeout120}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset/sample_privacy/${DATA_RUN_NAME}/rl}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${REPO_ROOT}/output_checkpoint/SFT}"
RELIK_MODEL="${RELIK_MODEL:-${REPO_ROOT}/models/relik}"
GATE_REPORT="${GATE_REPORT:-${REPO_ROOT}/outputs/relik_consistency/report.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output_checkpoint/RL-${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ppo/${RUN_NAME}}"

for path in \
  "${DATA_ROOT}/popqa_train_rl.jsonl" \
  "${DATA_ROOT}/hotpotqa_train_rl.jsonl" \
  "${SFT_CHECKPOINT}/config.json" \
  "${RELIK_MODEL}" \
  "${GATE_REPORT}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Refusing to overwrite existing PPO output: ${OUTPUT_DIR}" >&2
  echo "Set a different RUN_NAME or OUTPUT_DIR for a new run." >&2
  exit 1
}

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
cd "${REPO_ROOT}"

"${PYTHON_BIN}" RL_train.py \
  --data-dir-1 "${DATA_ROOT}/popqa_train_rl.jsonl" \
  --data-dir-2 "${DATA_ROOT}/hotpotqa_train_rl.jsonl" \
  --data-dir-3 "" \
  --data-dir-4 "" \
  --model-dir "${SFT_CHECKPOINT}" \
  --tokenizer-dir "${SFT_CHECKPOINT}" \
  --relik-model "${RELIK_MODEL}" \
  --relik-consistency-report "${GATE_REPORT}" \
  --output-dir "${OUTPUT_DIR}" \
  --logging-dir "${LOG_DIR}" \
  --seed 42 \
  --batch-size 8 \
  --mini-batch-size 4 \
  --ppo-epochs 4 \
  --learning-rate 1e-5 \
  --gamma 0.99 \
  --max-steps 3000 \
  --save-freq 500 \
  --show-freq 1 \
  --initial-p 20 \
  --p-increment 5 \
  --p-interval 350 \
  --p-max 40 \
  --relik-device cuda \
  --minimum-gate-samples 500

[[ -s "${OUTPUT_DIR}/step_final/config.json" ]] || {
  echo "PPO exited without step_final/config.json" >&2
  exit 2
}
echo "PPO completed: ${OUTPUT_DIR}/step_final"
