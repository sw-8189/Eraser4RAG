#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ERASER_PYTHON:-/root/autodl-tmp/conda/envs/eraser-main/bin/python}"
DATA_PATH="$REPO_ROOT/dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl"
MODEL_PATH="$REPO_ROOT/models/sft"
OUTPUT_PATH="$REPO_ROOT/output_checkpoint/SFT"
LOG_DIR="$REPO_ROOT/logs/sft-full"
LOG_PATH="$LOG_DIR/train.log"
STATUS_PATH="$LOG_DIR/exit_code.txt"

mkdir -p "$LOG_DIR"
if [[ -e "$OUTPUT_PATH" ]]; then
  echo "Refusing to overwrite existing formal SFT output: $OUTPUT_PATH" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$DATA_PATH" || ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "Formal SFT input data or model is missing" >&2
  exit 2
fi

rm -f "$STATUS_PATH"
{
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "python=$PYTHON_BIN"
  echo "output=$OUTPUT_PATH"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$LOG_DIR/run_metadata.txt"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT" || exit 2
"$PYTHON_BIN" finetune_rewrite_doc.py \
  --data-dir "$DATA_PATH" \
  --model-dir "$MODEL_PATH" \
  --tokenizer-dir "$MODEL_PATH" \
  --output-dir "$OUTPUT_PATH" \
  --seed 42 \
  >> "$LOG_PATH" 2>&1
exit_code=$?
printf '%s\n' "$exit_code" > "$STATUS_PATH"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/run_metadata.txt"
exit "$exit_code"
