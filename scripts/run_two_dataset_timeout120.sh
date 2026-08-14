#!/usr/bin/env bash
# Run the reduced PopQA + HotpotQA data pipeline on the AutoDL instance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COREF_PYTHON="${COREF_PYTHON:-/root/autodl-tmp/conda/envs/eraser-coref/bin/python}"
MAIN_PYTHON="${MAIN_PYTHON:-/root/autodl-tmp/conda/envs/eraser-main/bin/python}"
RELIK_MODEL="${RELIK_MODEL:-${REPO_ROOT}/models/relik}"
HF_CACHE="${HF_CACHE:-${REPO_ROOT}/hf_cache}"
RUN_NAME="${RUN_NAME:-two_dataset_v1_timeout120}"
CONTEXT_TIMEOUT_SECONDS="${CONTEXT_TIMEOUT_SECONDS:-120}"
COREF_WORKERS="${COREF_WORKERS:-16}"
RELIK_BATCH_SIZE="${RELIK_BATCH_SIZE:-8}"

DATA_DIR="${REPO_ROOT}/dataset/retrieved_data/two_dataset_v1"
COREF_DIR="${REPO_ROOT}/dataset/retrieved_data/${RUN_NAME}_coref"
RELIK_DIR="${REPO_ROOT}/dataset/retrieved_data/${RUN_NAME}_relik"
PRIVACY_DIR="${REPO_ROOT}/dataset/sample_privacy/${RUN_NAME}"
COREF_SHARD_DIR="${REPO_ROOT}/outputs/coref_shards/${RUN_NAME}"
VALIDATION_DIR="${REPO_ROOT}/outputs/validation/${RUN_NAME}"
MANIFEST_DIR="${REPO_ROOT}/outputs/manifests"
COREF_LOG_DIR="${REPO_ROOT}/logs/coref/${RUN_NAME}"
RELIK_LOG_DIR="${REPO_ROOT}/logs/relik/${RUN_NAME}"
RL_LOG_DIR="${REPO_ROOT}/logs/rl_postprocess/${RUN_NAME}"

for executable in "${COREF_PYTHON}" "${MAIN_PYTHON}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Python executable not found: ${executable}" >&2
        exit 1
    fi
done
for path in "${RELIK_MODEL}" "${HF_CACHE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required model/cache path not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p \
    "${COREF_DIR}" \
    "${RELIK_DIR}" \
    "${PRIVACY_DIR}/sampled" \
    "${PRIVACY_DIR}/rl" \
    "${COREF_SHARD_DIR}" \
    "${VALIDATION_DIR}" \
    "${MANIFEST_DIR}" \
    "${COREF_LOG_DIR}" \
    "${RELIK_LOG_DIR}" \
    "${RL_LOG_DIR}"

# Reuse the verified completed PopQA train clean output without copying 218 MB.
REUSABLE_POPQA_TRAIN="${REPO_ROOT}/dataset/retrieved_data/two_dataset_v1_coref/popqa_train_retrieved_clean.jsonl"
REUSED_POPQA_TRAIN="${COREF_DIR}/popqa_train_retrieved_clean.jsonl"
if [[ -f "${REUSABLE_POPQA_TRAIN}" && ! -e "${REUSED_POPQA_TRAIN}" ]]; then
    ln "${REUSABLE_POPQA_TRAIN}" "${REUSED_POPQA_TRAIN}"
fi

cd "${REPO_ROOT}"

"${COREF_PYTHON}" scripts/run_coreferee_shards.py \
    --inputs \
    "${DATA_DIR}/popqa_train_retrieved.jsonl" \
    "${DATA_DIR}/popqa_eval_retrieved.jsonl" \
    "${DATA_DIR}/hotpotqa_train_retrieved.jsonl" \
    "${DATA_DIR}/hotpotqa_eval_retrieved.jsonl" \
    --output-dir "${COREF_DIR}" \
    --shard-dir "${COREF_SHARD_DIR}" \
    --log-dir "${COREF_LOG_DIR}" \
    --coref-script dataset/retrieved_data/clean_retrieved.py \
    --workers "${COREF_WORKERS}" \
    --records-per-shard 250 \
    --per-context-timeout-seconds "${CONTEXT_TIMEOUT_SECONDS}" \
    --reuse-complete \
    --manifest "${MANIFEST_DIR}/coreferee_${RUN_NAME}.json" \
    > "${COREF_LOG_DIR}/full.log" 2>&1

HF_HOME="${HF_CACHE}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
"${MAIN_PYTHON}" scripts/run_relik_files.py \
    --inputs \
    "${COREF_DIR}/popqa_train_retrieved_clean.jsonl" \
    "${COREF_DIR}/popqa_eval_retrieved_clean.jsonl" \
    "${COREF_DIR}/hotpotqa_train_retrieved_clean.jsonl" \
    "${COREF_DIR}/hotpotqa_eval_retrieved_clean.jsonl" \
    --output-dir "${RELIK_DIR}" \
    --log-dir "${RELIK_LOG_DIR}" \
    --manifest "${MANIFEST_DIR}/relik_${RUN_NAME}.json" \
    --relik-script dataset/to_triplets/RELIK_re.py \
    --model "${RELIK_MODEL}" \
    --device cuda \
    --batch-size "${RELIK_BATCH_SIZE}" \
    --seed 42 \
    > "${RELIK_LOG_DIR}/full.log" 2>&1

"${MAIN_PYTHON}" scripts/run_rl_postprocess.py \
    --triples-inputs \
    "${RELIK_DIR}/popqa_train_with_triplets.jsonl" \
    "${RELIK_DIR}/popqa_eval_with_triplets.jsonl" \
    "${RELIK_DIR}/hotpotqa_train_with_triplets.jsonl" \
    "${RELIK_DIR}/hotpotqa_eval_with_triplets.jsonl" \
    --sampled-dir "${PRIVACY_DIR}/sampled" \
    --final-dir "${PRIVACY_DIR}/rl" \
    --validation-dir "${VALIDATION_DIR}" \
    --log-dir "${RL_LOG_DIR}" \
    --manifest "${MANIFEST_DIR}/rl_postprocess_${RUN_NAME}.json" \
    --process-script utils/process_triplets.py \
    --map-script utils/add_sampled_data.py \
    --validator-script scripts/validate_rl_dataset.py \
    --top-k 10 \
    --privacy-rate 0.25 \
    --seed 42 \
    --spacy-model en_core_web_sm \
    > "${RL_LOG_DIR}/full.log" 2>&1
