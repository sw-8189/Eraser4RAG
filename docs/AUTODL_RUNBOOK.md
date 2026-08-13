# AutoDL Runbook: Eraser4RAG v2 Core Reproduction

This runbook begins after the local P0–P7 preparation passes. It deliberately
uses two environments and enforces the ReLiK consistency gate before full PPO.
Commands assume the repository root as the working directory.

## 1. Obtain the prepared branch

```bash
git clone https://github.com/sw-8189/Eraser4RAG.git Eraser4RAG
cd Eraser4RAG
git checkout reproduce-v2
git rev-parse HEAD
```

The local preparation agent does not push automatically. Confirm that the
branch containing the P0–P7 changes has been pushed before using these commands.

## 2. Put environments and caches on the data disk

AutoDL's system disk is small. Keep environments, package caches, model
snapshots, and temporary files under the persistent data disk:

```bash
export ERASER_DATA_ROOT=/root/autodl-tmp
export CONDA_PKGS_DIRS="$ERASER_DATA_ROOT/conda/pkgs"
export PIP_CACHE_DIR="$ERASER_DATA_ROOT/pip_cache"
export HF_HOME="$ERASER_DATA_ROOT/hf_cache"
export TMPDIR="$ERASER_DATA_ROOT/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR" \
  "$ERASER_DATA_ROOT/model_wheels"
source /root/miniconda3/etc/profile.d/conda.sh
```

These exports must be repeated after reconnecting unless they are added to the
shell startup file. The commands below use prefix environments so that they
remain on the data disk.

## 3. Create the main environment

```bash
conda create -p "$ERASER_DATA_ROOT/conda/envs/eraser-main" python=3.10.14 -y
conda activate "$ERASER_DATA_ROOT/conda/envs/eraser-main"
```

Check the image and driver first:

```bash
nvidia-smi
```

This reconstruction targets Torch 2.3.1 with the CUDA 12.1 wheel. Install it
only on an AutoDL image whose NVIDIA driver supports CUDA 12.1, then install
the pinned top-level dependencies:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r requirements/eraser-main.txt
pip check

curl -L --fail --retry 8 --retry-delay 3 -C - \
  -o "$ERASER_DATA_ROOT/model_wheels/en_core_web_sm-3.7.1-py3-none-any.whl" \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
python -m zipfile -t \
  "$ERASER_DATA_ROOT/model_wheels/en_core_web_sm-3.7.1-py3-none-any.whl"
pip install --no-deps \
  "$ERASER_DATA_ROOT/model_wheels/en_core_web_sm-3.7.1-py3-none-any.whl"

# Valid in AutoDL no-GPU mode:
python scripts/check_environment.py --mode main --cpu-only

# Repeat this GPU gate after switching to a GPU instance:
python scripts/check_environment.py --mode main
pip freeze > requirements/eraser-main-lock.txt
```

Expected critical versions include Transformers 4.41.2, TRL 0.11.4, ReLiK
1.0.7, spaCy 3.7.5, and Torch 2.3.1. Do not upgrade TRL or Transformers.

The verified SHA-256 of the spaCy model wheel is
`86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889`.

## 4. Create the coreference environment

```bash
conda create -p "$ERASER_DATA_ROOT/conda/envs/eraser-coref" python=3.10.14 -y
conda activate "$ERASER_DATA_ROOT/conda/envs/eraser-coref"
pip install -r requirements/eraser-coref.txt

curl -L --fail --retry 8 --retry-delay 3 -C - \
  -o "$ERASER_DATA_ROOT/model_wheels/en_core_web_lg-3.5.0-py3-none-any.whl" \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl
python -m zipfile -t \
  "$ERASER_DATA_ROOT/model_wheels/en_core_web_lg-3.5.0-py3-none-any.whl"
pip install --no-deps \
  "$ERASER_DATA_ROOT/model_wheels/en_core_web_lg-3.5.0-py3-none-any.whl"

# Coreferee's built-in installer follows the mutable master branch. Pin the
# exact upstream commit containing the English model instead.
curl -L --fail --retry 8 --retry-delay 3 -C - \
  -o "$ERASER_DATA_ROOT/model_wheels/coreferee_model_en.zip" \
  https://raw.githubusercontent.com/richardpaulhudson/coreferee/aeb42a447484ad019fef4ea2dc6f5c952af29794/models/coreferee_model_en.zip
python -m zipfile -t "$ERASER_DATA_ROOT/model_wheels/coreferee_model_en.zip"
rm -rf "$TMPDIR/coreferee_model_en"
mkdir -p "$TMPDIR/coreferee_model_en"
unzip -q "$ERASER_DATA_ROOT/model_wheels/coreferee_model_en.zip" \
  -d "$TMPDIR/coreferee_model_en"
pip install --no-deps --force-reinstall "$TMPDIR/coreferee_model_en"
rm -rf "$TMPDIR/coreferee_model_en"

pip check

# The 2 GB no-GPU container can verify all package/model versions safely:
python scripts/check_environment.py --mode coref --skip-pipeline-load

# Run the real large spaCy + Coreferee pipeline after switching back to a GPU
# instance (the pipeline uses CPU, but that instance has a higher RAM limit):
python scripts/check_environment.py --mode coref
```

Coreference inference itself runs on CPU. However, AutoDL's 2 GB no-GPU cgroup
is too small to load `en_core_web_lg` and Coreferee together while VS Code and
the platform services are running; the process was killed at that boundary.
Use no-GPU mode for download, installation, and the version gate, then repeat
the full pipeline check on the GPU instance. The fixed Coreferee model commit
is an engineering reproducibility pin; the package's default command,
`python -m coreferee install en`, is deliberately not used because it downloads
from a floating branch.

Verified download SHA-256 values:

```text
en_core_web_lg-3.5.0-py3-none-any.whl
c8ac64840c1eb3e3ca7bd38bd1e1c48fb0faeb2449d54d01d5ce629af4595775

coreferee_model_en.zip
aec5662b4af38fbf4b8c67e4aada8b828c51d4a224b5e08f7b2b176c02d8780f
```

After the real pipeline check passes, capture the resolved environment:

```bash
pip freeze > requirements/eraser-coref-lock.txt
```

Return to the main environment for all steps except raw retrieved-document
coreference cleaning:

```bash
conda activate "$ERASER_DATA_ROOT/conda/envs/eraser-main"
```

Run a stage preflight whenever the workspace changes. A nonzero exit is a hard
stop, with every missing artifact listed in JSON:

```bash
python scripts/autodl_preflight.py --stage sft
python scripts/autodl_preflight.py --stage ppo
python scripts/autodl_preflight.py --stage full
```

## 5. Place and validate the author SFT dataset

Expected path:

```text
dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl
```

On a fresh clone the tracked input is the archive
`dataset/constructed_dataset/popqa_10_25_filtered_new.rar`; the JSONL is
ignored because it is a generated large artifact. Restore it before validation:

```bash
# RAR5 archives require a recent native RAR extractor on Ubuntu 22.04.
command -v unrar >/dev/null || \
  (apt-get update && apt-get install -y unrar)
python scripts/restore_sft_dataset.py
```

The script verifies the archive SHA-256 and refuses to overwrite a mismatching
existing JSONL. The expected archive SHA-256 is
`b32d7e45c457d27e398b69d58678fea5bc40271445d798c494c016a119493186`.
The restored JSONL must match
`c215e99f5e70b8cab8fa3631fb3b8708b90506527bc5d350fa6f8d658dcb9f36`.

Validate every row before model work:

```bash
python scripts/validate_sft_dataset.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --output-dir outputs/validation
```

Both JSON and Markdown reports must say `PASS`. Preserve the report with the
experiment artifacts.

## 6. Download and verify only the first required models

The repository configuration freezes one immutable commit for each model as
of 2026-08-12. These commits are an engineering reconstruction choice because
the paper/public snapshot did not report Hub revisions. The downloader reads
them automatically and records the same SHA in each model manifest:

```bash
python scripts/download_models.py --stage sft
python scripts/download_models.py --stage relik
python scripts/verify_models.py --stage sft --model models/sft \
  --load-model --device cuda
python scripts/verify_models.py --stage relik --model models/relik --device cuda
```

Each model directory must contain `download_manifest.json` with the resolved
immutable revision. If a commit cannot be chosen in advance, the explicit
`--allow-floating-revision` option resolves the current head once and records
it; this is a documented fallback, not the formal default.

For the Flan stages, the downloader intentionally fetches only configuration,
tokenizer files, and `model.safetensors`. PyTorch does not need the duplicate
`pytorch_model.bin`, TensorFlow, or Flax weights. An interrupted local-dir
download resumes only when its Hub metadata identifies the same immutable
commit and every completed visible file belongs to that stage's allow-list.

Flan-T5-base is optional for API smoke tests:

```bash
python scripts/download_models.py --stage smoke
python scripts/verify_models.py --stage smoke --model models/smoke --load-model
```

Do not download Llama 3 at this stage.

## 7. Mandatory ReLiK consistency gate

### 6.1 Fixed 50-sample smoke

```bash
python scripts/validate_relik_consistency.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --relik-model models/relik \
  --sample-size 50 \
  --seed 42 \
  --device cuda \
  --output-dir outputs/relik_consistency/smoke
```

Review loading, triple formatting, and mismatch output even if recall is low.

### 6.2 Fixed 500-sample formal engineering gate

```bash
python scripts/validate_relik_consistency.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --relik-model models/relik \
  --sample-size 500 \
  --seed 42 \
  --device cuda \
  --output-dir outputs/relik_consistency
```

Required artifacts:

```text
outputs/relik_consistency/report.json
outputs/relik_consistency/report.md
outputs/relik_consistency/mismatches.jsonl
outputs/relik_consistency/sample_indices.json
```

The 0.80 value is an engineering warning threshold, not a paper threshold. If
any key recall is materially below it, stop and check model ID/revision,
`relik==1.0.7`, `use_nme`, duplicate rules, relation labels, and serialization.
Do not launch full PPO without a reviewed 500-sample report.

## 8. SFT smoke and full SFT

First run a short Flan-T5-large smoke:

```bash
python finetune_rewrite_doc.py \
  --data-dir dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --model-dir models/sft \
  --tokenizer-dir models/sft \
  --output-dir output_checkpoint/SFT-smoke \
  --max-samples 100 \
  --seed 42 \
  --smoke-test
```

Verify forward/backward, eval, save, and reload. Then run the formal settings:

```bash
python finetune_rewrite_doc.py \
  --data-dir dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --model-dir models/sft \
  --tokenizer-dir models/sft \
  --output-dir output_checkpoint/SFT \
  --seed 42
```

Formal defaults are epochs 3, lr `5e-5`, input length 1300, and target length
128. Resume only from a checkpoint created by the same model/tokenizer setup:

```bash
python finetune_rewrite_doc.py <same-arguments> \
  --resume-from-checkpoint output_checkpoint/SFT/checkpoint-<step>
```

## 9. PPO engineering smoke from the SFT file

Build and validate the explicitly reconstructed smoke corpus:

```bash
python scripts/build_popqa_rl_from_sft.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --output outputs/smoke/popqa_regrouped_author_sft_rl.jsonl \
  --max-groups 100 \
  --force

python scripts/validate_rl_dataset.py \
  --data outputs/smoke/popqa_regrouped_author_sft_rl.jsonl \
  --output-dir outputs/validation
```

This dataset is for engineering smoke only. It has no author query/answer IDs.

Dry-run data loading without model initialization:

```bash
python RL_train.py \
  --data-dir-1 outputs/smoke/popqa_regrouped_author_sft_rl.jsonl \
  --data-dir-2 "" --data-dir-3 "" --data-dir-4 "" \
  --max-samples 16 --batch-size 16 --dry-run
```

Then test generation, reward, one update, and ten updates in order:

```bash
python RL_train.py <same-data-arguments> \
  --model-dir output_checkpoint/SFT \
  --tokenizer-dir output_checkpoint/SFT \
  --relik-model models/relik \
  --relik-consistency-report outputs/relik_consistency/report.json \
  --max-samples 16 --batch-size 16 --generate-only

python RL_train.py <same-data-and-model-arguments> \
  --max-samples 16 --batch-size 16 --reward-only

python RL_train.py <same-data-and-model-arguments> \
  --max-samples 16 --batch-size 16 --max-steps 1 \
  --output-dir output_checkpoint/RL-smoke-1

python RL_train.py <same-data-and-model-arguments> \
  --max-samples 160 --batch-size 16 --max-steps 10 \
  --output-dir output_checkpoint/RL-smoke-10
```

Replace the `<same-...-arguments>` placeholders with the exact arguments from
the preceding complete command when creating shell scripts or job manifests.

## 10. Build the formal four-dataset PPO data

Download Contriever only when the pinned corpus/index preparation begins:

```bash
python scripts/download_models.py --stage retrieval
python scripts/verify_models.py --stage retrieval --model models/retrieval \
  --device cuda
```

The required chain is:

```text
PopQA / TriviaQA / NQ-Open / HotpotQA
-> query and answers
-> Contriever-MS MARCO over a pinned Wikipedia corpus, top 10
-> eraser-coref cleaning
-> eraser-main ReLiK extraction
-> global graph merge
-> seeded 25% privacy sampling and connectivity filtering
-> global-to-local mapping
-> validated RL JSONL
```

The public snapshot does not include a complete pinned Wikipedia retrieval and
index build entry point. Record the corpus snapshot, passage segmentation,
index revision, retrieval commit, and document IDs before continuing. Do not
substitute arbitrary QA contexts and call them the formal corpus.

For already retrieved files, coreference runs only in `eraser-coref`:

```bash
conda activate eraser-coref
python dataset/retrieved_data/clean_retrieved.py --help
# Run once per dataset with explicit --input/--output paths.
```

Switch back for ReLiK and graph processing:

```bash
conda activate eraser-main
python dataset/to_triplets/RELIK_re.py --help
python utils/process_triplets.py --help
python utils/add_sampled_data.py --help
```

Validate each final corpus with QA fields required:

```bash
python scripts/validate_rl_dataset.py --data <dataset>.jsonl \
  --require-qa --output-dir outputs/validation
```

## 11. Formal PPO gate and launch

Before full PPO, all of these must exist or pass:

- main `pip check` and environment checker;
- ReLiK minimal inference;
- SFT dataset validation;
- reviewed 50- and 500-sample consistency reports;
- Flan-T5-large SFT smoke and full SFT checkpoint;
- PPO generate, reward, 1-step, and 10-step smoke;
- four validated formal RL datasets.

Formal launch template:

```bash
python RL_train.py \
  --data-dir-1 dataset/sample_privacy/popqa_10_25_trp.jsonl \
  --data-dir-2 dataset/sample_privacy/triviaqa_10_25_trp.jsonl \
  --data-dir-3 dataset/sample_privacy/NQ-open_10_25_trp.jsonl \
  --data-dir-4 dataset/sample_privacy/hotpotqa_10_25_trp.jsonl \
  --model-dir output_checkpoint/SFT \
  --tokenizer-dir output_checkpoint/SFT \
  --relik-model models/relik \
  --output-dir output_checkpoint/RL \
  --seed 42 \
  --batch-size 16 \
  --mini-batch-size 8 \
  --ppo-epochs 4 \
  --gamma 0.99 \
  --max-steps 200000
```

The paper does not report a definitive PPO stopping count. The configured
200,000 outer updates are an explicit reconstruction choice; preserve the
value in each run manifest and do not present it as an author-reported setting.

The training log must show `p=20,25,30,35,40` at the documented boundaries and
must never show 45.

`RL_train.py` enforces the consistency report and its sample manifest before
any non-dry model run. A `WARNING` report stops execution unless the mismatch
has been reviewed, documented, and `--allow-relik-gate-warning` is supplied.

## 12. Core evaluation and Llama 3

First produce rewritten documents, then compute `r_pri`, `r_pub`, and
`r_connect` with explicit model/data/output arguments. Inspect current CLI help
before each invocation:

```bash
python rewrite_docs_special.py --help
python test_special.py --help
python rewrite_docs_inferattack_inference.py --help
python test_inferattack.py --help
```

Only after privacy/utility evaluation is stable should Llama 3 be downloaded:

```bash
python scripts/download_models.py --stage eval
python scripts/verify_models.py --stage eval --model models/eval \
  --load-model --device cuda
```

The current public snapshot does not contain a complete downstream Llama-3 QA
evaluation entry point. Implementing or sourcing it must be recorded as a
separate reconstruction before claiming paper RAG accuracy.

## 13. Artifact and secret discipline

Do not commit model weights, checkpoints, Wikipedia indexes, generated large
JSONL files, `.env`, or access tokens. Preserve small manifests, config files,
validation reports, exact commands, Git SHA, GPU model, CUDA/driver details,
package locks, seeds, wall time, and checkpoint hashes with each experiment.

## 14. Readiness boundary

Passing the commands above proves that the prepared code can begin staged
AutoDL validation. It does not by itself prove full paper reproduction. The
formal four-dataset RL JSONL files, a pinned Wikipedia corpus and Contriever
index, the 500-sample ReLiK gate, trained checkpoints, and the downstream
Llama-3 QA evaluator must all exist and pass their respective checks before
claiming paper-table results.
