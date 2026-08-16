# AutoDL 运行手册

本文档说明如何在新的 AutoDL 实例上恢复本仓库的 PopQA + HotpotQA 两数据集实验，以及如何继续使用已经完成的云端产物。命令默认在仓库根目录执行。

## 1. 获取代码

```bash
cd /root/autodl-tmp
git clone https://github.com/sw-8189/Eraser4RAG.git
cd Eraser4RAG
git checkout main
git rev-parse HEAD
```

`main` 与 `reproduce-v2` 保存同一份正式交付。环境、缓存、模型和 checkpoint 都放在 `/root/autodl-tmp`，避免占满系统盘：

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

## 2. 创建主环境

主环境用于 SFT、ReLiK、PPO、改写和指标评估：

```bash
conda create -p /root/autodl-tmp/conda/envs/eraser-main python=3.10.14 -y
conda activate /root/autodl-tmp/conda/envs/eraser-main
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/eraser-main.txt

curl -L --fail --retry 8 --retry-delay 3 -C - \
  -o /root/autodl-tmp/model_wheels/en_core_web_sm-3.7.1-py3-none-any.whl \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
pip install --no-deps \
  /root/autodl-tmp/model_wheels/en_core_web_sm-3.7.1-py3-none-any.whl

pip check
python scripts/check_environment.py --mode main --cpu-only
```

切换到有 GPU 的实例后再执行完整检查：

```bash
nvidia-smi
python scripts/check_environment.py --mode main
```

关键版本为 Python 3.10.14、Torch 2.3.1+cu121、Transformers 4.41.2、TRL 0.11.4、ReLiK 1.0.7 和 spaCy 3.7.5。不要单独升级 Transformers 或 TRL。

## 3. 创建 Coreferee 环境

Coreferee 与 ReLiK 的 spaCy 依赖冲突，因此使用独立环境：

```bash
conda create -p /root/autodl-tmp/conda/envs/eraser-coref python=3.10.14 -y
conda activate /root/autodl-tmp/conda/envs/eraser-coref
pip install -r requirements/eraser-coref.txt

curl -L --fail --retry 8 --retry-delay 3 -C - \
  -o /root/autodl-tmp/model_wheels/en_core_web_lg-3.5.0-py3-none-any.whl \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl
pip install --no-deps \
  /root/autodl-tmp/model_wheels/en_core_web_lg-3.5.0-py3-none-any.whl

curl -L --fail --retry 8 --retry-delay 3 -C - \
  -o /root/autodl-tmp/model_wheels/coreferee_model_en.zip \
  https://raw.githubusercontent.com/richardpaulhudson/coreferee/aeb42a447484ad019fef4ea2dc6f5c952af29794/models/coreferee_model_en.zip
mkdir -p /root/autodl-tmp/tmp/coreferee_model_en
unzip -q /root/autodl-tmp/model_wheels/coreferee_model_en.zip \
  -d /root/autodl-tmp/tmp/coreferee_model_en
pip install --no-deps --force-reinstall /root/autodl-tmp/tmp/coreferee_model_en

pip check
python scripts/check_environment.py --mode coref --skip-pipeline-load
```

无卡模式通常只有 2 GiB 内存，只做版本检查。加载 `en_core_web_lg` 和 Coreferee 的完整检查应在 GPU 实例上执行：

```bash
python scripts/check_environment.py --mode coref
```

## 4. 无卡与有卡阶段

| 阶段 | 无卡模式 | GPU 实例 |
| --- | --- | --- |
| clone、下载、解压、静态测试 | 可用 | 可用 |
| 主环境 CPU 检查 | 可用 | 可用 |
| Coreferee 完整加载与批处理 | 内存通常不足 | 使用 CPU，但需要 GPU 实例提供的较大内存 |
| SFT、ReLiK、PPO、模型改写和指标提取 | 不可用 | 必须 |
| 查看、压缩和同步已有结果 | 可用 | 可用 |

## 5. 下载固定模型

回到主环境：

```bash
conda activate /root/autodl-tmp/conda/envs/eraser-main
python scripts/download_models.py --stage sft
python scripts/download_models.py --stage relik
python scripts/verify_models.py --stage sft --model models/sft --device cpu
python scripts/verify_models.py --stage relik --model models/relik --device cuda
```

模型 revision 在 `configs/reproduction.yaml` 中固定。下载 manifest 位于各模型目录，后续可在离线模式运行。

## 6. 恢复 SFT 数据

```bash
python scripts/restore_sft_dataset.py
python scripts/validate_sft_dataset.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --output-dir outputs/validation
```

正式输入应为 23,074 条。归档文件已在 Git 中，解压后的 108 MiB JSONL 不进入 Git。

## 7. 下载并构造两数据集检索输入

```bash
python scripts/download_reduced_qa_sources.py \
  --output-dir dataset/source_data/two_dataset_v1 \
  --cache-dir /root/autodl-tmp/hf_cache/datasets \
  --manifest outputs/manifests/two_dataset_sources.json

python scripts/prepare_reduced_retrieval_data.py \
  --popqa-input dataset/source_data/two_dataset_v1/popqa_train.parquet \
  --hotpot-train-input dataset/source_data/two_dataset_v1/hotpotqa_train.parquet \
  --hotpot-eval-input dataset/source_data/two_dataset_v1/hotpotqa_validation.parquet \
  --output-root dataset/retrieved_data/two_dataset_v1 \
  --manifest outputs/manifests/two_dataset_retrieval_manifest.json \
  --train-size 5000 --eval-size 1000 --top-k 10 --seed 42

python scripts/validate_retrieved_data.py \
  --data dataset/retrieved_data/two_dataset_v1/*.jsonl \
  --expected-contexts 10 \
  --output outputs/validation/two_dataset_retrieval.json
```

生成四个文件：PopQA/HotpotQA 各 5,000 条 train 和 1,000 条 eval，每条保留 10 个 context。

## 8. Coreferee、ReLiK 和 RL 数据

在 GPU 实例上用持久会话启动：

```bash
screen -dmS eraser-two-data bash scripts/run_two_dataset_timeout120.sh
screen -ls
tail -f logs/coref/two_dataset_v1_timeout120/full.log
```

该入口按顺序执行：

1. Coreferee 分片清洗；单个 context 最长 120 秒，超时保留原文并记录。
2. ReLiK CUDA 三元组抽取。
3. 按种子 42 进行 25% 隐私采样和图处理。
4. 生成并严格验证四个 RL JSONL。

验收文件：

```text
dataset/retrieved_data/two_dataset_v1_timeout120_relik/*_with_triplets.jsonl
dataset/sample_privacy/two_dataset_v1_timeout120/rl/*_rl.jsonl
outputs/manifests/coreferee_two_dataset_v1_timeout120.json
outputs/manifests/relik_two_dataset_v1_timeout120.json
outputs/manifests/rl_postprocess_two_dataset_v1_timeout120.json
```

正式 ReLiK 压缩产物也已收入 `results/two_dataset_b8_3000/artifacts/relik/`，可用 `scripts/verify_result_artifacts.py` 校验并恢复。

## 9. ReLiK 一致性门禁

```bash
conda activate /root/autodl-tmp/conda/envs/eraser-main
python scripts/validate_relik_consistency.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --relik-model models/relik \
  --sample-size 500 --seed 42 --device cuda \
  --output-dir outputs/relik_consistency
```

本次正式报告为 `pass`，public/private micro recall 分别为 0.9536468984 和 0.9653130288。非 dry-run PPO 会检查这份 500 样本报告。

## 10. 正式 SFT

```bash
screen -dmS eraser-sft-full bash scripts/run_sft_full.sh
screen -ls
tail -f logs/sft-full/train.log
```

参数为 Flan-T5-large、3 epochs、learning rate `5e-5`、输入长度 1300、目标长度 128、seed 42。完成条件：

```bash
test -s output_checkpoint/SFT/model.safetensors
test -s output_checkpoint/SFT/training_manifest.json
```

本次 SFT 的 `train_runtime` 为 18,781.7946 秒。

## 11. 正式 PPO

```bash
export MAIN_PYTHON=/root/autodl-tmp/conda/envs/eraser-main/bin/python
screen -dmS eraser-ppo-two bash scripts/run_two_dataset_ppo.sh
screen -ls
tail -f logs/ppo/two_dataset_b8_3000/run.log
```

默认参数为 batch 8、mini-batch 4、4 个 PPO epochs、`gamma=0.99`、3,000 个 outer steps。`p` 在 step 1–349 为 20，之后每 350 步加 5，step 1,400 起保持 40。48 GiB RTX 4090 已完成该配置；batch 16 会 OOM。

新运行默认输出到 `output_checkpoint/RL-two_dataset_b8_3000/`。本次已完成云端运行使用的实际目录是：

```text
output_checkpoint/RL-two-dataset-b8-3000/step_final/
```

两者名称不同只源于当次启动时显式设置了 `OUTPUT_DIR`。完成条件是 `step_final/model.safetensors` 和 `step_final/config.json` 均存在。当前 launcher 不提供 PPO 自动断点续训，因此不要在训练期间关闭云实例。

## 12. 训练后评估

先生成 `D_special` 和 inference-attack 数据：

```bash
bash scripts/build_two_dataset_eval_sets.sh
```

若评估刚完成的新运行，直接执行：

```bash
screen -dmS eraser-eval-two bash scripts/run_two_dataset_eval.sh
```

若复用本次已有的云端 checkpoint，显式指定真实路径：

```bash
FINAL_CHECKPOINT=/root/autodl-tmp/Eraser4RAG/output_checkpoint/RL-two-dataset-b8-3000/step_final \
  screen -dmS eraser-eval-two bash scripts/run_two_dataset_eval.sh
```

查看进度：

```bash
screen -ls
tail -f logs/evaluation/two_dataset_b8_3000/*
```

评估完成必须同时满足：

```text
outputs/evaluation/two_dataset_b8_3000/metrics/complete.txt
outputs/evaluation/two_dataset_b8_3000/metrics/{popqa,hotpotqa}_{eval,special,inferattack}.json
```

launcher 会在写 `complete.txt` 前校验六个改写文件的记录数、有效分母和指标范围。

## 13. 已完成云端产物

云端项目根目录为 `/root/autodl-tmp/Eraser4RAG`，正式产物位置如下：

| 产物 | 相对路径 |
| --- | --- |
| SFT | `output_checkpoint/SFT/` |
| PPO | `output_checkpoint/RL-two-dataset-b8-3000/step_final/` |
| PPO 输入 | `dataset/sample_privacy/two_dataset_v1_timeout120/rl/` |
| ReLiK 三元组 | `dataset/retrieved_data/two_dataset_v1_timeout120_relik/` |
| 最终改写 | `outputs/evaluation/two_dataset_b8_3000/rewritten/` |
| 指标 | `outputs/evaluation/two_dataset_b8_3000/metrics/` |
| PPO 日志 | `logs/ppo/two_dataset_b8_3000/run.log` |

其脱敏副本、训练曲线、ReLiK 三元组和最终改写压缩文件位于仓库的 `results/two_dataset_b8_3000/`。

## 14. 会话与关机

- SSH、浏览器或 VS Code 断开不会终止 `screen` 中的任务。
- AutoDL 关机或释放实例会终止正在运行的进程。
- SFT 可从 Trainer checkpoint 手动恢复；当前 PPO launcher 不自动续跑。
- 无卡模式可查看和同步结果，但不能继续 GPU 推理或训练。
- 只有看到对应完成文件并校验通过后，才能关机。

常用命令：

```bash
screen -ls
screen -r eraser-ppo-two
# 在 screen 内按 Ctrl+A，再按 D，可退出但不终止任务
```

## 15. 最终检查

```bash
python -m pytest -q
python -m compileall -q RL_train.py data_structure.py finetune_rewrite_doc.py \
  rewrite_docs_special.py rewrite_docs_inferattack_inference.py scripts utils tests
python scripts/verify_result_artifacts.py \
  --artifact-dir results/two_dataset_b8_3000/artifacts/relik \
  --manifest results/two_dataset_b8_3000/manifests/relik_two_dataset_v1_timeout120.json
python scripts/verify_result_artifacts.py \
  --artifact-dir results/two_dataset_b8_3000/artifacts/rewritten \
  --manifest results/two_dataset_b8_3000/manifests/rewritten_outputs.json
```

算法、数据源和论文设置之间的差异见 `docs/REPRODUCTION_DEVIATIONS.md`。
