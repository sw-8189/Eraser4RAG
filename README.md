# Eraser4RAG 复现实验

本仓库整理了 Eraser4RAG（*Learning to Erase Private Knowledge from Multi-Documents for Retrieval-Augmented Large Language Models*）的可运行代码、固定依赖、AutoDL 运行流程，以及一次已经完成的 PopQA + HotpotQA 两数据集核心复现实验结果。

> **复现范围**：正式结果覆盖 PopQA 和 HotpotQA 两数据集的 SFT、PPO 与 ReLiK 指标评估；论文四数据集实验和下游 RAG QA 指标不在本次结果内。

## 当前交付内容

- 可运行的 SFT、PPO、ReLiK 抽取、隐私采样、数据校验和指标评估代码。
- 两个相互隔离的 Python 3.10.14 环境依赖清单，以及版本和模型 revision 配置。
- [`docs/AUTODL_RUNBOOK.md`](docs/AUTODL_RUNBOOK.md)：中文 AutoDL 环境、运行、监控和验收手册。
- [`docs/REPRODUCTION_DEVIATIONS.md`](docs/REPRODUCTION_DEVIATIONS.md)：中文说明论文、作者代码与本次实验之间的数据和实现差异。
- `results/two_dataset_b8_3000/` 中从 AutoDL 导出的正式指标、运行 manifest、训练曲线、ReLiK 三元组和最终改写压缩文件。
- `results/popqa_sft_only/` 中已有 SFT checkpoint 的 PopQA-only 对照评估、六项指标、验证清单和压缩改写输出；该目录不再单独维护说明文档，结果摘要统一见本文“已完成结果”章节。
- `tests/` 中的单元和契约测试源码。测试源码是可复现性的一部分，pytest 缓存、测试输出和 smoke 产物不提交。

## 目录说明

```text
configs/                  固定实验参数和模型 revision
dataset/                  代码与可恢复的 SFT RAR 输入
docs/                     AutoDL runbook 和复现偏差说明
requirements/             原作者依赖及两个可安装环境清单
scripts/                  下载、恢复、预处理、校验和正式 launcher
tests/                    不依赖大模型的回归/契约测试
utils/                    三元组、奖励和后处理工具
results/                  脱敏结果、运行记录和重要中间产物
```

`dataset/constructed_dataset/popqa_10_25_filtered_new.rar` 是 7.36 MiB 的正式 SFT 输入归档，可恢复解压后的 SFT JSONL。模型权重、checkpoint、缓存、日志、过程记录和可重建的大型 JSONL 由 `.gitignore` 排除。

## 环境

不要直接安装根目录的 `requirements.txt`：它是作者原始快照，Coreferee 与 ReLiK 的 spaCy 约束互相冲突。使用两个 Python 3.10.14 环境：

| 环境 | 用途 | 依赖文件 |
| --- | --- | --- |
| `eraser-main` | SFT、ReLiK、PPO、改写和指标评估 | `requirements/eraser-main.txt` |
| `eraser-coref` | Coreferee + spaCy 检索文本清洗 | `requirements/eraser-coref.txt` |

关键版本是 Torch 2.3.1（CUDA 12.1 wheel）、Transformers 4.41.2、TRL 0.11.4、ReLiK 1.0.7、spaCy 3.7.5（main）和 Coreferee 1.4.1 + spaCy 3.5.0（coref）。AutoDL 的完整安装、固定模型 wheel、缓存位置和检查命令见 [`docs/AUTODL_RUNBOOK.md`](docs/AUTODL_RUNBOOK.md)。

## 从 GitHub 开始

```bash
git clone https://github.com/sw-8189/Eraser4RAG.git
cd Eraser4RAG
git checkout main
```

`main` 是当前交付分支；`reproduce-v2` 保留提交历史和同一份可运行代码。若只复现实验，建议使用 `main`。

在 AutoDL 上建议把 Conda 环境、HF cache、临时目录和模型放到 `/root/autodl-tmp`，不要占满系统盘。安装完成后先执行：

```bash
python scripts/check_environment.py --mode main --cpu-only
python -m pytest -q
```

切换到有 GPU 的实例后，再执行 `python scripts/check_environment.py --mode main` 和 Coreferee 的完整 pipeline 检查。无卡模式适合下载、解压、版本检查和静态测试；完整 Coreferee、SFT、ReLiK、PPO 和评估需要 GPU 实例（Coreferee 本身主要占 CPU/RAM，但无卡容器通常只有 2 GiB 内存）。

## 数据恢复和前处理

恢复作者 SFT 输入并验证 SHA-256：

```bash
python scripts/restore_sft_dataset.py
python scripts/validate_sft_dataset.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --output-dir outputs/validation
```

两数据集缩减实验固定使用以下公开源：

- PopQA：`MinaGabriel/popqa-with-retrieval-20@dcc3f4f72fab2f7bca386c51e5cf329109727919`。
- HotpotQA distractor：`hotpotqa/hotpot_qa@1908d6afbbead072334abe2965f91bd2709910ab`。

先下载固定 revision 并转换为本项目的 5,000 train / 1,000 eval、top-10 retrieved schema。源文件、转换结果和 manifest 都被 Git 忽略，但可由以下命令重新生成：

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

验证通过后运行完整缩减前处理入口：

```bash
bash scripts/run_two_dataset_timeout120.sh
```

该 launcher 依次运行 Coreferee 分片清洗、ReLiK CUDA 抽取、25% 隐私采样、全局/局部三元组映射和严格 RL JSONL 校验；它不会启动 PPO。正式数据的来源、样本数、超时处理和与论文的差异记录在 `docs/REPRODUCTION_DEVIATIONS.md`。

## SFT

先完成 ReLiK 50/500 样本一致性门禁和 SFT smoke，再运行正式 SFT。正式默认值为 Flan-T5-large、3 epochs、learning rate `5e-5`、输入长度 1300、目标长度 128：

```bash
python scripts/validate_relik_consistency.py \
  --data dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl \
  --relik-model models/relik --sample-size 500 --seed 42 \
  --device cuda --output-dir outputs/relik_consistency

screen -dmS eraser-sft-full bash scripts/run_sft_full.sh
screen -ls
tail -f logs/sft-full/train.log
```

SFT checkpoint 保存在 `output_checkpoint/SFT`，训练 manifest 的脱敏副本已收入正式结果目录。

## 两数据集 PPO

本仓库提供参数化入口，默认值就是本次已经完成的核心实验：PopQA/HotpotQA 训练集各 5,000 条，batch size 8、mini-batch size 4、4 个 PPO epochs、`gamma=0.99`、3,000 个 outer steps。`p` 从 20 开始，每 350 步加 5，封顶 40：

```bash
screen -dmS eraser-ppo-two bash scripts/run_two_dataset_ppo.sh
screen -ls
tail -f logs/ppo/two_dataset_b8_3000/run.log
```

launcher 会在训练前检查 500 样本 ReLiK consistency report。48 GiB RTX 4090 上 batch 8 已完成，batch 16 会 OOM。`screen` 中的任务不受 SSH 或 VS Code 断开影响，但云实例关机会终止进程。

## 训练后评估

PPO 完成并出现 `step_final` 后，生成 held-out、`D_special` 和 inference-attack 改写，再计算 `r_pub`、`r_pri`、`r_connect`：

```bash
bash scripts/build_two_dataset_eval_sets.sh
screen -dmS eraser-eval-two bash scripts/run_two_dataset_eval.sh
screen -ls
tail -f logs/evaluation/two_dataset_b8_3000/*
```

已完成实验在云端使用的 PPO 路径是 `output_checkpoint/RL-two-dataset-b8-3000/step_final`。若直接复用该 checkpoint 重新评估，应显式指定：

```bash
FINAL_CHECKPOINT=/root/autodl-tmp/Eraser4RAG/output_checkpoint/RL-two-dataset-b8-3000/step_final \
  screen -dmS eraser-eval-two bash scripts/run_two_dataset_eval.sh
```

其中 `test_special.py` 是 retention 指标入口，`test_inferattack.py` 是 connectivity 指标入口；二者支持 `--output-json`，输出的机器可读结果默认写到被忽略的 `outputs/evaluation/.../metrics/`。

## SFT-only PopQA 对照

该对照实验直接评估已有的 `output_checkpoint/SFT`，不重新训练 SFT，不运行 PPO，也不包含 HotpotQA。它用于与两数据集 PPO 最终策略结果区分，判断仅经过 SFT 的模型在 PopQA 上保留公共知识和残留私有知识的情况。

| 子集 | 评估类型 | 记录数 | `r_pub` | `r_pri` | `r_connect` |
| --- | --- | ---: | ---: | ---: | ---: |
| PopQA held-out（留出集） | 保留率 | 1,000 | 0.3032580512 | 0.1167518096 | - |
| PopQA `D_special`（特殊集） | 保留率 | 954 | 0.2156079682 | 0.1382274612 | - |
| PopQA inference attack（推断攻击） | 连通性 | 167 | - | - | 0.1392665750 macro；0.0884520885 micro |

held-out 使用 10,000 个文档，`D_special` 使用 6,105 个文档；inference-attack 的私有三元组分母为 21,164，其中 1,872 个被连接。六项指标的原始 JSON 在 `results/popqa_sft_only/metrics/`，三份改写 JSONL 以 gzip 形式保存在 `results/popqa_sft_only/artifacts/rewritten/`，记录数与解压后 SHA-256 清单在 `results/popqa_sft_only/manifests/rewritten_outputs.json`。

本次结果来自 AutoDL 项目 `/root/autodl-tmp/Eraser4RAG`，运行时 Git 提交为 `b11c998ee9aae0105ad34c6f3f0f5c77a37ce421`，设备为 48 GiB RTX 4090，随机种子为 42，ReLiK 使用 CUDA。SFT checkpoint 不提交到 Git；模型文件 SHA-256 为 `50401656cdf284a404f48f806cdca79e4d73ca2c795d6ac5ae7a9984dce70e9d`，配置文件 SHA-256 为 `b64f112c4ace1990c3ba1c9da7731c1b09509bf70a42c883795d23055115c15c`。

SFT-only 产物校验命令：

```bash
python scripts/verify_result_artifacts.py \
  --artifact-dir results/popqa_sft_only/artifacts/rewritten \
  --manifest results/popqa_sft_only/manifests/rewritten_outputs.json
```

在 AutoDL 评估完成、或已把未压缩 JSONL 和 checkpoint 放回本地后，再运行严格的
三份 JSONL/指标校验：

```bash
python scripts/validate_popqa_sft_only_eval.py \
  --rewrite-root outputs/local_artifacts/popqa_sft_only/rewritten \
  --metric-root results/popqa_sft_only/metrics \
  --output outputs/local_artifacts/popqa_sft_only/validation.json \
  --manifest outputs/local_artifacts/popqa_sft_only/run_manifest.json \
  --checkpoint output_checkpoint/SFT
```

## 已完成结果

完整的两数据集脱敏结果表和 JSON 见 [`results/two_dataset_b8_3000/README.md`](results/two_dataset_b8_3000/README.md) 与 [`results/two_dataset_b8_3000/metrics.json`](results/two_dataset_b8_3000/metrics.json)。上面的“SFT-only PopQA 对照”章节同时给出 SFT 控制组结果，避免把不同训练阶段的指标混在同一张表中。

| 子集 | `r_pub` | `r_pri` | `r_connect` |
| --- | ---: | ---: | ---: |
| PopQA held-out | 0.2980656501 | 0.0963874887 | - |
| HotpotQA held-out | 0.5326244030 | 0.1286557895 | - |
| PopQA `D_special` | 0.2081317544 | 0.1147655179 | - |
| HotpotQA `D_special` | 0.4602122605 | 0.2146164628 | - |
| PopQA inference attack | - | - | 0.1244024661 macro / 0.0827820828 micro |
| HotpotQA inference attack | - | - | 0.3623177934 macro / 0.3783319003 micro |

最终 PPO `step_final/model.safetensors` 的 SHA-256 为 `489453a9d640e4914462fa7bd2e221bff274b956fc0390e7413ee54f8920569f`。权重不进入 Git，四个 ReLiK 三元组、六个最终改写、六个原始指标 JSON、运行 manifest 和 3,000 步训练曲线已收入结果目录。

## 验证

建议在无卡模式执行：

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
git diff --check
```

模型、checkpoint、缓存、完整日志和过程记录不进入 Git。算法与数据处理差异见 [`docs/REPRODUCTION_DEVIATIONS.md`](docs/REPRODUCTION_DEVIATIONS.md)。

## 参考

- [Hugging Face TRL](https://github.com/huggingface/trl)
- [Self-RAG retrieval code](https://github.com/AkariAsai/self-rag)
