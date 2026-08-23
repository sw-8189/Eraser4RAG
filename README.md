# Eraser4RAG 复现实验

本仓库整理了 Eraser4RAG（*Learning to Erase Private Knowledge from Multi-Documents for Retrieval-Augmented Large Language Models*）的可运行代码、固定依赖、AutoDL 运行流程，以及一次已经完成的 PopQA + HotpotQA 两数据集核心复现实验结果。

> **复现范围**：正式结果覆盖 PopQA 和 HotpotQA 两数据集的 SFT、PPO 与 ReLiK 指标评估。

## 当前交付内容

- 可运行的 SFT、PPO、ReLiK 抽取、隐私采样、数据校验和指标评估代码。
- 两个相互隔离的 Python 3.10.14 环境依赖清单，以及版本和模型 revision 配置。
- [`docs/AUTODL_RUNBOOK.md`](docs/AUTODL_RUNBOOK.md)：中文 AutoDL 环境、运行、监控和验收手册。
- [`docs/REPRODUCTION_DEVIATIONS.md`](docs/REPRODUCTION_DEVIATIONS.md)：中文说明论文、作者代码与本次实验之间的数据和实现差异。
- `results/two_dataset_b8_3000/` 中从 AutoDL 导出的正式指标、运行 manifest、训练曲线、ReLiK 三元组和最终改写压缩文件。
- `results/popqa_sft_only/` 中已有 SFT checkpoint 的 PopQA-only 对照评估、六项指标、验证清单和压缩改写输出；该目录不再单独维护说明文档，结果摘要统一见本文“全部评价结果对比”章节。
- `tests/` 中的单元和契约测试源码。
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

`dataset/constructed_dataset/popqa_10_25_filtered_new.rar` 是 7.36 MiB 的正式 SFT 输入归档，可恢复解压后的 SFT JSONL。

## 环境

不要直接安装根目录的 `requirements.txt`：Coreferee 与 ReLiK 的 spaCy 约束会互相冲突。使用两个 Python 3.10.14 环境：

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


在 AutoDL 上建议把 Conda 环境、HF cache、临时目录和模型放到 `/root/autodl-tmp`，不要占满系统盘。安装完成后先执行：

```bash
python scripts/check_environment.py --mode main --cpu-only
python -m pytest -q
```


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

## 全部评价结果对比

下表把已有 SFT checkpoint 的 PopQA-only 对照和以 SFT 为初始化、继续进行 PPO 的两数据集实验放在一起。`r_pub` 越高表示公共知识保留越多，`r_pri` 越低表示私有知识残留越少；`r_connect` 越低表示推断攻击中的私有三元组越不容易连通。注意：只有相同数据集、相同评估子集和相同有效分母的行适合直接比较。

| 训练阶段 | 数据集 | 评估子集 | 记录数 / 有效分母 | `r_pub` | `r_pri` | `r_connect` macro | `r_connect` micro |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| SFT-only | PopQA | 留出集 | 1,000 / 10,000 文档 | 0.3032580512 | 0.1167518096 | - | - |
| SFT-only | PopQA | `D_special` 特殊集 | 954 / 6,105 文档 | 0.2156079682 | 0.1382274612 | - | - |
| SFT-only | PopQA | 推断攻击 | 167 / 21,164 私有三元组（1,872 个连接） | - | - | 0.1392665750 | 0.0884520885 |
| SFT+PPO | PopQA | 留出集 | 1,000 / 10,000 文档 | 0.2980656501 | 0.0963874887 | - | - |
| SFT+PPO | HotpotQA | 留出集 | 1,000 / 10,000 文档 | 0.5326244030 | 0.1286557895 | - | - |
| SFT+PPO | PopQA | `D_special` 特殊集 | 954 / 6,105 文档 | 0.2081317544 | 0.1147655179 | - | - |
| SFT+PPO | HotpotQA | `D_special` 特殊集 | 862 / 4,483 文档 | 0.4602122605 | 0.2146164628 | - | - |
| SFT+PPO | PopQA | 推断攻击 | 167 / 21,164 私有三元组（1,752 个连接） | - | - | 0.1244024661 | 0.0827820828 |
| SFT+PPO | HotpotQA | 推断攻击 | 45 / 1,163 私有三元组（440 个连接） | - | - | 0.3623177934 | 0.3783319003 |

SFT-only 只评估 PopQA，未重新训练 SFT、未运行 PPO，也未处理 HotpotQA；因此 HotpotQA 没有 SFT-only 对照行。六项 SFT-only 指标的原始 JSON 在 `results/popqa_sft_only/metrics/`，三份改写 JSONL 以 gzip 形式保存在 `results/popqa_sft_only/artifacts/rewritten/`，记录数与解压后 SHA-256 清单在 `results/popqa_sft_only/manifests/rewritten_outputs.json`。

本次结果来自 AutoDL 项目 `/root/autodl-tmp/Eraser4RAG`，运行时 Git 提交为 `b11c998ee9aae0105ad34c6f3f0f5c77a37ce421`，设备为 48 GiB RTX 4090，随机种子为 42，ReLiK 使用 CUDA。模型文件 SHA-256 为 `50401656cdf284a404f48f806cdca79e4d73ca2c795d6ac5ae7a9984dce70e9d`，配置文件 SHA-256 为 `b64f112c4ace1990c3ba1c9da7731c1b09509bf70a42c883795d23055115c15c`。


## 结果文件与验收

完整的两数据集指标 JSON、运行 manifest、训练曲线、ReLiK 三元组和最终改写压缩文件见 [`results/two_dataset_b8_3000/README.md`](results/two_dataset_b8_3000/README.md) 与 [`results/two_dataset_b8_3000/metrics.json`](results/two_dataset_b8_3000/metrics.json)。上面的统一表是所有已完成评价结果的汇总，机器可读文件仍按训练阶段分别保存，避免覆盖或混淆原始产物。

最终 PPO `step_final/model.safetensors` 的 SHA-256 为 `489453a9d640e4914462fa7bd2e221bff274b956fc0390e7413ee54f8920569f`。四个 ReLiK 三元组、六个最终改写、六个原始指标 JSON、运行 manifest 和 3,000 步训练曲线已收入结果目录。

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

算法与数据处理差异见 [`docs/REPRODUCTION_DEVIATIONS.md`](docs/REPRODUCTION_DEVIATIONS.md)。

## 参考

- [Hugging Face TRL](https://github.com/huggingface/trl)
- [Self-RAG retrieval code](https://github.com/AkariAsai/self-rag)
