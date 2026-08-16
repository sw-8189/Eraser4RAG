# Eraser4RAG 复现实验

本仓库整理了 Eraser4RAG（*Learning to Erase Private Knowledge from Multi-Documents for Retrieval-Augmented Large Language Models*）的可运行代码、固定依赖、AutoDL 运行流程，以及一次已经完成的 PopQA + HotpotQA 两数据集核心复现实验结果。

> **范围声明**：本仓库的正式结果是论文方法的两数据集缩减复现。论文描述的四个 PPO 数据集为 PopQA、TriviaQA、NQ-Open 和 HotpotQA；本次为了控制算力和时间只使用 PopQA、HotpotQA。因此不能把本仓库的数值直接称为论文四数据集表格的完整复现。论文下游 Llama-3 RAG QA 评估入口在公开代码中也不完整，本仓库没有虚构该部分结果。

## 当前交付内容

- 可运行的 SFT、PPO、ReLiK 抽取、隐私采样、数据校验和指标评估代码。
- 两个相互隔离的 Python 3.10.14 环境依赖清单，以及版本和模型 revision 配置。
- AutoDL 从数据恢复到评估的命令行 runbook。
- `results/two_dataset_b8_3000/` 中脱敏后的正式指标、样本分母和 checkpoint/config 哈希。
- `tests/` 中的单元和契约测试源码。测试源码是可复现性的一部分，pytest 缓存、测试输出和 smoke 产物不提交。

过程性计划、机器状态记录、日志、模型权重、checkpoint、完整 JSONL、检索索引和本机环境 lock 均由 `.gitignore` 排除，保留在 `plan/` 或 AutoDL 持久盘中，不作为 GitHub 交付物。

## 目录说明

```text
configs/                  固定实验参数和模型 revision
dataset/                  代码与可恢复的 SFT RAR 输入
docs/                     AutoDL runbook 和复现偏差说明
requirements/             原作者依赖及两个可安装环境清单
scripts/                  下载、恢复、预处理、校验和正式 launcher
tests/                    不依赖大模型的回归/契约测试
utils/                    三元组、奖励和后处理工具
results/                  小型、脱敏、可提交的实验结果摘要
```

`outputs/`、`logs/`、`models/`、`output_checkpoint/`、生成的 JSONL 和 `plan/` 不应上传。`dataset/constructed_dataset/popqa_10_25_filtered_new.rar` 是 7.36 MiB 的正式 SFT 输入归档，保留它是为了让新环境能够恢复输入；解压后的 108 MiB JSONL 被忽略。

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
git checkout reproduce-v2
```

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

完整论文四数据集流程还需要固定的 Wikipedia/Contriever corpus 和 index。公开作者快照没有提供完整、可固定 revision 的构建入口，不能用任意 QA context 冒充四数据集正式语料。

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

SFT checkpoint 应保存在 `output_checkpoint/SFT`。正式运行会写入被忽略的 manifest 和 checkpoint；提交时只保留运行说明和结果摘要，不提交权重。

## 两数据集 PPO

本仓库提供参数化入口，默认值就是本次已经完成的核心实验：PopQA/HotpotQA 训练集各 5,000 条，batch size 8、mini-batch size 4、4 个 PPO epochs、`gamma=0.99`、3,000 个 outer steps。`p` 从 20 开始，每 350 步加 5，封顶 40：

```bash
screen -dmS eraser-ppo-two bash scripts/run_two_dataset_ppo.sh
screen -ls
tail -f logs/ppo/two_dataset_b8_3000/run.log
```

launcher 会在非 dry-run 前强制检查 500 样本 ReLiK consistency report。48 GiB RTX 4090 上本次使用 batch 8；batch 16 曾发生 OOM。正式四数据集实验不要把本次两数据集 checkpoint 当作论文默认结果。

## 训练后评估

PPO 完成并出现 `step_final` 后，生成 held-out、`D_special` 和 inference-attack 改写，再计算 `r_pub`、`r_pri`、`r_connect`：

```bash
bash scripts/build_two_dataset_eval_sets.sh
screen -dmS eraser-eval-two bash scripts/run_two_dataset_eval.sh
screen -ls
tail -f logs/evaluation/two_dataset_b8_3000/*
```

其中 `test_special.py` 是 retention 指标入口，`test_inferattack.py` 是 connectivity 指标入口；二者支持 `--output-json`，输出的机器可读结果默认写到被忽略的 `outputs/evaluation/.../metrics/`。

## 已完成结果

完整的脱敏结果表和 JSON 见 [`results/two_dataset_b8_3000/README.md`](results/two_dataset_b8_3000/README.md) 与 [`results/two_dataset_b8_3000/metrics.json`](results/two_dataset_b8_3000/metrics.json)。核心数值如下：

| 子集 | `r_pub` | `r_pri` | `r_connect` |
| --- | ---: | ---: | ---: |
| PopQA held-out | 0.2980656501 | 0.0963874887 | - |
| HotpotQA held-out | 0.5326244030 | 0.1286557895 | - |
| PopQA `D_special` | 0.2081317544 | 0.1147655179 | - |
| HotpotQA `D_special` | 0.4602122605 | 0.2146164628 | - |
| PopQA inference attack | - | - | 0.1244024661 macro / 0.0827820828 micro |
| HotpotQA inference attack | - | - | 0.3623177934 macro / 0.3783319003 micro |

最终 PPO 权重不在 GitHub；其 SHA-256 为 `489453a9d640e4914462fa7bd2e221bff274b956fc0390e7413ee54f8920569f`。这个哈希用于核对 AutoDL 上的本地 checkpoint 与结果摘要是否一致。

## 测试和提交边界

建议在无卡模式执行：

```bash
python -m pytest -q
python -m compileall -q RL_train.py data_structure.py finetune_rewrite_doc.py \
  rewrite_docs_special.py rewrite_docs_inferattack_inference.py scripts utils tests
git diff --check
```

应提交：源码、`requirements/`、`configs/`、正式 launcher、`tests/`、docs、中文 README、`results/` 和可恢复的 RAR 输入。不要提交：pytest/cache、smoke 输出、过程 Markdown、机器专属 lock、日志、密码/token、模型/索引/checkpoint、大型生成数据。`.gitignore` 已覆盖这些目录；提交前可用 `git status --ignored` 复核。

## 复现偏差和限制

请先阅读 [`docs/REPRODUCTION_DEVIATIONS.md`](docs/REPRODUCTION_DEVIATIONS.md)。其中区分作者原始行为、论文 v2 对齐、兼容性重建和纯工程改动，并明确说明：

1. 本次正式结果只覆盖 PopQA + HotpotQA。
2. 检索输入和 Coreferee 超时处理是有 provenance 的工程重建，不等价于作者隐藏数据。
3. PPO 的 3,000 steps 是本次核心复现实验选择，不是论文报告的停止步数。
4. `r_pub/r_pri/r_connect` 是 ReLiK 图和连接性评估，不是隐私安全保证。
5. 下游 Llama-3 QA 评估仍需补齐固定 Wikipedia/index 和公开可审计的 evaluator 后才能声称完整论文表格复现。

## 参考

- [Hugging Face TRL](https://github.com/huggingface/trl)
- [Self-RAG retrieval code](https://github.com/AkariAsai/self-rag)
