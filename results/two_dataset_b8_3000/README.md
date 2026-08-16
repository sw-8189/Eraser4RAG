# 两数据集核心复现实验结果

本目录保存一次已经完成的、可公开提交的结果摘要。原始改写 JSONL、模型权重、checkpoint、日志和 ReLiK/Coreferee 中间文件均没有提交；它们仍应在 AutoDL 持久盘中按运行说明重新生成。

## 实验范围

- 数据集：PopQA 和 HotpotQA。
- PPO 训练集：每个数据集 5,000 条记录，共 10,000 条记录；展平后 41,284 个训练样例。
- 评估：每个数据集 1,000 条 held-out 记录；另有独立的 `D_special` 和 inference-attack 子集。
- 训练设备：NVIDIA RTX 4090，约 48 GiB 显存；Torch 2.3.1 + CUDA 12.1。
- 随机种子：42；ReLiK 使用 `use_nme=true`。
- PPO：batch size 8、mini-batch size 4、4 个 PPO epochs、learning rate `1e-5`、`gamma=0.99`、3,000 个 outer steps。
- 隐私惩罚：`p=20` 起，每 350 步增加 5，封顶 `p=40`；step 1,400 之后保持 40。

这是论文方法的两数据集缩减核心复现，不是论文声称的 PopQA、TriviaQA、NQ-Open、HotpotQA 四数据集完整实验，也没有声称复现论文下游 Llama-3 RAG QA 表格。

## 指标

| 数据/子集 | 记录或文档分母 | `r_pub` | `r_pri` | `r_connect` |
| --- | ---: | ---: | ---: | ---: |
| PopQA held-out | 1,000 / 10,000 documents | 0.2980656501 | 0.0963874887 | - |
| HotpotQA held-out | 1,000 / 10,000 documents | 0.5326244030 | 0.1286557895 | - |
| PopQA `D_special` | 954 / 6,105 documents | 0.2081317544 | 0.1147655179 | - |
| HotpotQA `D_special` | 862 / 4,483 documents | 0.4602122605 | 0.2146164628 | - |
| PopQA inference attack | 167 records; 1,752 / 21,164 connected/private triples | - | - | 0.1244024661 macro; 0.0827820828 micro |
| HotpotQA inference attack | 45 records; 440 / 1,163 connected/private triples | - | - | 0.3623177934 macro; 0.3783319003 micro |

结果文件 `metrics.json` 保留了同一批指标的机器可读版本。评估入口是仓库中的 `test_special.py` 和 `test_inferattack.py`；它们不是 pytest 临时脚本。

## 产物校验

- 最终 PPO 权重 SHA-256：`489453a9d640e4914462fa7bd2e221bff274b956fc0390e7413ee54f8920569f`
- 最终 PPO `config.json` SHA-256：`38a78d5bd5c0ec51dbc30eca32309cc68c16499923bcb26804d4493aad5d5179`
- 实验完成时间：2026-08-16 08:53:12 CST

`artifact_hashes.sha256` 还保存了六个原始指标 JSON 和六个改写 JSONL 的脱敏路径及 SHA-256，用于把本摘要绑定到 AutoDL 原始产物。原始指标 JSON 含机器绝对路径，因此没有直接提交；这里的哈希只用于核对，不代表把大文件、权重或机器路径上传到 GitHub。
