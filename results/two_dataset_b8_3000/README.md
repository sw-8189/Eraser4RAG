# 两数据集核心复现实验结果

本目录来自 AutoDL 持久盘上的正式 SFT、PPO 和评估产物。提交前只做了绝对路径脱敏，没有改动指标、训练记录、样本数或哈希值。

本目录只说明两数据集 PPO 核心复现。PopQA SFT-only 控制组的指标不在本目录重复，统一见仓库根目录 `README.md` 的“全部评价结果对比”章节。

## 实验设置

- 数据集：PopQA 和 HotpotQA。
- PPO 输入：每个数据集 5,000 条，共 10,000 条记录；过滤并展平后为 41,284 个训练样例。
- 评估：每个数据集 1,000 条 held-out 记录，另有 `D_special` 和 inference-attack 子集。
- 设备：48 GiB NVIDIA RTX 4090，Torch 2.3.1，CUDA 12.1。
- PPO：batch size 8、mini-batch size 4、4 个 PPO epochs、learning rate `1e-5`、`gamma=0.99`、3,000 个 outer steps。
- 隐私惩罚：`p=20` 起，每 350 步增加 5，step 1,400 起保持 `p=40`。

## 指标

| 数据/子集 | 有效分母 | `r_pub` | `r_pri` | `r_connect` |
| --- | ---: | ---: | ---: | ---: |
| PopQA held-out | 1,000 records / 10,000 documents | 0.2980656501 | 0.0963874887 | - |
| HotpotQA held-out | 1,000 records / 10,000 documents | 0.5326244030 | 0.1286557895 | - |
| PopQA `D_special` | 954 records / 6,105 documents | 0.2081317544 | 0.1147655179 | - |
| HotpotQA `D_special` | 862 records / 4,483 documents | 0.4602122605 | 0.2146164628 | - |
| PopQA inference attack | 167 records；1,752 / 21,164 connected/private triples | - | - | 0.1244024661 macro；0.0827820828 micro |
| HotpotQA inference attack | 45 records；440 / 1,163 connected/private triples | - | - | 0.3623177934 macro；0.3783319003 micro |

最终评估完成时间为 2026-08-16 08:53:12（Asia/Shanghai）。最终 PPO `step_final/model.safetensors` 的 SHA-256 为 `489453a9d640e4914462fa7bd2e221bff274b956fc0390e7413ee54f8920569f`；权重本身不进入 Git。

## 目录内容

```text
metrics.json                 汇总指标和正式运行参数
metrics/                     AutoDL 评估器直接输出的六个脱敏 JSON
manifests/                   retrieval、Coreferee、ReLiK、RL 后处理记录
validation/                  500 样本 ReLiK 一致性报告与固定采样索引
training/                    SFT manifest、PPO 逐步曲线和 PPO 摘要
artifacts/relik/             四个完整 ReLiK 三元组 JSONL 压缩文件
artifacts/rewritten/         六个最终策略改写 JSONL 压缩文件
```

PPO 日志共解析出 3,000 条 reward 行和 3,000 条 update 行。`training/ppo_training_curve.csv` 保留每一步的 `p`、`mean_reward`、`mean_r_pub`、`mean_r_pri`、learning rate、KL 和更新时间；完整噪声日志不提交。

## 云端原始位置

| 产物 | AutoDL 持久盘路径 |
| --- | --- |
| SFT checkpoint 与 manifest | `output_checkpoint/SFT/` |
| PPO 最终 checkpoint | `output_checkpoint/RL-two-dataset-b8-3000/step_final/` |
| ReLiK 三元组 | `dataset/retrieved_data/two_dataset_v1_timeout120_relik/` |
| PPO 输入 JSONL | `dataset/sample_privacy/two_dataset_v1_timeout120/rl/` |
| 最终改写和六项指标 | `outputs/evaluation/two_dataset_b8_3000/` |
| PPO 日志 | `logs/ppo/two_dataset_b8_3000/run.log` |

上述路径均相对于云端项目根目录 `/root/autodl-tmp/Eraser4RAG`。最终改写和 ReLiK 三元组已压缩收入仓库；完整 RL 输入保留在云端及本地 Git 外备份中，可由已提交三元组和后处理代码重建。

## 校验

```bash
python scripts/verify_result_artifacts.py \
  --artifact-dir results/two_dataset_b8_3000/artifacts/relik \
  --manifest results/two_dataset_b8_3000/manifests/relik_two_dataset_v1_timeout120.json

python scripts/verify_result_artifacts.py \
  --artifact-dir results/two_dataset_b8_3000/artifacts/rewritten \
  --manifest results/two_dataset_b8_3000/manifests/rewritten_outputs.json

python -m pytest -q
```
