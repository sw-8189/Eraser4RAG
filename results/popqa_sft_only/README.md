# PopQA SFT-only 对照结果

本目录保存已有 SFT checkpoint 的 PopQA-only 直接评估结果。这里的
“SFT-only”表示不重新训练 SFT，也不运行 PPO；评估直接使用 AutoDL 上已经生成的
`output_checkpoint/SFT`，因此它是 PPO 结果之外的 SFT 控制组。

## 范围与指标

| 子集 | 类型 | 记录数 | 指标 |
| --- | --- | ---: | --- |
| PopQA held-out | retention | 1,000 | `r_pri=0.1167518096`；`r_pub=0.3032580512` |
| PopQA `D_special` | retention | 954 | `r_pri=0.1382274612`；`r_pub=0.2156079682` |
| PopQA inference attack | connectivity | 167 | `r_connect` macro=`0.1392665750`；micro=`0.0884520885` |

六个指标的原始 JSON 位于 `metrics/`。其中 held-out 使用 10,000 个文档，
`D_special` 使用 6,105 个文档；inference-attack 的私有三元组分母为 21,164，
其中 1,872 个被连接。

## 产物

```text
artifacts/rewritten/*.jsonl.gz   三份 SFT 改写输出，Git 中只保留 gzip 压缩副本
metrics/*.json                    三份指标 JSON、validation.json 和完成标记
manifests/run_manifest.json       运行、checkpoint 和输出哈希清单
manifests/rewritten_outputs.json  压缩文件记录数、字节数和解压后 SHA-256
```

未压缩 JSONL 保留在本地的
`outputs/local_artifacts/popqa_sft_only/rewritten/`，该路径被 `.gitignore` 忽略，
便于本机直接打开检查。模型权重不提交到 Git；checkpoint 模型文件的 SHA-256 为
`50401656cdf284a404f48f806cdca79e4d73ca2c795d6ac5ae7a9984dce70e9d`，配置文件
SHA-256 为 `b64f112c4ace1990c3ba1c9da7731c1b09509bf70a42c883795d23055115c15c`。

## 来源与复核

结果来自 AutoDL 项目 `/root/autodl-tmp/Eraser4RAG` 的
`reproduce-v2` 提交 `b11c998ee9aae0105ad34c6f3f0f5c77a37ce421`，运行清单和
远端文件哈希保存在 `metrics/outputs.sha256` 与 `manifests/run_manifest.json`。
评估设备为 48 GiB RTX 4090，随机种子为 42，ReLiK 使用 CUDA。

在仓库根目录执行以下命令可检查 Git 中的压缩结果：

```bash
python scripts/verify_result_artifacts.py \
  --artifact-dir results/popqa_sft_only/artifacts/rewritten \
  --manifest results/popqa_sft_only/manifests/rewritten_outputs.json
python -m pytest -q
```

要恢复单份未压缩文件：

```bash
gzip -dk results/popqa_sft_only/artifacts/rewritten/popqa_eval.jsonl.gz
```
