# 最终改写输出

本目录保存 PPO `step_final` 对 held-out、`D_special` 和 inference-attack 数据生成的六个最终 JSONL。文件仅做 gzip 压缩，指标评估使用的就是这些内容。

| 文件 | 记录数 | 压缩大小 | 解压后 SHA-256 |
| --- | ---: | ---: | --- |
| `popqa_eval.jsonl.gz` | 1,000 | 25,638,421 B | `f76af5667a1847aceb68825b7eb33040b4c2e6b486d17e06fe50a8bcc8fd867f` |
| `hotpotqa_eval.jsonl.gz` | 1,000 | 5,040,053 B | `c7fa12de3b0d212d4da6f393f9f85e8d560ca30f348cb8295ef0f3ae9580c62e` |
| `popqa_special.jsonl.gz` | 954 | 21,944,770 B | `3bc3871702ab4a01e0bed389900f3f96d04907d1d352c2f43549774c04d28cfa` |
| `hotpotqa_special.jsonl.gz` | 862 | 2,689,214 B | `e9c80e238d848afbd737929b78c0879ef331b24a332d2ed44b9fd78443965fd5` |
| `popqa_inferattack.jsonl.gz` | 167 | 4,897,072 B | `7ebf57382387bc9048dd45d8d5c01dd683fce57b14491cff97acd2a56840b741` |
| `hotpotqa_inferattack.jsonl.gz` | 45 | 186,039 B | `715c1786e4697d8288626f2bd973a1940ef096de938062326601505ebe171c44` |

校验命令：

```bash
python scripts/verify_result_artifacts.py \
  --artifact-dir results/two_dataset_b8_3000/artifacts/rewritten \
  --manifest results/two_dataset_b8_3000/manifests/rewritten_outputs.json
```
