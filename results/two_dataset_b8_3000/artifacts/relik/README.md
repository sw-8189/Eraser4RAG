# ReLiK 三元组产物

本目录保存两数据集正式前处理在 AutoDL 持久盘上生成的四个 ReLiK JSONL 文件。文件采用 gzip 压缩；解压后仍是逐行 JSON，可直接作为隐私采样和 RL 后处理的输入。

| 文件 | 记录数 | 压缩大小 | 解压后 SHA-256 |
| --- | ---: | ---: | --- |
| `popqa_train_with_triplets.jsonl.gz` | 5,000 | 97,601,605 B | `9863c6ec174970a4d3a3e88b7a0b3122b10e5a72cb65f203985468ba904cdf4e` |
| `popqa_eval_with_triplets.jsonl.gz` | 1,000 | 21,818,358 B | `fc5c0c678f5c0658c49cd5e5f79ad948071bdb3811335e3145e70fc1049d7bdf` |
| `hotpotqa_train_with_triplets.jsonl.gz` | 5,000 | 19,481,387 B | `2f33aa43d9b64948ea2e98dc478c129bab9fc5814a3d35306858a1d5218a8044` |
| `hotpotqa_eval_with_triplets.jsonl.gz` | 1,000 | 3,922,697 B | `6d520632404d86a57402adde131d991e27f4609834bf553f50b0e3b40a45454b` |

哈希和记录数来自同次运行的 `manifests/relik_two_dataset_v1_timeout120.json`，并已在下载后重新解压校验。可随时复核：

```bash
python scripts/verify_result_artifacts.py \
  --artifact-dir results/two_dataset_b8_3000/artifacts/relik \
  --manifest results/two_dataset_b8_3000/manifests/relik_two_dataset_v1_timeout120.json
```

如需恢复原文件，可执行 `gzip -dk <file>.jsonl.gz`。PopQA train 使用 gzip 最高压缩级别以满足 GitHub 单文件限制，其余文件使用快速压缩；压缩级别不影响解压后的内容和 SHA-256。
