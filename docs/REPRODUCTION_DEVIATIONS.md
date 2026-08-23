# Eraser4RAG v2 复现差异说明

_本文说明公开代码、论文 v2[^1] 与本仓库两数据集复现之间的差异，供结果解释和后续复跑使用。_

---

## 分类

本仓库以作者公开代码提交 `2fb526451ad49735cd6d1826b8ff882e40c49a5f`[^2] 为基线。以下标签不代表优劣，只说明设置来源：

| 标签 | 含义 |
| --- | --- |
| `[AUTHOR-CODE]` | 作者公开快照中可以直接确认的行为 |
| `[PAPER-V2]` | 论文 v2 明确给出，但公开代码缺失或已过时的设置 |
| `[RECONSTRUCTION]` | 为使公开快照可运行、兼容或内部一致而补建的行为，不视为作者未公开设置 |
| `[ENGINEERING]` | 路径、校验、日志和续跑等不改变算法目标的工程处理 |

## 作者行为与论文 v2 对齐

| 项目 | 分类 | 本仓库采用的口径 |
| --- | --- | --- |
| 改写提示与特殊标记 | `[AUTHOR-CODE]` | 保留作者的公有/私有提示格式和 8 个特殊 token |
| PPO 固定参照 | `[AUTHOR-CODE]` | 始终使用原始文档的局部 `public/private` 三元组；不把改写文本重新抽取的三元组当作固定参照 |
| PPO 动态预测 | `[AUTHOR-CODE]` + `[RECONSTRUCTION]` | 仅对当前策略生成的改写文本运行 ReLiK，并与固定参照计算奖励；固定和动态三元组统一使用“无序实体对保留第一条关系”的去重规则 |
| 奖励函数 | `[AUTHOR-CODE]` + `[PAPER-V2]` | 保持 `r_pub * exp(-p * r_pri)`，ReLiK 默认 `use_nme=True` |
| PPO 样本过滤 | `[AUTHOR-CODE]` | 保留局部公有三元组不少于 5、局部私有三元组不少于 2 的条件 |
| SFT 设置 | `[PAPER-V2]` | Flan-T5-large，3 epochs，学习率 `5e-5`，输入长度 1300，目标长度 128 |
| PPO 折扣 | `[PAPER-V2]` | 使用 `gamma=0.99`；公开代码只设置了学习率调度器的 gamma |
| 隐私惩罚 `p` | `[PAPER-V2]` + `[RECONSTRUCTION]` | step 1-349/350-699/700-1049/1050-1399 分别使用 20/25/30/35，step 1400 起封顶 40，修正公开代码可能递增到 45 的边界问题 |

## 数据口径

论文所用的固定 Wikipedia/Contriever 检索索引未随公开仓库发布，因此本次缩减实验使用以下可追踪替代。这些差异会改变模型看到的文档和三元组分布，结果不应视为论文原表的同条件重跑。

| 数据集/步骤 | 本次设置 | 对结果解释的影响 |
| --- | --- | --- |
| PopQA 检索 | 固定 `MinaGabriel/popqa-with-retrieval-20` revision `dcc3f4f72fab2f7bca386c51e5cf329109727919`；每条记录已有 top-20 文档，本次按原顺序截断为 top-10 | 使用第三方预检索结果，不是重新运行论文检索器 |
| HotpotQA 检索 | 固定 `hotpotqa/hotpot_qa` revision `1908d6afbbead072334abe2965f91bd2709910ab`；使用 official `distractor` contexts，少于 10 个 context 的记录先排除 | official distractor 文档替代论文检索结果，文档难度和覆盖率可能不同 |
| 数据规模 | seed 42；PopQA 和 HotpotQA 各固定抽样 5,000 条 train、1,000 条 eval | 这是两数据集缩减实验，不是论文的 PopQA、TriviaQA、NQ、HotpotQA 四数据集全量训练 |
| Coreferee | 每个 context 最多处理 120 秒；超时后保留原文，并在分片 provenance 中记录 record/context 身份 | 超时 context 未完成指代消解，可能影响后续 ReLiK 三元组及指标；样本不会因超时静默丢失 |
| 隐私划分 | seed 42，按 25% 采样公有图三元组；若候选会损害 QA 连通性，则返回公有候选集，并在每次转移后重建私有图 | 属于对公开后处理缺口的确定性重建，不保证与作者未公开流水线逐字节一致 |

## 兼容性重建

- `[RECONSTRUCTION]` 使用两个 Python 3.10.14 环境：`eraser-main` 运行 SFT、ReLiK、PPO 和评估，`eraser-coref` 单独运行 Coreferee，避免二者 spaCy 版本冲突。
- `[RECONSTRUCTION]` 缺失的私有三元组处理模块由统一的三元组解析、序列化和无序实体对去重实现替代；文档 ID 在对齐边界统一为字符串。
- `[RECONSTRUCTION]` 被采样进私有图的三元组会即时更新连通图；因 QA 约束被拒绝的候选不会从公有和私有集合同时消失。
- `[RECONSTRUCTION]` 空公有参照定义为 `r_pub=1`，空私有参照定义为 `r_pri=0`，避免除零。正式指标中若存在空参照，其数量必须与指标一起报告。
- `[RECONSTRUCTION]` `max_steps` 表示精确的外层 rollout/update 次数。本次 3,000 steps 是缩减复现的算力选择，不是论文报告的停止条件。
- `[RECONSTRUCTION]` 特殊集改写保留源 `ctxs`，推断攻击输出统一使用全局 `privacy` 字段，修复公开生产端与评估端的 schema 不一致。
- `[ENGINEERING]` 模型 revision 固定在 `configs/reproduction.yaml`。论文和公开代码未给出原始模型快照 commit，因此这些 revision 只代表本次复现环境。

## 指标解释

| 指标 | 本仓库含义 | 使用限制 |
| --- | --- | --- |
| `r_pub` | ReLiK 在改写文本中保留的参照公有三元组比例 | 越高表示公共知识保留越多；需结合空公有参照数量和评估子集说明 |
| `r_pri` | ReLiK 在改写文本中仍能抽取的参照私有三元组比例 | 越低表示私有三元组残留越少，但不等同于形式化隐私保证 |
| `r_connect` macro | 先按样本计算连通比例，再对样本平均 | 易受每条记录私有三元组数量差异影响，只作为补充统计 |
| `r_connect` micro | 汇总所有 connected/private triples 后计算比例 | 与论文的全局比例口径对照时使用该值 |

`r_connect` 保留作者的无向图连通性和实体 substring 短名匹配规则，因此它衡量结构可连接性，不等同于语义推断。固定参照、PPO 奖励和最终评估均依赖同一 ReLiK 模型族，500 样本一致性检查可以量化实现一致性，但不能消除这种测量循环性。

held-out、`D_special` 和 inference-attack 是三个不同分母的评估集，不能把它们的数值混为同一指标。`D_special` 只保留符合特殊关系条件的记录，inference-attack 只保留可构造跨文档攻击的记录。

## 已完成的 AutoDL 实验

本仓库已在 48 GiB RTX 4090 上完成一次 PopQA + HotpotQA 核心复现，正式产物见 [`results/two_dataset_b8_3000/`](../results/two_dataset_b8_3000/)。

- SFT 使用作者 PopQA 23,074 行输入，按上述论文 v2 参数完成训练。
- 固定 50/500 样本 ReLiK 一致性门禁均通过；500 样本公有/私有 micro recall 为 `0.9536/0.9653`。
- PPO 使用每个数据集 5,000 条训练记录，过滤并展平为 41,284 个样例；batch 8、mini-batch 4、4 PPO epochs、学习率 `1e-5`、`gamma=0.99`，完成 3,000 个 outer steps。
- 最终 checkpoint 和六项 final-policy ReLiK 评估均完成；权重 SHA-256 为 `489453a9d640e4914462fa7bd2e221bff274b956fc0390e7413ee54f8920569f`。
- 另外使用同一 AutoDL 环境直接评估了已有的 PopQA SFT checkpoint，作为不含 PPO 的 SFT-only 控制组；该对照不处理 HotpotQA，六项指标和哈希统一记录在仓库根目录 `README.md` 的“全部评价结果对比”章节。

| 数据/子集 | `r_pub` | `r_pri` | `r_connect` |
| --- | ---: | ---: | ---: |
| PopQA held-out | 0.2980656501 | 0.0963874887 | - |
| HotpotQA held-out | 0.5326244030 | 0.1286557895 | - |
| PopQA `D_special` | 0.2081317544 | 0.1147655179 | - |
| HotpotQA `D_special` | 0.4602122605 | 0.2146164628 | - |
| PopQA inference-attack | - | - | 0.1244024661 macro / 0.0827820828 micro |
| HotpotQA inference-attack | - | - | 0.3623177934 macro / 0.3783319003 micro |

## 未覆盖范围

- 未使用 TriviaQA 和 NQ，未复现论文四数据集联合强化学习。
- 未实现下游 Llama-3 RAG QA accuracy、QA 隐私攻击、实体删除/PPL、完整基线和消融实验。
- 仅完成 seed 42 的一次缩减 PPO，不能据此报告跨 seed 方差或统计显著性。
- Coreferee 的 token 级文本重建可能改变空格和标点；substring 实体匹配也可能对短实体产生误匹配。
- 由于缺少论文固定检索索引和原始模型 snapshot，本结果证明核心流水线可运行，但不构成论文全部表格的严格复现。

## 参考

[^1]: “Learning to Erase Private Knowledge from Multi-Documents for Retrieval-Augmented Large Language Models.” _arXiv:2504.09910v2_ (2025). https://arxiv.org/abs/2504.09910

[^2]: Eraser4RAG author repository. “Public source snapshot.” https://github.com/yjEugenia/Eraser4RAG/tree/2fb526451ad49735cd6d1826b8ff882e40c49a5f
