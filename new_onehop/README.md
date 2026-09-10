# new_onehop 数据处理说明

本目录用于构建动态 Top-k 预测任务的数据集。

当前阶段主要负责：

- 整理已有 RAG 问答结果；
- 构建每个 query 在不同 k 下的性能曲线；
- 根据性能曲线构造 Hard Oracle；
- 根据 F1 和 token cost 构造 Soft Oracle Distribution；
- 生成可直接用于后续模型训练的 Oracle 数据集。

数据处理阶段已经完成；Phase 1 在不改动这些数据的前提下，使用 question-only 的
15 分类 CE baseline 训练动态 Top-k 预测器。Phase 2 在同一 Dataset、划分和
DeBERTa encoder 上新增严格有序 threshold head，并只使用由 Hard Oracle 在线构造的
plain ordinal BCE；不读取 Soft Oracle 作为监督。训练与评测命令见
`scripts/README.md`。

---

# 1. 原始实验结果与数据划分

RAG 问答实验已经提前完成，原始预测结果分为两部分：

```text
kbqa/predictions/dev_3000/   # train 数据来源
kbqa/predictions/test/       # test 数据来源
```

两部分均包含 3 个数据集：

```text
nq
squad
trivia
```

每个数据集分别运行：

```text
bm25_retrieval_count = 1, 2, ..., 15
```

每个数据划分各有：

```text
3 datasets × 15 k = 45 experiments
```

当前生成的数据规模为：

- `data/train`：来自 `predictions/dev_3000`，每个数据集 3000 条，共 9000 条；
- `data/test`：来自 `predictions/test`，每个数据集 500 条，共 1500 条；
- `data/splits`：从 9000 条训练池中按数据集分层划分出的 8100 个 train ID 和 900 个 val ID。

Oracle 只为 train 数据生成；test 只保留 curves，不计算 Oracle。

例如 NQ 在 `k=5` 时的实验目录：

```text
kbqa/predictions/dev_3000/
└── oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__5___distractor_count__1/
    └── per_question_eval__nq_to_nq__dev_3000_subsampled.json
```

`predictions/dev_3000` 和 `predictions/test` 中的已有实验目录共同作为原始数据来源。

处理原则：

- 不移动；
- 不修改；
- 不复制到 `new_onehop`；
- `new_onehop` 中的数据处理脚本直接读取这些实验结果。

---

# 2. 目录结构

```text
kbqa/new_onehop/
├── README.md
│
├── configs/
│   ├── data_config.yaml
│   └── train_config.yaml
│
├── scripts/
│   ├── README.md
│   ├── data/
│   │   ├── build_query_curves.py
│   │   ├── validate_curves.py
│   │   └── build_oracle_labels.py
│   ├── train/
│   └── eval/
│
└── data/
    ├── train/
    │   ├── curves/
    │   │   ├── nq.jsonl
    │   │   ├── squad.jsonl
    │   │   ├── trivia.jsonl
    │   │   └── all.jsonl
    │   └── oracle/
    │       ├── nq.jsonl
    │       ├── squad.jsonl
    │       ├── trivia.jsonl
    │       └── all.jsonl
    ├── test/
    │   └── curves/
    │       ├── nq.jsonl
    │       ├── squad.jsonl
    │       ├── trivia.jsonl
    │       └── all.jsonl
    └── splits/
        ├── train_ids.json
        └── val_ids.json
```

整体数据处理流程：

```text
kbqa/predictions/dev_3000/
        │
        │ 45 个 fixed-k RAG 实验结果（9000 个问题）
        ▼
build_query_curves.py --split train
        │
        ▼
data/train/curves/
        │
        ├── validate_curves.py --split train
        │
        ▼
build_oracle_labels.py
        │
        ▼
data/train/oracle/ + data/splits/
        │
        ▼
后续直接用于模型训练

kbqa/predictions/test/
        │
        │ 45 个 fixed-k RAG 实验结果（1500 个问题）
        ▼
build_query_curves.py --split test
        │
        ▼
data/test/curves/
        │
        └── validate_curves.py --split test

test 流程到 curves 为止，不生成 Oracle。
```

---

# 3. `data/train/curves/` 与 `data/test/curves/`

两个 `curves/` 目录都保存从已有 RAG 实验中整理得到的 **query-level retrieval curve**：

- `data/train/curves/`：9000 条训练池曲线，可用于构造 Oracle；
- `data/test/curves/`：1500 条测试曲线，仅用于最终评测，不构造 Oracle。

对于同一个问题，将所有 retrieval budget：

$$
k = 1,2,\ldots,15
$$

下的实验结果按照 `question_id` 对齐，并合并成一条记录。

每条记录包含：

- `question_id`
- `dataset`
- `question`
- `gold_answer`
- 每个 k 下的：
  - `prediction`
  - `em`
  - `f1`
  - `context_tokens`
  - `context_tokens_validated`

示例：

```json
{
  "question_id": "nq_000001",
  "dataset": "nq",
  "question": "Who wrote The Old Man and the Sea?",
  "gold_answer": ["Ernest Hemingway"],
  "results": {
    "1": {
      "prediction": "Ernest Hemingway",
      "em": 1.0,
      "f1": 1.0,
      "context_tokens": 186,
      "context_tokens_validated": true
    },
    "2": {
      "prediction": "Ernest Hemingway",
      "em": 1.0,
      "f1": 1.0,
      "context_tokens": 374,
      "context_tokens_validated": true
    },
    "3": {
      "prediction": "Ernest Hemingway",
      "em": 1.0,
      "f1": 1.0,
      "context_tokens": 551,
      "context_tokens_validated": true
    },
    "4": {
      "prediction": "Ernest Hemingway",
      "em": 1.0,
      "f1": 1.0,
      "context_tokens": 742,
      "context_tokens_validated": true
    }
  }
}
```

实际文件中的 `results` 必须完整包含：

```text
1, 2, 3, ..., 15
```

共 15 个 retrieval budget。

---

# 4. 回答质量分数

本项目统一使用 **F1** 作为回答质量分数。

对于第 i 个 query，在 retrieval budget 为 k 时：

$$
S_i(k) = F1_i(k)
$$

因此：

- F1 用于 Hard Oracle 构造；
- F1 用于 Utility 计算；
- EM 保留在 `curves` 中，但不参与 Oracle 构造。

对于每个 query，`curves` 保存的核心信息可以表示为：

$$
q_i
\rightarrow
\left\{
F1_i(k), T_i(k)
\right\}_{k=1}^{15}
$$

其中：

- `F1_i(k)` 表示问题 i 在 retrieval budget 为 k 时的回答 F1；
- `T_i(k)` 表示问题 i 在 retrieval budget 为 k 时的检索上下文 token 数。

---

# 5. `context_tokens`

对于问题 i 和 retrieval budget k，定义：

$$
T_i(k) = \mathrm{Tokens}(C_i^k)
$$

其中：

$$
C_i^k
$$

表示问题 i 在 retrieval budget 为 k 时，实际送入生成模型的 retrieved context。

因此 `context_tokens` 统计的是：

> retrieved context 实际包含的 token 数量。

当前不统计：

- question tokens；
- prompt template tokens；
- LLM output tokens。

应尽可能使用与实际 RAG 生成过程一致的 tokenizer。

不要使用类似：

```text
k × 平均文档长度
```

的估计方式。

应记录每个 query、每个 k 对应的真实 context token 数。

---

# 6. train/test 中 `curves` 与 `oracle` 的职责

`data/train/curves/` 和 `data/test/curves/` 只保存真实实验结果。

其中不要保存：

```text
hard_oracle_k
utility
soft_oracle_distribution
```

这些内容属于从训练 retrieval curve 中计算得到的派生数据，应写入：

```text
data/train/oracle/
```

`data/test/` 下不设置 `oracle/`，避免将测试结果误用于监督标签构造。

因此两层数据的职责为：

```text
curves
  │
  │ 真实 RAG 实验结果
  ▼
oracle
  │
  │ 根据 curves 构造出的监督数据
  ▼
直接用于后续模型训练
```

这样，如果以后修改 Oracle 定义或 Utility 参数，只需要重新生成 `oracle`，不需要重新运行 RAG。

---

# 7. Hard Oracle

Hard Oracle 表示每个 query 对应的目标 retrieval budget。

记为：

$$
k_i^*
$$

候选 retrieval budget 为：

$$
\mathcal{K} = \{1,2,\ldots,15\}
$$

首先找到该问题在所有 k 下能够达到的最高 F1：

$$
S_i^{\max}
=
\max_{k \in \mathcal{K}} F1_i(k)
$$

然后找到所有达到最高 F1 的 retrieval budget：

$$
\mathcal{K}_i^{\max}
=
\left\{
k \in \mathcal{K}
\;:\;
F1_i(k)=S_i^{\max}
\right\}
$$

如果只有一个 k 达到最高 F1，则该 k 直接作为 Hard Oracle。

如果多个 k 达到相同的最高 F1，则在这些候选中选择 `context_tokens` 最少的一个：

$$
k_i^*
=
\arg\min_{k \in \mathcal{K}_i^{\max}}
T_i(k)
$$

因此 Hard Oracle 的规则为：

> **首先保证 F1 达到该问题在所有 k 下的最高值；如果多个 k 的 F1 同为最高，则选择 context token 消耗最少的 k。**

Hard Oracle 不设置 F1 下降容忍范围，也不使用 `epsilon` 参数。

---

## 7.1 Hard Oracle 示例

假设某个问题的结果为：

| k | F1 | context_tokens |
|---:|---:|---:|
| 1 | 0.50 | 200 |
| 2 | 0.80 | 400 |
| 3 | 1.00 | 600 |
| 4 | 1.00 | 800 |
| 5 | 0.95 | 1000 |

最高 F1 为：

$$
S_i^{\max}=1.00
$$

达到最高 F1 的候选为：

$$
\mathcal{K}_i^{\max}=\{3,4\}
$$

由于：

$$
T_i(3) < T_i(4)
$$

因此：

$$
k_i^*=3
$$

---

# 8. Soft Oracle Distribution

Hard Oracle 只提供一个离散目标：

$$
k_i^*
$$

但它无法描述其他 retrieval budget 的相对优劣。

因此，还需要根据不同 k 下的 F1 和 token cost 构造 Utility。

对于问题 i 和 retrieval budget k：

$$
U_i(k)
=
F1_i(k)
-
\lambda_T
\frac{T_i(k)}{T_{\max}}
$$

其中：

- `F1_i(k)`：该 query 在 k 下的回答 F1；
- `T_i(k)`：该 query 在 k 下的 context token 数；
- `T_max`：token normalization 使用的最大 token 数；
- `lambda_T`：token cost 权重。

当前暂时不考虑 latency，因此 Utility 中不包含 latency 项。

---

# 9. Utility 转换为 Soft Oracle Distribution

对于同一个 query，将所有 k 下的 Utility 通过 temperature softmax 转换为概率分布：

$$
P_i^*(k)
=
\frac{
\exp\left(U_i(k)/\tau\right)
}{
\sum_{k' \in \mathcal{K}}
\exp\left(U_i(k')/\tau\right)
}
$$

其中：

- `tau` 为 temperature；
- `P_i^*(k)` 表示 query i 对 retrieval budget k 的 soft target probability。

对于每一个 query，都有：

$$
\sum_{k \in \mathcal{K}} P_i^*(k)=1
$$

需要特别注意：

> Soft Oracle Distribution 不是所有 query 共用的全局分布。

而是每一个 query 都单独根据自己的 F1 和 token curve 得到一个分布：

$$
q_i
\rightarrow
P_i^*(k)
$$

因此不同 query 的 Soft Oracle Distribution 可以完全不同。

---

# 10. `data/train/oracle/`

`data/train/oracle/` 是从 `data/train/curves/` 中生成的 **Oracle training dataset**。

该目录中的数据后续可以直接用于预测模型训练，不再额外生成 `final/` 数据目录。

每条记录至少包含：

- `question_id`
- `dataset`
- `question`
- `hard_oracle_k`
- `soft_oracle_distribution`

示例：

```json
{
  "question_id": "nq_000001",
  "dataset": "nq",
  "question": "Who wrote The Old Man and the Sea?",
  "hard_oracle_k": 3,
  "soft_oracle_distribution": {
    "1": 0.081,
    "2": 0.132,
    "3": 0.127,
    "4": 0.119,
    "5": 0.108,
    "6": 0.095,
    "7": 0.081,
    "8": 0.067,
    "9": 0.054,
    "10": 0.042,
    "11": 0.031,
    "12": 0.022,
    "13": 0.015,
    "14": 0.010,
    "15": 0.006
  }
}
```

其中：

```text
hard_oracle_k
```

用于 Hard Oracle supervision。

```text
soft_oracle_distribution
```

用于 Utility Distribution supervision。

`curves` 中的：

- prediction；
- EM；
- F1；
- context_tokens；

不需要再次复制到 Oracle training dataset 中。

如果后续需要进行错误分析，可以通过 `question_id` 回到 `curves` 中查询完整实验结果。

---

# 11. `build_query_curves.py`

该脚本通过 `--split` 分别读取：

```text
--split train  -> kbqa/predictions/dev_3000/
--split test   -> kbqa/predictions/test/
```

每个 split 都包含 45 个 fixed-k RAG 实验结果。

对于每个数据集：

```text
nq
squad
trivia
```

分别读取：

```text
k = 1, 2, ..., 15
```

对应的实验结果。

然后按照 `question_id` 严格对齐。

最终分别生成：

```text
data/train/curves/{nq,squad,trivia,all}.jsonl
data/test/curves/{nq,squad,trivia,all}.jsonl
```

该脚本负责提取和整理：

- `question_id`
- `dataset`
- `question`
- `gold_answer`
- `prediction`
- `em`
- `f1`
- `context_tokens`

该脚本不负责：

- 计算 Hard Oracle；
- 计算 Utility；
- 计算 Soft Oracle Distribution。

test/NQ 中有 31 个问题在部分 k 上无法与历史 chain 严格对齐。为保持 1500 条测试集完整，
对应记录仍保留当前重建的 token 数，但 `context_tokens_validated` 为 `false`；这些值不能视为
已精确复现的历史 token 消耗。train 中不允许出现未验证的 token 记录。

---

# 12. `validate_curves.py`

该脚本负责检查生成的 query curve 是否完整、正确。

至少检查以下内容：

- 每个数据集的 query 数量；
- 是否存在重复 `question_id`；
- 每个 query 是否完整包含 k=1 到 k=15；
- 不同 k 下同一 `question_id` 的 question 是否一致；
- gold answer 是否一致；
- 是否缺失 prediction；
- EM 是否合法；
- F1 是否位于 `[0, 1]`；
- `context_tokens` 是否为合法正值。
- `context_tokens_validated` 是否为布尔值；train 中必须全部为 `true`。

正常情况下，每个 query 应完整包含：

```text
15 / 15
```

个 retrieval budget。

验证输出示例：

```text
Dataset: nq
Questions: 3000
Expected k values: 1-15

Complete questions: 3000 / 3000
Missing k: 0
Duplicate IDs: 0
Question mismatches: 0
Gold answer mismatches: 0
Invalid F1: 0
Invalid EM: 0
Missing token counts: 0

PASS
```

---

# 13. `build_oracle_labels.py`

该脚本读取：

```text
data/train/curves/*.jsonl
```

并根据每个 query 的完整 retrieval curve 计算：

```text
hard_oracle_k
soft_oracle_distribution
```

最终输出：

```text
data/train/oracle/
├── nq.jsonl
├── squad.jsonl
├── trivia.jsonl
└── all.jsonl
```

同时按照 `configs/data_config.yaml` 中的固定种子和验证比例生成：

```text
data/splits/train_ids.json
data/splits/val_ids.json
```

其中 Hard Oracle 根据：

```text
最高 F1
    ↓
若并列
    ↓
选择 context_tokens 最少的 k
```

进行构造。

Soft Oracle Distribution 根据：

```text
F1
 +
token cost
 ↓
Utility
 ↓
temperature softmax
 ↓
soft_oracle_distribution
```

进行构造。

生成的 `data/train/oracle/*.jsonl` 后续可以直接用于模型训练。该脚本不读取
`data/test/curves/`，也不会生成 test Oracle。

---

# 14. 配置文件

数据生成相关路径和参数保存在：

```text
configs/data_config.yaml
```

后续训练入口使用的数据路径保存在：

```text
configs/train_config.yaml
```

示例：

```yaml
k_values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

splits:
  train:
    prediction_root: predictions/dev_3000
    curve_output_dir: new_onehop/data/train/curves
    expected_questions_per_dataset: 3000
    allow_unvalidated_context_tokens: false
  test:
    prediction_root: predictions/test
    curve_output_dir: new_onehop/data/test/curves
    expected_questions_per_dataset: 500
    allow_unvalidated_context_tokens: true

oracle:
  curve_input_dir: new_onehop/data/train/curves
  output_dir: new_onehop/data/train/oracle
  lambda_T: 0.1
  temperature: 0.1

id_splits:
  train_ids_file: new_onehop/data/splits/train_ids.json
  val_ids_file: new_onehop/data/splits/val_ids.json
  validation_fraction: 0.1
  seed: 42
```

其中：

- Hard Oracle 不需要额外超参数；
- `lambda_T` 用于控制 Soft Oracle 中 token cost 的权重；
- `temperature` 用于控制 Soft Oracle Distribution 的平滑程度。
- `allow_unvalidated_context_tokens` 只在 test 中开启，用于保留无法严格复现历史检索顺序的测试曲线。

具体参数值后续通过实验确定。

---

# 15. 数据层级原则

整个数据处理过程严格按照：

```text
train RAG outputs                test RAG outputs
        ↓                               ↓
data/train/curves              data/test/curves
        ↓
data/train/oracle
        ↓
model training
```

进行。

---

## RAG outputs

已经完成 train 45 组、test 45 组 fixed-k RAG 问答实验。

属于原始实验输出，不修改。

---

## curves

`data/train/curves/` 和 `data/test/curves/` 分别保存每个 query 在不同 retrieval
budget 下的真实实验结果。

核心形式为：

$$
q_i
\rightarrow
\left\{
F1_i(k), T_i(k)
\right\}_{k=1}^{15}
$$

这是整个数据处理过程中最重要、最可复用的中间数据。

---

## oracle

仅根据 `data/train/curves/` 构造得到的模型监督数据。

主要包括：

$$
k_i^*
$$

以及：

$$
P_i^*(k)
$$

其中：

- Hard Oracle 由最高 F1 和 token cost 确定；
- Soft Oracle Distribution 由 F1、token cost 和 temperature softmax 构造。

`data/train/oracle/` 可以直接作为后续模型训练的数据来源。

---

# 16. 可复用性

采用该数据结构后，如果未来只修改：

- Hard Oracle 构造方法；
- `lambda_T`；
- temperature；
- Utility 公式；

都不需要重新运行 45 组 train RAG 问答实验。

只需要重新执行：

```text
build_oracle_labels.py
```

即可从已有：

```text
data/train/curves/
```

重新生成：

```text
data/train/oracle/
```

从而得到新的训练数据。
