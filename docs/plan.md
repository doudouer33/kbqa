# new_onehop 动态 Top-k 预测模型训练方案

本文档给出 `new_onehop` 在现有数据处理完成之后的模型训练、验证与测试实施方案。

当前阶段只处理 **k=1..15 的 retrieval budget 预测**，暂不讨论 `k=0` / 是否检索路由问题。

---

# 1. 当前仓库状态

现有数据流程已经完成：

```text
predictions/dev_3000
        ↓
new_onehop/data/train/curves
        ↓
new_onehop/data/train/oracle
        ↓
8100 train IDs + 900 val IDs

predictions/test
        ↓
new_onehop/data/test/curves
        ↓
1500 final test questions
```

当前关键文件：

```text
new_onehop/
├── configs/
│   ├── data_config.yaml
│   └── train_config.yaml
├── data/
│   ├── train/
│   │   ├── curves/all.jsonl
│   │   └── oracle/all.jsonl
│   ├── test/
│   │   └── curves/all.jsonl
│   └── splits/
│       ├── train_ids.json
│       └── val_ids.json
└── scripts/
    ├── data/
    ├── train/
    └── eval/
```

当前数据规模：

- train pool：9000 条，NQ / SQuAD / Trivia 各 3000 条；
- train IDs：8100 条；
- val IDs：900 条；
- test：1500 条，三个数据集各 500 条。

训练 Oracle 已包含：

```text
question_id
dataset
question
hard_oracle_k
soft_oracle_distribution
```

其中：

- `hard_oracle_k` 用于 Hard / Ordinal supervision；
- `soft_oracle_distribution` 用于 Utility Distribution supervision；
- test 不生成 Oracle，只通过 retrieval curves 做最终 RAG replay evaluation。

现有 Oracle 构造中：

```text
T_max = 9000 条 train-pool curves 中所有 k 的最大 context_tokens
```

并使用：

```text
lambda_T = 0.1
temperature = 0.1
```

作为当前初始 Oracle 参数。

---

# 2. 训练任务

模型在线阶段只读取问题文本：

```text
question
   ↓
Top-k Predictor
   ↓
predicted_k ∈ {1, 2, ..., 15}
```

模型输入中禁止使用：

- gold answer；
- F1；
- context tokens；
- retrieved documents；
- prediction；
- test curve 中的任何结果。

这些信息只能用于离线监督或评测。

任务可以表示为：

$$
q_i \rightarrow \hat{k}_i,\qquad \hat{k}_i\in\{1,2,\ldots,15\}.
$$

最终目标不是单纯最大化 `k` 分类准确率，而是在 RAG replay 中实现：

> 在保持或提高 QA F1 的同时，降低平均 retrieval budget 和 context token 消耗。

---

# 3. 模型实验顺序

第一轮不一次性加入所有复杂机制，按以下顺序实现。

```text
Model A: Multiclass CE baseline
            ↓
Model B: Ordinal predictor
            ↓
Model C: Ordinal + Utility Distribution
            ↓
后续再决定是否加入 Risk-sensitive loss / calibration
```

核心 ablation：

| Model | Head | Hard Oracle | Soft Oracle | 作用 |
|---|---|---:|---:|---|
| A | 15-way classifier | ✓ | ✗ | 普通分类 baseline |
| B | Ordinal head | ✓ | ✗ | 验证 k 的序关系是否有帮助 |
| C | Ordinal head | ✓ | ✓ | 验证完整 quality-cost distribution 是否有帮助 |

第一阶段先不加入 Risk Loss，避免在基础模型尚未验证时引入过多超参数。

---

# 4. Backbone

第一版统一使用：

```text
microsoft/deberta-v3-base
```

理由：

- 任务只输入 query 文本，不需要生成能力；
- 9000 条训练池规模不需要大模型；
- encoder-only backbone 足够；
- 三个模型使用相同 backbone，保证 ablation 主要比较 head / loss，而不是 backbone 大小。

初始建议：

```yaml
model_name: microsoft/deberta-v3-base
max_length: 128
dropout: 0.1
```

如果个别 query 超过 128 tokens，再根据实际长度统计决定是否提高到 192 / 256，不预先扩大。

问题表示优先使用 encoder 的首 token hidden state：

$$
h_i = \mathrm{Encoder}(q_i)_{[0]}.
$$

---

# 5. Model A：Multiclass CE baseline

最简单 baseline：

```text
question
   ↓
DeBERTa-v3-base
   ↓
h_i
   ↓
Dropout
   ↓
Linear(hidden_size, 15)
   ↓
Softmax
```

预测：

$$
P_\theta(k\mid q_i),\qquad k=1,\ldots,15.
$$

Hard Oracle 类别索引：

$$
y_i=k_i^*-1.
$$

Loss：

$$
\mathcal{L}_{CE}
=
-\log P_\theta(k_i^*\mid q_i).
$$

第一版先使用普通 CE，不默认加 class weight。训练后如果发现 Hard Oracle 分布严重不均衡，再增加 weighted CE 作为单独实验，避免 baseline 被过多技巧污染。

默认预测：

$$
\hat{k}_i=\arg\max_k P_\theta(k\mid q_i).
$$

---

# 6. Model B：Ordinal Predictor

由于：

$$
1<2<3<\cdots<15,
$$

Top-k 不是普通的互斥类别，而具有天然序关系。

## 6.1 Ordinal target

对每个 threshold：

$$
j\in\{1,2,\ldots,14\},
$$

定义：

$$
y_{ij}=\mathbf{1}[k_i^*>j].
$$

例如：

```text
hard_oracle_k = 5
```

则 target 为：

```text
[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## 6.2 Ordinal head

先从 query representation 得到一个 scalar retrieval-demand score：

$$
s_i=w^T h_i+c.
$$

定义 14 个有序 threshold：

$$
b_1<b_2<\cdots<b_{14}.
$$

模型预测：

$$
p_{ij}
=
\sigma(s_i-b_j)
=
P_\theta(k_i^*>j\mid q_i).
$$

为了保证 threshold 顺序，不直接训练 14 个完全独立的参数。实现建议：

$$
b_1=a_1,
$$

$$
b_j=b_{j-1}+\operatorname{softplus}(d_j),\qquad j=2,\ldots,14.
$$

这样天然满足：

$$
b_1<b_2<\cdots<b_{14},
$$

并进一步保证：

$$
p_{i1}\ge p_{i2}\ge\cdots\ge p_{i14}.
$$

## 6.3 Ordinal Loss

$$
\mathcal{L}_{ord}
=
-\frac{1}{14}
\sum_{j=1}^{14}
\left[
 y_{ij}\log p_{ij}
 +(1-y_{ij})\log(1-p_{ij})
\right].
$$

batch loss 再对 batch 中所有样本取平均。

---

# 7. 从 Ordinal 输出得到 15-way distribution

Utility Loss 需要完整的：

$$
\pi_\theta(k\mid q_i),\qquad k=1,\ldots,15.
$$

Ordinal cumulative probability 转换方式：

$$
\pi_1=1-p_1,
$$

$$
\pi_k=p_{k-1}-p_k,\qquad k=2,\ldots,14,
$$

$$
\pi_{15}=p_{14}.
$$

由于 ordinal thresholds 单调，理论上：

$$
\pi_k\ge0,
$$

并且：

$$
\sum_{k=1}^{15}\pi_k=1.
$$

实现时仍需进行数值检查，并在 `eps` 范围内 clamp 后再计算 log。

默认 decoding：

$$
\hat{k}_i=\arg\max_k\pi_\theta(k\mid q_i).
$$

第一轮所有主结果统一使用该 decoding，避免不同模型使用不同后处理规则造成比较不公平。

---

# 8. Model C：Ordinal + Utility Distribution

训练 Oracle 已提供每个 query 的：

```text
soft_oracle_distribution
```

记为：

$$
P_i^*(k),\qquad k=1,\ldots,15.
$$

该分布已经将：

```text
F1 + token cost
```

通过 Utility 注入监督：

$$
U_i(k)
=
F1_i(k)
-
\lambda_T\frac{T_i(k)}{T_{\max}}.
$$

随后：

$$
P_i^*(k)
=
\frac{\exp(U_i(k)/\tau)}
{\sum_{k'}\exp(U_i(k')/\tau)}.
$$

因此模型训练时 **不需要再次读取 context_tokens 计算 Utility**。

Utility Loss 使用 soft-label cross entropy：

$$
\mathcal{L}_{utility}
=
-\sum_{k=1}^{15}
P_i^*(k)\log\pi_\theta(k\mid q_i).
$$

它与：

$$
KL(P_i^*\Vert\pi_\theta)
$$

对模型参数具有相同的优化方向，因为 Oracle distribution 是固定 target。

总 Loss：

$$
\mathcal{L}
=
\mathcal{L}_{ord}
+
\alpha\mathcal{L}_{utility}.
$$

第一轮建议实验：

```text
alpha = 0.0   # 等价于 Model B
alpha = 0.5
alpha = 1.0
```

不建议一开始进行过密的 alpha 网格搜索。

---

# 9. Hard / Soft Oracle 一致性检查

Hard Oracle 的目标是：

```text
最高 F1优先；并列时选择 token 最少的 k
```

Soft Oracle 的目标是：

```text
F1 - token penalty
```

因此可能出现：

$$
k_i^*\ne\arg\max_k P_i^*(k).
$$

这并不自动意味着数据错误，但必须在训练前量化。

建议新增统计：

```text
hard_soft_conflict_rate
mean_argmax_distance
median_argmax_distance
```

其中：

$$
\mathrm{ConflictRate}
=
\frac{1}{N}
\sum_i
\mathbf{1}
\left[
 k_i^*\ne\arg\max_kP_i^*(k)
\right],
$$

$$
\mathrm{MeanDistance}
=
\frac{1}{N}
\sum_i
\left|
 k_i^*-\arg\max_kP_i^*(k)
\right|.
$$

同时统计：

- Hard Oracle 的 k 分布；
- Soft Oracle 平均 entropy；
- 按 NQ / SQuAD / Trivia 分数据集统计冲突率。

如果 Hard / Soft 冲突异常高，再回到 Oracle 参数 `lambda_T` / `temperature` 检查，不应直接通过增大 loss 权重掩盖数据目标冲突。

---

# 10. Train / Validation 数据使用

当前仓库已经有：

```text
new_onehop/data/splits/train_ids.json  # 8100
new_onehop/data/splits/val_ids.json    # 900
```

训练与验证都从同一个：

```text
new_onehop/data/train/oracle/all.jsonl
```

按 `question_id` 过滤，不再复制 train / val JSONL。

同时加载：

```text
new_onehop/data/train/curves/all.jsonl
```

用于 validation RAG replay。

数据职责：

```text
train IDs
  └── oracle → 反向传播

val IDs
  ├── oracle → predictor-level metrics
  └── curves → RAG-level replay metrics
```

最终 test：

```text
test question
   ↓
final model
   ↓
predicted_k
   ↓
data/test/curves
   ↓
F1(predicted_k) + context_tokens(predicted_k)
```

不使用 test 数据进行：

- early stopping；
- checkpoint 选择；
- alpha 选择；
- learning-rate 选择；
- decoding rule 选择。

---

# 11. Validation 指标

不能只看 classification accuracy。

## 11.1 Predictor-level metrics

至少计算：

```text
accuracy
mae
rmse
within_1_accuracy
within_2_accuracy
under_retrieval_rate
over_retrieval_rate
avg_predicted_k
avg_oracle_k
```

其中：

$$
MAE
=
\frac{1}{N}
\sum_i|\hat{k}_i-k_i^*|.
$$

$$
Within1
=
\frac{1}{N}
\sum_i\mathbf{1}[|\hat{k}_i-k_i^*|\le1].
$$

under-retrieval：

$$
\hat{k}_i<k_i^*.
$$

over-retrieval：

$$
\hat{k}_i>k_i^*.
$$

所有指标同时输出：

- overall；
- NQ；
- SQuAD；
- Trivia。

## 11.2 RAG replay metrics

对于每个 validation query，模型给出：

$$
\hat{k}_i.
$$

直接从 train curve 查询：

$$
F1_i(\hat{k}_i),
$$

$$
T_i(\hat{k}_i).
$$

至少计算：

```text
avg_f1
avg_context_tokens
avg_k
```

并同样按数据集输出。

**最终 checkpoint 的主选择依据应以 RAG-level trade-off 为主，而不是单纯 predictor accuracy。**

第一版建议：

1. 先要求 `avg_f1` 不明显低于最佳 fixed-k baseline；
2. 在 F1 相近模型中优先选择 `avg_context_tokens` 更低的模型；
3. predictor MAE / accuracy 作为辅助解释指标。

不要在 test 上执行该选择过程。

---

# 12. Fixed-k baseline 与 Oracle upper bound

对 validation / test curves 可以直接 replay 所有固定 k：

```text
fixed_k_1
fixed_k_2
...
fixed_k_15
```

每个 fixed-k 输出：

```text
avg_f1
avg_context_tokens
avg_k
```

最终动态模型必须与 fixed-k Pareto frontier 比较，而不是只与一个固定 k 比。

可以在评测脚本内部临时计算 test Hard Oracle 作为 upper bound：

```text
最高 F1
  ↓
并列时 token 最少
```

但：

- 不生成 `data/test/oracle/`；
- 不把 test Oracle 用于训练；
- 不用 test Oracle 调参。

---

# 13. test/NQ 未验证 token 的处理

当前 test/NQ 有 31 个问题在部分 k 上：

```text
context_tokens_validated = false
```

F1 曲线仍可以正常用于 QA performance evaluation。

最终 token 指标建议同时报告：

```text
avg_context_tokens_all
avg_context_tokens_validated_only
num_queries_with_unvalidated_tokens
```

主表可以保留完整 1500 条结果，但实验日志必须记录 unvalidated token 数量。

如果 `all` 与 `validated_only` 的 token 结果非常接近，则可以在论文/报告中说明该差异对结论影响很小。

---

# 14. 建议新增目录

在当前项目结构上新增：

```text
new_onehop/
├── configs/
│   ├── data_config.yaml
│   ├── train_config.yaml
│   ├── train_multiclass.yaml
│   ├── train_ordinal.yaml
│   └── train_ordinal_utility.yaml
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multiclass.py
│   │   └── ordinal.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── ordinal_loss.py
│   │   └── utility_loss.py
│   ├── metrics.py
│   └── utils.py
│
├── scripts/
│   ├── data/
│   ├── train/
│   │   ├── train_multiclass.py
│   │   ├── train_ordinal.py
│   │   └── train_ordinal_utility.py
│   └── eval/
│       ├── evaluate_predictor.py
│       └── evaluate_rag.py
│
└── outputs/
    ├── checkpoints/
    │   ├── multiclass/
    │   ├── ordinal/
    │   └── ordinal_utility/
    ├── predictions/
    │   ├── val/
    │   └── test/
    ├── metrics/
    │   ├── val/
    │   └── test/
    └── logs/
```

职责：

- `src/`：可复用实现；
- `scripts/train/`：训练入口；
- `scripts/eval/`：评测入口；
- `configs/`：实验参数；
- `outputs/`：模型和实验输出；
- `data/`：只保存数据，不保存模型预测结果。

---

# 15. Dataset loader 设计

建议统一实现：

```text
new_onehop/src/dataset.py
```

负责：

1. 读取 `oracle/all.jsonl`；
2. 按 train / val ID 过滤；
3. tokenizer 编码 question；
4. 返回 Hard Oracle target；
5. 返回 Soft Oracle 15-way vector；
6. 保留 `question_id` 和 `dataset` 供评测。

样本逻辑结构：

```python
{
    "question_id": str,
    "dataset": str,
    "input_ids": ...,
    "attention_mask": ...,
    "hard_k": int,
    "ordinal_target": float[14],
    "soft_target": float[15],
}
```

Model A 只使用：

```text
hard_k
```

Model B 使用：

```text
ordinal_target
```

Model C 使用：

```text
ordinal_target + soft_target
```

---

# 16. 初始训练超参数

第一轮建议统一：

```yaml
seed: 42

model:
  name: microsoft/deberta-v3-base
  max_length: 128
  dropout: 0.1

training:
  epochs: 5
  batch_size: 32
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_clip_norm: 1.0
  mixed_precision: true

optimizer:
  name: AdamW

scheduler:
  name: linear

checkpoint:
  save_best_only: true
```

如果显存不足：

```text
batch_size = 16
gradient_accumulation_steps = 2
```

保持 effective batch size 为 32。

不建议第一轮同时搜索过多：

- backbone；
- max length；
- learning rate；
- batch size；
- alpha；
- class weight；
- decoding rule。

先保证三个核心模型的可比较性。

---

# 17. Checkpoint 与日志

每次实验生成唯一 `run_name`，例如：

```text
multiclass_deberta_seed42
ordinal_deberta_seed42
ordinal_utility_a05_deberta_seed42
```

每个 run 保存：

```text
config.yaml
best_model.pt / HuggingFace checkpoint
train_log.jsonl
val_predictions.jsonl
val_metrics.json
```

`val_predictions.jsonl` 建议保存：

```json
{
  "question_id": "...",
  "dataset": "nq",
  "predicted_k": 5,
  "hard_oracle_k": 4,
  "probabilities": [0.01, 0.03, 0.08, 0.12, 0.31, ...]
}
```

test prediction 不写 Hard Oracle：

```json
{
  "question_id": "...",
  "dataset": "nq",
  "predicted_k": 5,
  "probabilities": [0.01, 0.03, 0.08, 0.12, 0.31, ...]
}
```

---

# 18. 训练脚本行为

## 18.1 `train_multiclass.py`

执行：

```text
load config
  ↓
load train / val IDs
  ↓
load oracle
  ↓
DeBERTa + 15-way classifier
  ↓
CE training
  ↓
每 epoch validation
  ↓
保存最佳 checkpoint
```

## 18.2 `train_ordinal.py`

执行：

```text
load oracle
  ↓
build ordinal targets
  ↓
DeBERTa + ordered-threshold head
  ↓
Ordinal BCE
  ↓
convert cumulative probabilities to 15-way distribution
  ↓
validation
```

## 18.3 `train_ordinal_utility.py`

执行：

```text
Ordinal branch
  +
Soft Oracle Distribution
  ↓
L = L_ord + alpha * L_utility
  ↓
validation
```

最好让三个训练入口尽量复用同一套 trainer / evaluator，而不是复制三份训练循环。

---

# 19. 评测脚本

## 19.1 `evaluate_predictor.py`

主要用于 validation，因为 validation 有 Oracle labels。

输出：

```text
accuracy
mae
rmse
within_1_accuracy
within_2_accuracy
under_retrieval_rate
over_retrieval_rate
avg_predicted_k
avg_oracle_k
```

同时输出 overall + per-dataset。

## 19.2 `evaluate_rag.py`

输入：

```text
prediction jsonl
+
curve jsonl
```

根据：

```text
question_id + predicted_k
```

查找对应：

```text
f1
context_tokens
context_tokens_validated
```

输出：

```text
avg_f1
avg_context_tokens
avg_k
```

并同时生成：

```text
fixed_k_1 ... fixed_k_15
```

用于最终对比。

---

# 20. 模型选择规则

模型与超参数只在 900 条 validation 上选择。

建议按以下优先级：

1. 首先比较 validation `avg_f1`；
2. F1 相近时比较 `avg_context_tokens`；
3. 再参考 `avg_k`、MAE、under-retrieval rate；
4. 最终确定一个 checkpoint；
5. test 只对最终候选模型进行正式报告。

不要将：

```text
test avg_f1
```

作为挑选 Model A / B / C 或 alpha 的依据。

如果开发阶段不可避免地查看了 test 结果，应明确将其视为开发信息，而不能再把该 1500 条描述成完全 untouched final test。

---

# 21. 最终 Test 结果

最终至少生成如下主表：

| Method | Avg F1 | Avg k | Avg Context Tokens |
|---|---:|---:|---:|
| Fixed k=1 | | 1 | |
| Fixed k=2 | | 2 | |
| ... | | | |
| Fixed k=15 | | 15 | |
| Multiclass Dynamic | | | |
| Ordinal Dynamic | | | |
| Ordinal + Utility | | | |
| Oracle Upper Bound | | | |

另外分别报告：

```text
NQ
SQuAD
Trivia
Overall
```

关键结论不是要求动态模型精确命中 Oracle k，而是检查动态模型是否位于更好的：

```text
QA quality ↔ retrieval cost
```

trade-off 上。

---

# 22. 推荐实施阶段

## Phase 0：训练前数据 sanity check

实现 / 输出：

```text
Hard Oracle k distribution
Hard/Soft conflict rate
Soft Oracle entropy
train/val dataset counts
```

验收：

- train ID = 8100；
- val ID = 900；
- 无 ID overlap；
- 每个 ID 都能在 Oracle 中找到；
- soft target 每行和为 1；
- k 范围均为 1..15。

## Phase 1：Multiclass baseline

实现：

```text
dataset.py
multiclass.py
train_multiclass.py
evaluate_predictor.py
evaluate_rag.py
```

验收：

- 可以完成 1 个 epoch；
- loss 正常下降；
- val prediction 数量严格为 900；
- RAG replay 可以输出 avg F1 / tokens。

## Phase 2：Ordinal model

实现：

```text
ordinal.py
ordinal_loss.py
train_ordinal.py
```

额外验收：

- thresholds 严格单调；
- cumulative probabilities 单调；
- 15-way probabilities 非负且和约为 1；
- 与 multiclass baseline 做 validation 对比。

## Phase 3：Utility supervision

实现：

```text
utility_loss.py
train_ordinal_utility.py
```

实验：

```text
alpha = 0.5
alpha = 1.0
```

比较：

```text
Ordinal
vs
Ordinal + Utility
```

重点看：

```text
avg_f1
avg_context_tokens
under_retrieval_rate
```

## Phase 4：Final model selection

根据 validation 选择：

```text
model type
alpha
checkpoint
```

冻结后进入 test。

## Phase 5：Final test

仅使用：

```text
question → model → predicted_k
```

然后从 `data/test/curves/all.jsonl` replay。

生成最终：

- overall results；
- per-dataset results；
- fixed-k baseline comparison；
- Oracle upper bound；
- token validation sensitivity results。

---

# 23. 暂不实现的内容

第一轮暂不实现：

```text
k=0 retrieval gate
Risk-sensitive loss
Conformal calibration
Multi-environment Oracle
大规模 backbone 搜索
复杂 ensemble
```

这些内容只有在：

```text
Multiclass → Ordinal → Ordinal + Utility
```

主链路跑通并确认有效后再加入。

---

# 24. 第一轮实验完成标准

认为训练阶段第一轮完成，需要至少得到：

```text
1. Multiclass CE baseline
2. Ordinal baseline
3. Ordinal + Utility model
4. 三者的 val predictor metrics
5. 三者的 val RAG replay metrics
6. fixed-k 1..15 validation baselines
7. 最优模型的 final test RAG metrics
8. fixed-k 1..15 test baselines
9. overall + NQ + SQuAD + Trivia 分项结果
10. test token validated-only sensitivity result
```

如果 Model C 最终能够在与较大 fixed-k 相近的 F1 下显著减少：

```text
avg_k
avg_context_tokens
```

则当前 query-only 动态 Top-k 预测方案得到初步验证，可以继续引入 Risk / Calibration 或扩展到其他 RAG framework。
