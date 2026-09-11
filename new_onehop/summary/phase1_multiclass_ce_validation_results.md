# Phase 1: DeBERTa + 15-way Multiclass Cross Entropy

本文记录 Phase 1 已完成的 question-only、15 类 Multiclass Cross Entropy
validation 实验。报告数值以 Phase 1 已有 checkpoint、training history、prediction、
predictor metrics 和 RAG replay metrics 为准；没有重新训练模型，也没有进入 test set。

实现代码实际位于 `new_onehop/src/models/multiclass.py`；仓库中不存在单文件
`new_onehop/src/models.py`，模型通过 `new_onehop/src/models/__init__.py` 导出。

## 1. 实验范围

### 1.1 任务与数据

模型输入只包含 Oracle 数据中的 question text。模型不读取 retrieved context、答案、
retrieval curve、F1、token cost 或数据集名称作为输入特征；`question_id` 和 `dataset`
只用于对齐与分组评测。

预测目标为：

$$
k\in\{1,2,\ldots,15\}.
$$

Hard Oracle 对每个问题先求：

$$
F1_i^{\max}=\max_{k\in\{1,\ldots,15\}}F1_i(k),
$$

再取所有达到最高 F1 的候选集合：

$$
\mathcal K_i^{\max}
=
\{k:F1_i(k)=F1_i^{\max}\}.
$$

最终标签为其中 `context_tokens` 最少的候选；如果 token 数也相同，再取较小的
`k`：

$$
k_i^*
=
\arg\min_{k\in\mathcal K_i^{\max}}
\bigl(T_i(k),k\bigr).
$$

Hard Oracle 不使用 F1 容忍区间或 `epsilon`。训练池包含 NQ、SQuAD、Trivia 各
3000 条，共 9000 条；按数据集分层划分为 8100 条 train 和 900 条 validation：

| Split | NQ | SQuAD | Trivia | Total |
|---|---:|---:|---:|---:|
| Train | 2700 | 2700 | 2700 | 8100 |
| Validation | 300 | 300 | 300 | 900 |

### 1.2 模型与训练配置

| 项目 | 实际设置 |
|---|---|
| 输入 | question text only |
| 目标 | Hard Oracle `k=1..15` |
| Backbone | `microsoft/deberta-v3-base` |
| Classification head | `Dropout(0.1)` + `Linear(hidden_size, 15)` |
| Loss | 无权重 `nn.CrossEntropyLoss()` |
| Epochs | 5 |
| Batch size | 32 |
| Gradient accumulation | 1 |
| Effective batch size | 32 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.1` |
| Max length | 128 |
| Dropout | `0.1` |
| Gradient clipping | global norm `1.0` |
| Mixed precision config | `true`；仅在 CUDA device 上启用 AMP |
| Seed | 42 |

本阶段没有使用：

- Ordinal modeling 或 ordered-threshold head；
- Soft Oracle supervision；数据加载时会校验 source 中的 soft distribution，但不会保留
  或送入模型；
- Utility Loss 或 Risk Loss；
- class weight 或 weighted Cross Entropy；
- oversampling 或 balanced sampler；train loader 采用普通 seeded shuffle；
- label smoothing；`nn.CrossEntropyLoss()` 使用默认设置；
- calibration 或 ensemble；
- test labels、test training 或 test evaluation。

配置中虽然保留了 `data.test_curves` 路径，Phase 1 训练与本报告中的 validation
评测没有读取该文件。

## 2. Multiclass 实现

### 2.1 前向流程

```text
Question text
   ↓
AutoTokenizer（truncation，max_length=128，dynamic padding）
   ↓
microsoft/deberta-v3-base
   ↓
last_hidden_state[:, 0, :]（首 token 表示）
   ↓
Dropout(p=0.1)
   ↓
Linear(hidden_size, 15)
   ↓
15-way logits，shape = [batch_size, 15]
   ↓
softmax / argmax
   ↓
predicted k ∈ {1,...,15}
```

代码直接使用 encoder 最后一层的首 token 表示：

$$
h_q=\operatorname{DeBERTa}(q)_{0}.
$$

classification head 只有 dropout 和一个线性层：

$$
\mathbf z=W\operatorname{Dropout}(h_q)+b,
$$

其中：

$$
\mathbf z\in\mathbb R^{15},
\qquad
W\in\mathbb R^{15\times d}.
$$

没有额外 MLP、pooling layer 或 ordinal threshold 参数。

### 2.2 Label、loss 与 decoding

Dataset 返回零基分类标签，实际关系为：

$$
y=k^*-1,
\qquad
y\in\{0,1,\ldots,14\}.
$$

单样本 Cross Entropy Loss 为：

$$
\mathcal L_{CE}
=
-\log
\frac{\exp z_y}
{\sum_{c=0}^{14}\exp z_c}.
$$

训练使用 PyTorch 默认 mean reduction。推理先对 logits 做 softmax，再取零基类别
argmax：

$$
\hat y=\arg\max_{c\in\{0,\ldots,14\}}z_c,
$$

最后恢复为一基 retrieval budget：

$$
\hat k=\hat y+1.
$$

保存的 prediction 按 `k=1..15` 顺序写入 15 个 softmax probability，并同时保存
`question_id`、`dataset`、`hard_oracle_k` 和 `predicted_k`。

## 3. Smoke Test

仓库中存在独立的 Phase 1 smoke-test checkpoint、日志、900 条 validation prediction、
predictor metrics 和 RAG replay metrics。仓库 README 记录的命令为：

```bash
HTTP_PROXY=http://127.0.0.1:17890 \
HTTPS_PROXY=http://127.0.0.1:17890 \
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_multiclass.py \
  --config new_onehop/configs/train_multiclass.yaml \
  --max_train_steps 10 \
  --run-name multiclass_deberta_seed42_smoke
```

可核验结果：

| 项目 | 数值/状态 |
|---|---:|
| Completed epochs | 1 |
| Optimizer steps | 10 |
| Train loss | 2.531790 |
| Val loss | 2.415587 |
| Training seconds | 8.396590 |
| Epoch seconds | 7.287588 |
| Validation predictions | 900 |
| Avg predicted k | 14.000000 |
| RAG avg F1 | 0.488794 |
| RAG avg context tokens | 1932.586 |

smoke prediction 的分布为 `k=14: 900 / 900`。best checkpoint 成功保存于 epoch 1、
optimizer step 10；独立 predictor evaluator 生成了 900 条结果，RAG evaluator 生成了
dynamic replay 和完整 fixed-`k=1..15` baselines。smoke RAG 输出中的所有 token 记录均
为 validated，unvalidated query 数为 0。

## 4. 正式训练

### 4.1 命令与训练方法

仓库 README 记录的正式训练命令为：

```bash
HTTP_PROXY=http://127.0.0.1:17890 \
HTTPS_PROXY=http://127.0.0.1:17890 \
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_multiclass.py \
  --config new_onehop/configs/train_multiclass.yaml
```

训练入口使用 `torch.optim.AdamW` 更新 DeBERTa encoder 和 classification head 的全部
参数。每个 epoch 有 254 个 optimizer updates，总计 1270 个 updates；linear scheduler
的总步数为 1270，按 `warmup_ratio=0.1` 得到 127 个 warmup steps，之后线性衰减。
每次 update 前执行 global gradient norm clipping，阈值为 `1.0`。

train loader 使用 seed 42 的 generator 做 shuffle；validation loader 不 shuffle。
tokenization 使用 fast tokenizer、截断长度 128、batch 内 dynamic padding；当 CUDA AMP
启用时 padding 到 8 的倍数。每个 epoch 结束后计算 validation CE、predictor metrics
以及基于已有 validation curves 的 RAG replay。checkpoint 只按最低 validation loss
选择。

Phase 1 持久化产物没有记录 stdout 或具体 GPU 型号，因此本报告不从 Phase 2 的硬件
记录反推 Phase 1 的 GPU。可直接核验的运行设置是 `mixed_precision: true`，而代码仅在
自动选择到 CUDA device 时实际启用 AMP。

### 4.2 总体信息

| 项目 | 数值 |
|---|---:|
| Epochs | 5 |
| Optimizer steps | 1270 |
| Updates per epoch | 254 |
| Warmup steps | 127 |
| Training seconds | 157.042063 |
| Checkpoint selection metric | minimum validation loss |
| Best epoch | 1 |
| Best optimizer step | 254 |
| Best validation loss | 1.506004978 |

### 4.3 每个 epoch 的已有记录

| Epoch | Step | Train loss | Val loss | Accuracy | Macro F1 | MAE | Avg predicted k | RAG F1 | Avg tokens | Epoch seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 254 | 1.718504 | 1.506005 | 0.631111 | 0.051589 | 1.805556 | 1.000000 | 0.452164 | 140.708 | 35.161 |
| 2 | 508 | 1.476988 | 1.511631 | 0.631111 | 0.051589 | 1.805556 | 1.000000 | 0.452164 | 140.708 | 30.327 |
| 3 | 762 | 1.411564 | 1.514067 | 0.631111 | 0.051589 | 1.805556 | 1.000000 | 0.452164 | 140.708 | 30.207 |
| 4 | 1016 | 1.303741 | 1.574202 | 0.617778 | 0.057097 | 1.815556 | 1.030000 | 0.456850 | 144.637 | 30.101 |
| 5 | 1270 | 1.200570 | 1.634142 | 0.607778 | 0.066652 | 1.805556 | 1.122222 | 0.459487 | 157.141 | 30.126 |

表中仅包含 `training_history.json` / `training_history.jsonl` 已记录的字段。正式 run
没有保存每个 epoch 的独立 checkpoint 或 prediction；保存的是 validation loss 最低的
epoch 1 checkpoint 及其 prediction。

## 5. 最优 checkpoint 的 Predictor 指标

最优 checkpoint：

```text
new_onehop/outputs/checkpoints/multiclass_deberta_seed42/best/
epoch: 1
optimizer step: 254
selection metric: validation loss
validation loss: 1.5060049777560764
```

独立 `evaluate_predictor.py` 输出的 validation 指标如下。Macro F1 对全部 15 个类别
求平均；未获得 true positive 的类别 F1 记为 0。Under-retrieval 定义为
`predicted_k < hard_oracle_k`，over-retrieval 定义为
`predicted_k > hard_oracle_k`。

| Scope | Examples | Accuracy | Macro F1 | MAE | RMSE | Within-1 | Within-2 | Under rate | Over rate | Avg predicted k | Avg Oracle k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 900 | 0.631111 | 0.051589 | 1.805556 | 3.843754 | 0.727778 | 0.785556 | 0.368889 | 0.000000 | 1.000000 | 2.805556 |
| NQ | 300 | 0.550000 | 0.047312 | 2.053333 | 4.084116 | 0.700000 | 0.763333 | 0.450000 | 0.000000 | 1.000000 | 3.053333 |
| SQuAD | 300 | 0.570000 | 0.048408 | 2.380000 | 4.495183 | 0.640000 | 0.713333 | 0.430000 | 0.000000 | 1.000000 | 3.380000 |
| Trivia | 300 | 0.773333 | 0.058145 | 0.983333 | 2.727025 | 0.843333 | 0.880000 | 0.226667 | 0.000000 | 1.000000 | 1.983333 |

predictor metrics 文件还保存了 `always_k_1_predictor_baseline`。由于该 checkpoint 的
900 条 validation prediction 均为 `k=1`，该 baseline 的 overall 及三个数据集分组
指标与上表逐项相同。

## 6. RAG Replay 结果

RAG evaluator 不重新运行 retrieval 或 generation，而是按每条 prediction 的
`predicted_k` 从已有 validation query curves 中读取对应的 F1 和
`context_tokens`。正式 best checkpoint 的 dynamic multiclass replay 为：

| Scope | Examples | Avg F1 | Avg context tokens | Avg k | Validated-only avg tokens | Unvalidated queries |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 900 | 0.452164 | 140.708 | 1.000000 | 140.708 | 0 |
| NQ | 300 | 0.409567 | 140.230 | 1.000000 | 140.230 | 0 |
| SQuAD | 300 | 0.262684 | 139.923 | 1.000000 | 139.923 | 0 |
| Trivia | 300 | 0.684242 | 141.970 | 1.000000 | 141.970 | 0 |

raw Phase 1 RAG metrics 同时保存 `dynamic_multiclass`、`fixed_k_baselines` 和
`always_k_1_rag_baseline`。由于所有 prediction 都是 `k=1`，
`dynamic_multiclass` 与 `fixed_k_1` / `always_k_1_rag_baseline` 的所有分组字段精确
一致。

Phase 1 的原始 RAG JSON 没有保存 RAG utility、`lambda_T` 或 `T_max` 字段，且
Phase 1 checkpoint 选择没有使用 RAG utility。Phase 2 报告中引用的 Phase 1 utility
`0.448134` 是按 Phase 2 报告的 utility 口径列出的派生值，不是本 Phase 1 原始 RAG
文件中的字段。

## 7. Fixed-k Baseline

下表为 Phase 1 RAG metrics 原始文件中完整的 validation fixed-`k=1..15` overall
baseline。所有 15 个 baseline 都覆盖 900 条 validation examples；
`avg_context_tokens_validated_only` 与 `avg_context_tokens` 相同，且每个 baseline 的
unvalidated query 数均为 0。

| Fixed k | Avg F1 | Avg context tokens | Avg k |
|---:|---:|---:|---:|
| 1 | 0.452164 | 140.708 | 1.000000 |
| 2 | 0.470858 | 275.479 | 2.000000 |
| 3 | 0.498634 | 413.708 | 3.000000 |
| 4 | 0.488471 | 551.428 | 4.000000 |
| 5 | 0.484183 | 688.292 | 5.000000 |
| 6 | 0.487101 | 824.954 | 6.000000 |
| 7 | 0.483109 | 964.207 | 7.000000 |
| 8 | 0.475652 | 1102.221 | 8.000000 |
| 9 | 0.487658 | 1242.560 | 9.000000 |
| 10 | 0.472882 | 1380.151 | 10.000000 |
| 11 | 0.469947 | 1518.150 | 11.000000 |
| 12 | 0.474323 | 1656.421 | 12.000000 |
| 13 | 0.474743 | 1794.994 | 13.000000 |
| 14 | 0.488794 | 1932.586 | 14.000000 |
| 15 | 0.482791 | 2071.223 | 15.000000 |

原始文件还按数据集保存了每个 fixed k 的结果。对应的 avg F1 如下：

| Fixed k | NQ F1 | SQuAD F1 | Trivia F1 |
|---:|---:|---:|---:|
| 1 | 0.409567 | 0.262684 | 0.684242 |
| 2 | 0.460951 | 0.259504 | 0.692118 |
| 3 | 0.481167 | 0.303890 | 0.710845 |
| 4 | 0.468731 | 0.301610 | 0.695074 |
| 5 | 0.466878 | 0.311825 | 0.673846 |
| 6 | 0.456070 | 0.325051 | 0.680183 |
| 7 | 0.462182 | 0.330102 | 0.657041 |
| 8 | 0.457763 | 0.315682 | 0.653512 |
| 9 | 0.468518 | 0.334765 | 0.659692 |
| 10 | 0.453526 | 0.330327 | 0.634792 |
| 11 | 0.428855 | 0.344853 | 0.636133 |
| 12 | 0.435744 | 0.334986 | 0.652237 |
| 13 | 0.425531 | 0.344002 | 0.654698 |
| 14 | 0.463919 | 0.354043 | 0.648421 |
| 15 | 0.454349 | 0.349236 | 0.644789 |

## 8. Prediction Distribution

正式 best checkpoint 的 prediction distribution 为：

| Predicted k | Overall | Percent | NQ | SQuAD | Trivia |
|---:|---:|---:|---:|---:|---:|
| 1 | 900 | 100.00% | 300 | 300 | 300 |
| 2 | 0 | 0.00% | 0 | 0 | 0 |
| 3 | 0 | 0.00% | 0 | 0 | 0 |
| 4 | 0 | 0.00% | 0 | 0 | 0 |
| 5 | 0 | 0.00% | 0 | 0 | 0 |
| 6 | 0 | 0.00% | 0 | 0 | 0 |
| 7 | 0 | 0.00% | 0 | 0 | 0 |
| 8 | 0 | 0.00% | 0 | 0 | 0 |
| 9 | 0 | 0.00% | 0 | 0 | 0 |
| 10 | 0 | 0.00% | 0 | 0 | 0 |
| 11 | 0 | 0.00% | 0 | 0 | 0 |
| 12 | 0 | 0.00% | 0 | 0 | 0 |
| 13 | 0 | 0.00% | 0 | 0 | 0 |
| 14 | 0 | 0.00% | 0 | 0 | 0 |
| 15 | 0 | 0.00% | 0 | 0 | 0 |

作为数据记录，Hard Oracle label 的 train/validation 分布为：

| Hard Oracle k | Train | Validation |
|---:|---:|---:|
| 1 | 5166 | 568 |
| 2 | 726 | 87 |
| 3 | 490 | 52 |
| 4 | 282 | 26 |
| 5 | 227 | 26 |
| 6 | 189 | 25 |
| 7 | 168 | 14 |
| 8 | 141 | 17 |
| 9 | 123 | 17 |
| 10 | 99 | 9 |
| 11 | 126 | 10 |
| 12 | 90 | 12 |
| 13 | 91 | 15 |
| 14 | 83 | 13 |
| 15 | 99 | 9 |
| **Total** | **8100** | **900** |

## 9. 数据与实验完整性

### 9.1 数据 topology

对现有文件进行只读核验所得结果：

- Oracle 共 9000 条且 `question_id` 唯一；
- train IDs 共 8100 个且唯一；
- validation IDs 共 900 个且唯一；
- train / validation ID overlap 为 0；
- 两个 split 的 ID 并集与 9000 条 Oracle ID 精确相同；
- validation 按数据集为 NQ 300、SQuAD 300、Trivia 300；
- 正式 prediction 共 900 条且 `question_id` 唯一，ID 集合与 validation split 精确相同；
- prediction 中的 `dataset` 和 `hard_oracle_k` 与 Oracle 对应记录精确一致。

当前四个核心数据文件的 SHA-256 为：

| 文件 | SHA-256 |
|---|---|
| `new_onehop/data/train/oracle/all.jsonl` | `dc5d455e9fff3886c45c8acfd64d8423aa732672699a8dad9156050396603004` |
| `new_onehop/data/train/curves/all.jsonl` | `48787ef1322b3848dc5919198f63eb5cdd18353ad40fcb8a749635b18852b7c0` |
| `new_onehop/data/splits/train_ids.json` | `0722a28624133a4826037e3057d3835e96956e36d6f2737dfc090640fb971bbe` |
| `new_onehop/data/splits/val_ids.json` | `4dbcf15942ade8a1ffa7c6cf89b62dacf67227e85863149ec51d55db49fcb4ae` |

这些当前 hash 与 Phase 2 报告中记录的四个 hash 相同。

### 9.2 Prediction 与 metrics 完整性

900 条正式 prediction 的只读校验结果：

| Check | Result |
|---|---:|
| Rows | 900 |
| Probability count per row | 15 |
| Non-finite / out-of-range probability rows | 0 |
| Max probability-sum error | `1.6065314412117004e-07` |
| `predicted_k` / probability argmax mismatches | 0 |
| Invalid `predicted_k` range rows | 0 |
| Prediction / validation ID mismatch | 0 |
| Prediction / Oracle label mismatch | 0 |
| Prediction / Oracle dataset mismatch | 0 |

RAG metrics 中 dynamic replay 和 15 个 fixed-k baseline 的 overall、NQ、SQuAD、
Trivia 共 64 个分组记录，`num_queries_with_unvalidated_tokens` 全部为 0。

`training_history.json` 与 `training_history.jsonl` 解析后内容完全一致。以下字段在
`summary.json`、best checkpoint `metadata.json` 和独立 predictor metrics 之间一致：

- best epoch：1；
- optimizer step：254；
- validation loss：`1.5060049777560764`。

Phase 2 报告中引用的 Phase 1 predictor、RAG F1、avg context tokens 和 avg predicted
`k` 与 Phase 1 原始文件一致，未发现共有数值不一致。Phase 2 报告另列的 Phase 1
utility 字段来源已在第 6 节说明。

## 10. 主要实验产物

```text
new_onehop/outputs/checkpoints/multiclass_deberta_seed42/
├── config.yaml
└── best/
    ├── config.yaml
    ├── metadata.json
    ├── model.pt
    └── tokenizer/

new_onehop/outputs/predictions/val/
└── multiclass_deberta_seed42.jsonl

new_onehop/outputs/metrics/val/
├── multiclass_deberta_seed42_predictor.json
└── multiclass_deberta_seed42_rag.json

new_onehop/outputs/logs/multiclass_deberta_seed42/
├── config.yaml
├── summary.json
├── training_history.json
└── training_history.jsonl
```

独立 smoke-test 产物为：

```text
new_onehop/outputs/checkpoints/multiclass_deberta_seed42_smoke/
new_onehop/outputs/predictions/val/multiclass_deberta_seed42_smoke.jsonl
new_onehop/outputs/metrics/val/multiclass_deberta_seed42_smoke_predictor.json
new_onehop/outputs/metrics/val/multiclass_deberta_seed42_smoke_rag.json
new_onehop/outputs/logs/multiclass_deberta_seed42_smoke/
```

本报告仅记录 Phase 1 的 train / validation 实验与已有 validation replay 产物。
