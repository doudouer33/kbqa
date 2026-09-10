# Phase 2: DeBERTa + Ordinal Head + Plain Ordinal BCE

## 1. 实验范围

本阶段在 Phase 1 现有数据、Dataset、DeBERTa encoder、训练配置和评测基础设施上，
增量实现 ordered-threshold ordinal predictor。

本实验只使用：

```text
microsoft/deberta-v3-base
+ ordered ordinal head
+ plain BCEWithLogitsLoss
```

没有使用：

- Soft Oracle supervision；
- Utility Loss 或 Risk Loss；
- class weight、`pos_weight`、weighted BCE 或 focal loss；
- oversampling 或 balanced sampler；
- label smoothing、calibration、ensemble；
- test 数据。

数据规模和主要训练参数保持为：

```text
train examples: 8100
validation examples: 900
epochs: 5
batch size: 32
learning rate: 2e-5
weight decay: 0.01
warmup ratio: 0.1
max length: 128
dropout: 0.1
seed: 42
```

## 2. Ordinal 实现

DeBERTa 首 token 表示经过 dropout 和 `Linear(hidden_size, 1)` 得到一个标量
retrieval-demand score：

$$
s(q)=w^Th_q+c.
$$

14 个 threshold 通过下式参数化：

$$
b_1=a,
$$

$$
b_j=b_{j-1}+\operatorname{softplus}(d_j)+10^{-6}.
$$

因此在训练前后始终满足：

$$
b_1<b_2<\cdots<b_{14}.
$$

实际 threshold 初始化为 `[-2,2]` 上的等间隔序列，内部 raw gap 通过 softplus
反函数初始化。

现有 Dataset 的零基 label 为：

$$
\text{label}=k^*-1.
$$

训练时在线构造：

$$
y_j=\mathbf{1}[k^*>j],\qquad j=1,\ldots,14.
$$

Loss 为无权重、mean reduction 的普通 ordinal BCE：

$$
\mathcal{L}_{ord}
=
-\frac{1}{14}
\sum_{j=1}^{14}
\left[
y_j\log\sigma(z_j)
+(1-y_j)\log(1-\sigma(z_j))
\right].
$$

其中：

$$
z_j=s-b_j.
$$

由累计概率：

$$
p_j=P(k>j)
$$

构造 15-way class distribution：

$$
\pi_1=1-p_1,
$$

$$
\pi_k=p_{k-1}-p_k,\qquad k=2,\ldots,14,
$$

$$
\pi_{15}=p_{14}.
$$

主 decoding 为：

$$
\hat{k}=\arg\max_k\pi_k.
$$

没有使用 threshold-count decoding 替代主结果。

## 3. Smoke Test

命令：

```bash
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_ordinal.py \
  --config new_onehop/configs/train_ordinal.yaml \
  --run-name ordinal_deberta_seed42_smoke \
  --max_train_steps 10
```

结果：

```text
optimizer steps: 10
training seconds: 6.586
train loss: 0.807657
val loss: 0.652026
validation predictions: 900
```

`best_val_loss` 和 `best_rag_utility` 两套 checkpoint 均成功保存并通过独立
predictor evaluator 和 RAG replay。Fixed-k 1..15 完整输出。

独立数值测试命令：

```bash
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  -m unittest -v new_onehop.tests.test_ordinal
```

3 项测试全部通过；compileall 和 isolated Ruff check 也通过。

## 4. 正式训练

命令：

```bash
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_ordinal.py \
  --config new_onehop/configs/train_ordinal.yaml
```

训练环境：

```text
GPU: NVIDIA GeForce RTX 4090
epochs: 5
optimizer steps: 1270
training seconds: 167.937
validation utility lambda_T: 0.1
train-pool T_max: 3491
```

`T_max` 来自全部 9000 条 train-pool curves、所有 `k=1..15` 的全局最大
`context_tokens`，没有使用 test curves。

| Epoch | Train loss | Val loss | Avg predicted k | RAG F1 | Avg tokens | RAG utility |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.408235 | 0.329227 | 1.000000 | 0.452164 | 140.708 | 0.448134 |
| 2 | 0.314208 | 0.338801 | 1.000000 | 0.452164 | 140.708 | 0.448134 |
| 3 | 0.289345 | 0.348635 | 1.000000 | 0.452164 | 140.708 | 0.448134 |
| 4 | 0.247386 | 0.376520 | 1.388889 | 0.453980 | 193.940 | 0.448424 |
| 5 | 0.212255 | 0.402062 | 1.902222 | 0.456264 | 263.548 | 0.448714 |

## 5. Best validation-loss checkpoint

Checkpoint：epoch 1，optimizer step 254。

| Metric | Value |
|---|---:|
| Val loss | 0.329227 |
| Accuracy | 0.631111 |
| Macro F1 | 0.051589 |
| MAE | 1.805556 |
| RMSE | 3.843754 |
| Within-1 | 0.727778 |
| Within-2 | 0.785556 |
| Under-retrieval rate | 0.368889 |
| Over-retrieval rate | 0.000000 |
| Avg predicted k | 1.000000 |
| Avg Oracle k | 2.805556 |
| RAG avg F1 | 0.452164 |
| RAG avg context tokens | 140.708 |
| RAG avg k | 1.000000 |
| RAG utility | 0.448134 |

Prediction 分布：

```text
k=1: 900 / 900
```

因此 best-val-loss checkpoint 仍然完全 collapse 到 `k=1`。

## 6. Best RAG-utility checkpoint

Checkpoint：epoch 5，optimizer step 1270。

| Metric | Value |
|---|---:|
| Val loss | 0.402062 |
| Accuracy | 0.597778 |
| Macro F1 | 0.050875 |
| MAE | 2.403333 |
| RMSE | 4.810752 |
| Within-1 | 0.690000 |
| Within-2 | 0.741111 |
| Under-retrieval rate | 0.337778 |
| Over-retrieval rate | 0.064444 |
| Avg predicted k | 1.902222 |
| Avg Oracle k | 2.805556 |
| RAG avg F1 | 0.456264 |
| RAG avg context tokens | 263.548 |
| RAG avg k | 1.902222 |
| RAG utility | 0.448714 |

Prediction 分布：

```text
k=1:   842 / 900 = 93.56%
k=15:   58 / 900 =  6.44%
k=2..14: 0
```

该 checkpoint 已不再全部预测 `k=1`，但动态变化只表现为从 `k=1` 跳到
`k=15`，没有产生中间 retrieval budget。

## 7. 与 Phase 1 Multiclass CE 对比

| Metric | Phase 1 CE | Ordinal best loss | Ordinal best utility |
|---|---:|---:|---:|
| Accuracy | 0.631111 | 0.631111 | 0.597778 |
| Macro F1 | 0.051589 | 0.051589 | 0.050875 |
| MAE | 1.805556 | 1.805556 | 2.403333 |
| Under rate | 0.368889 | 0.368889 | 0.337778 |
| Over rate | 0.000000 | 0.000000 | 0.064444 |
| Avg predicted k | 1.000000 | 1.000000 | 1.902222 |
| RAG F1 | 0.452164 | 0.452164 | 0.456264 |
| Avg context tokens | 140.708 | 140.708 | 263.548 |
| RAG utility | 0.448134 | 0.448134 | 0.448714 |

对比结论：

1. Best-val-loss 仍全部预测 `k=1`。
2. Macro-F1 没有改善；best-utility 下降 `0.000715`。
3. MAE 没有改善；best-utility 增加 `0.597778`。
4. Best-utility under-retrieval 降低 `0.031111`，但 over-retrieval 增加到
   `0.064444`。
5. Best-utility RAG F1 提高 `0.004099`。
6. Context tokens 增加 `122.84`，约为 `87.30%`。
7. 与 fixed-k=3 的 F1 差距从 `0.046469` 缩小到 `0.042370`，缩小
   `0.004099`。
8. 按 validation utility，best-utility 相比 Phase 1 提高 `0.000580`，属于非常
   轻微的 trade-off 改善。

Fixed baseline utility：

```text
fixed k=1: 0.448134
fixed k=2: 0.462966
fixed k=3: 0.486783
```

因此当前 Ordinal best-utility 的 `0.448714` 仍明显低于 fixed-k=2 和
fixed-k=3，尚未形成有竞争力的整体 F1-token trade-off。

核心实验结论：

> 在当前 hard Oracle 分布、plain BCE、ordered threshold 和 categorical argmax
> decoding 下，仅利用 `k=1..15` 的有序结构不足以可靠缓解 majority-class
> collapse。较晚 checkpoint 虽开始动态预测，但形成了 `k=1/k=15` 的端点模式。

## 8. Numerical sanity

两个正式 checkpoint 均满足：

| Check | Result |
|---|---:|
| Threshold count | 14 |
| Threshold order violations | 0 |
| Cumulative monotonicity violations | 0 |
| Negative class-probability rows | 0 |
| Max probability sum error | `1.19209e-07` |

所有 900 条 prediction 均满足：

- 15 个 class probabilities；
- probability finite 且非负；
- 每行 probability sum 约为 1；
- `predicted_k` 与 15-way categorical argmax 一致；
- `predicted_k` 位于 `[1,15]`。

## 9. Phase 1 regression

已有 Phase 1 multiclass checkpoint 使用 `strict=True` 成功加载。

回归结果：

- validation predictions 为 900 条；
- predictor loss 和 overall metrics 与原 Phase 1 文件精确一致；
- dynamic multiclass RAG metrics 精确一致；
- fixed-k 1..15 精确一致；
- 回归临时输出写入 `/tmp`，没有覆盖 Phase 1 产物；
- `predict_multiclass()` 行为保持不变；
- 通用 RAG evaluator 新增 `dynamic_model`，并为 Phase 1 保留
  `dynamic_multiclass` alias。

## 10. 数据完整性

以下四个文件训练前后的 SHA-256 完全一致：

| 文件 | SHA-256 |
|---|---|
| `new_onehop/data/train/oracle/all.jsonl` | `dc5d455e9fff3886c45c8acfd64d8423aa732672699a8dad9156050396603004` |
| `new_onehop/data/train/curves/all.jsonl` | `48787ef1322b3848dc5919198f63eb5cdd18353ad40fcb8a749635b18852b7c0` |
| `new_onehop/data/splits/train_ids.json` | `0722a28624133a4826037e3057d3835e96956e36d6f2737dfc090640fb971bbe` |
| `new_onehop/data/splits/val_ids.json` | `4dbcf15942ade8a1ffa7c6cf89b62dacf67227e85863149ec51d55db49fcb4ae` |

## 11. 主要产物

```text
new_onehop/outputs/checkpoints/ordinal_deberta_seed42/
├── config.yaml
├── best_val_loss/
└── best_rag_utility/

new_onehop/outputs/predictions/val/
├── ordinal_deberta_seed42_best_val_loss.jsonl
└── ordinal_deberta_seed42_best_rag_utility.jsonl

new_onehop/outputs/metrics/val/
├── ordinal_deberta_seed42_best_val_loss_predictor.json
├── ordinal_deberta_seed42_best_val_loss_rag.json
├── ordinal_deberta_seed42_best_rag_utility_predictor.json
└── ordinal_deberta_seed42_best_rag_utility_rag.json

new_onehop/outputs/logs/ordinal_deberta_seed42/
├── config.yaml
├── summary.json
├── training_history.json
└── training_history.jsonl
```

本报告只覆盖 Phase 2，没有实现或进入 Phase 3 Utility Loss。
