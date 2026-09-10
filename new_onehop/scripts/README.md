# new_onehop 数据处理与生成说明

本目录用于构建动态 Top-k（`k=1..15`）预测任务的数据。

## 数据划分

- `data/train`：来自 `predictions/dev_3000`，NQ、SQuAD、Trivia 各 3000 条，
  共 9000 条。训练曲线用于生成 Oracle 标签。
- `data/test`：来自 `predictions/test`，三个数据集各 500 条，共 1500 条。
  test 只生成 curves，不生成 Oracle。
- `data/splits`：从 9000 条训练池中按数据集分层、固定种子划分的训练和验证 ID。
  当前比例为 90/10，即 8100 个 train ID 和 900 个 val ID。

## 目录结构

```text
new_onehop/
├── README.md
├── configs/
│   ├── data_config.yaml
│   └── train_config.yaml
├── scripts/
│   ├── data/
│   │   ├── build_query_curves.py
│   │   ├── validate_curves.py
│   │   └── build_oracle_labels.py
│   ├── train/
│   └── eval/
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

## Curve 格式

每条 curve 以 `question_id` 对齐同一问题在 `k=1..15` 下的实验结果：

```json
{
  "question_id": "single_nq_dev_2393",
  "dataset": "nq",
  "question": "who won the us open golf in 2017",
  "gold_answer": ["Brooks Koepka"],
  "results": {
    "1": {
      "prediction": "Park Sung-hyun",
      "em": 0.0,
      "f1": 0.0,
      "context_tokens": 151,
      "context_tokens_validated": true
    }
  }
}
```

`context_tokens` 只统计实际重建出的 retrieved context，不包含 question、prompt
模板和输出 token。`context_tokens_validated` 表示当前重建是否与历史 chain 的标题及
顺序严格一致。

训练集的所有 token 记录都必须通过校验。test/NQ 有 31 个问题在部分 k 上无法与
历史 chain 严格对齐；为了保持 1500 条 test 完整，曲线保留当前重建的 token 数，
并将对应的 `context_tokens_validated` 标为 `false`。这些值不能视为已复现的历史
token 消耗。

## Oracle

Oracle 仅由 `data/train/curves` 生成：

- Hard Oracle：先选择最高 F1；并列时选择 `context_tokens` 最少的 k。
- Soft Oracle：

  ```text
  utility(k) = F1(k) - lambda_T * context_tokens(k) / T_max
  ```

  再使用 temperature softmax 得到 `soft_oracle_distribution`。

test 数据不生成 Oracle，也不会写入 `data/train/oracle`。

## 生成和校验

从仓库根目录运行：

```bash
# 9000 条训练曲线
python new_onehop/scripts/data/build_query_curves.py --split train
python new_onehop/scripts/data/validate_curves.py --split train

# 只基于训练曲线生成 9000 条 Oracle，并生成 8100/900 ID 划分
python new_onehop/scripts/data/build_oracle_labels.py

# 1500 条测试曲线；不运行 Oracle
python new_onehop/scripts/data/build_query_curves.py --split test
python new_onehop/scripts/data/validate_curves.py --split test
```

所有输入、输出路径和数据规模定义在 `configs/data_config.yaml`；训练入口所需的
路径定义在 `configs/train_config.yaml`。

## Phase 1：Multiclass CE baseline

Phase 1 只使用问题文本预测 `k=1..15`，训练标签为
`label = hard_oracle_k - 1`。默认实验配置位于
`configs/train_multiclass.yaml`，所有 checkpoint、prediction 和 metrics 都写入
`new_onehop/outputs/`，不会写入或重建 `data/`。

从仓库根目录执行短 smoke test：

```bash
HTTP_PROXY=http://127.0.0.1:17890 \
HTTPS_PROXY=http://127.0.0.1:17890 \
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_multiclass.py \
  --config new_onehop/configs/train_multiclass.yaml \
  --max_train_steps 10 \
  --run-name multiclass_deberta_seed42_smoke
```

独立重跑 validation predictor 指标与 prediction：

```bash
HTTP_PROXY=http://127.0.0.1:17890 \
HTTPS_PROXY=http://127.0.0.1:17890 \
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/eval/evaluate_predictor.py \
  --checkpoint new_onehop/outputs/checkpoints/multiclass_deberta_seed42_smoke/best \
  --split val
```

使用保存的 prediction 做 validation RAG replay（包含 fixed k=1..15）：

```bash
HTTP_PROXY=http://127.0.0.1:17890 \
HTTPS_PROXY=http://127.0.0.1:17890 \
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/eval/evaluate_rag.py \
  --predictions new_onehop/outputs/predictions/val/multiclass_deberta_seed42_smoke.jsonl \
  --split val
```

正式训练去掉 smoke 参数：

```bash
HTTP_PROXY=http://127.0.0.1:17890 \
HTTPS_PROXY=http://127.0.0.1:17890 \
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_multiclass.py \
  --config new_onehop/configs/train_multiclass.yaml
```

## Phase 2：Ordered-threshold Ordinal BCE

Phase 2 复用 Phase 1 的 `OracleQuestionDataset`、dynamic padding、predictor metrics
和 RAG replay。零基 `labels` 在线转换为 14 个 `1[hard_k > j]` target；模型输出一个
question retrieval-demand score，并通过 softplus gap 参数化保证 14 个 threshold
始终严格递增。训练只使用无权重的 `BCEWithLogitsLoss(reduction="mean")`。

主 decoding 将 14 个 cumulative probability 转换为 `P(k=1)..P(k=15)` 后取
categorical argmax。每个 epoch 同时维护 `best_val_loss` 和
`best_rag_utility`（validation-only，`lambda_T=0.1`；`T_max` 来自完整 9000 条
train-pool curves）。

先运行独立数值检查：

```bash
/home/dengxin/miniconda3/envs/adaptiverag/bin/python -m unittest -v \
  new_onehop.tests.test_ordinal
```

运行 smoke test：

```bash
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_ordinal.py \
  --config new_onehop/configs/train_ordinal.yaml \
  --run-name ordinal_deberta_seed42_smoke \
  --max_train_steps 10
```

正式训练：

```bash
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  new_onehop/scripts/train/train_ordinal.py \
  --config new_onehop/configs/train_ordinal.yaml
```

分别评估两个 validation checkpoint：

```bash
for selection in best_val_loss best_rag_utility; do
  /home/dengxin/miniconda3/envs/adaptiverag/bin/python \
    new_onehop/scripts/eval/evaluate_ordinal_predictor.py \
    --checkpoint \
      new_onehop/outputs/checkpoints/ordinal_deberta_seed42/${selection} \
    --split val

  /home/dengxin/miniconda3/envs/adaptiverag/bin/python \
    new_onehop/scripts/eval/evaluate_rag.py \
    --config new_onehop/configs/train_ordinal.yaml \
    --predictions \
      new_onehop/outputs/predictions/val/ordinal_deberta_seed42_${selection}.jsonl \
    --split val
done
```

这些命令只使用 train/validation 资源，不进入 test。Ordinal prediction 中的
`probabilities` 仍是兼容现有 loader 的 15-way class distribution；额外的
`ordinal_cumulative_probabilities` 仅用于诊断。
