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
