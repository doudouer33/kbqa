# HotpotQA 两阶段动态 Top-K 数据准备

后续分类器第一层区分 `k≤3` 和 `k>3`。低档由第二层在 `0、1、2、3` 中四分类，高档固定使用 `k=10`。
本阶段对同一批冻结的 5,000 道训练题整理现有 `k=0～15` 预测，并汇总固定 k 的描述性统计。

## 数据来源与校验

- 原始题目来自 [`../hotpotqa_train_5000/data/train_5000.jsonl`](../hotpotqa_train_5000/data/train_5000.jsonl)。该批样本使用 seed 42 从 HotpotQA Train 抽取。
- 正式 `k=1～15` 结果位于 `../hotpotqa_train_5000/runs/predictions/hotpotqa_train_5000_seed42/`，正式 `k=0` 结果位于同级的 `hotpotqa_train_5000_seed42_k0/`。
- `prepare_data.py` 只读原实验文件，核对样本和运行 Manifest、运行身份、完成标记、成功的 attempt、配置快照和产物 SHA-256。它按 QID 对齐预测和上下文，检查问题、预测、实际上下文和推理链记录的一致性。缺失、重复或不一致时会非零退出并指出出处。
- 逐题 JSONL 保留冻结样本的原顺序，每行包含 `qid`、`question`、`gold_answers`、`level`、`type` 以及完整的 16 个 `results`。每个结果包含原预测字符串、F1、EM 和 Context Tokens。

## 指标口径

- Gold 为 `answers_objects[0].spans` 中唯一的答案。评分直接调用 [HotpotQA 评分实现](../../official_evaluation/hotpotqa/hotpot_evaluate_v1.py) 的 `f1_score(prediction, gold)[0]` 和 `int(exact_match_score(prediction, gold))`。
- Context Tokens 使用 [此前 Train 5000 分析](../hotpotqa_train_5000_bert/src/build_data.py) 的 Qwen fast tokenizer。脚本从本地缓存加载配置中固定的 `Qwen/Qwen2.5-7B-Instruct` 快照，计算 `len(tokenizer.encode(effective_context_text, add_special_tokens=False))`，保留完整文本长度。该字段是 `contexts.jsonl` 中已经在最终 Prompt 中逐字验证过的实际检索上下文。`k=0` 逐题核验原检索序列、Reader 上下文、`effective_context_text` 及上下文后缀均为空，且 tokenizer 计数为 0。
- 空答案按 `not prediction.strip()` 计数。评分和 JSONL 均使用正式产物中的原字符串。`k=0` 的完成标记记录了一条长度截断后按原策略保存的空答案和一条标记为手动空答案的记录。两条均纳入 5,000 题分母，来源 QID 见校验报告。
- CSV 和图表的 F1、EM 使用 0 到 1 的尺度。Context Tokens 与 Answer F1 图的每个点使用该固定 k 的平均 Context Tokens 和平均 F1。各组统计均基于同一训练样本，用于描述数据特征。

## 结果

完整样本通过校验，共 5,000 道题和 80,000 条逐题逐 k 结果。`k=1～15` 的 75,000 条预测、F1、EM 和 Context Tokens 与此前 Train 5000 实验保留的逐题指标完全一致。全表有 3 条空答案，其中 `k=0` 有 2 条，`k=11` 有 1 条。[固定 k 统计](runs/analysis/fixed_k_metrics.csv) · [校验报告](runs/analysis/validation_report.json)

`k=0` 的平均 F1 为 0.3111，平均 EM 为 0.2490。5,000 题的 Context Tokens 均为 0。

| k=0 F1 区间 | 题数 | 比例 |
| --- | ---: | ---: |
| `F1 = 0` | 3,161 | 63.22% |
| `0 < F1 < 1` | 589 | 11.78% |
| `F1 = 1` | 1,250 | 25.00% |

| 分组 | 题数 | 平均 F1 | 平均 EM |
| --- | ---: | ---: | ---: |
| level · easy | 929 | 0.3699 | 0.2756 |
| level · medium | 3,221 | 0.3036 | 0.2509 |
| level · hard | 850 | 0.2755 | 0.2129 |
| type · bridge | 4,012 | 0.2308 | 0.1707 |
| type · comparison | 988 | 0.6374 | 0.5668 |

## 目录

```text
hotpotqa_hierarchical_topk/
├── README.md
├── .gitignore
├── configs/settings.json
├── scripts/
│   ├── prepare_data.py
│   └── analyze_data.py
├── data/per_question_metrics.jsonl
└── runs/analysis/
    ├── fixed_k_metrics.csv
    ├── validation_report.json
    └── figures/
        ├── mean_f1_vs_k.png
        ├── mean_em_vs_k.png
        ├── mean_context_tokens_vs_k.png
        └── mean_f1_vs_context_tokens.png
```

`data/*.jsonl` 由本实验的 Git 规则忽略。CSV、校验报告与 PNG 图表可随代码追踪。

## 复现

从仓库根目录运行。使用已有 `adaptiverag` Python 环境及其中的 Transformers、Tokenizers、ujson、Matplotlib，服务端预测产物和指定 tokenizer 快照需已在本机。

```sh
PYTHON=/home/dengxin/miniconda3/envs/adaptiverag/bin/python
EXPERIMENT=experiments/hotpotqa_hierarchical_topk
"$PYTHON" -B "$EXPERIMENT/scripts/prepare_data.py" --settings "$EXPERIMENT/configs/settings.json"
"$PYTHON" -B "$EXPERIMENT/scripts/analyze_data.py" --settings "$EXPERIMENT/configs/settings.json"
```

两条命令均离线执行。配置中的 `source_dir` 相对于仓库根目录，`inputs` 相对于 `source_dir`，`outputs` 相对于本实验目录。`--settings` 省略时读取本实验的 `configs/settings.json`。重复执行会根据相同输入重新生成输出。分析脚本会将 `k=0` 的总体、F1 分布、level 与 type 分组统计以 JSON 打印到终端。
