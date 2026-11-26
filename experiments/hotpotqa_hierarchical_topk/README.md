# HotpotQA 两阶段动态 Top-K

后续分类器第一层区分 `k≤3` 和 `k>3`。低档由第二层在 `0、1、2、3` 中四分类，高档固定使用 `k=10`。
实验先整理冻结的 5,000 道 HotpotQA Train 样本在 `k=0～15` 下的预测，再按固定划分比较两种标签策略。

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
├── configs/
│   ├── settings.json
│   └── label_settings.json
├── scripts/
│   ├── prepare_data.py
│   ├── analyze_data.py
│   ├── split_data.py
│   └── analyze_labels.py
├── data/
│   ├── per_question_metrics.jsonl
│   └── splits/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── split_manifest.json
└── runs/
    ├── analysis/
    │   ├── fixed_k_metrics.csv
    │   ├── validation_report.json
    │   └── figures/
    └── label_analysis/
        ├── oracle_metrics.csv
        ├── label_distribution.csv
        ├── mapping_analysis.csv
        ├── split_statistics.json
        └── figures/
```

逐题指标和由 Manifest 导出的三个 QID-only JSONL 由 Git 忽略。代码、配置、完整划分 Manifest、汇总 CSV/JSON 和 PNG 图表纳入 Git。

## 划分与标签规则

`split_data.py` 按 `level × type` 联合分层。每格配额取理想值的向下或向上取整，在保留每个分层总数及各集合总数的约束下，最小化与理想配额的平方误差。同分取配额矩阵的字典序最小值。用 `random.Random(42)` 依次打乱各分层内的 QID，按配额分配，各集合按原样本顺序输出。首次执行保存全部有序 QID、输入与 QID 文件的 SHA-256、划分配置和计数。后续执行核对现有 Manifest，可恢复缺失的 QID JSONL，遇到冲突即报错。[划分 Manifest](data/splits/split_manifest.json) · [划分统计](runs/label_analysis/split_statistics.json)

| level / type | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| easy / bridge | 592 | 74 | 74 |
| easy / comparison | 151 | 19 | 19 |
| medium / bridge | 2,082 | 260 | 260 |
| medium / comparison | 495 | 62 | 62 |
| hard / bridge | 536 | 67 | 67 |
| hard / comparison | 144 | 18 | 18 |
| 合计 | 4,000 | 500 | 500 |

Oracle 和固定 k 基线分别使用 Train、Validation 两个集合。Test 仅参加划分和分布核验，留作最终评估。分析脚本读取现有逐题指标。Oracle 表中的 F1、EM、Context Tokens、Selected k 和 Utility 均按实际执行的 k 取值，再在相应集合上求均值。

| 策略 | 选择与执行 |
| --- | --- |
| Oracle-Full | 在 `0～15` 中选 k 并执行，给出该目标在完整范围内的参考上限 |
| Oracle-Mapped（A） | 在 `0～15` 中选择，若 `k>3` 则执行 `k=10` |
| Oracle-Candidate（B） | 在 `{0,1,2,3,10}` 中选择并执行，给出该目标在可执行候选集合内的参考上限 |
| Fixed k | 分别固定执行 `0、1、2、3、10` |

两个选择目标分别为 Answer F1 和 `U(k) = F1(k) − λ × ContextTokens(k) / 1000`。若与最大值的差处于 `1e-12` 数值容差内，优先较少 Context Tokens，再优先较小 k。λ 为 `0、0.025、0.05、0.1、0.2、0.4、0.6、0.8、1.0`。λ=0 与最大 F1 的选择相同。第一层 `0` 为低档，`1` 为固定执行 k=10 的高档。第二层仅在第一层真实标签为 0 的题中统计四类占比。

此前 BERT 的 Scheme D 同样使用 F1 减去成本惩罚，但先要求 `F1(k) ≥ max F1 − ε` 且句级 Evidence Recall 下界至少为 `τ`，按每道题在全 15 个 k 中的最大 token 数归一化，并在通过筛选的 k 上做温度 softmax。旧配置 `ε=0.1`、`τ=0.5`、`λ=0.1`、`T=0.1`，全零 F1 和空候选的软标签为 null。本次在各策略的固定搜索空间内直接做确定性选择，所有题均保留，固定除数 1,000 使 λ 具有跨题一致的单位。若只对齐成本项，对最大成本为 `M` 的某道题，旧 λ 折算为本次单位是 `旧 λ × 1000 / M`，该值因题而异。[Scheme D 实现](../hotpotqa_train_5000_bert/src/soft_oracle.py) · [本阶段配置](configs/label_settings.json)

## Oracle 与映射结果

| 集合 | 最大 F1 策略 | F1 | EM | 平均 Context Tokens | 平均执行 k |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | Full | 0.7150 | 0.62125 | 128.13 | 1.4545 |
| Train | Mapped | 0.6690 | 0.58250 | 143.49 | 1.6570 |
| Train | Candidate | 0.6733 | 0.58250 | 96.62 | 1.0558 |
| Validation | Full | 0.7075 | 0.61200 | 112.50 | 1.2940 |
| Validation | Mapped | 0.6729 | 0.57800 | 125.72 | 1.4480 |
| Validation | Candidate | 0.6764 | 0.57800 | 87.23 | 0.9680 |

| 集合 | 策略 | k=0 | k=1 | k=2 | k=3 | k=10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | A | 2,058 | 1,042 | 320 | 122 | 458 |
| Train | B | 2,250 | 1,077 | 336 | 128 | 209 |
| Validation | A | 265 | 133 | 37 | 19 | 46 |
| Validation | B | 284 | 136 | 39 | 20 | 21 |

- 最大 F1 下，Train 有 458 题的全范围最优 `k>3`（11.45%），其中 275 题映射后 F1 下降（60.04%）。Validation 对应为 46 题（9.20%）和 27 题（58.70%）。A/B 全集最终标签不一致率分别为 6.23% 和 5.00%。
- 在这些高 k 样本上，Train 的 Full / Mapped / Candidate 平均 F1 为 `0.8368 / 0.4355 / 0.4726`，平均 Tokens 为 `669.96 / 804.10 / 394.78`。Validation 对应为 `0.8422 / 0.4659 / 0.5038` 和 `664.30 / 807.96 / 389.63`。
- Train 有 865 题（21.63%）、Validation 有 114 题（22.80%）在全部 k 下 F1 为 0。两种目标和所有 λ 下，这些样本在 A、B 两方案均选 k=0。最大 F1 下，Train 去掉这部分后，A、B 的 k=0 占比分别为 38.05%、44.18%，含全零题时为 51.45%、56.25%。
- A 在最大 F1 下提供更多高档样本，便于保留全范围最优 k 的信号。统一执行 k=10 后，缺失的高 k 候选可能使 F1 下降并增加成本。B 按部署候选优化 F1 或效用，两个集合在所有扫描点的平均 Context Tokens 均更低。高 λ 下 B 会用 F1 换成本，高档标签也更稀少。以 `λ=1` 为例，B 仅有 93 道 Train 高档题和 9 道 Validation 高档题，Validation 的 comparison 高档仅 1 道。
- 两个集合中，Candidate 在全部 λ 下的平均效用均高于 Mapped。较大 λ 下 Mapped 的平均 F1 可以超过 Candidate，例如 Validation 的 `λ=1` 下为 `0.6700 / 0.6607`，平均 Tokens 为 `88.07 / 63.67`，效用为 `0.5819 / 0.5970`。最终标签方案与 λ 留待后续确定。

### 完整输出

- `oracle_metrics.csv` 每行对应一个集合、目标、λ 和策略，含五项均值及所选 k 计数。最大 F1 行的 λ 和效用留空。所有 λ 下的基线也有对应效用。[Oracle 与基线指标](runs/label_analysis/oracle_metrics.csv)
- `label_distribution.csv` 按方案、目标、λ、集合、整体或 level/type、全样本或零/非零 F1 子群列出标签计数和比例。`original_full_k` 仅用于 A。`denominator` 是对应分组和阶段的题数，第二层分母为该组的低档题数。空分母的比例留空。[标签分布](runs/label_analysis/label_distribution.csv)
- `mapping_analysis.csv` 用相同题目比较 Full、Mapped、Candidate，含三组绝对均值、`Mapped − Full` 和 `Mapped − Candidate`。`all` 为当前集合全部题目，`full_high_k` 含原本已选 k=10 的题，`changed_k` 仅含执行 k 实际变化的高档题。方向与标签分歧的比例以该行子集为分母。[映射影响](runs/label_analysis/mapping_analysis.csv)
- [最大 F1 的最终标签分布](runs/label_analysis/figures/max_f1_label_distribution.png) · [F1 随 λ 变化](runs/label_analysis/figures/f1_vs_lambda.png) · [Tokens 随 λ 变化](runs/label_analysis/figures/context_tokens_vs_lambda.png) · [F1 与 Tokens](runs/label_analysis/figures/f1_vs_context_tokens.png) · [五类标签随 λ 变化](runs/label_analysis/figures/labels_vs_lambda.png) · [映射后 F1 变化方向](runs/label_analysis/figures/mapping_f1_change.png)

## 复现

从仓库根目录运行。使用已有 `adaptiverag` Python 环境及其中的 Transformers、Tokenizers、ujson、Matplotlib，服务端预测产物和指定 tokenizer 快照需已在本机。

```sh
PYTHON=/home/dengxin/miniconda3/envs/adaptiverag/bin/python
EXPERIMENT=experiments/hotpotqa_hierarchical_topk
"$PYTHON" -B "$EXPERIMENT/scripts/prepare_data.py" --settings "$EXPERIMENT/configs/settings.json"
"$PYTHON" -B "$EXPERIMENT/scripts/analyze_data.py" --settings "$EXPERIMENT/configs/settings.json"
```

两条命令均离线执行。配置中的 `source_dir` 相对于仓库根目录，`inputs` 相对于 `source_dir`，`outputs` 相对于本实验目录。`--settings` 省略时读取本实验的 `configs/settings.json`。重复执行会根据相同输入重新生成输出。分析脚本会将 `k=0` 的总体、F1 分布、level 与 type 分组统计以 JSON 打印到终端。

已有 `per_question_metrics.jsonl` 和基础校验报告时，只需执行以下两条命令。复用环境中的 NumPy 与 Matplotlib。

```sh
PYTHON=/home/dengxin/miniconda3/envs/adaptiverag/bin/python
EXPERIMENT=experiments/hotpotqa_hierarchical_topk
"$PYTHON" -B "$EXPERIMENT/scripts/split_data.py"
"$PYTHON" -B "$EXPERIMENT/scripts/analyze_labels.py"
```

两条命令默认读取 `configs/settings.json` 和 `configs/label_settings.json`。可分别通过 `--settings` 与 `--label-settings` 指定。划分在同一 Manifest 下保持固定，分析产物按当前标签配置重建。
