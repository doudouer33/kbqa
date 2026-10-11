# HotpotQA 500+500 实验新增文件说明

本文档说明本次 HotpotQA 实验过程中新增或生成的文件。实验目标是从 HotpotQA dev distractor 数据中固定抽取 bridge 500 条和 comparison 500 条，共 1000 条样本，然后使用 Qwen RAG 流程测试检索数量 `k=1..15`。

## 一、原始数据

### `raw_data/hotpotqa/hotpot_dev_distractor_v1.json`

HotpotQA 官方 dev distractor 原始数据文件，共 7405 条样本。它保留了问题、答案、问题类型、支持事实和候选段落，是后续转换和抽样的输入。

## 二、处理后的数据

### `processed_data/hotpotqa/dev.jsonl`

把官方 JSON 原始数据转换成项目 RAG 流程可以读取的 JSONL 格式。每行对应一个问题。

### `processed_data/hotpotqa/dev_1000_subsampled.jsonl`

正式实验数据集，共 1000 条：

- bridge：500 条
- comparison：500 条
- 抽样 seed：`13370`

后续 k=1 到 k=15 的正式预测都使用这个文件。

### `processed_data/hotpotqa/dev_1000_sample_manifest.json`

1000 条正式样本的抽样清单，记录随机种子、样本类型和 question ID，用于复现实验和确认抽样没有变化。

### `processed_data/hotpotqa/dev_2_subsampled.jsonl`

2 条 smoke 测试数据，只用于验证 RAG 流程、服务连接和配置是否可以正常运行，不属于正式统计样本。

### `processed_data/hotpotqa/dev_2_sample_manifest.json`

2 条 smoke 测试样本的抽样记录。

## 三、抽样脚本

### `processing_scripts/subsample_hotpot_by_type.py`

按照问题类型进行固定抽样：

- `bridge` 抽取 500 条
- `comparison` 抽取 500 条
- 使用固定随机种子 `13370`

这个脚本用于重新生成 `dev_1000_subsampled.jsonl` 和对应的 manifest。

## 四、实例化配置

### `instantiated_configs/oner_qa_qwen_hotpotqa____prompt_set_1___bm25_retrieval_count__K___distractor_count__1.jsonnet`

这里的 `K` 从 1 到 15，共 15 个配置文件。每个配置文件固定一个检索数量，例如：

- `retrieval_count__1`：k=1
- `retrieval_count__8`：k=8
- `retrieval_count__15`：k=15

配置中记录模型、提示模板、检索数量和数据集映射关系。

## 五、正式预测结果

### `predictions/dev_1000/`

正式的 1000 条样本预测结果。目录下有 k=1 到 k=15 共 15 个子目录，每个子目录对应一个检索预算。

每个 k 目录通常包含以下文件：

| 文件名模式 | 含义 |
|---|---|
| `prediction__...json` | 模型对 1000 个问题生成的预测答案及推理结果 |
| `evaluation_metrics__...json` | 项目普通评测指标，如 F1、EM、precision、recall |
| `official_evaluation_metrics__...json` | HotpotQA 官方评测脚本计算的 F1、EM、precision、recall |
| `ground_truth__...json` | 对应的标准答案和评测真值 |
| `zero_single_multi_classification__...json` | 问题在零跳、单跳、多跳类别上的辅助分类结果 |
| `prediction__..._chains.txt` | 每道题的检索和推理链记录 |
| `prediction__..._time_taken.txt` | 本次预测耗时 |
| `prediction__..._full_eval_path.txt` | 完整评测输出路径记录 |
| `prediction__..._variable_replacements.json` | 运行时配置变量替换记录 |
| `config__...jsonnet` | 本次预测实际使用的配置副本 |

当前官方评测中最好的固定 k 是 `k=8`：

- Official F1：`0.18202`
- Official EM：`0.098`

### `predictions/dev_2/`

smoke 测试预测结果，共 20 个文件：

- k=1：10 个文件
- k=15：10 个文件

这些结果只用于确认流程可运行，不应与正式 1000 条结果混合统计。

## 六、分析结果

### `analysis/analyze_hotpot_500_500.py`

读取 k=1 到 k=15 的预测和评测文件，计算每道题的最佳 k、各问题类型的 k 分布、固定 k 的 F1 和 token 统计，并生成汇总文件和图表。

### `analysis/hotpot_500_500/固定结果文件`

| 文件 | 含义 |
|---|---|
| `fixed_k_results.csv` | bridge、comparison、combined 在每个固定 k 下的平均 F1 和上下文 token 统计 |
| `oracle_distribution.csv` | 每道题达到最佳 F1 时的 oracle k 分布 |
| `hotpot_basic_statistics.json` | bridge、comparison、combined 的数量、均值、中位数、熵等统计 |
| `per_question_curves.jsonl` | 每道题从 k=1 到 k=15 的 F1 曲线和 token 曲线 |
| `hotpot_500_500_analysis.md` | 最终文字汇总报告 |
| `fixed_k_f1.png` | 固定 k 的平均 F1 曲线图 |
| `token_curve.png` | 检索数量与上下文 token 数量关系图 |
| `bridge_oracle_distribution.png` | bridge 样本最佳 k 分布图 |
| `comparison_oracle_distribution.png` | comparison 样本最佳 k 分布图 |
| `combined_oracle_distribution.png` | 全部 1000 条样本最佳 k 分布图 |

## 七、代码和测试修改

### `base_configs/oner_qa_qwen_hotpotqa.jsonnet`

配置本地 Qwen2.5-7B-Instruct 模型名、上下文长度和生成 token 数，使项目适配本地 vLLM 服务。

### `commaqa/models/gpt3generator.py`

让项目通过 OpenAI 兼容接口连接本地 Qwen 服务，并支持本地 tokenizer 加载。

### `requirements.txt`

补充 `matplotlib`，用于生成分析 PNG 图。

### `tests/`

包含抽样脚本、Qwen 客户端和分析流程的测试文件，用于验证新增代码。

### `hotpot_500_500_rag_pipeline.md`

记录完整实验流程、服务地址、运行命令、结果位置和注意事项。

## 八、哪些文件是正式结果

正式实验应主要保留和使用：

```text
processed_data/hotpotqa/dev_1000_subsampled.jsonl
processed_data/hotpotqa/dev_1000_sample_manifest.json
predictions/dev_1000/
analysis/hotpot_500_500/
```

`dev_2` 文件和 `predictions/dev_2/` 只用于 smoke 测试；`cache/`、`.venv/`、`__pycache__/` 等属于运行环境或缓存，不是实验结果。
