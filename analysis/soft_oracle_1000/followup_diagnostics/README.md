# Soft Oracle 后续离线诊断

本目录新增三项分析：Minimum Feasible与Utility预算比较、λ/T尖锐度扫描、非零空候选A/B原因及阈值转换。
复用上级Soft Oracle的数据加载、校验、候选筛选、Utility和Softmax，不修改原实现及两个旧实验目录。
无检索、LLM、RAG、训练或新增数据调用。Python >=3.8；图表使用已安装的matplotlib 3.7.5。

## 运行命令

在仓库根目录执行：

```sh
python3 -m unittest discover -s analysis/soft_oracle_1000/followup_diagnostics/tests -p 'test_*.py' -v
/home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/soft_oracle_1000/followup_diagnostics/analyze.py --config analysis/soft_oracle_1000/followup_diagnostics/config.json
```

默认创建本子目录 `outputs/experiment_NNN/` 的下一个编号；明确指定 `--output` 时该路径必须不存在，即使是空目录也拒绝覆盖。
所有输入按仓库根目录定位，可以从其他工作目录调用脚本绝对路径。
完整运行会读取原4个统计/数据文件，核对原1000题逐题标签和原参数扫描的重叠结果；不会运行上级分析main流程。
metadata保存SHA256、复用与新增代码版本、命令、参数、已有文件快照；原文件发生变化将报错。

## 配置

`config.json`指定上级配置与原experiment_002，以及：

- λ：0.1、0.2、0.4、0.8；T：0.01、0.02、0.05、0.1，共16组，不自动选最佳。
- ε、参考τ、Evidence和成本复用上级配置（0.1、0.5、sentence_recall_lower、context_tokens）。
- 空候选分析τ：0.5、0.75、1.0。
- 接近One-hot的描述阈值pmax≥0.95，同时报告全有效集和多候选集，避免单候选影响解释。

## 指标定义

- Minimum Feasible：候选中实际Context Tokens最少，成本并列取较小k，不假设Token单调。
- Hard、Minimum、Utility只在同一有效题集比较；逐题CSV保留全部1000题，无标签预算字段为空。
- Utility argmax复用原实现最大Utility、并列较小k；Temperature不影响预算排序。
- H自然对数；H/ln候选数、top1/top2概率和Utility差仅在多候选时定义，单候选存null。
- Utility跨度/T分布也仅统计多候选；pmax、H、期望质量成本的总体均值包括单候选。
- A=`evidence_unreachable`：非零F1，但全部k的Evidence均低于门槛。
- B=`constraint_conflict`：非零F1，有Evidence达标k，但无同时满足F1近优要求的k。
- 全零F1状态优先，不混入A/B。边界沿用原实现1e-12容差；unknown保留lower/upper及数量，不回退。
- 提高τ的新失效样本以先前仍有效的题为分母；另计旧B→A，防止总数相减混淆原因。
- 所有质量/成本结果为Offline Oracle / Label Analysis，不能直接推断Query-only模型泛化。

## 新增文件与输出

代码为 `analyze.py`、`config.json`、本README与 `tests/test_followup.py`，无旧文件修改。

| 输出 | 内容 |
| --- | --- |
| min_feasible_comparison.csv | 3策略×3组，公平的质量/成本/证据与配对差 |
| min_feasible_per_question.csv | 全1000题的3策略预算与状态，无标签不回退 |
| sharpness_sweep.csv | 16组λ/T×3组，48行完整统计 |
| sharpness_per_question.jsonl | 16×752=12032行逐题概率、Utility、top2差与成本 |
| empty_candidate_analysis.csv | 3个τ×3组的A/B分布和冲突F1差 |
| empty_candidate_per_question.jsonl | 3×1000行全部状态、完整F1/Evidence/Token曲线及预测 |
| empty_candidate_transitions.csv | 相邻τ间的新失效A/B与旧B→A转换 |
| distribution_summary.json | 分位数/直方图、按候选数分层熵、预算比较细节 |
| examples.jsonl | 全部预算差异题，A/B按类型取qid顺序前两例 |
| report.md | 新中文完整分析报告，含6张PNG图 |
| validation.json | 输入复用、旧扫描交叉验证、落盘重读和参数不变量 |
| metadata.json | 输入/旧结果/代码/配置/输出SHA256，已有文件快照与环境 |
| figures/*.png | 三策略、尖锐度、概率差/分布、质量成本、原因分布、案例曲线 |
| tests.log / delivery_audit.json | 补充交付核验：新旧测试、原文件SHA256、所有分组统计独立重算和Git检查 |

父目录CSV Git属性保证标准CRLF字节保持不变，不改变旧数据。图内用英文避免本机缺少中文字体，报告用中文图注。

本次完整输出为 `outputs/experiment_001/`。实际运行命令为：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/soft_oracle_1000/followup_diagnostics/analyze.py --config analysis/soft_oracle_1000/followup_diagnostics/config.json --output analysis/soft_oracle_1000/followup_diagnostics/outputs/experiment_001
```

该目录现已存在，再运行会拒绝覆盖；重跑请省略`--output`或使用新编号。tests.log与delivery_audit.json为完成后附加的交付证据，不属于脚本主运行的必需产物。
