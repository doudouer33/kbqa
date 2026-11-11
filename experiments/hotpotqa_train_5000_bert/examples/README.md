# 数据格式示例

这 7 道真实题目展示 `data/` 中完整产物的格式：3 道有效标签、各 1 道 all_zero_f1 / empty_candidates，以及 2 道按用户确认保留 unknown 支持句的题目。选样用于展示格式和边界情形，不代表完整数据分布。

- `queries.jsonl`：模型输入，只含 question_id 和 question_text。
- `labels.jsonl`：Original Soft Label、候选集合、状态和熵；无效标签概率保持 null。
- `curves.jsonl`：每题 k=1..15 的官方 F1、Context Tokens 和句子级 Evidence Recall 等离线曲线。

完整 5000 道标签和曲线保留在本地 `data/`，统计见 `../reports/label_distribution.json`。完整 Validation/Test 预测、测试决策和划分 ID 已提交到 `data/`，可以结合 `../reports/evaluation.json` 分析结果。
