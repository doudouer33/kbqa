# GitHub 阅读入口：HotpotQA Smoke 示例

这是从固定 5000 题中选出的 6 道真实样本，覆盖 easy/medium/hard × bridge/comparison。
同样的 6 道题分别在 k=1、8、15 下运行，共有 18 条最终预测和上下文记录。
这份历史快照用于阅读代码和理解数据格式；正式 5000×15 的结果位于本地，未提交。

## 建议阅读顺序

1. [实验说明](../../README.md)：完整目录树、命令和字段含义。
2. [输入 smoke.jsonl](smoke.jsonl)：每行一题，含 question_id、question_text、answers_objects、contexts 等字段。
3. [pipeline.py](../../scripts/pipeline.py)：配置准备、运行、校验和完成标记。
4. [hooks.py](../../scripts/hooks.py)：捕获实际检索上下文、模型调用参数和原始生成。
5. [Smoke 输出目录](../predictions/hotpotqa_train_5000_seed42_smoke/)：按 k 分开的真实输出和逐题记录。
6. [选中的 15 个示例](../selected_prompt.txt)：模型提示词中的 Few-shot 部分。

## 真实目录结构

```text
runs/
├── run_manifest.json           # 冻结数据、配置、代码及依赖信息
├── directory_migration.json    # 历史路径到当前路径的映射
├── empty_answer_policy.json    # 允许空字符串预测的规则变更
├── selected_prompt.txt         # 15 个 Few-shot 示例
├── smoke/
│   ├── smoke.jsonl             # 6 道输入题
│   ├── smoke_acceptance_report.json
│   └── cache_validation_report.json
└── predictions/hotpotqa_train_5000_seed42_smoke/
    ├── run_identity.json
    └── oner_qa_qwen_hotpotqa_train_5000____prompt_set_1___bm25_retrieval_count__K___distractor_count__1/
        ├── prediction__hotpotqa_to_hotpotqa__smoke.json
        ├── prediction__hotpotqa_to_hotpotqa__smoke_chains.txt
        ├── contexts.jsonl
        ├── config__hotpotqa_to_hotpotqa__smoke.jsonnet
        ├── complete.json
        ├── observations/QID.json
        └── attempts/UTC_TIMESTAMP_ID/
            ├── attempt.json
            └── records/QID.json
```

K 在此为 1、8、15。每个 `contexts.jsonl` 有 6 行，包含原始检索段落 `raw_contexts`、
模型实际读取的 `reader_contexts` / `effective_context_text`、问题 `input_question`、
原始生成 `raw_generation`、提取答案 `prediction` 和 API / 缓存来源。
输入数据自带的 `contexts` 与实际检索得到的上下文应按各自字段含义区分。

原始完成标记、失败或中断的历史 attempt 和报告均按原文保留；日志、锁文件没有提交。
历史报告可能引用迁移前路径，使用 `directory_migration.json` 查找当前路径。
Smoke 输出产生于允许空答案规则变更之前；当前正式流程允许模型生成空字符串答案，
仍要求完整的逐题记录、正确 ID 和正常推理结束状态。完整源码见上面的 pipeline 和 hooks。
