# Qwen HotpotQA Prompt 与 Token 预算诊断

本轮复用 `predictions/hotpot_stop_40_20261013_1050/samples.jsonl` 的原始字节、
40 个 IDs 与顺序，只运行 k=1、8、15；不重新抽样，不改公共推理、答案提取或评分代码。

独立配置为 `base_configs/oner_qa_qwen_hotpotqa_prompt_token_diagnostic.jsonnet`，
独立 Prompt 为 `prompts/hotpotqa/gold_with_1_distractors_context_cot_qa_prompt_token.txt`。
唯一 Prompt 差异是在首个选定示例 `5abb14bd5542992ccd8e7f07` 的 metadata 后插入格式指令。
`read_prompt` 筛选后，这段指令成为 15 个原有示例之前的前缀；原示例正文不变。
不能把前缀放在 metadata 之外，否则原有筛选器会删除它。

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_prompt_token_diagnostic/run_diagnostic.py \
  --set-name hotpot_prompt_token_40_UNIQUE_RUN_NAME

/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_prompt_token_diagnostic/analyze.py \
  predictions/hotpot_prompt_token_40_UNIQUE_RUN_NAME
```

输出目录和新样本文件必须不存在。运行器从 `/v1/models` 确认 32768 上下文上限，
对全部完整 Prompt 调用 `/tokenize`（包含 chat template，无模型生成），
检查 16384 本地预算、512 生成预算和 256 安全余量，然后启动原有
`run.py write → run.py predict → predict.py → configurable_inference`。
`estimated_generation_length` 与 `max_tokens` 均为 512，继续 `stop=[]`、temperature=0、top_p=1。

日志复用上一轮的 `analysis.hotpot_stop_diagnostic.hooks.install`，保留完整 HTTP
请求与响应、Prompt 初始化和运行时拟合、原始输出、提取器捕获与最终答案。
生成缓存通过原函数 `__wrapped__` 绕过，保留原 API 调用体。
新增独立守卫在生成前检查全部 15 个示例内容、顺序、指令和当前题目/检索正文，
并核对真实 BM25 请求及完整结果与上轮一致。一旦裁剪或内容变化就退出，不能静默降级。
启动钩子失败也会退出，避免 Python 忽略 `sitecustomize` 异常后继续未记录的推理。

`preflight.json` 记录每条 Prompt 的 SHA-256、正文 token 数、服务端完整 chat token 数和开销。
`guard_events.jsonl` 记录运行时守卫；`http_events.jsonl` 与 `diagnostics.jsonl` 保存原始证据。
后者在新实验内部补充题型、gold、示例 IDs、格式观察和原评分器的逐题分数。
`comparison.jsonl/csv` 与 `summary.json` 对照上轮 stop 修复结果；
`raw_output_review.md` 展示全部 120 条配对原始输出。

离线语法检查只观察格式，不参与答案提取。`format_ok` 检查单行、精确标记、末尾句点及
Yes/No 的纯 yes/no；普通实体是否冗长、是否正确仍需人工检查。
`single_marker_format_ok` 另要求只有一个精确答案标记，用于观察重复结论。
原官方聚合结果保留 5 位小数，仓库结果保留 3 位；分析同时保留逐题均值的精度并核对舍入结果。

历史结果、原 Prompt、公共代码、评分器和源数据均记录运行前后的 SHA-256，
不覆盖任何历史文件，不提交 Git commit，不启动 500+500 实验。
