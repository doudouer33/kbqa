# HotpotQA 500+500 正式运行

复用 `run.py write/predict` 及 `analysis/analyze_hotpot_500_500.py`。正式配置是
`base_configs/oner_qa_qwen_hotpotqa_formal.jsonnet`，内容与验证过的 candidate 完全一致。

从仓库根目录执行：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_formal_runner/run_experiment.py \
  --set-name hotpot_formal_1000_20261014
```

默认先执行回归测试、正式数据中的 4 bridge + 4 comparison × k=1/15
真实 Smoke Test，以及不新增模型调用的断点恢复演练。全部通过后自动顺序运行
1000×15，逐 k 验证，再调用原分析脚本及验收检查。

恢复相同实验使用同一命令加 `--resume`。禁止使用 `--force`；代码、配置、Prompt、
数据发生变化时拒绝恢复。checkpoint 原子保存最终答案、原始链路、实际上下文、
API Token 等；模型响应在答案提取之前独立原子落盘。Token journal 在每题完成后
fsync，恢复时仅从完成 checkpoint 重建。存在已发送请求但没有持久响应时停止，
避免未知完成状态导致重复推理。预算超限、裁剪、HTTP 失败和异常结束原因均停止。

已有实验的长时间恢复运行使用独立 tmux 管理：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_formal_runner/launch_tmux.py \
  --set-name hotpot_formal_1000_20261014
```

启动器使用专用 tmux socket `kbqa_<set-name>` 和 `runner` 会话，运行已冻结的
`run_experiment.py --resume`；退出码和收到的信号写入
`analysis/<set-name>_run/process_lifecycle.jsonl`，完整外层日志为 `runner_tmux_*.log`。
它默认阻止重发已发送但未保存响应的请求。若用户明确要求继续这类中断，保留
原始请求日志归档及 `interrupted_request_attempts.jsonl` 中的 unknown 状态记录，
再将这一未完成题纳入真实重试；失联请求不伪造 usage，不计作有效完成记录。
2026-10-14 的一次授权恢复仅重试 k=4 第 827 题，原有 3826 条完成结果保持不变，
验证见 `analysis/hotpot_formal_1000_20261014_run/tmux_resume_validation.json`。

每 k 保留原预测 JSON 格式，新增 `token_usage.jsonl`、`checkpoints/`、
`api_responses/`、`request_events.jsonl`。不保存 15000 份完整 Few-shot Prompt。
上下文统计使用保存的实际上下文及原 `count_context_tokens()`；不重新请求 BM25。
`context_tokens` 与 API `prompt_tokens` 各自独立。

进度、Smoke 报告、运行日志、配置快照及中文报告位于
`analysis/<set-name>_run/`；全部 15 个 k 验证后，原有十项分析产物位于
`analysis/<set-name>/`。历史预测、Prompt、原始配置和历史报告不覆盖。
