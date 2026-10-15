# HotpotQA 500+500 正式实验报告

- 更新时间：2026-10-15T01:20:11.624555+08:00
- 状态：complete
- Git 基线：`13f9a52dd283c2346fa113419cf703c8706f53aa`；修改未 commit/push。
- Python：`/home/dengxin/miniconda3/envs/adaptiverag/bin/python`。
- 依赖版本：`{"openai": "2.2.0", "httpx": "0.28.1", "transformers": "4.26.0.dev0", "torch": "1.13.1+cu117", "diskcache": "5.6.3", "jsonnet": "0.22.0", "matplotlib": "3.7.5"}`。
- 模型：Qwen2.5-7B-Instruct；vLLM http://127.0.0.1:8002/v1；BM25。
- 正式配置：`/media/hnu/hnu2021/dengxin/luoyichen/kbqa/base_configs/oner_qa_qwen_hotpotqa_formal.jsonnet`；与基线 candidate 在 k=1～15 语义一致。
- 数据：`/media/hnu/hnu2021/dengxin/luoyichen/kbqa/processed_data/hotpotqa/dev_1000_subsampled.jsonl`；500 bridge + 500 comparison；1000 唯一 qid；manifest 顺序及原始官方答案均匹配。
- 数据 SHA-256：`b9798320af36fbeedf1e81ed7690673016ba77b3cdc085d268c36b50669b04af`。
- Prompt SHA-256：`fa018fc3c35f7ca527acee717e070864b7f99a73bf8319502cedc909ce1597d9`；固定 15 Few-shot，与评测无重叠。
- 配置：model_tokens_limit=16384；max_tokens=estimated_generation_length=512；stop=[]；temperature=0；top_p=1。
- 真实 API usage、结束原因、耗时、Prompt hash 和原始生成文本逐题落盘；上下文快照与实际 Prompt 的上下文 hash 一致。
- 启动时恢复原子 checkpoint，重建 Token journal；已完成题不再调用模型；已保存响应在 Prompt/config 一致时恢复提取。
- 代理异常：仅为本次实验进程设置 NO_PROXY，并为模型 HTTP 客户端关闭环境代理。未重启/修改服务。
- 子进程：run.py/predict.py 传播失败退出状态，最终文件另做逐题完整性核验。

## Smoke Test

- 结果：通过，16/16 真实调用完成；无裁剪、异常缓存或缺失记录。
- 目录：`/media/hnu/hnu2021/dengxin/luoyichen/kbqa/predictions/hotpot_formal_1000_20261014_smoke`。
- 回归测试：全部 unittest 通过；包含 Unicode/JSON、多行 CoT、Yes/No、官方评分和断点恢复。

| qid | type | question | gold |
|---|---|---|---|
| 5a7f40de5542992e7d278cc8 | bridge | What star if the Burbs was Canadian? | Richard "Rick" Ducommun |
| 5a84a7f95542992a431d1a85 | bridge | Who was the wife of the United States Army lieutenant general who received the Distinguished Service Cross, and was the first of his West Point class to be promoted to brigadier general, major general, and lieutenant general | Julia Compton Moore |
| 5a7248665542990c210a40b1 | bridge | What is the colloquial name of the professional football team in which striker Frederico Chaves Guedos played?  | Rooster |
| 5a79b495554299148911fa3b | bridge | Which serial killer was known to start killing first, Wayne Boden or Ted Bundy? | Wayne Clifford Boden |
| 5ae14def55429901ffe4ae4a | comparison | Do both Lee Tamahori and Robert Redford work as a director? | yes |
| 5ae497595542995ad6573db7 | comparison | What are both Stauntonia and Pimelea? | plant |
| 5ab80bbf5542991d32223818 | comparison | Which airport is operated by the Coos Country Airport District, Southwest Oregon Regional Airport or Lawton–Fort Sill Regional Airport? | Southwest Oregon Regional Airport |
| 5abdc0ad55429933744ab81b | comparison | Which band was formed first, Goldfinger or The Fray? | Goldfinger |

## 正式执行

- 完成：15/15 个 k；15000/15000 个有效 (qid,k)。
- 预测目录：`/media/hnu/hnu2021/dengxin/luoyichen/kbqa/predictions/hotpot_formal_1000_20261014`。
- 逐题和 Token：每个 k 的 prediction__*.json、token_usage.jsonl、checkpoints/*.json、api_responses/*.json。
- 当前状态详见 `status.json` 与 `formal_k_*.log`；尚未通过验证的 k 不计为完成。

| k | count | official F1 | EM | API prompt tokens | completion tokens | total tokens | latency s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 0.508705 | 0.411000 | 6512.2 | 46.8 | 6558.9 | 1.137 |
| 2 | 1000 | 0.531057 | 0.443000 | 6596.5 | 47.0 | 6643.4 | 1.137 |
| 3 | 1000 | 0.537971 | 0.444000 | 6677.4 | 47.3 | 6724.7 | 1.177 |
| 4 | 1000 | 0.549323 | 0.450000 | 6755.6 | 47.0 | 6802.6 | 1.228 |
| 5 | 1000 | 0.552761 | 0.450000 | 6832.1 | 47.6 | 6879.8 | 1.215 |
| 6 | 1000 | 0.560942 | 0.460000 | 6906.8 | 46.7 | 6953.5 | 1.261 |
| 7 | 1000 | 0.554640 | 0.459000 | 6980.2 | 47.1 | 7027.3 | 1.258 |
| 8 | 1000 | 0.561491 | 0.461000 | 7055.6 | 47.2 | 7102.8 | 1.259 |
| 9 | 1000 | 0.574987 | 0.477000 | 7132.0 | 47.1 | 7179.1 | 1.276 |
| 10 | 1000 | 0.574353 | 0.478000 | 7206.7 | 47.7 | 7254.4 | 1.275 |
| 11 | 1000 | 0.570683 | 0.471000 | 7283.6 | 47.0 | 7330.7 | 1.278 |
| 12 | 1000 | 0.573988 | 0.475000 | 7359.7 | 46.8 | 7406.5 | 1.280 |
| 13 | 1000 | 0.584750 | 0.480000 | 7437.0 | 46.9 | 7483.9 | 1.228 |
| 14 | 1000 | 0.573159 | 0.473000 | 7510.5 | 47.2 | 7557.7 | 1.215 |
| 15 | 1000 | 0.578012 | 0.479000 | 7585.5 | 47.4 | 7633.0 | 1.230 |

## 分析与研究用途

- 评分全部使用 official_evaluation/hotpotqa/hotpot_evaluate_v1.py；Oracle 最大化官方 F1，并列取最小 k，保留全部并列 k 和完整 F1 曲线。
- 分析复用 analysis/analyze_hotpot_500_500.py；context_tokens 保持原函数计算，API prompt/completion/total tokens 独立保存。
- 历史 DROP F1 与新官方 F1 不能直接比较；历史预测和分析保持不变。
- Dynamic Top-K 标签分析：可以使用；15000 条完整性验收及原有十项分析产物已通过。
- 原有分析产物与 Bridge/Comparison/Combined Fixed-k、Oracle 并列和差距统计：`/media/hnu/hnu2021/dengxin/luoyichen/kbqa/analysis/hotpot_formal_1000_20261014`。

## Smoke 工程修复记录

- 2026-10-14T06:20:02.514973+08:00：修正检查以遵循原 CopyQuestionParticipant.strip()；增加完整预测已落盘后的安全恢复及流式文件哈希。变化文件：analysis/hotpot_formal_runner/hooks.py, analysis/hotpot_formal_runner/run_experiment.py, analysis/hotpot_formal_runner/storage.py。配置、Prompt、数据、评分和模型生成参数保持一致；失败记录保留，重跑 Smoke gate。
- 2026-10-14T17:51:35.241386+08:00：模型 Token 预算检查连接断开导致 k=4 第 138 题在生成前停止。现服务恢复，同模型 ID/上下文上限；独立恢复 Smoke Test 16 次真实调用通过，Prompt、原始生成、预测及 prompt_tokens 全部与首轮一致。源码、数据和配置哈希不变；保留 3137 条记录，从断点续跑，未重启或修改服务。
- 2026-10-14T19:13:58.510710+08:00：查到第二次中断的 Codex 执行记录：18:15:25 命令会话 14122 结束，status=failed、exit_code=-1，无程序异常输出；同秒 app-server 报当前 chat 会话不存在。所属 cgroup 的 oom_kill=0。证据最符合临时 Codex 执行会话失效导致终止，具体信号/发起方未记录。原启动方式未与临时会话隔离；新增独立 tmux 启动器和退出/信号日志，12 秒脱离工具调用的存活探针通过，冻结推理源码哈希保持一致。第 827 题缺失 API 响应，启动器保持阻止重发。证据：execution_exit_evidence.json。
- 2026-10-14T19:17:46.779650+08:00：用户要求在原实验基础上继续。k=4 第 827 题未持久保存响应，经用户本次续跑指令，对这一个缺失结果发起真实重试；失联请求完成状态仍标记 unknown、usage 为 null，仅保留在 interrupted_request_attempts.jsonl 和原始请求日志归档中，不计作有效结果、不伪造 Token。已完成 3826 条结果及快照哈希保存，冻结源码/配置/数据哈希不变。恢复采用独立 tmux 管理与退出日志。
