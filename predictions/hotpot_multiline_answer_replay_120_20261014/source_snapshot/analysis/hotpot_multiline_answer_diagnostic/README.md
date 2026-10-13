# Qwen HotpotQA 多行 CoT 答案提取验证

候选配置：[oner_qa_qwen_hotpotqa_multiline_candidate.jsonnet](/media/hnu/hnu2021/dengxin/luoyichen/kbqa/base_configs/oner_qa_qwen_hotpotqa_multiline_candidate.jsonnet)。只更换提取正则，引用现有强化 Prompt；历史配置和公共提取器均保持不变。

从仓库根目录运行新增测试：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  -m unittest discover -s tests -p test_hotpot_multiline_answer.py -v
```

从相同的 120 条保存输出离线重放，输出目录必须尚不存在：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_multiline_answer_diagnostic/replay.py \
  --source predictions/hotpot_prompt_token_40_20261013 \
  --baseline predictions/hotpot_unicode_json_replay_120_20261014 \
  --output predictions/hotpot_multiline_answer_replay_120_20261014_new
```

脚本复用上一轮的真实答案处理与评分工具，两边都使用当前公共提取器及 Unicode 修复，只改变配置中的正则。生成器仅返回保存文本，HTTP 调用被禁止。它会检查旧路径逐题复现 Unicode 修复后的结果，并检查候选配置的 k=1～15 求值结果只有正则不同。原评分器的 `set` 累加顺序通过 `PYTHONHASHSEED=0` 固定，不改变评分方法。

脚本输出完整逐题轨迹、变化记录、两套评测输入和结果、k=1～15 的配置求值快照、源代码快照及保护文件哈希。没有运行 `run.py write/predict`、模型请求或正式实验。

结果见 [中文报告](/media/hnu/hnu2021/dengxin/luoyichen/kbqa/analysis/hotpot_multiline_answer_diagnostic/report_20261014.md)。
