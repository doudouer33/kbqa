# Unicode / JSON 答案边界修复与离线回归

只从已有 `diagnostics.jsonl` 读取模型原始文本，不调用 Qwen、vLLM、检索器或 tokenizer。修复前的 `AnswerExtractor` 从本目录的 `ircot_before.py.txt` 精确加载；修复后使用实际公共模块。两条路径都执行实际的 LLM QA、participant、答案提取及最终预测转换。

针对性测试：

```sh
OPENAI_API_KEY=EMPTY /home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  -m unittest discover -s tests -p test_answer_extractor_json.py -v
```

从仓库根目录重放，输出目录必须是尚不存在的新目录：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_unicode_json_diagnostic/replay.py \
  --source predictions/hotpot_prompt_token_40_20261013 \
  --output predictions/hotpot_unicode_json_replay_120_20261014_new
```

脚本固定 `PYTHONHASHSEED=0`，使原评测器按 set 遍历问题时的累加顺序可复现；不改变评分或舍入公式。父进程中的 HTTP 请求被禁止，生成器由已保存文本替代。Gold 只进入评分和一致性检查，不进入 `replay_raw()`。

默认源文件和原始代码受 SHA-256 守卫保护。若保护的文件后来改变，脚本会退出；此时应先核对变更，不能跳过守卫。运行目录内包含源代码快照、差异、原始输入哈希、逐题处理轨迹、两套评测结果、测试日志和复现核对。

完整结论见 [中文诊断报告](/media/hnu/hnu2021/dengxin/luoyichen/kbqa/analysis/hotpot_unicode_json_diagnostic/report_20261014.md)。
