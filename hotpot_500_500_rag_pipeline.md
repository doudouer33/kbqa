# HotpotQA 500+500 数据构造与 RAG 运行流程

## 1. 目标

本阶段使用 HotpotQA 的两类问题：

- `bridge`
- `comparison`

从 HotpotQA dev 集中分别固定抽取：

```text
bridge:      500
comparison:  500
----------------
total:      1000
```

构造一个固定的 diagnostic dev set，用于后续：

- `k = 1...15` 的 fixed-k RAG 实验；
- 每个问题的 F1 曲线统计；
- retrieved context token 统计；
- Oracle best-k 分布分析；
- `bridge` / `comparison` / `combined` 三组结果比较。

本阶段的重点不是训练 Dynamic Top-K 模型，而是先确认 HotpotQA 上的 best-k 分布是否比原有单跳数据更加合理。

---

# 2. 现有 HotpotQA 数据

当前项目已经有 HotpotQA 的 processed data：

```text
processed_data/hotpotqa/
├── train.jsonl
├── dev.jsonl
├── dev_3000_subsampled.jsonl
└── ...
```

其中：

```text
processed_data/hotpotqa/dev_3000_subsampled.jsonl
```

可以作为本次数据处理的参考。

它原本由类似下面的命令生成：

```bash
python processing_scripts/subsample_dataset_and_remap_paras.py \
    hotpotqa dev_diff_size 3000
```

现有脚本的核心逻辑是：

```python
instances = read_jsonl("processed_data/hotpotqa/dev.jsonl")
instances = random.sample(instances, sample_size)
```

即：

> 从整个 HotpotQA dev 中随机抽取 `sample_size` 条。

现有脚本没有区分：

```text
bridge
comparison
```

因此不能直接满足本阶段的 `500 + 500` 分层采样要求。

---

# 3. HotpotQA processed data 已经保留 type

HotpotQA 的预处理脚本：

```text
processing_scripts/process_hotpotqa.py
```

在生成 `processed_data/hotpotqa/dev.jsonl` 时已经保留：

```python
processed_instance["question_id"] = raw_instance["id"]
processed_instance["question_text"] = raw_instance["question"]
processed_instance["level"] = raw_instance["level"]
processed_instance["type"] = raw_instance["type"]
```

因此现有 processed data 中已经包含：

```json
{
  "question_id": "...",
  "question_text": "...",
  "level": "...",
  "type": "bridge"
}
```

或者：

```json
{
  "question_id": "...",
  "question_text": "...",
  "level": "...",
  "type": "comparison"
}
```

所以不需要重新处理 raw HotpotQA。

本阶段直接从：

```text
processed_data/hotpotqa/dev.jsonl
```

进行分层抽样即可。

---

# 4. 新的数据构造方式

建议新建一个专门的脚本，例如：

```text
processing_scripts/subsample_hotpot_by_type.py
```

不要直接大幅修改原来的：

```text
processing_scripts/subsample_dataset_and_remap_paras.py
```

这样可以避免影响之前的实验流程。

---

## 4.1 分层采样逻辑

核心逻辑：

```python
instances = read_jsonl("processed_data/hotpotqa/dev.jsonl")

bridge = [
    x for x in instances
    if x["type"] == "bridge"
]

comparison = [
    x for x in instances
    if x["type"] == "comparison"
]

random.seed(13370)

bridge = random.sample(bridge, 500)
comparison = random.sample(comparison, 500)

instances = bridge + comparison
random.shuffle(instances)
```

建议继续沿用现有脚本中的随机种子：

```python
seed = 13370
```

一旦样本确定后，后续所有实验必须使用相同的 question IDs。

禁止不同 `k` 重新抽样。

---

## 4.2 输出文件

建议最终文件严格遵守现有项目命名方式：

```text
processed_data/hotpotqa/dev_1000_subsampled.jsonl
```

而不是：

```text
hotpot_dev_1000.jsonl
```

原因是当前 `runner.py` 已经按照：

```text
dev_<sample_size>_subsampled.jsonl
```

自动构造 evaluation path。

例如：

```text
--sample_size 1000
```

对应：

```text
processed_data/hotpotqa/dev_1000_subsampled.jsonl
```

这样可以最大程度复用现有 RAG 运行链路。

---

# 5. 数据构造后的验证

生成：

```text
processed_data/hotpotqa/dev_1000_subsampled.jsonl
```

后，需要至少检查：

```text
total        = 1000
bridge       = 500
comparison   = 500
```

同时检查：

- `question_id` 是否唯一；
- 是否存在缺失的 `type`；
- 是否存在未知 type；
- 每条样本是否仍保留原有 `contexts`；
- `answers_objects` 是否存在；
- 文件是否能够被现有 reader 正常读取。

建议额外保存：

```text
sampled question IDs
random seed
采样脚本
```

保证实验可复现。

---

# 6. Smoke Test

正式跑 1000 条之前，可以先构造一个小规模 smoke set：

```text
bridge:      100
comparison:  100
total:       200
```

主要用于检查：

- 数据读取；
- Retriever；
- LLM；
- prediction 输出；
- evaluation；
- token 统计；
- 不同 `k` 之间 question ID 对齐。

Smoke test 只用于检查 pipeline，不作为正式统计结果。

---

# 7. RAG 运行方式

本阶段虽然使用的是 HotpotQA，但不使用 IRCoT 的 iterative multi-hop retrieval。

仍然采用：

```text
Question
   ↓
Single BM25 Retrieval
   ↓
Top-k Contexts
   ↓
LLM QA
   ↓
Prediction
```

也就是说使用当前项目里的：

```text
oner_qa
```

而不是：

```text
ircot_qa
```

这样可以保证：

> 改变的是问题复杂度，而不是 retrieval framework。

---

# 8. k 的实验范围

本阶段计划运行：

```text
k = 1, 2, 3, ..., 15
```

共：

```text
1000 × 15 = 15000
```

条 QA inference。

所有实验必须保持：

```text
Retriever       相同
LLM             相同
Prompt          相同
Evaluator       相同
Dataset         相同
```

唯一变化：

```text
bm25_retrieval_count = k
```

---

# 9. 使用 runner.py 运行

当前工程已经支持：

```text
--sample_size
--topk_values
```

因此建议通过：

```text
runner.py
```

调用：

```text
run.py
```

例如使用 Qwen：

```bash
python runner.py \
    oner_qa \
    qwen \
    hotpotqa \
    predict \
    --sample_size 1000 \
    --topk_values 1-15 \
    --llm_port_num YOUR_PORT
```

其中：

```text
--sample_size 1000
```

会自动对应：

```text
processed_data/hotpotqa/dev_1000_subsampled.jsonl
```

而：

```text
--topk_values 1-15
```

会依次运行：

```text
k=1
k=2
...
k=15
```

---

# 10. runner.py / run.py 的 evaluation path

对于：

```text
dataset = hotpotqa
sample_size = 1000
```

当前运行链路会自动使用：

```text
processed_data/hotpotqa/dev_1000_subsampled.jsonl
```

等价于：

```text
set_name = dev_1000
```

所以本阶段不需要为了新的 1000 条数据创建一个新的 dataset name。

仍然使用：

```text
dataset = hotpotqa
```

即可。

---

# 11. Prediction 输出位置

当前 `predict.py` / `run.py` 会根据：

```text
args.set_name
```

构造 prediction directory。

对于：

```text
set_name = dev_1000
```

结果会统一放到：

```text
predictions/dev_1000/
```

例如：

```text
predictions/
└── dev_1000/
    ├── oner_qa_qwen_hotpotqa____...bm25_retrieval_count__1...
    ├── oner_qa_qwen_hotpotqa____...bm25_retrieval_count__2...
    ├── oner_qa_qwen_hotpotqa____...bm25_retrieval_count__3...
    ├── ...
    └── oner_qa_qwen_hotpotqa____...bm25_retrieval_count__15...
```

因此 prediction 目录组织方式可以完全沿用现有项目。

---

# 12. 是否需要修改 RAG 主代码

当前结论：

> 第一阶段原则上不需要修改 RAG 主流程。

现有代码已经支持：

```text
HotpotQA
+
custom evaluation_path
+
sample_size
+
topk sweep
+
predictions/<set_name>/
```

因此现在主要需要增加的是：

```text
HotpotQA 500 bridge + 500 comparison
分层抽样脚本
```

而不是重构：

```text
Retriever
Reader
run.py
predict.py
Adaptive-RAG 主流程
```

如果实际 smoke test 中发现某些路径被硬编码，再做局部修改即可。

---

# 13. 推荐的完整流程

```text
processed_data/hotpotqa/dev.jsonl
            │
            │
            │ 按 type 分层采样
            │
            ├── bridge      500
            └── comparison  500
            │
            ↓
processed_data/hotpotqa/
dev_1000_subsampled.jsonl
            │
            │
            │ runner.py
            │ oner_qa
            │ BM25
            │
            ↓
       k = 1...15
            │
            ↓
predictions/dev_1000/
├── k=1
├── k=2
├── k=3
├── ...
└── k=15
            │
            ↓
evaluate_per_question
            │
            ↓
每个 question 获得：
├── F1(k)
├── context_tokens(k)
└── prediction(k)
            │
            ↓
Oracle / Utility
            │
      ┌─────┼─────────┐
      ↓     ↓         ↓
   bridge comparison combined
      │     │         │
      └─────┴─────────┘
            ↓
best-k distribution
majority ratio
entropy
mean best-k
fixed-k F1 curve
token cost curve
```

---

# 14. 后处理分析

RAG 跑完后，需要分别分析：

```text
bridge_500
comparison_500
combined_1000
```

虽然 RAG 是一次性对 1000 条混合数据运行，但由于每条 processed sample 已经保留：

```text
type
```

所以后处理阶段可以直接：

```python
if sample["type"] == "bridge":
    ...

if sample["type"] == "comparison":
    ...
```

因此没有必要把 bridge 和 comparison 分成两个独立的 RAG 数据文件分别跑。

---

# 15. 第一阶段需要得到的结果

至少输出：

## Fixed-k F1

```text
k = 1...15
```

下：

```text
bridge average F1
comparison average F1
combined average F1
```

---

## Token Cost

不同 `k` 下：

```text
average context tokens
median context tokens
P90
P95
```

---

## Oracle Best-k

分别得到：

```text
bridge best-k distribution
comparison best-k distribution
combined best-k distribution
```

---

## 分布统计

至少包括：

```text
majority ratio
entropy
mean best-k
median best-k
std best-k
```

---

# 16. 当前最重要的问题

这一阶段主要回答：

> HotpotQA 是否能缓解原单跳数据中 best-k 严重集中于小 k 的问题？

尤其需要观察：

```text
bridge
comparison
combined
```

三种情况下：

1. `k=1` 是否仍然占绝对多数；
2. best-k 分布是否明显展开；
3. bridge 是否比 comparison 更需要较大的 k；
4. token penalty 加入后分布是否迅速塌缩；
5. 是否值得继续构造更大的训练集。

---

# 17. 是否立即决定训练集大小

暂时不要。

当前阶段先固定：

```text
Diagnostic Dev:
Bridge      500
Comparison  500
Total      1000
```

跑完：

```text
k = 1...15
```

并分析 Oracle distribution。

之后再决定正式训练数据规模，例如：

```text
2000
5000
10000
```

如果 500+500 上 best-k 仍严重不平衡，则不应该直接扩大训练集。

应该优先重新检查：

```text
数据选择
utility function
lambda
label definition
k range
```

---

# 18. 当前建议的第一项开发任务

新增：

```text
processing_scripts/subsample_hotpot_by_type.py
```

负责：

```text
输入：
processed_data/hotpotqa/dev.jsonl

处理：
bridge 随机抽 500
comparison 随机抽 500
固定 seed
合并并 shuffle

输出：
processed_data/hotpotqa/dev_1000_subsampled.jsonl
```

然后验证：

```bash
python runner.py \
    oner_qa \
    qwen \
    hotpotqa \
    predict \
    --sample_size 1000 \
    --topk_values 1-15 \
    --llm_port_num YOUR_PORT
```

能够直接读取新数据，并将结果写入：

```text
predictions/dev_1000/
```

---

# 19. 本阶段最终交付物

建议包括：

```text
processing_scripts/
└── subsample_hotpot_by_type.py

processed_data/hotpotqa/
└── dev_1000_subsampled.jsonl

predictions/
└── dev_1000/
    ├── k=1
    ├── k=2
    ├── ...
    └── k=15

analysis/
├── fixed_k_results.csv
├── oracle_distribution.csv
└── hotpot_basic_statistics.json

figures/
├── fixed_k_f1.png
├── token_curve.png
├── bridge_oracle_distribution.png
├── comparison_oracle_distribution.png
└── combined_oracle_distribution.png

summary/
└── hotpot_500_500_analysis.md
```

当前优先完成：

```text
500 bridge
+
500 comparison
→ dev_1000_subsampled.jsonl
→ runner.py
→ predictions/dev_1000
```

之后再进入 Oracle 与 Dynamic Top-K 训练阶段。

---

# 20. 2026-10-10 实际运行记录

本机已确认以下服务可用：

```text
Retriever:     http://127.0.0.1:8000
Elasticsearch: http://127.0.0.1:9200
Qwen/vLLM:     http://127.0.0.1:8010/v1
Model ID:      Qwen2.5-7B-Instruct
```

注意：vLLM 的 model ID 区分大小写，配置中必须使用：

```text
Qwen2.5-7B-Instruct
```

本项目的 `predict` 不会自动创建 `instantiated_configs`，因此完整命令必须分两步：

```bash
python runner.py oner_qa qwen hotpotqa write \
    --sample_size 1000 \
    --topk_values 1-15 \
    --llm_port_num 8010

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python runner.py oner_qa qwen hotpotqa predict \
    --sample_size 1000 \
    --topk_values 1-15 \
    --llm_port_num 8010
```

Qwen 服务的最大上下文是 4096 token。项目使用 OpenAI chat API，服务端还会添加 chat template token，
所以配置中的 prompt+generation 总预算使用 4000，而不是恰好 4096；generation 上限为 300。

正式运行前已经完成两类端到端 smoke test：

```text
2 questions, k=1:  retrieval + generation + evaluation passed
2 questions, k=15: retrieval + generation + evaluation passed
```

正式扫描使用最多 8 个客户端分片向支持 4 个并发 sequence 的 vLLM 提交请求。各 k 写入独立目录，
不会互相覆盖；实验数据、Retriever、模型、prompt 和 evaluator 均相同，唯一变化仍是
`bm25_retrieval_count`。

完成后执行：

```bash
python analysis/analyze_hotpot_500_500.py
```

输出目录：

```text
analysis/hotpot_500_500/
├── fixed_k_results.csv
├── oracle_distribution.csv
├── hotpot_basic_statistics.json
├── per_question_curves.jsonl
├── fixed_k_f1.png
├── token_curve.png
├── bridge_oracle_distribution.png
├── comparison_oracle_distribution.png
├── combined_oracle_distribution.png
└── hotpot_500_500_analysis.md
```
