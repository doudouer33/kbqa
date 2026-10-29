# HotpotQA 1000 Soft Oracle（独立离线分析）

仅读取 2026-10-14 正式实验的已生成统计；不导入 RAG、检索器、LLM 或训练模块。
所有新增代码和结果位于本目录，旧实验和主流程保持原样。默认参数只是示例，不代表最优配置。

## 运行

从仓库根目录运行：

```sh
python3 -m unittest discover -s analysis/soft_oracle_1000/tests -p 'test_*.py' -v
/home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/soft_oracle_1000/analyze_soft_oracle.py --config analysis/soft_oracle_1000/config.json
```

完整流程需要 Python >=3.8 和 matplotlib；已使用现有 `adaptiverag` 环境的 matplotlib 3.7.5，无需安装依赖。
核心构造、数据校验和单元测试只依赖 Python 标准库。无网络调用，可在无网络环境运行。
路径相对于仓库定位，与调用时工作目录无关。

指定的输出目录必须不存在，即使是空目录也拒绝覆盖。省略 `--output` 会自动分配新的 `experiment_NNN`。
失败的运行目录也不会被覆盖，可另用新编号，保留其失败现场。

仅生成一个配置的逐题标签（无需 matplotlib）：

```sh
python3 analysis/soft_oracle_1000/build_soft_oracle.py --config analysis/soft_oracle_1000/config.json --output analysis/soft_oracle_1000/outputs/labels_reference_001
```

其他配置请另存一个 JSON 配置，再在独立目录生成，不能混写标签。例如生成 D、ε=.05、τ=.75、λ=.2、T=.05：

```sh
python3 - <<'PY'
import json
from pathlib import Path
base = Path('analysis/soft_oracle_1000')
config = json.loads((base / 'config.json').read_text())
config['reference'].update(epsilon=.05, tau=.75, **{'lambda': .2, 'temperature': .05})
path = base / 'config_D_e005_t075_l020_T005.json'
with path.open('x', encoding='utf-8') as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
PY
python3 analysis/soft_oracle_1000/build_soft_oracle.py --config analysis/soft_oracle_1000/config_D_e005_t075_l020_T005.json --output analysis/soft_oracle_1000/outputs/labels_D_e005_t075_l020_T005
```

完整分析要求参考方案为D、参考参数在 sweep 内，输出所有数值组合（108组）×四方案×三类型×自身/共同有效集（2592行）。
无约束的方案仍记录不启用的参数，重复结果是消融设计的预期。

## 输入和校验

- `analysis/hotpot_formal_1000_20261014/per_question_curves.jsonl`
- `analysis/hotpot_formal_1000_20261014/evidence_diagnostic/per_k_evidence.csv`
- `analysis/hotpot_formal_1000_20261014/evidence_diagnostic/per_question.csv`
- `processed_data/hotpotqa/dev_1000_subsampled.jsonl`

实际字段以仓库数据为准：`scores[str(k)].f1/em`、`context_tokens[str(k)]`、`api_token_usage[str(k)]`、`best_k`、`tied_best_ks`。
问题文本取 dataset 的 `question_text`；Gold 答案取已有 per_question 诊断。
校验四个来源的 qid 集、完整唯一 (qid,k)、500+500 类型、官方 F1/EM、原 Oracle/并列、Token qid/k 与总量、Evidence 计数/上下界/状态。
缺失输入会直接报错，不启动补全实验。JSON 重复键、CSV 重复键、数值 NaN/Inf、异常 Token 都拒绝。

证据定义复用原诊断：过滤后实际 contexts 的 350 词渲染文本，旧诊断已核对 context hash。
本次不读取大量 checkpoint，不重做证据匹配；可追溯定义见原 `analyze_evidence.py` 和 `summary.md`。
默认 `sentence_recall_lower=hit/total`。保留 upper、unknown 数量/状态及未知事实；unknown 不能解释为确定未命中。

## 标签和分母

- A：无约束；B：仅 F1 约束；C：仅 Evidence 约束；D：双重约束。
- Smax 始终从全部15点取最大值；F1≥Smax−ε，E≥τ，边界比较允许1e-12绝对舍入误差。
- C(k)=Tokens(k)/max_j Tokens(j)，分母从全部15点取最大值；U=F1−λC。
- 仅在候选中减最大值后 Softmax；非候选概率精确为0。Utility并列时argmax取最小k。
- `valid`：生成15维概率。单候选 One-hot。
- `empty_candidates`：非零F1但候选为空，`soft_label=null`、argmax=null。
- `all_zero_f1`：所有F1为0，优先于空集状态；保留候选mask但`soft_label=null`。四方案一致。
- `valid_candidate_count` 是原约束通过数量；全零题可能有候选，但仍无可训练标签，必须同时判断状态。
- 标签训练只可消费 `label_status == 'valid'` 的行，不将 null 转成全零向量。
- 全组有效率/空集率/全零率分母1000或500；三状态互斥。`constraint_empty_all_n` 是包括全零的原始空集，可与全零重叠。
- 候选数提供 all/nonzero/valid 三个直方图；`mean_candidate_count_valid` 使用有效题作分母。
- 排除候选按(qid,k)计数，提供仅F1失败/仅Evidence失败/两者失败/均通过的互斥分解，另有重叠的边际计数。
- `own_valid` 各方案自己的有效题；`common_valid` 同ε/τ的四方案交集（等于D有效题）。各集合中的固定k/Hard基线使用完全相同的qid。
- H 使用自然对数，另输出 H/ln15 与 H/ln候选数。单候选后者无定义，保存null，均值排除，计数单独保留。
- Soft概率期望仅是对已有逐k结果加权；不代表一次部署回答的得分。

可配置成本为 context/prompt/completion/total_tokens；默认 context_tokens。
成本分母为0时报错。正式context要求正整数，API Token非负且total=prompt+completion。
Evidence指标可选sentence/paragraph的lower/upper；替代指标需独立配置/输出，不能把upper当成已确认覆盖。

## 输出

| 文件 | 内容 |
| --- | --- |
| labels.jsonl | 参考配置全部1000题，包括曲线、原始/API成本、候选mask、概率/null、状态、unknown、参数 |
| per_k_details.csv | 15000行逐k的F1/EM、证据、成本、约束、Utility、概率；无标签概率字段为空 |
| feasibility_summary.csv | 12组ε/τ×4方案×3类型=144行，含完整候选数/排除统计 |
| parameter_sweep.csv | 108组参数×4方案×3类型×2评估集=2592行；嵌套列用JSON编码 |
| distribution_summary.json | 全参数熵/最大概率分位数与直方图、按候选数分层熵、Hard→Soft转移、全部固定k基线 |
| examples.jsonl | 六类典型案例及unknown；按类型、qid固定规则选取，包含空类别记录 |
| report.md | 中文完整分析报告，说明覆盖率、类型偏差与离线限制 |
| metadata.json | 输入/代码/配置SHA256、命令、版本、成本/证据定义、产物SHA256 |
| validation.json | 输入验证与真实落盘文件重新读取后的概率、数量、分母、一致性验证 |
| figures/*.png | 有效率、候选数、标签分布、熵、质量成本五张图 |

PNG图内用英文避免本机缺少中文字体，报告图注用中文。
`metadata.json` 不将自身列入校验和；运行时元数据含时间，数值结果可复现。
本目录 `.gitattributes` 保留标准 CSV 的 CRLF 字节并声明相应行尾检查规则，
避免 Git 换行转换改变输出 SHA256；其他文本仍按通常的空白规则检查。

## 交付核验

本次最终完整输出为 `outputs/experiment_002/`。`experiment_001/` 保留首次通过的扫描结果；第二次运行补充报告解释、落盘统计核验并统一代码格式，未覆盖第一次结果。
最终执行命令（该目录已存在，再执行会明确拒绝；重跑请省略输出参数或指定新编号）：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/soft_oracle_1000/analyze_soft_oracle.py --config analysis/soft_oracle_1000/config.json --output analysis/soft_oracle_1000/outputs/experiment_002
```

第一次完成后额外保存 `tests.log`、`delivery_audit.json`（原始tracked文件/旧正式数据/已有未跟踪文档的SHA256核验和Git状态），供本次交付追溯。
这些交付审计不属于核心运行必需文件。后续重新运行不会覆盖它们。
旧实验目录、旧分析报告和训练/推理代码均不应发生 diff。
