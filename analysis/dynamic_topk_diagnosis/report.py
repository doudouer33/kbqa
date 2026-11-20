"""Render the Chinese diagnosis from the computed statistics and saved case data."""
from analyze import BERT, FALLBACK, FIXED, KS, ORACLES, SOFT

Q = ORACLES[-1]
NAMES = {BERT: "BERT Argmax", FALLBACK: "Scheme D Argmax + k=1 回退", SOFT: "原 Scheme D Argmax"}


def display(method):
    return NAMES.get(method, method)


def pp(value):
    return f"{100 * value:+.2f}"


def share(count, n):
    return f"{count}（{count / n:.1%}）"


def markdown_table(headers, records):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(v) for v in record) + " |" for record in records),
    ])


def method_table(cohort, methods, validation_k=None):
    reference = cohort[FIXED[0]]
    valid_reference = cohort[f"Fixed k={validation_k}"] if validation_k else None
    head = ["方法", "F1", "Context Token", "平均 k", "对 k=1 ΔF1（百分点）", "对 k=1 ΔToken"]
    if valid_reference:
        head += [f"对 k={validation_k} ΔF1（百分点）", f"对 k={validation_k} ΔToken"]
    records = []
    for method in methods:
        result = cohort[method]
        f, t = result["f1"], result["context_tokens"]
        row = [display(method), f"{f:.5f}", f"{t:.2f}", f"{result['mean_k']:.3f}"]
        for base in (reference, valid_reference):
            if base:
                row += [pp(f - base["f1"]),
                        f"{t - base['context_tokens']:+.2f}（{t / base['context_tokens'] - 1:+.1%}）"]
        records.append(row)
    return markdown_table(head, records)


def inline(text):
    return " ".join(text.split()).replace("|", "&#124;").replace("<", "&lt;").replace(">", "&gt;")


def case_sections(cases):
    sections = []
    for number, c in enumerate(cases, 1):
        b = c["bert_k"]
        q = c["oracles"][Q]
        s = c["soft_argmax_k"]
        f, t, e = c["f1_curve"], c["context_tokens"], c["sentence_recall"]
        category = c["category"]
        certain = [1, b, q, s]
        if category.startswith("曲线波动"):
            # Show the first answer/F1 flip for the oscillating example.
            flip = next(k for k in range(2, len(KS) + 1) if f[k - 1] != f[k - 2])
            certain += [flip - 1, flip]
        key_k = sorted(set(k for k in certain if k is not None))
        ordered_oracles = [c["oracles"][name] for name in ORACLES]
        description = (
            f"ID `{c['question_id']}` · {c['type']} · 金标准 **{inline(c['gold_answer'])}**。"
            f"BERT 选 **{b}**（置信度 {c['bert_confidence']:.3f}）。"
            f"Scheme D 选 **{s if s is not None else '无标签'}**（`{c['label_status']}`）。"
            f"质量 Oracle 在 ε=0、0.05、0.10 下依次选 **{' / '.join(map(str, ordered_oracles))}**。"
        )
        contexts = c["contexts"]
        context_rows = []
        for k in key_k:
            raw = contexts[k - 1]
            titles = "、".join(inline(title) for title in raw["titles"]) if raw else "原始上下文缺失"
            context_rows.append((k, titles))
        e1, eq = e[0], e[q - 1]
        if category.startswith("k=1 严重不足"):
            note = f"从 k=1 到 k={q}，保存的句证据召回由 {e1:.2f} 变为 {eq:.2f}，F1 由 {f[0]:.3f} 变为 {f[q-1]:.3f}。"
            note += "句证据增加支持检索改善这一解释。" if eq > e1 else "保存的句证据召回没有增加，具体改善原因待验证。"
            if c["predictions"][0].lower() == "animation" and c["predictions"][q-1].lower() == "animated":
                note += "此外，评分区分 animation 与 animated，F1 的跳升也反映了答案词形变化。"
        elif category.startswith("k=1 已答对"):
            low, high = contexts[0], contexts[b - 1]
            keep = low and high and high["reader_context_sha256"][:len(low["reader_context_sha256"])] == low["reader_context_sha256"]
            note = (f"BERT 多用了 {t[b-1] - t[0]} 个 Context Token，F1 由 {f[0]:.3f} 变为 {f[b-1]:.3f}。")
            if f[b-1] < f[0] and keep:
                note += "原 k=1 阅读文本仍是新上下文的前缀。新增内容干扰阅读器是一个待验证的假设。"
            elif c["predictions"][0] == c["predictions"][b-1]:
                note += "两档 k 的保存答案完全相同。"
        elif category.startswith("软标签"):
            note = (f"质量 Oracle 与软标签的 F1 分别为 {f[q-1]:.3f}、{f[s-1]:.3f}，"
                    f"句证据召回分别为 {eq:.2f}、{e[s-1]:.2f}，Scheme D 多用 {t[s-1] - t[q-1]} 个 Context Token。")
            note += "质量 Oracle 的 k 被 0.50 证据门槛排除。" if eq < 0.5 else "该题差异来自效用排序。"
        elif category.startswith("曲线波动"):
            note = (f"相邻 k 之间 F1 变动 {sum(x != y for x, y in zip(f, f[1:]))} 次，"
                    f"句证据召回范围为 {min(e):.2f}–{max(e):.2f}。"
                    "变动反映了保存输出对上下文变化的敏感性，生成稳定性需要同一 k 的重复观测才能判断。")
        else:
            note = (f"质量最优 F1 为 {max(f):.3f}，在满足 ε=0.10 的 k 中，"
                    f"最高句证据召回为 {max(ev for score, ev in zip(f, e) if score + 1e-12 >= max(f) - 0.1):.2f}。"
                    "证据门槛使所有近最优 k 失去标签资格。")
        labels = c["soft_label"]
        summary = markdown_table(["k", "F1", "Context Token", "句证据召回", "软标签 P(k)", "保存的答案"], [
            (k, f"{f[i]:.3f}", t[i], f"{e[i]:.3f}", f"{labels[i]:.4f}" if labels is not None else "—",
             inline(c["predictions"][i]) or "（空）") for i, k in enumerate(KS)
        ])
        sections.append("\n".join([
            f"### {number}. {c['category']}", "", f"**Question** · {inline(c['question'])}", "", description,
            "", note, "", markdown_table(["对照 k", "实际进入 Reader 的检索标题（原顺序）"], context_rows),
            "", "<details>", "<summary>展开 k=1–15 的 F1、Token、软标签和保存答案</summary>", "",
            summary, "", "</details>",
        ]))
    return "\n\n".join(sections)


def write_report(out, result, cases):
    audit = result["audit"]
    metrics = result["metrics"]
    m = metrics["all_test"]
    valid = metrics["valid_test"]
    n, nv = m[BERT]["n"], valid[BERT]["n"]
    oq, bert, soft = m[Q], m[BERT], valid[SOFT]
    vq = valid[Q]
    d = result["demands"]["counts"]
    demand = result["demands"]
    sc = result["label_comparison"]
    effect = result["bert"]["effects"]
    eq = effect[Q]
    train = result["training"]
    usage = audit["api_usage"]
    qc = audit["saved_eval_diagnostics"]["query_specificity"]
    val_fixed = audit["saved_eval_diagnostics"]["validation_selected_fixed_k"]["k"]
    bp = result["bert"]["probability"]
    mix = result["bert"]["equal_budget_fixed_1_2_mix"]
    sw = result["bert"]["vs_fixed1_switch"]
    prior = audit["saved_bert_vs_mean_distribution"]
    ce_interval = audit["saved_eval_diagnostics"]["bert_minus_train_mean"]["soft_ce"]
    condition = f"{nv} 道有效标签题"
    bq_under = eq["below_and_violate"]
    bq_over = eq["above_and_violate"]
    surplus = eq["feasible_extra_cost_no_f1_gain"]
    best_at_one = d["best_at_k1_positive"]
    needs = d["gain_gt_0.1"]
    zeros = d["all_zero"]
    parts = [
        "# HotpotQA Dynamic Top-K RAG 离线诊断",
        (f"测试集共 {n:,} 题。**{share(needs, n)} 在允许 F1 损失 0.10 时仍需要提高 k**，"
         f"**{share(best_at_one, n)} 在 k=1 已取得正的最佳 F1**，另有 **{share(zeros, n)} 在所有 k 下 F1 为零**。"
         f"质量约束 Oracle（ε=0.10）达到 **F1 {oq['f1']:.5f} / {oq['context_tokens']:.2f} Context Token**，"
         f"BERT 为 **{bert['f1']:.5f} / {bert['context_tokens']:.2f}**。"
         "当前最直接的性能差距出现在选 k 策略的逐题分配上，证据门槛和现有 RAG 的全零题构成另外两处可量化的限制。"),
        "## 1. 当前实验配置与数据说明",
        ("诊断采用已完成的 HotpotQA Train5000 + BERT。原实验从官方 Train 的预处理样本中去除 15 个 Few-shot ID，"
         "以种子 42 无放回抽取 5,000 题。单轮 BM25 请求 k=1–15，Reader 使用实际保存的检索段落，"
         "Qwen2.5-7B-Instruct 使用固定 15-shot Prompt、temperature=0。过滤和裁剪后实际段数可能小于请求 k。"
         "原研究笔记记录的是较早的 Dev 分层实验，本文使用本地正式 Train5000 产物。"
         " [RAG 实验说明](../../experiments/hotpotqa_train_5000/README.md)"
         " · [BERT 实验说明](../../experiments/hotpotqa_train_5000_bert/README.md)"),
        ("数据按 `random.Random(42).shuffle` 对全部 ID 洗牌，按 70/10/20 划分，划分与标签无关。"
         "测试是官方 Train 内部的保留集。BERT 输入仅为 Question，使用 `bert-base-uncased` 全参数微调"
         "和 15 类分类头（AdamW、学习率 2e-5、batch 16、最长 128 tokens）。"
         f"训练 5 个 epoch，按验证集 Soft CE 选中第 {train['best_epoch']} 轮。"
         " [配置](../../experiments/hotpotqa_train_5000_bert/config/experiment.json)"
         " · [划分 ID](../../experiments/hotpotqa_train_5000_bert/data/splits.json)"),
        markdown_table(["划分", "总计", "有效标签", "全零 F1", "证据/质量联合候选为空"], [
            (role, sum(r.values()), r.get("valid", 0), r.get("all_zero_f1", 0), r.get("empty_candidates", 0))
            for role, r in audit["split_counts"].items()
        ]),
        ("- 逐题 F1 是保存的生成答案相对于 HotpotQA 金标准的官方 Answer F1，保留 yes/no/noanswer 的特殊规则。"
         " `curves.jsonl` 每题一行，按 k 记录 F1、Context Token、句证据召回和生成答案。"
         " [曲线](../../experiments/hotpotqa_train_5000_bert/data/curves.jsonl)"
         " · [评分实现](../../experiments/hotpotqa_train_5000_bert/src/metrics.py)\n"
         "- Context Token 是 Qwen tokenizer 对实际 `effective_context_text` 的计数。API Prompt、Completion、Total Token"
         " 取自原始 `contexts.jsonl` 中保存的 API Usage，包含对应调用范围的开销。"
         " [原始运行](../../experiments/hotpotqa_train_5000/runs/predictions/hotpotqa_train_5000_seed42/)\n"
         "- 原标签含候选掩码、15 维概率和 Argmax。BERT 文件有 500 个验证集和 1,000 个测试集概率向量及预测 k，"
         " 另有测试决策和既有评测。"
         " [标签](../../experiments/hotpotqa_train_5000_bert/data/labels.jsonl)"
         " · [BERT 概率](../../experiments/hotpotqa_train_5000_bert/data/bert_predictions.jsonl)"
         " · [测试决策](../../experiments/hotpotqa_train_5000_bert/data/test_decisions.jsonl)"
         " · [原评测](../../experiments/hotpotqa_train_5000_bert/reports/evaluation.json)"),
        ("实际标签为 Scheme D。对每题以全部 15 个 k 的最大 Context Token 归一化成本，"
         "即 `c_i(k) = C_i(k) / max_j C_i(j)`。候选需同时满足 "
         "`F1_i(k) ≥ max_j F1_i(j) − 0.10` 及 `句证据召回下界 ≥ 0.50`。"
         "句证据下界使用确认匹配的原标注支持句数除以原标注支持句总数，两处越界索引按 unknown 保留原分母。"
         "在候选内取 `U_i(k) = F1_i(k) − 0.10 × c_i(k)`，再用 `softmax(U_i / 0.10)` 分配概率，候选外概率为零。"
         "效用平手选较小 k，全零 F1 或候选为空时概率与 Argmax 均为空。仅有效标签参与训练。"
         "实现记录 Soft CE，即 `−Σ p log q`；相对于固定标签分布，优化它与优化 `KL(p||q)` 等价，两者相差标签熵。"
         " [标签实现](../../experiments/hotpotqa_train_5000_bert/src/soft_oracle.py)"
         " · [训练实现](../../experiments/hotpotqa_train_5000_bert/src/train.py)"),
        ("以 `question_id` 连接划分、曲线、标签、输入、BERT 概率和决策，原始 RAG 以 `qid` 对应同一值。"
         "核验各划分互斥、每个测试 ID 在每个 k 恰好出现一次、问题文本和预测答案一致，校验冻结输入与原始上下文指纹、"
         "标签公式及历史评测汇总。所有 Context Token 主对比均在注明的统一集合上完成。"
         " [逐题核验](test_audit.jsonl) · [汇总和指纹](summary.json)"),
        "## 2. 固定 k、当前软标签、BERT、质量约束 Oracle 的性能对比",
        ("质量 Oracle 对每题在 `F1_i(k) ≥ F1_i^max − ε` 的候选中最小化实际 Context Token，"
         "等成本时取最小 k，F1 边界误差容许 1e-12。ε=0 给出保存曲线可达到的最高平均 F1。"
         "所有质量 Oracle 和 Scheme D Argmax 都依赖真实答案的事后 F1，仅作离线参照。"
         "原有验证集最高平均 F1 选中了固定 " + f"k={val_fixed}。测试上的 15 个固定点用于描述权衡。"),
        "### 全测试集",
        (f"原 Scheme D 只有 {nv} 道题可决策。为给出全测试集对照，表中的混合规则对其余 {n-nv} 题显式回退到固定 k=1。"
         "这项回退只用于诊断。每个方法相对全部 15 个固定 k 的变化和完整选 k 计数都在 CSV 中。 [方法统计](method_metrics.csv)"),
        method_table(m, [*FIXED, BERT, FALLBACK, *ORACLES], val_fixed),
        "### 有效标签子集",
        f"以下所有方法均在相同的 {condition}上比较。全量固定 k 数据见上面的 CSV。",
        method_table(valid, [FIXED[0], FIXED[1], f"Fixed k={val_fixed}", BERT, SOFT, *ORACLES]),
        "![同一集合上的 F1–Context Token 散点图](f1_context_tradeoff.png)",
        "### API Token 单独统计",
    ]
    if "api_complete_test" in metrics:
        api = metrics["api_complete_test"]
        na = usage["complete_test_questions"]
        parts += [
            (f"测试集 {15*n:,} 个题目–k 组合中，{usage['known_pairs']:,} 个有原始 API Usage，"
             f"缺失 {usage['missing_pairs']:,} 个。以下所有方法只比较 15 个 k 均有 Usage 的同一批 **{na} 题**，"
             "选择规则仍以 Context Token 为成本。保留 API 返回或可绑定到同一调用的历史 Usage，不用 tokenizer 估计补齐。"
             "子集可能随旧缓存覆盖产生选择偏差，表中结果适用于该子集。"),
            markdown_table(["方法", "F1", "Context Token", "API Prompt", "API Completion", "API Total",
                            "API Total 对固定 k=1"],
                           [(display(method), f"{r['f1']:.5f}", f"{r['context_tokens']:.2f}",
                             f"{r['prompt_tokens']:.2f}", f"{r['completion_tokens']:.2f}",
                             f"{r['total_tokens']:.2f}",
                             f"{r['total_tokens'] / api[FIXED[0]]['total_tokens'] - 1:+.2%}")
                            for method, r in api.items()]),
            (f"在统一的 {na} 题上，ε=0.10 Oracle 相对固定 k={val_fixed} 的 Context Token 节省 "
             f"{1 - api[Q]['context_tokens'] / api[f'Fixed k={val_fixed}']['context_tokens']:.1%}，"
             f"API Total Token 节省 {1 - api[Q]['total_tokens'] / api[f'Fixed k={val_fixed}']['total_tokens']:.1%}。"
             "API Total 包含大体稳定的 Few-shot 与其他 Prompt 开销。"),
            "![完整 Usage 子集上的 F1–API Total Token 散点图](f1_api_total_tradeoff.png)",
        ]
    else:
        parts.append("缺少可以覆盖全部候选 k 的 API Usage 交集，无法生成可比的 API Token 统计。")

    parts += [
        "## 3. 不同问题对 k 的需求分布",
        ("令 ΔF1 为逐题最大 F1 减 k=1 的 F1。以 ΔF1 > 0.10 界定有明显收益，以 "
         "`0 < F1_max ≤ 0.20` 标记全程低分的正分题。表内部分诊断指标彼此重叠。"),
        markdown_table(["现象", f"题数（占 {n:,} 题）"], [
            (label, share(d[key], n)) for label, key in [
                ("k=1 达到正的逐题最佳 F1", "best_at_k1_positive"),
                ("k=1 的 F1 就是 1", "perfect_at_k1"),
                ("任何 k 的 F1 均为 0", "all_zero"),
                ("逐题最佳 F1 在 (0, 0.20] 内", "positive_best_le_0.2"),
                ("存在 k>1 带来任意正收益", "strict_gain"),
                ("ΔF1 在 (0, 0.05] 内", "gain_0_to_0.05"),
                ("ΔF1 在 (0.05, 0.10] 内", "gain_0.05_to_0.1"),
                ("ΔF1 > 0.10", "gain_gt_0.1"),
                ("至少一个 k>1 的 F1 低于 k=1", "any_more_k_worse_than_k1"),
                ("k=15 的 F1 低于 k=1", "k15_worse_than_k1"),
                ("至少一次相邻 k 增加使 F1 下降", "any_adjacent_drop"),
                ("相邻 k 同时出现过上升和下降", "adjacent_rise_and_drop"),
            ]
        ]),
        (f"按正分最优定义，固定 k=1 已足够的比例是 **{share(best_at_one, n)}**，"
         f"在 {n - zeros} 道至少有一个 k 得到正分的题中占 **{best_at_one / (n - zeros):.1%}**。"
         f"当 ε=0、0.05、0.10 时，k=1 达到质量门槛且逐题最佳 F1 为正的题数分别为 "
         f"**{' / '.join(str(d[f'{name}: k1 meets quality and peak > 0']) for name in ORACLES)}**，"
         f"必须增加 k 才达门槛的分别为 **{' / '.join(str(d[f'{name}: k1 misses quality']) for name in ORACLES)}**。"
         f"其余 {zeros} 道全零题对这批保存输出没有可观察到的质量收益。"
         f"全测试题平均 ΔF1 是 {demand['gain_mean']:.4f}；超过 0.10 的子组平均提升 {demand['gain_gt_0.1_mean']:.4f}。"),
        markdown_table(["问题类型", "题数", "k=1 正分最优", "ΔF1 > 0.10", "全零"], [
            (tag, row["n"], share(row["best_at_k1_positive"], row["n"]), share(row["gain_gt_0.1"], row["n"]),
             share(row["all_zero"], row["n"])) for tag, row in demand["by_type"].items()
        ]),
        markdown_table(["k", "Oracle ε=0", "Oracle ε=0.05", "Oracle ε=0.10", "BERT（全测试）",
                        "BERT（有效标签）", "Scheme D（有效标签）"], [
            (int(k), *(m[name]["k_counts"][j] for name in ORACLES), m[BERT]["k_counts"][j],
             valid[BERT]["k_counts"][j], soft["k_counts"][j]) for j, k in enumerate(KS)
        ]),
        (f"Context Token 曲线逐题检查发现 {d['any_adjacent_cost_drop']} 题出现过相邻 k 成本回落。"
         "Oracle 始终直接比较 Token，在成本相等时才以较小 k 决胜。"
         f"原始 Reader 文本序列中，{usage['adjacent_context_prefix_changed_questions']} 题至少一次出现相邻 k "
         f"没有保留此前文本作为完整前缀（{usage['adjacent_context_prefix_changed_pairs']} 对）。"
         "F1 变化同时反映了已检索内容和 Reader 输出对上下文的响应。单题每个 k 只有一个保存答案。"),
        "![ΔF1 直方图，首柱单独标出全零题](f1_gain_histogram.png)",
        "![Oracle、BERT 和原软标签的 k 分布](selected_k.png)",
        "![固定 k 的平均 F1 和 Context Token 曲线](fixed_k_curves.png)",
        "## 4. BERT 的选 k 错误分析",
        (f"BERT 只选了 k=1、2、3、7，其中 k=2 占 {bert['k_counts'][1] / n:.1%}，k≤2 占 "
         f"{sum(bert['k_counts'][:2]) / n:.1%}。ε=0.10 的 Oracle 有 "
         f"{result['bert']['coverage_quality_oracle_eps_0.1_ge_4']} 题选 k≥4。"
         "混淆矩阵行是目标 k、列是 BERT k，原 Scheme D 只用有效标签题。"
         " [混淆矩阵数值](confusion.csv)"),
        "![BERT 与各目标的混淆矩阵](bert_confusion.png)",
        markdown_table(["目标", "n", "相同 k", "BERT k 偏小", "BERT k 偏大", "k 不同但 F1 持平",
                        "F1 更低", "F1 更高"],
                        [(display(name), sum(groups[x]["n"] for x in ("below_target", "same_k", "above_target")),
                          *(groups[key]["n"] for key in ("same_k", "below_target", "above_target",
                                                        "different_k_same_f1", "lower_f1", "higher_f1")))
                         for name, groups in effect.items()]),
        (f"以 ε=0.10 的质量 Oracle 为参照，有 **{eq['violates_quality']['n']} 题**未达到质量门槛。"
         f"其中 **{bq_under['n']} 题 k 偏小**，平均少用 {-bq_under['context_delta_mean']:.1f} Context Token，"
         f"平均少 {-bq_under['f1_delta_mean']:.3f} F1，约占净 F1 差距的 "
         f"{bq_under['f1_delta_sum'] / (n * (bert['f1'] - oq['f1'])):.1%}。"
         f"另 **{bq_over['n']} 题 k 偏大仍失分**，平均多用 {bq_over['context_delta_mean']:.1f} Token。"
         f"还有 **{surplus['n']} 题已达质量门槛、额外花费 Token 且 F1 没有提升**，"
         f"平均多花 {surplus['context_delta_mean']:.1f} Token，折合全测试集每题 "
         f"{surplus['context_delta_sum'] / n:.2f} Token。"
         "主要质量损失发生在 k 偏小的题上，额外成本则广泛出现在质量已经达标的题上。"
         f"仅看总均值时，欠检索省下的成本掩盖了这部分额外花费，BERT 平均较 Oracle 少用 "
         f"{oq['context_tokens'] - bert['context_tokens']:.2f} 个 Context Token，同时 F1 低 "
         f"{100 * (oq['f1'] - bert['f1']):.2f} 个百分点。"),
        (f"在 {needs} 道 k=1 达不到质量门槛的题里，BERT 仍选 k=1 的有 "
         f"{result['bert']['k1_selected_when_k1_misses_quality_eps_0.1']} 道。"),
        (f"BERT 相对于固定 k=1 共在 {sum(x['n'] for x in sw.values())} 题上提高 k。"
         f"其中 {sw['improved']['n']} 题 F1 上升，{sw['worse']['n']} 题下降，{sw['tied']['n']} 题持平。"
         f"全体平均相对固定 k=1 的 F1 变化为 {pp(bert['f1'] - m[FIXED[0]]['f1'])} 个百分点，"
         f"相对 k=2 为 {pp(bert['f1'] - m[FIXED[1]]['f1'])} 个百分点。"
         "下表的配对 Bootstrap 按测试题有放回抽样 1,000 次。"),
        markdown_table(["基线", "BERT − 基线 F1（百分点）及 95% 区间", "Context Token 差及 95% 区间"], [
            (name, f"{pp(r['f1']['mean_delta'])} [{pp(r['f1']['ci95_low'])}, {pp(r['f1']['ci95_high'])}]",
             f"{r['context']['mean_delta']:+.2f} [{r['context']['ci95_low']:+.2f}, {r['context']['ci95_high']:+.2f}]")
            for name, r in result["bert"]["paired_vs_fixed"].items()
        ]),
        (f"在等平均 Context Token 的离线对照中，以 {mix['k2_probability']:.1%} 的概率独立选 k=2，"
         f"其余选 k=1，期望 F1 为 {mix['expected_f1']:.5f}。BERT 高 "
         f"{pp(mix['bert_minus_mix_f1']['mean_delta']).lstrip('+')} 个百分点，"
         f"配对 95% 区间 [{pp(mix['bert_minus_mix_f1']['ci95_low'])}, "
         f"{pp(mix['bert_minus_mix_f1']['ci95_high'])}] 包含零。"
         "混合权重来自在测试曲线上对齐 BERT 的平均预算，用于事后比较。"),
        "### 预测概率和训练日志",
        (f"BERT 平均最大类概率为 {bp['mean_confidence']:.3f}，平均熵为 {bp['mean_entropy']:.4f} nats，"
         f"15 类均匀分布的熵为 {bp['max_possible_entropy']:.4f}。"
         f"有效软标签的平均候选数为 {train['test_valid_mean_candidate_count']:.2f}，"
         f"平均最大概率为 {train['test_valid_mean_max_label_probability']:.3f}，熵为 "
         f"{train['test_valid_label_entropy']:.4f} nats。确定性选 k 集中与模型概率接近均匀同时存在。"
         f"若从保存的 BERT 概率采样，期望 k={prior['bert']['mean_expected_k']:.2f}，"
         f"期望 F1={prior['bert']['expected_answer_f1']:.5f}，期望 Context Token="
         f"{prior['bert']['expected_context_tokens']:.2f}。这些是同一组曲线的概率加权值。"),
        markdown_table(["置信度四分位（低到高）", "n", "最大概率范围", "平均熵", "与质量 Oracle k 一致",
                        "低于质量门槛", "到逐题最佳平均 F1 差"], [
            (i, r["n"], f"{r['confidence_min']:.3f}–{r['confidence_max']:.3f}", f"{r['entropy_mean']:.4f}",
             f"{r['k_match_rate']:.1%}", f"{r['quality_violation_rate']:.1%}", f"{r['f1_regret_mean']:.3f}")
            for i, r in enumerate(bp["confidence_quartiles_oracle"], 1)
        ]),
        "高置信度组与 Oracle 的 k 更常一致，质量违反率和平均 F1 缺口随置信度的变化没有单调趋势。",
        (f"保留 BERT 的 k 频数并打乱题目对应关系，原评测 200 次置换的平均 F1 为 "
         f"{qc['argmax_k_histogram_preserving_shuffle_f1_mean']:.5f}，Context Token 为 "
         f"{qc['argmax_k_histogram_preserving_shuffle_tokens_mean']:.2f}。"
         f"真实对应的 F1 高 {100 * qc['aligned_argmax_f1_minus_shuffled']:.2f} 个百分点，"
         f"单侧置换 p={qc['permutation_p_greater_argmax_f1']:.3f}，逐题分配带来的回答收益证据较弱。"
         f"概率与问题对应时，Soft CE 比打乱概率低 {-qc['aligned_minus_shuffled_soft_ce']:.5f}"
         f"（p={qc['permutation_p_lower_ce']:.3f}），说明存在弱的逐题标签信号。"
         f"与只用训练集拟合的平均软标签相比，BERT 测试 KL 为 {prior['bert']['soft_kl']:.5f}，"
         f"训练均值策略为 {prior['train_mean_distribution']['soft_kl']:.5f}。BERT−训练均值 Soft CE 的配对 "
         f"95% 区间为 [{ce_interval['ci95_low']:+.5f}, {ce_interval['ci95_high']:+.5f}]。"
         "置换结果和区间均来自这一划分及模型种子。"),
        markdown_table(["Epoch", "训练在线 Soft CE", "训练在线 KL（推导）", "验证 Soft CE", "验证 KL"], [
            (r["epoch"], f"{r['train_soft_ce']:.5f}", f"{r['train_online_kl_derived']:.5f}",
             f"{r['validation_soft_ce']:.5f}", f"{r['validation_kl']:.5f}") for r in train["epochs"]
        ]),
        (f"日志直接记录训练在线 Soft CE 和验证 KL。训练在线 KL 用每轮 Soft CE 减去有效训练标签平均熵 "
         f"{train['train_label_entropy']:.5f} 推导，训练值来自参数不断更新的 batch。"
         "第 2 轮后训练损失继续下降，验证 KL 开始回升，有轻度过拟合迹象。"
         "测试概率熵接近均匀、Soft CE 接近训练均值且 Argmax 集中，说明当前目标下的逐题预测信号较弱。"
         "这组日志无法区分 Query 本身信息不足和模型表达、训练样本或监督目标的影响。"
         " [Epoch 日志](../../experiments/hotpotqa_train_5000_bert/logs/epochs.jsonl)"),
        "## 5. 典型案例",
        ("以下为定向选取的案例，用于查看错误机制。检索标题、句证据召回与答案来自保存记录。"
         "每个 k 只有一次输出，关于新增文本干扰或 Reader 表现的解释均为待验证假设。"
         " [案例 JSON](cases.json)"),
        case_sections(cases),
    ]

    drift = sc["different_k"]
    blocked = sc["oracle_k_blocked_by_evidence"]
    utility_only = sc["different_k_oracle_passes_evidence"]
    cost_ci = sc["paired_delta"]["context_tokens"]
    best_fixed = max(FIXED, key=lambda k: m[k]["f1"])
    parts += [
        "## 6. 原因分析与结论",
        "### 数据存在逐题收益，可靠监督仍受检索与生成链路限制",
        (f"在这批保存曲线上，{share(needs, n)} 的问题增加 k 后收益超过 0.10，且 "
         f"{share(d['any_more_k_worse_than_k1'], n)} 的题至少一次扩大 k 后比 k=1 更差。"
         f"ε=0.10 的质量 Oracle 相对验证集选定的 k={val_fixed} 提高 "
         f"{100 * (oq['f1'] - m[f'Fixed k={val_fixed}']['f1']):.2f} 个 F1 百分点，"
         f"节省 {1 - oq['context_tokens'] / m[f'Fixed k={val_fixed}']['context_tokens']:.1%} 的 Context Token。"
         f"在测试集上描述性最高的固定点是 {best_fixed}，平均 F1 {m[best_fixed]['f1']:.5f}。"
         "这些观测支持继续研究动态选择，也给出了离线机会的量级。"),
        (f"{zeros} 道全零题中，{d['all_zero_and_no_sentence_hit']} 道所有 k 都没有确认匹配的支持句，"
         f"{d['all_zero_but_full_evidence_at_some_k']} 道则至少一个 k 的句证据召回达到 1。"
         f"测试数据有 {audit['test_anomalies']['answer_extraction_fallback']} 个答案提取回退记录、"
         f"{audit['test_anomalies']['empty_answer']} 个空答案。"
         "全零和波动可能涉及检索覆盖、阅读器利用证据、答案措辞或评分。"
         "单次观测无法拆分这些因素，也无法判断哪些差异能由 Question text 稳定预测。"),
        "### 软标签主要改变证据要求和成本",
        (f"在同一批 {nv} 道有标签题上，Scheme D 相比 ε=0.10 Oracle 改变 "
         f"**{share(drift['n'], nv)}** 的 k，平均 **多用 {soft['context_tokens'] - vq['context_tokens']:.2f} "
         f"个 Context Token（{soft['context_tokens'] / vq['context_tokens'] - 1:+.1%}）**，"
         f"F1 差只有 **{pp(soft['f1'] - vq['f1'])} 个百分点**。"
         f"配对 Bootstrap 的平均 Token 差 95% 区间为 [{cost_ci['ci95_low']:.2f}, {cost_ci['ci95_high']:.2f}]。"
         f"其中 {blocked['n']} 题由证据门槛排除了质量 Oracle 的 k，F1 持平且共多用 "
         f"{blocked['context_delta_sum']:,} 个 Token，占差额的 "
         f"{blocked['context_delta_sum'] / drift['context_delta_sum']:.1%}；"
         f"另有 {utility_only['n']} 题通过门槛后因效用排序而选了更贵、F1 更高的 k。"
         f"此外，{audit['split_counts']['test']['empty_candidates']} 道有正分的测试题没有同时满足质量和证据要求的候选，因此原标签无 Argmax。"
         "证据门槛适合单独审视，因为它提供了答案 F1 之外的约束。"),
        markdown_table(["保持 ε=0.10 的离线规则", "F1", "Context Token", "平均 k"], [
            (name, f"{row['f1']:.5f}", f"{row['context_tokens']:.2f}", f"{row['mean_k']:.3f}")
            for name, row in sc["four_policies"].items()
        ]),
        (f"回退混合规则在 {n} 题上的 F1 为 {m[FALLBACK]['f1']:.5f}，比质量 Oracle 低 "
         f"{100 * (oq['f1'] - m[FALLBACK]['f1']):.2f} 个百分点。这个全测试集差异包含所选回退策略的影响。"
         "在共同有效子集上，原软标签的事后 Argmax 能保持接近逐题最优的 F1。"),
        "### 当前主要落差位于决策效果，模型与输入的原因仍待对照",
        (f"在 {nv} 道同一有效标签题上，BERT 与原 Scheme D Argmax 的 F1 分别为 "
         f"{valid[BERT]['f1']:.5f} 和 {soft['f1']:.5f}，差 "
         f"**{100 * (soft['f1'] - valid[BERT]['f1']):.2f} 个百分点**。"
         "这一落差明显大于两个事后目标的 F1 差异。BERT 的题目与训练标签间有弱对应信号，"
         "但保留 k 频数的置换对照没有给出明确的实际问答收益证据。当前性能缺口首先表现在 Question-only "
         "BERT 策略未能把高预算分给收益最大的题。单个 checkpoint、单一数据划分和现有监督只能定位到这一环节，"
         "无法将其归因于 BERT 架构、训练数据数量、Query 特征不足或软目标散度中的某一项。"),
        "### 离线 λ 敏感度",
        ("保持原证据和质量候选集，扫描如下 λ。对任一正温度 T，效用排序和 Argmax 保持不变，"
         "温度只改变分布形状及训练信号。该扫描的 F1 和成本来自事后 Argmax，无需重新训练。"),
        markdown_table(["λ", "相对当前 Argmax 改变的题数", "F1", "Context Token"], [
            (penalty, row["changed_from_actual"], f"{row['f1']:.5f}", f"{row['context_tokens']:.2f}")
            for penalty, row in result["lambda_sweep"].items()
        ]),
        "## 7. 后续实验建议",
        ("1. 在现有划分和曲线上做受控标签对照，比较 Scheme D、直接监督最省 Token 的质量 Oracle，以及先预测 "
         "`k=1 是否距最优超过 0.10` 再预测预算的两阶段目标。为全零题单独建状态，对正分但证据候选为空的题保留质量监督。"
         "同时记录每种规则的 F1 和句证据召回，明确项目是否要求 grounding。\n"
         "2. 在相同上下文预算下比较固定 k、独立随机混合、现有 Question-only BERT 和加入低成本 BM25 特征的选择器。"
         "用多种子报告质量门槛违反率、无 F1 收益的超额 Token 和等预算 F1。超参数和阈值用验证集预选。\n"
         "3. 在后续实验中优先抽查全零和 F1 非单调的题，用相同 k 的重复或小规模受控 Reader 对照估计标签稳定性。"
         "若部署目标是 API 总消耗，使用覆盖一致的 Usage 重新定义目标，并同时报告 Context Token 与 API Total Token。"),
        "从仓库根目录执行以下离线命令。脚本需要本地完整实验产物、NumPy 和 Matplotlib。\n\n"
        "```sh\n"
        "PYTHONDONTWRITEBYTECODE=1 /home/dengxin/miniconda3/envs/adaptiverag/bin/python "
        "analysis/dynamic_topk_diagnosis/analyze.py\n"
        "```",
    ]
    (out / "report.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
