#!/usr/bin/env python3
"""Offline-only HotpotQA evidence diagnostic. Python standard library only.

Run from any directory: python3 /absolute/path/to/analyze_evidence.py
No experiment modules with network clients are imported. Fail closed on bad joins.
"""

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import re
import statistics
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "analysis/hotpot_formal_1000_20261014"
PRED = ROOT / "predictions/hotpot_formal_1000_20261014"
KS = tuple(range(1, 16))
GROUPS = ("bridge", "comparison", "combined")
STATES = ("complete", "partial", "zero", "unknown")
FOCUS = (1, 2, 5, 10, 15, "oracle")
MAX_WORDS = 350


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    # JSON object duplicates must not be silently collapsed by the parser.
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, f"Duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def index_rows(rows, field, source):
    result = {}
    for row in rows:
        qid = row[field]
        require(qid not in result, f"Duplicate {field}={qid} in {source}")
        result[qid] = row
    return result


def read_jsonl(path, field):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        require(bool(line.strip()), f"Empty JSONL record {path}:{number}")
        # Reuse the duplicate-key check for JSONL objects too.
        def unique(pairs):
            out = {}
            for key, value in pairs:
                require(key not in out, f"Duplicate key {path}:{number}: {key}")
                out[key] = value
            return out
        rows.append(json.loads(line, object_pairs_hook=unique))
    return index_rows(rows, field, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_official_scores():
    """Load the unchanged official functions without its unused ujson dependency.

    AST extraction executes only the three named pure functions, with re/string/
    Counter provided explicitly. The official evaluator file is fingerprinted.
    """
    path = ROOT / "official_evaluation/hotpotqa/hotpot_evaluate_v1.py"
    tree = ast.parse(path.read_text())
    names = {"normalize_answer", "f1_score", "exact_match_score"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, "Official scoring functions missing")
    scope = {"re": re, "string": string, "Counter": Counter}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def norm(text):
    # Preserve punctuation, digits and accents: title punctuation can identify
    # different entities (Money, Money, Money vs Money! Money! Money!).
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return " ".join(text.split())


def compact(text):
    # Only whitespace is removed. No fuzzy/token-overlap matching is used.
    return "".join(norm(text).split())


def loose_title(text):
    # Used to mark uncertainty, NEVER to award a match.
    return "".join(c for c in norm(text) if c.isalnum())


def render_context(doc):
    body = " ".join(doc["paragraph_text"].split(" ")[:MAX_WORDS]).strip()
    return body if body.startswith("Wikipedia Title: ") else "Wikipedia Title: " + doc["title"] + "\n" + body


def model_doc(doc):
    text = render_context(doc)
    # The added title is not counted as a supporting sentence.
    body = text.split("\n", 1)[1] if text.startswith("Wikipedia Title: ") and "\n" in text else ""
    return {"title": doc["title"], "paragraph_text": body}


def coverage_state(hit, unknown, total):
    require(total > 0 and 0 <= hit <= total and 0 <= unknown <= total - hit, "Bad evidence counts")
    if hit == total:
        return "complete"
    if unknown:
        return "unknown"
    return "partial" if hit else "zero"


def prepare_gold(sample, raw):
    qid = sample["question_id"]
    for key, rawkey in (("question_text", "question"), ("type", "type"), ("level", "level")):
        require(sample[key] == raw[rawkey], f"Raw/processed {key} mismatch {qid}")
    answers = [a for obj in sample["answers_objects"] for a in obj["spans"]]
    require(answers == [raw["answer"]], f"Gold answer mismatch {qid}")
    raw_context = index_rows([{"title": t, "sentences": ss} for t, ss in raw["context"]], "title", qid)
    facts = [tuple(f) for f in raw["supporting_facts"]]
    require(len(facts) == len(set(facts)), f"Duplicate supporting fact {qid}")
    require(facts, f"No supporting facts {qid}")
    processed = index_rows(sample["contexts"], "title", qid)
    require(set(processed) == set(raw_context), f"Raw/processed titles differ {qid}")
    titles = {f[0] for f in facts}
    for title, row in processed.items():
        require(isinstance(row["is_supporting"], bool), f"Invalid is_supporting {qid}")
        require(row["is_supporting"] == (title in titles), f"Support flag mismatch {qid}")
        require(compact(row["paragraph_text"]) == compact("".join(raw_context[title]["sentences"])),
                f"Raw/processed text differs {qid}/{title}")
    paragraphs = []
    for title in sorted(titles):
        require(title in raw_context, f"Fact title absent from raw context {qid}/{title}")
        sentences = raw_context[title]["sentences"]
        indices = [i for t, i in facts if t == title]
        for i in indices:
            require(isinstance(i, int) and 0 <= i < len(sentences), f"Bad fact index {qid}/{title}/{i}")
            require(bool(sentences[i].strip()), f"Empty support sentence {qid}/{title}/{i}")
        text = "".join(sentences).strip()
        paragraphs.append({"title": title, "text": text, "norm": norm(text), "compact": compact(text),
                           "facts": [{"sentence_index": i, "text": sentences[i].strip()} for i in indices]})
    return paragraphs


def match_evidence(gold, documents):
    """Conservative literal coverage, with explicit per-item lower/upper bounds.

    Paragraph identity requires exact normalized text or a substantial same-title
    prefix. A title-only/text-version conflict is unknown. Sentence occurrence is
    checked separately in the supplied (already rendered) text. Long exact
    cross-title occurrences count as sentence evidence but not as paragraph hits.
    """
    docs = [{"title": d["title"], "n": norm(d["paragraph_text"]), "c": compact(d["paragraph_text"])} for d in documents]
    paragraphs, facts = [], []
    for g in gold:
        methods, candidates = [], []
        full_text = False
        for j, d in enumerate(docs):
            same = norm(d["title"]) == norm(g["title"])
            near = loose_title(d["title"]) == loose_title(g["title"])
            exact = d["n"] == g["norm"]
            spacing = d["c"] == g["compact"]
            if (exact or spacing) and (same or len(g["compact"]) >= 80):
                methods.append((j, "exact" if exact else "whitespace_only"))
                full_text = True
            elif same and len(d["c"]) >= 80 and len(d["n"].split()) >= 12 and g["compact"].startswith(d["c"]):
                methods.append((j, "same_title_prefix"))
            elif same or near:
                candidates.append(j)
        status = "present" if methods else "unknown" if candidates else "absent"
        paragraphs.append({"title": g["title"], "status": status, "full_text_present": full_text,
                           "matches": [{"document_index": j, "method": m} for j, m in methods],
                           "ambiguous_document_indices": candidates})
        for fact in g["facts"]:
            fn, fc = norm(fact["text"]), compact(fact["text"])
            matches, short_unanchored = [], False
            for j, d in enumerate(docs):
                same = norm(d["title"]) == norm(g["title"])
                exact, spacing = fn in d["n"], fc in d["c"]
                if not (exact or spacing):
                    continue
                anchored = same or any(j == index for index, _ in methods)
                long_literal = len(fc) >= 40 and len(fn.split()) >= 6
                if anchored or long_literal:
                    matches.append({"document_index": j, "method": "exact" if exact else "whitespace_only",
                                    "cross_title": not same})
                else:
                    short_unanchored = True
            fs = "present" if matches else "unknown" if status == "unknown" or short_unanchored else "absent"
            facts.append({"title": g["title"], "sentence_index": fact["sentence_index"], "status": fs, "matches": matches})
    result = {"paragraphs": paragraphs, "facts": facts}
    for unit, rows in (("paragraph", paragraphs), ("sentence", facts)):
        hit = sum(x["status"] == "present" for x in rows)
        unknown = sum(x["status"] == "unknown" for x in rows)
        result.update({unit + "_hit": hit, unit + "_unknown": unknown, unit + "_total": len(rows),
                       unit + "_status": coverage_state(hit, unknown, len(rows)),
                       unit + "_recall_lower": hit / len(rows),
                       unit + "_recall_upper": (hit + unknown) / len(rows)})
    result["full_paragraph_text_count"] = sum(x["full_text_present"] for x in paragraphs)
    return result


def curve_features(scores):
    ks = sorted(scores)
    fs = [scores[k]["f1"] for k in ks]
    es = [scores[k]["em"] for k in ks]
    maximum = max(fs)
    tied = [k for k in ks if scores[k]["f1"] == maximum]
    return {"oracle_k": min(tied), "k1_f1": fs[0], "k1_em": es[0], "max_f1": maximum,
            "max_em": max(es), "delta_f1": maximum - fs[0], "tied_best_count": len(tied),
            "tied_best_ks": tied, "all_zero_f1": maximum == 0, "flat_positive_f1": len(set(fs)) == 1 and maximum > 0,
            "flat_f1": len(set(fs)) == 1, "variable_f1": len(set(fs)) > 1,
            "k1_unique_best": tied == [1], "k1_tied_best": 1 in tied and len(tied) > 1,
            "k1_not_best": 1 not in tied, "all_em_zero": max(es) == 0,
            "positive_f1_no_em": maximum > 0 and max(es) == 0,
            "all_em_one": min(es) == 1, "any_f1_drop": any(b < a for a, b in zip(fs, fs[1:])),
            "k15_worse_than_k1": fs[-1] < fs[0]}


def doc_keys(documents, with_id=False):
    return [(d.get("id"), d["title"], d["paragraph_text"]) if with_id else (d["title"], d["paragraph_text"]) for d in documents]


def nesting(lists):
    failures, order_changes, replacements = [], [], []
    for small in KS:
        for large in KS:
            if small >= large:
                continue
            a, b = lists[small], lists[large]
            if a != b[:len(a)]:
                failures.append([small, large])
            common = set(a) & set(b)
            if [x for x in a if x in common] != [x for x in b if x in common]:
                order_changes.append([small, large])
            if not set(a).issubset(b):
                replacements.append([small, large])
    return {"nested": not failures, "non_prefix_pairs": failures, "order_change_pairs": order_changes,
            "lost_document_pairs": replacements, "k1_equals_k15_first": lists[1] == lists[15][:len(lists[1])]}


def ratio_count(count, denominator):
    return {"count": count, "denominator": denominator, "ratio": count / denominator if denominator else None}


def histogram(values, domain=None):
    c = Counter(values)
    return {str(k): c[k] for k in (domain if domain is not None else sorted(c))}


def oracle_distribution(rows):
    return {"n": len(rows), "counts": histogram([r["oracle_k"] for r in rows], KS),
            "k1": ratio_count(sum(r["oracle_k"] == 1 for r in rows), len(rows))}


def average(values):
    return statistics.mean(values) if values else None


def distribution(values):
    values = sorted(values)
    def quantile(p):
        if not values:
            return None
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return {"n": len(values), "mean": average(values), "min": values[0] if values else None,
            "max": values[-1] if values else None, "quantiles": {str(p): quantile(p) for p in (0, .25, .5, .75, .9, 1)},
            "histogram": histogram(values)}


def delta_bin(value):
    # Keep the original float in every record/histogram. Tolerance is only for
    # diagnostic interval boundaries (e.g. .8-.5 == .30000000000000004).
    if value == 0:
        return "zero"
    if value <= .1 + 1e-12:
        return "(0,0.1]"
    if value <= .3 + 1e-12:
        return "(0.1,0.3]"
    return "(0.3,1]"


def performance(rows, k_selector):
    records = [r["per_k"][r["oracle_k"] if k_selector == "oracle" else k_selector] for r in rows]
    return {"n": len(rows), "mean_f1": average([x["f1"] for x in records]), "em": average([x["em"] for x in records]),
            "k1_em": average([r["k1_em"] for r in rows]), "oracle_distribution": oracle_distribution(rows)}


def coverage_summary(rows, selector):
    result = performance(rows, selector)
    for unit in ("paragraph", "sentence"):
        records = [r["per_k"][r["oracle_k"] if selector == "oracle" else selector] for r in rows]
        unit_result = {"macro_recall_lower": average([x[unit + "_recall_lower"] for x in records]),
                       "macro_recall_upper": average([x[unit + "_recall_upper"] for x in records]),
                       "gold_item_total": sum(x[unit + "_total"] for x in records),
                       "hit_total": sum(x[unit + "_hit"] for x in records),
                       "unknown_total": sum(x[unit + "_unknown"] for x in records)}
        unit_result["micro_recall_lower"] = unit_result["hit_total"] / unit_result["gold_item_total"] if unit_result["gold_item_total"] else None
        unit_result["states"] = {}
        for state in STATES:
            selected = [r for r, x in zip(rows, records) if x[unit + "_status"] == state]
            unit_result["states"][state] = {**ratio_count(len(selected), len(rows)), **performance(selected, selector)}
        result[unit] = unit_result
    return result


def analyze_transitions(rows):
    """Within-question associations; unknown endpoints are kept separate."""
    output = {}
    for left, right in ((1, 2), (1, 5), (1, 10), (1, 15)):
        akey = f"{left}->{right}"
        table = {name: [] for name in ("coverage_increased", "coverage_same", "coverage_decreased", "unknown")}
        for r in rows:
            a, b = r["per_k"][left], r["per_k"][right]
            if a["sentence_unknown"] or b["sentence_unknown"]:
                key = "unknown"
            else:
                delta = b["sentence_hit"] - a["sentence_hit"]
                key = "coverage_increased" if delta > 0 else "coverage_decreased" if delta < 0 else "coverage_same"
            table[key].append((b["f1"] - a["f1"], a["em"], b["em"]))
        output[akey] = {}
        for key, vals in table.items():
            output[akey][key] = {"n": len(vals), "f1_increased": sum(v[0] > 0 for v in vals),
                                "f1_same": sum(v[0] == 0 for v in vals), "f1_decreased": sum(v[0] < 0 for v in vals),
                                "mean_delta_f1": average([v[0] for v in vals]),
                                "em_0_to_1": sum(a == 0 and b == 1 for _, a, b in vals)}
    first_complete = [r for r in rows if r["first_complete_sentence_k"] and r["first_complete_sentence_k"] > 1]
    output["first_complete_after_k1"] = {"n": len(first_complete),
        "f1_increased_at_first_complete": sum(r["per_k"][r["first_complete_sentence_k"]]["f1"] > r["k1_f1"] for r in first_complete),
        "f1_same_at_first_complete": sum(r["per_k"][r["first_complete_sentence_k"]]["f1"] == r["k1_f1"] for r in first_complete),
        "f1_decreased_at_first_complete": sum(r["per_k"][r["first_complete_sentence_k"]]["f1"] < r["k1_f1"] for r in first_complete),
        "ever_improved_after_first_complete": sum(any(r["per_k"][k]["f1"] > r["k1_f1"] for k in KS if k >= r["first_complete_sentence_k"]) for r in first_complete)}
    return output


def summarize_group(rows):
    n = len(rows)
    flags = ("all_zero_f1", "flat_positive_f1", "flat_f1", "variable_f1", "k1_unique_best", "k1_tied_best", "k1_not_best",
             "all_em_zero", "positive_f1_no_em", "all_em_one", "any_f1_drop", "k15_worse_than_k1")
    oracle1 = [r for r in rows if r["oracle_k"] == 1]
    deltas = [r["delta_f1"] for r in rows]
    out = {"n": n, "categories": {f: ratio_count(sum(r[f] for r in rows), n) for f in flags},
           "oracle": oracle_distribution(rows), "exclude_all_zero": oracle_distribution([r for r in rows if not r["all_zero_f1"]]),
           "exclude_flat": oracle_distribution([r for r in rows if r["variable_f1"]]),
           "oracle1_composition": {f: ratio_count(sum(r[f] for r in oracle1), len(oracle1)) for f in flags},
           "tied_best_counts": histogram([r["tied_best_count"] for r in rows], KS),
           "tied_best": ratio_count(sum(r["tied_best_count"] > 1 for r in rows), n),
           "delta_f1": {**distribution(deltas), "bins": {
               b: ratio_count(sum(delta_bin(d) == b for d in deltas), n) for b in ("zero", "(0,0.1]", "(0.1,0.3]", "(0.3,1]")}},
           "coverage_by_k": {str(k): coverage_summary(rows, k) for k in (*KS, "oracle")},
           "oracle1_k1_em1": coverage_summary([r for r in oracle1 if r["k1_em"] == 1], 1),
           "all_k1_em1": coverage_summary([r for r in rows if r["k1_em"] == 1], 1),
           "oracle1_k1_em1_rate": ratio_count(sum(r["k1_em"] == 1 for r in oracle1), len(oracle1)),
           "transitions": analyze_transitions(rows)}
    out["k1_correct_incomplete"] = ratio_count(sum(r["k1_em"] == 1 and r["per_k"][1]["sentence_status"] in ("partial", "zero") for r in rows), n)
    out["answer_types"] = {}
    for atype in ("yes_no", "other"):
        subset = [r for r in rows if r["answer_type"] == atype]
        out["answer_types"][atype] = {"n": len(subset), "oracle": oracle_distribution(subset),
            "flat_f1": ratio_count(sum(r["flat_f1"] for r in subset), len(subset)),
            "k1": coverage_summary(subset, 1), "k15": coverage_summary(subset, 15),
            "k1_em1": coverage_summary([r for r in subset if r["k1_em"]], 1)}
    out["levels"] = {level: {"n": len(part), "k1": performance(part, 1), "oracle": oracle_distribution(part)}
                     for level in sorted({r["level"] for r in rows})
                     for part in [[r for r in rows if r["level"] == level]]}
    audit_flags = ("raw_nested", "saved_contexts_nested", "model_contexts_nested", "raw_k1_equals_k15_first", "model_k1_equals_k15_first",
                   "any_raw_order_change", "any_model_order_change", "any_raw_duplicates", "any_context_duplicates",
                   "any_raw_short", "any_context_short", "any_filtered", "any_ambiguous", "any_title_body_conflict", "any_runtime_truncation")
    out["retrieval_audit"] = {f: ratio_count(sum(r[f] for r in rows), n) for f in audit_flags}
    out["retrieval_audit"]["per_k"] = {}
    for k in KS:
        part = [r["per_k"][k] for r in rows]
        keys = ("raw_count", "context_count", "omitted_count", "omitted_over_600_words", "runtime_truncated_count",
                "raw_duplicate_ids", "raw_duplicate_texts", "context_duplicate_texts", "raw_only_sentence_hits",
                "raw_only_paragraph_hits", "title_body_conflict_candidates")
        out["retrieval_audit"]["per_k"][str(k)] = {key: sum(p[key] for p in part) for key in keys}
        for key in ("raw_short", "context_short", "retrieval_equals_contexts", "contexts_are_raw_subsequence", "raw_only_sentence_hits"):
            out["retrieval_audit"]["per_k"][str(k)][key + "_records"] = sum(bool(p[key]) for p in part)
    complete = [r for r in rows if r["first_complete_sentence_k"]]
    out["first_complete_sentence"] = {"counts": histogram([r["first_complete_sentence_k"] for r in complete], KS),
        "ever_complete": ratio_count(len(complete), n), "never_complete": ratio_count(n - len(complete), n),
        "oracle_before_first_complete": ratio_count(sum(r["oracle_k"] < r["first_complete_sentence_k"] for r in complete), len(complete))}
    return out


CASE_LABELS = {
    "k1_correct_complete": "k=1 答对且 Gold 支持句完整",
    "k1_correct_partial": "k=1 答对但 Gold 支持句仅部分出现",
    "k1_correct_zero": "k=1 答对但未匹配任何 Gold 支持句",
    "later_correct": "k=1 答错，较大 k 出现 EM=1",
    "more_evidence_no_gain": "支持句覆盖增加但该次 F1 未提高",
    "always_wrong": "所有 k 均 EM=0",
    "always_correct": "所有 k 均 EM=1",
    "degradation": "存在相邻 k 增加但 F1 下降",
    "yes_no": "Yes/No 答案",
    "uncertain_or_non_nested": "匹配歧义、截断或检索非嵌套",
}

# Close readings of actual saved text, retained in the reproducible generator.
# These are observations about the specific contexts, not inferred model causes.
CLOSE_READINGS = {
    "5a7283365542991f9a20c4b2": "问题已描述“characters in threes”，k1 的 Rule of three 段直接给出术语定义；Allison Gross 故事链未覆盖。该题展示问句线索与单段定义可能已足以选择答案。",
    "5a7286865542991f9a20c4cb": "Skindles 正文直接写出酒店出现在 Kind Hearts and Coronets 中；缺少影片的 Gold 属性段仍可抽出片名。k3 起答案变错，不能将正确的 k1 标签解释为完成了全部 Gold 链。",
    "5a76f45a5542994aec3b719b": "k1 为非 Gold 的 Welcome Interstate Managers 专辑页，正文直接指明乐队 Fountains of Wayne。Gold 句零匹配仍有回答线索；形成于1995这一条件未由该段验证。",
    "5a7780e855429949eeb29e9f": "非 Gold 的 Kick-Ass 角色页明确写出演员 Aaron Taylor-Johnson，但未验证 Count Vronsky 的另一半条件。该例支持“零 Gold 匹配不等于零相关事实”，不证明模型完成了两个角色的交集判断。",
    "5a7222b155429971e9dc92c4": "k1 是 The Happiest Millionaire 的同名专辑页，提及相关电影和1967年录音；两部电影的 Gold 首句均未出现。候选答案已在问题中，相关年份线索也在正文中，不能将答对直接归因于参数知识。",
    "5a733caa5542992359bc32a3": "k1 只给 Battle Sheep 的2010年，问题中已列出 Obsession 与 Battle Sheep 两个答案候选；缺少 Obsession 的1977年仍全程答对。缺失比较事实来自哪里无法由本次数据确定。",
    "5a7ec61a55429934daa2fc5f": "该 Yes/No 题只在 k12 答对；k12 上下文仍未命中 This Is Elvis 的 Gold 句，新增标题主要是 Staten Island 相关条目。答案改善没有伴随 Gold 句覆盖增加，不能用 Oracle=12 证明第12篇提供了所需 Gold 事实。",
    "5a7300b85542994cef4bc41a": "问题本身已说明 Gaia 的性质，实际只询问 Astrosat；k1 的 Astrosat 首句直接给出答案。虽被标为 Comparison，表面问句结构允许单实体描述直接作答。",
    "5a79c1095542996c55b2dc62": "比较死亡时间时，k2～5 已答 Sam Taylor，k6 开始转为错误，k13 又恢复；Gold 句覆盖不会因此下降。该例体现保留原有证据仍可能发生答案退化。",
    "5abc7bc45542993a06baf8ef": "k7 的 CASSIOPE 是加拿大卫星，Gold 的 Cassiope 是植物属；大小写折叠后的同名不能判命中。k8 才出现正确植物正文。k1 的 Deutzia silvestrii 未给两个属的物种数，正确答案的来源仍未确定。",
    "5ab3abe9554299753aec597e": "k8 两个 Gold 段落身份均可确认，但赛季段的第8句在350词之外，模型实际只见3/4条支持句。k4已答 Virginia Tech，k10起反而错答 Illinois State；段落命中、句可见与最终答案需分开考察。",
    "5ab804065542990e739ec7e0": "k1 只给 Balance and Composure 是美国乐队，缺 Black Grape 的英国属性仍答 no；k2两条 Gold 句齐全后答案保持正确。较大 k 的截断发生于额外无关段，不改变这两条支持句的覆盖。",
}


def contains_phrase(phrase, text):
    return re.search(r"(?<!\w)" + re.escape(norm(phrase)) + r"(?!\w)", norm(text)) is not None


def case_categories(row):
    p = row["per_k"]
    more_no_gain = [k for k in KS[1:] if not p[k]["sentence_unknown"] and not p[1]["sentence_unknown"]
                    and p[k]["sentence_hit"] > p[1]["sentence_hit"] and p[k]["f1"] <= p[1]["f1"]]
    return {"k1_correct_complete": row["k1_em"] == 1 and p[1]["sentence_status"] == "complete",
            "k1_correct_partial": row["k1_em"] == 1 and p[1]["sentence_status"] == "partial",
            "k1_correct_zero": row["k1_em"] == 1 and p[1]["sentence_status"] == "zero",
            "later_correct": row["k1_em"] == 0 and row["max_em"] == 1,
            "more_evidence_no_gain": bool(more_no_gain), "always_wrong": row["all_em_zero"],
            "always_correct": row["all_em_one"], "degradation": row["any_f1_drop"],
            "yes_no": row["answer_type"] == "yes_no",
            "uncertain_or_non_nested": row["any_ambiguous"] or row["any_runtime_truncation"] or not row["raw_nested"] or not row["model_contexts_nested"]}


def make_cases(rows, directories):
    selected, availability = [], {}
    # Fixed deterministic quotas, stratified by question type. Empty categories
    # remain empty; do not manufacture a full-evidence-at-k1 example.
    for group in GROUPS[:2]:
        candidates = [r for r in rows if r["type"] == group]
        used = set()
        availability[group] = {key: sum(case_categories(r)[key] for r in candidates) for key in CASE_LABELS}
        for key in CASE_LABELS:
            pool = [r for r in candidates if case_categories(r)[key] and r["qid"] not in used]
            pool.sort(key=lambda r: (-int(r["all_zero_f1"] if key == "always_wrong" else False),
                                     -r["delta_f1"], r["qid"]))
            if key == "yes_no":
                # Include both a correct-at-k1 and a retrieval-benefit example.
                correct = [r for r in pool if r["k1_em"] == 1]
                incorrect = [r for r in pool if r["k1_em"] == 0]
                pool = correct[:1] + incorrect[:1] + correct[1:] + incorrect[1:]
            if key == "uncertain_or_non_nested":
                pool.sort(key=lambda r: (-int(r["any_ambiguous"]), -sum(p["raw_only_sentence_hits"] for p in r["per_k"].values()), r["qid"]))
            for r in pool[:2]:
                selected.append((key, r))
                used.add(r["qid"])
        while len(used) < 24:
            # Prefer underrepresented primary categories before filling a quota.
            counts = Counter(key for key, r in selected if r["type"] == group)
            options = []
            for key in CASE_LABELS:
                pool = [r for r in candidates if r["qid"] not in used and case_categories(r)[key]]
                if pool:
                    pool.sort(key=lambda r: (-r["delta_f1"], r["qid"]))
                    options.append((counts[key], list(CASE_LABELS).index(key), key, pool[0]))
            require(options, "Insufficient distinct representative cases")
            _, _, key, r = min(options, key=lambda x: x[:2])
            selected.append((key, r))
            used.add(r["qid"])
    cases = []
    for key, r in selected:
        qid, p = r["qid"], r["per_k"]
        larger = r["oracle_k"] if r["oracle_k"] > 1 else r["first_complete_sentence_k"] or 15
        if key == "degradation":
            larger = next(k for k in KS[1:] if p[k]["f1"] < p[k - 1]["f1"])
        if key == "more_evidence_no_gain":
            larger = next(k for k in KS[1:] if p[k]["sentence_hit"] > p[1]["sentence_hit"] and p[k]["f1"] <= p[1]["f1"])
        if key == "uncertain_or_non_nested":
            larger = next((k for k in KS if p[k]["paragraph_unknown"] or p[k]["sentence_unknown"]),
                          next((k for k in KS if p[k]["paragraph_status"] == "complete" and p[k]["sentence_status"] != "complete"),
                               next((k for k in KS if p[k]["runtime_truncated_count"]), larger)))
        def checkpoint(k):
            return read_json(directories[k] / "checkpoints" / (qid + ".json"))
        c1, cl = checkpoint(1), checkpoint(larger)
        actual1 = [model_doc(d) for d in c1["contexts"]]
        actual_larger = [model_doc(d) for d in cl["contexts"]]
        evidence = p[larger]["evidence"]
        relevant_indices = {m["document_index"] for item in evidence["paragraphs"] + evidence["facts"] for m in item["matches"]}
        relevant_indices |= {j for item in evidence["paragraphs"] for j in item["ambiguous_document_indices"]}
        relevant = []
        extra_indices = [j for j in range(len(actual_larger)) if actual_larger[j] not in actual1 and j not in relevant_indices]
        chosen_indices = sorted(relevant_indices) + extra_indices[-2:]
        for j in chosen_indices:
            d = actual_larger[j]
            if d in actual1:
                continue
            text = d["paragraph_text"]
            matched = [{"title": g["title"], **f} for g in r["gold"] for f in g["facts"]
                       if compact(f["text"]) in compact(text)]
            relevant.append({"document_index": j, "title": d["title"], "context_role": "gold_match_or_candidate" if j in relevant_indices else "additional_context",
                             "text_excerpt": text[:1400],
                             "excerpt_truncated": len(text) > 1400, "matched_support_sentences": matched})
        answer = r["gold_answer"]
        answer_in_question = r["answer_type"] != "yes_no" and contains_phrase(answer, r["question"])
        answer_in_context = r["answer_type"] != "yes_no" and any(contains_phrase(answer, d["paragraph_text"]) for d in actual1)
        observations = [f"k=1: F1={r['k1_f1']:.4f}, EM={r['k1_em']}; 支持句 {p[1]['sentence_hit']}/{p[1]['sentence_total']}，类别 {p[1]['sentence_status']}。",
                        f"k={larger}: F1={p[larger]['f1']:.4f}, EM={p[larger]['em']}; 支持句 {p[larger]['sentence_hit']}/{p[larger]['sentence_total']}，类别 {p[larger]['sentence_status']}。",
                        f"最大 F1={r['max_f1']:.4f}，{r['tied_best_count']} 个 k 并列；按原规则 Oracle k={r['oracle_k']}。"]
        if answer_in_context:
            observations.append("标准答案字符串已在 k=1 正文中出现；这仅说明可见的词面线索，不证明完整推理链。")
        if answer_in_question:
            observations.append("标准答案字符串已在问题中出现；该词面线索可能降低答案选择难度。")
        if r["all_zero_f1"]:
            observations.append("全部 F1=0，该题的 Oracle=1 来自全零并列，不能作为低检索需求的正面证据。")
        if r["all_em_zero"] and r["max_f1"] > 0:
            observations.append("此例的“始终答错”仅指全部官方 EM=0；部分答案可能涉及名称缩写、别名或答案形式差异，不能直接认定所有预测在语义上错误。")
        if r["any_ambiguous"]:
            uncertain_ks = [k for k in KS if p[k]["paragraph_unknown"] or p[k]["sentence_unknown"]]
            observations.append(f"k={uncertain_ks} 存在标题候选但正文不一致，相关证据保持 unknown；其他 k 可能出现可靠命中。")
        if p[larger]["raw_only_sentence_hits"]:
            observations.append(f"k={larger} 有 {p[larger]['raw_only_sentence_hits']} 条 Gold 支持句仅存在于原始返回全文，未进入实际模型文本；不能计为模型证据。")
        if qid in CLOSE_READINGS:
            observations.append(CLOSE_READINGS[qid])
        cases.append({"question_id": qid, "type": r["type"], "level": r["level"], "answer_type": r["answer_type"],
            "primary_category": key, "categories": [k for k, yes in case_categories(r).items() if yes],
            "question": r["question"], "gold_answer": answer, "oracle_k": r["oracle_k"],
            "close_reading": CLOSE_READINGS.get(qid),
            "f1_curve": [p[k]["f1"] for k in KS], "em_curve": [p[k]["em"] for k in KS],
            "predictions_by_k": {str(k): r["scores"][k]["prediction"] for k in KS},
            "k1_prediction": r["scores"][1]["prediction"], "oracle_prediction": r["scores"][r["oracle_k"]]["prediction"],
            "gold_supporting_paragraphs": [{"title": g["title"], "facts": g["facts"]} for g in r["gold"]],
            "k1_actual_model_contexts": actual1,
            "larger_k": larger, "larger_k_new_evidence_contexts": relevant[:4],
            "larger_k_all_context_titles": [d["title"] for d in actual_larger],
            "k1_evidence": p[1]["evidence"], "oracle_evidence": p[r["oracle_k"]]["evidence"], "larger_k_evidence": evidence,
            "retrieval_nested": r["raw_nested"], "model_contexts_nested": r["model_contexts_nested"],
            "ambiguity_by_k": {str(k): {"unknown_titles": p[k]["unknown_gold_titles"], "unknown_facts": p[k]["unknown_gold_facts"]} for k in KS if p[k]["paragraph_unknown"] or p[k]["sentence_unknown"]},
            "larger_k_raw_sentence_hits": p[larger]["raw_sentence_hit"],
            "larger_k_model_sentence_hits": p[larger]["sentence_hit"],
            "answer_literal_in_question": answer_in_question, "answer_literal_in_k1_context": answer_in_context,
            "observations": observations,
            "unconfirmed_explanations": ["答对且 Gold 不完整时，候选解释包括参数知识、问题线索、非 Gold 等价证据和猜测；本次离线数据不能区分因果。",
                "Gold 完整仅表示标注句文本可见，不证明模型使用了它；文本未匹配也不证明不存在其他充分证据。"],
            "checkpoint_reference": str((directories[larger] / "checkpoints" / (qid + ".json")).relative_to(ROOT))})
    return cases, {"availability": availability, "selected": {g: histogram([c["primary_category"] for c in cases if c["type"] == g]) for g in GROUPS[:2]},
                   "n": len(cases), "empty_categories": {g: [k for k, n in availability[g].items() if not n] for g in GROUPS[:2]}}


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path, rows):
    require(bool(rows), f"No CSV rows {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            require(set(row) == set(rows[0]), f"Inconsistent output CSV fields {path}")
            writer.writerow({k: csv_value(v) for k, v in row.items()})


def load_experiment():
    data_path = ROOT / "processed_data/hotpotqa/dev_1000_subsampled.jsonl"
    raw_path = ROOT / "raw_data/hotpotqa/hotpot_dev_distractor_v1.json"
    paths = [data_path, raw_path, *[p for p in BASE.iterdir() if p.is_file()],
             ROOT / "official_evaluation/hotpotqa/hotpot_evaluate_v1.py",
             ROOT / "commaqa/inference/ircot.py", ROOT / "commaqa/inference/participant_qa.py",
             ROOT / "analysis/hotpot_formal_runner/hooks.py"]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    samples = read_jsonl(data_path, "question_id")
    raw = index_rows(read_json(raw_path), "_id", raw_path)
    curves = read_jsonl(BASE / "per_question_curves.jsonl", "question_id")
    require(len(samples) == 1000 and Counter(s["type"] for s in samples.values()) == {"bridge": 500, "comparison": 500}, "Expected 500 bridge + 500 comparison")
    require(set(samples) == set(curves), "Curve/dataset qids differ")
    require(set(samples).issubset(raw), "Raw data missing sampled qids")
    directories = {}
    for directory in PRED.iterdir():
        if not directory.is_dir():
            continue
        match = re.search(r"bm25_retrieval_count__(\d+)___", directory.name)
        require(match, f"Unexpected prediction directory {directory}")
        k = int(match[1])
        require(k in KS and k not in directories, f"Unexpected/duplicate k={k}")
        directories[k] = directory
    require(set(directories) == set(KS), "Missing k directories")
    predictions, journals, config_hashes = {}, {}, {}
    source_stats = {}
    for k in KS:
        directory = directories[k]
        predictions_path = directory / "prediction__hotpotqa_to_hotpotqa__dev_1000_subsampled.json"
        journal_path = directory / "token_usage.jsonl"
        config_path = ROOT / "instantiated_configs" / (directory.name + ".jsonnet")
        config_text = config_path.read_text()
        config_hashes[k] = digest(config_path)
        for field, expected in (("max_para_num_words", "350"), ("shuffle_paras", "false")):
            override = re.search(r'"' + field + r'"\s*:\s*([^,\n}]+)', config_text)
            require(override is None or override[1].strip() == expected, f"Unexpected renderer override {config_path}/{field}")
        predictions[k], journals[k] = read_json(predictions_path), read_jsonl(journal_path, "qid")
        require(set(predictions[k]) == set(samples) == set(journals[k]), f"Prediction/token qids differ at k={k}")
        checkpoints = list((directory / "checkpoints").glob("*.json"))
        require({p.stem for p in checkpoints} == set(samples) and len(checkpoints) == 1000, f"Checkpoint qids differ at k={k}")
        for path in checkpoints:
            st = path.stat()
            source_stats[path] = (st.st_size, st.st_mtime_ns)
        for path in (predictions_path, journal_path, config_path):
            source_hashes[str(path.relative_to(ROOT))] = digest(path)
    scores_api = load_official_scores()
    # Fail if the renderer contract used to reconstruct evidence changes.
    renderer_tree = ast.parse((ROOT / "commaqa/inference/ircot.py").read_text())
    renderer = next(n for n in renderer_tree.body if isinstance(n, ast.FunctionDef) and n.name == "para_to_text")
    scope = {}
    exec(compile(ast.Module(body=[renderer], type_ignores=[]), "ircot.para_to_text", "exec"), scope)
    for title, text in (("Title", "a  b\n c " * 150), ("Title", "Wikipedia Title: Existing\nBody"), ("T", "")):
        require(render_context({"title": title, "paragraph_text": text}) == scope["para_to_text"](title, text, MAX_WORDS), "Renderer mismatch")
    model_tree = ast.parse((ROOT / "commaqa/inference/participant_qa.py").read_text())
    model_class = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMQAParticipantModel")
    init = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
    require(ast.literal_eval(defaults["max_para_num_words"]) == MAX_WORDS, "Paragraph limit changed")
    require(ast.literal_eval(defaults["shuffle_paras"]) is False, "Paragraph shuffle default changed")
    records, per_k_rows, audit_details = [], [], {}
    checkpoint_hash = hashlib.sha256()
    methods = Counter()
    full_input_hash = digest(data_path)
    for number, qid in enumerate(sorted(samples), 1):
        sample, curve = samples[qid], curves[qid]
        gold = prepare_gold(sample, raw[qid])
        require(set(curve["scores"]) == {str(k) for k in KS}, f"Incomplete F1 curve {qid}")
        require(curve["type"] == sample["type"], f"Curve type mismatch {qid}")
        scores = {int(k): v for k, v in curve["scores"].items()}
        features = curve_features(scores)
        require(features["oracle_k"] == curve["best_k"] and features["tied_best_ks"] == curve["tied_best_ks"], f"Oracle mismatch {qid}")
        answer = raw[qid]["answer"]
        record = {"qid": qid, "question_id": qid, "type": sample["type"], "level": sample["level"],
                  "question": sample["question_text"], "gold_answer": answer,
                  "answer_type": "yes_no" if scores_api["normalize_answer"](answer) in ("yes", "no") else "other",
                  **features, "gold": gold, "scores": scores, "per_k": {}}
        raw_lists, saved_lists, model_lists = {}, {}, {}
        for k in KS:
            path = directories[k] / "checkpoints" / (qid + ".json")
            checkpoint_bytes = path.read_bytes()
            checkpoint_hash.update(f"{qid}\t{k}\t".encode() + hashlib.sha256(checkpoint_bytes).digest())
            c = read_json(path)
            require(c["qid"] == qid and c["k"] == k, f"Checkpoint identity mismatch {path}")
            require(c["config_sha256"] == config_hashes[k], f"Checkpoint config mismatch {qid}/{k}")
            require(c["input_sha256"] == full_input_hash, f"Checkpoint dataset mismatch {qid}/{k}")
            token = c["tokens"]
            require(token == journals[k][qid], f"Token journal/checkpoint mismatch {qid}/{k}")
            require(token["qid"] == qid and token["k"] == k and token["config_sha256"] == c["config_sha256"], f"Token identity mismatch {qid}/{k}")
            require(token["finish_reason"] == "stop" and token["prompt_cropped"] is False and token["fewshot_count"] == 15, f"Abnormal generation {qid}/{k}")
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                require(isinstance(token[field], int) and token[field] >= 0, f"Bad tokens {qid}/{k}")
            require(token["total_tokens"] == token["prompt_tokens"] + token["completion_tokens"], f"Token total mismatch {qid}/{k}")
            require(isinstance(token["raw_generation"], str), f"Missing raw generation {qid}/{k}")
            prediction = c["result"][1]
            require(c["result"][0] == qid and isinstance(prediction, str), f"Bad checkpoint result {qid}/{k}")
            require(prediction == token["prediction"] == scores[k]["prediction"] == predictions[k][qid], f"Prediction mismatch {qid}/{k}")
            require(scores_api["exact_match_score"](prediction, answer) == scores[k]["em"], f"EM mismatch {qid}/{k}")
            require(scores_api["f1_score"](prediction, answer)[0] == scores[k]["f1"], f"F1 mismatch {qid}/{k}")
            curve_tokens = curve["api_token_usage"][str(k)]
            require(curve_tokens == {key: value for key, value in token.items() if key != "raw_generation"}, f"Curve token mismatch {qid}/{k}")
            contexts, retrieval = c["contexts"], c["retrieval"]["documents"]
            require(isinstance(contexts, list) and contexts and isinstance(retrieval, list) and retrieval, f"Missing contexts/retrieval {qid}/{k}")
            for doc in contexts + retrieval:
                require(isinstance(doc["title"], str) and isinstance(doc["paragraph_text"], str) and bool(doc["paragraph_text"].strip()), f"Bad document {qid}/{k}")
            request = c["retrieval"]["request"]
            require(request["max_hits_count"] == k and request["corpus_name"] == "hotpotqa", f"Retrieval request mismatch {qid}/{k}")
            rendered = "\n\n".join(render_context(d) for d in contexts)
            require(hashlib.sha256(rendered.encode()).hexdigest() == token["context_sha256"], f"Actual context hash mismatch {qid}/{k}")
            model = [model_doc(d) for d in contexts]
            ev, raw_ev = match_evidence(gold, model), match_evidence(gold, retrieval)
            for unit in ("paragraphs", "facts"):
                for item in ev[unit]:
                    for m in item["matches"]:
                        methods[unit + "/" + m["method"]] += 1
                        if m.get("cross_title"):
                            methods["sentence_cross_title"] += 1
            raw_keys, saved_keys, model_keys = doc_keys(retrieval, True), doc_keys(contexts), doc_keys(model)
            raw_lists[k], saved_lists[k], model_lists[k] = raw_keys, saved_keys, model_keys
            remaining = iter(doc_keys(retrieval))
            subsequence = all(any(a == b for b in remaining) for a in saved_keys)
            require(subsequence, f"Saved contexts are not an ordered subset of retrieval {qid}/{k}")
            omitted = [d for d in retrieval if (d["title"], d["paragraph_text"]) not in saved_keys]
            row = {"qid": qid, "question_id": qid, "type": sample["type"], "k": k, "oracle_k": record["oracle_k"],
                   "f1": scores[k]["f1"], "em": scores[k]["em"],
                   **{key: value for key, value in ev.items() if key not in ("paragraphs", "facts")},
                   "raw_count": len(retrieval), "context_count": len(contexts), "omitted_count": len(retrieval) - len(contexts),
                   "omitted_over_600_words": sum(len(d["paragraph_text"].split(" ")) > 600 for d in omitted),
                   "runtime_truncated_count": sum(len(d["paragraph_text"].split(" ")) > MAX_WORDS for d in contexts),
                   "title_body_conflict_candidates": sum(len(g["ambiguous_document_indices"]) for g in ev["paragraphs"]),
                   "raw_duplicate_ids": len(raw_keys) - len({d["id"] for d in retrieval}),
                   "raw_duplicate_texts": len(raw_keys) - len(set(doc_keys(retrieval))),
                   "context_duplicate_texts": len(saved_keys) - len(set(saved_keys)),
                   "raw_duplicate_titles": len(retrieval) - len({norm(d["title"]) for d in retrieval}),
                   "raw_short": len(retrieval) < k, "context_short": len(contexts) < k,
                   "retrieval_equals_contexts": doc_keys(retrieval) == saved_keys,
                   "contexts_are_raw_subsequence": subsequence,
                   "raw_paragraph_hit": raw_ev["paragraph_hit"], "raw_sentence_hit": raw_ev["sentence_hit"],
                   "raw_paragraph_status": raw_ev["paragraph_status"], "raw_sentence_status": raw_ev["sentence_status"],
                   "raw_only_sentence_hits": raw_ev["sentence_hit"] - ev["sentence_hit"],
                   "raw_only_paragraph_hits": raw_ev["paragraph_hit"] - ev["paragraph_hit"],
                   "matched_gold_titles": [g["title"] for g in ev["paragraphs"] if g["status"] == "present"],
                   "unknown_gold_titles": [g["title"] for g in ev["paragraphs"] if g["status"] == "unknown"],
                   "sentence_matches": [{"title": f["title"], "sentence_index": f["sentence_index"], "matches": f["matches"]} for f in ev["facts"] if f["status"] == "present"],
                   "unknown_gold_facts": [[f["title"], f["sentence_index"]] for f in ev["facts"] if f["status"] == "unknown"],
                   "omitted_titles": [d["title"] for d in omitted]}
            per_k_rows.append(row)
            record["per_k"][k] = {**row, "evidence": ev}
        audits = {"raw": nesting(raw_lists), "saved": nesting(saved_lists), "model": nesting(model_lists)}
        audit_details[qid] = audits
        record.update(raw_nested=audits["raw"]["nested"], saved_contexts_nested=audits["saved"]["nested"],
                      model_contexts_nested=audits["model"]["nested"], raw_k1_equals_k15_first=audits["raw"]["k1_equals_k15_first"],
                      model_k1_equals_k15_first=audits["model"]["k1_equals_k15_first"],
                      any_raw_order_change=bool(audits["raw"]["order_change_pairs"]), any_model_order_change=bool(audits["model"]["order_change_pairs"]))
        for name, key in (("any_raw_duplicates", "raw_duplicate_ids"), ("any_context_duplicates", "context_duplicate_texts"),
                          ("any_raw_short", "raw_short"), ("any_context_short", "context_short")):
            record[name] = any(p[key] for p in record["per_k"].values())
        record["any_filtered"] = any(p["omitted_count"] for p in record["per_k"].values())
        record["any_ambiguous"] = any(p["paragraph_unknown"] or p["sentence_unknown"] for p in record["per_k"].values())
        record["any_title_body_conflict"] = any(p["title_body_conflict_candidates"] for p in record["per_k"].values())
        record["any_runtime_truncation"] = any(p["runtime_truncated_count"] for p in record["per_k"].values())
        for unit in ("paragraph", "sentence"):
            record[f"first_complete_{unit}_k"] = next((k for k in KS if record["per_k"][k][unit + "_status"] == "complete"), None)
        records.append(record)
        if number % 200 == 0:
            print(f"Validated and matched {number}/1000 questions ({number * 15}/15000 checkpoints)", flush=True)
    require(len(per_k_rows) == 15000 and len({(r['qid'], r['k']) for r in per_k_rows}) == 15000, "Pair count mismatch")
    for path, stat in source_stats.items():
        current = path.stat()
        require(stat == (current.st_size, current.st_mtime_ns), f"Checkpoint changed during analysis: {path}")
    for path, value in source_hashes.items():
        require(digest(ROOT / path) == value, f"Source changed during analysis: {path}")
    return records, per_k_rows, directories, {"source_sha256": source_hashes,
        "checkpoint_aggregate_sha256": checkpoint_hash.hexdigest(), "checkpoint_digest_order": "sorted qid, then k; sha256(qid TAB k TAB binary_sha256(file))",
        "counts": {"bridge": 500, "comparison": 500, "combined": 1000, "qid_k_pairs": 15000, "raw_questions": len(raw)},
        "validation_errors": [], "literal_match_methods": dict(methods), "nesting_details": audit_details,
        "field_inventory": {"processed": ["question_id", "question_text", "type", "level", "answers_objects", "contexts[].is_supporting"],
                            "raw": ["_id", "question", "answer", "type", "level", "context[][sentences]", "supporting_facts[][title,index]"],
                            "checkpoint": ["qid", "k", "result", "contexts[].title", "contexts[].paragraph_text", "retrieval.documents", "retrieval.request", "tokens", "config_sha256", "input_sha256"],
                            "all_required_fields_valid_for_all_records": True},
        "validated": ["unique qid/k in dataset, curves, predictions, checkpoints and token journals",
                      "raw/processed question, answer, type, level, contexts and support flags",
                      "all official supporting-fact indices and unique fact identities",
                      "official answer F1/EM for all 15000 predictions; unchanged Oracle and ties",
                      "checkpoint/token/curve/prediction alignment; context SHA256 after 350-word rendering; config file SHA256 and renderer defaults",
                      "exact journal/checkpoint token equality; complete generation and no prompt cropping",
                      "all checkpoints unchanged by size/mtime; other read sources unchanged by SHA256"]}


def verify_summaries(summary, rows):
    old = read_json(BASE / "hotpot_basic_statistics.json")
    with (BASE / "oracle_distribution.csv").open() as stream:
        old_csv = list(csv.DictReader(stream))
    require(len(old_csv) == 45 and len({(r["group"], int(r["k"])) for r in old_csv}) == 45, "Old Oracle CSV duplicate/missing records")
    for row in old_csv:
        group, k = row["group"], row["k"]
        item = summary["groups"][group]
        require(item["oracle"]["counts"][k] == int(row["count"]), "Old Oracle CSV count mismatch")
        require(abs(int(row["count"]) / item["n"] - float(row["proportion"])) < 1e-12, "Old Oracle CSV proportion mismatch")
    for group in GROUPS:
        g, o = summary["groups"][group], old["oracle"][group]
        require(g["oracle"]["counts"] == o["distribution"], f"Old Oracle JSON mismatch {group}")
        require(g["tied_best_counts"] == {str(k): o["tied_best_k_count_distribution"].get(str(k), 0) for k in KS}, f"Old tie counts mismatch {group}")
        require(g["categories"]["all_zero_f1"]["ratio"] == o["all_zero_f1_ratio"], f"Old all-zero mismatch {group}")
        require(g["tied_best"]["ratio"] == o["tied_best_k_ratio"], f"Old tied ratio mismatch {group}")
        selected = [r for r in rows if group == "combined" or r["type"] == group]
        require(abs(average([r["max_f1"] for r in selected]) - o["oracle_average_f1"]) < 1e-12, "Old Oracle mean mismatch")
        require(sum(g["delta_f1"]["bins"][b]["count"] for b in g["delta_f1"]["bins"]) == g["n"], "Delta bin partition mismatch")
        require(sum(g["categories"][f]["count"] for f in ("all_zero_f1", "flat_positive_f1", "variable_f1")) == g["n"], "Curve categories do not partition")
        require(sum(g["categories"][f]["count"] for f in ("k1_unique_best", "k1_tied_best", "k1_not_best")) == g["n"], "Best categories do not partition")
        for selector in (*KS, "oracle"):
            c = g["coverage_by_k"][str(selector)]
            for unit in ("paragraph", "sentence"):
                require(sum(x["count"] for x in c[unit]["states"].values()) == g["n"], "Coverage denominator mismatch")
    require(len(old["fixed_k"]) == 45, "Old fixed-k stats incomplete")
    for oldrow in old["fixed_k"]:
        c = summary["groups"][oldrow["group"]]["coverage_by_k"][str(oldrow["k"]) ]
        require(c["n"] == oldrow["count"] and abs(c["mean_f1"] - oldrow["average_f1"]) < 1e-12 and abs(c["em"] - oldrow["average_em"]) < 1e-12, "Old fixed-k performance mismatch")
    with (BASE / "fixed_k_results.csv").open() as stream:
        fixed_csv = list(csv.DictReader(stream))
    require(len(fixed_csv) == 45 and len({(r["group"], r["k"]) for r in fixed_csv}) == 45, "Old fixed-k CSV duplicate/missing records")
    for oldrow in fixed_csv:
        c = summary["groups"][oldrow["group"]]["coverage_by_k"][oldrow["k"]]
        require(c["n"] == int(oldrow["count"]) and abs(c["mean_f1"] - float(oldrow["average_f1"])) < 1e-12 and abs(c["em"] - float(oldrow["average_em"])) < 1e-12, "Old fixed-k CSV performance mismatch")
    summary["validation"]["old_statistics_reproduced"] = True


def pct(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def count_pct(value):
    return f"{value['count']}/{value['denominator']} ({pct(value['ratio'])})"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def write_report(summary, cases, output):
    groups = summary["groups"]
    g = groups["combined"]
    lines = ["# HotpotQA 500+500：Oracle 标签与检索证据依赖离线诊断", "",
             "分析范围：本地 500 Bridge + 500 Comparison，Qwen2.5-7B-Instruct，BM25 Single Retrieval，k=1～15；分析日期 2026-10-17（Asia/Shanghai）。未运行任何推理、检索、训练或 Utility 修改。原 Oracle 始终为最大答案 F1、并列选最小 k。",
             "", "## 核心结论", ""]
    tied = g["categories"]["k1_tied_best"]["count"]
    unique = g["categories"]["k1_unique_best"]["count"]
    oracle1 = g["oracle"]["counts"]["1"]
    lines += [f"1. **k=1 标签主要受并列和答案指标限制。** {oracle1}/1000 的 Oracle=1，其中 {tied}/{oracle1}（{pct(tied / oracle1)}）有多个最优 k，只有 {unique}/{oracle1}（{pct(unique / oracle1)}）是 k=1 唯一最优。全零 {g['categories']['all_zero_f1']['count']} 题、非零平坦 {g['categories']['flat_positive_f1']['count']} 题均由并列规则取 1。排除所有平坦曲线后，Oracle=1 为 {count_pct(g['exclude_flat']['k1'])}。",
              f"2. **答案正确经常伴随 Gold 标注证据不完整。** 全部 k=1、EM=1 有 {g['all_k1_em1']['n']} 题，其支持句类别为完整/部分/零/未知 = " + "/".join(str(g["all_k1_em1"]["sentence"]["states"][s]["count"]) for s in STATES) + "。这说明答案标签与 Gold 支持句覆盖并不等价，不能由此确认参数知识的因果贡献。",
              "3. **Comparison 更容易答对，但其更高的 Oracle=1 不能简单归因于 Yes/No。** k1 EM 为57.8%，Bridge为24.4%；Comparison中 Yes/No 的 k1 EM为71.5%，非 Yes/No 为51.5%，两者 Oracle=1却都约73.4%。Comparison的正值平坦题216道、Bridge为101道，而Bridge全零题更多（154 vs 27）。这些是答案形式和表现的描述性差异，尚不识别真实证据需求的因果差异。",
              "4. **覆盖提升和答案提升并非同步。** 同题 k1→15，653题支持句覆盖增加，其中146题 F1提高、455题不变、52题下降。k15完整支持句比例为Bridge25.8%、Comparison81.4%；Gold检索失败在Bridge更常见，完整证据到达后也不保证回答改进。",
              "5. **当前 Oracle 可表示这套模型和提示词在观测 k 范围内的答案效用最优点，但不足以单独监督实际证据需求。** 全零题提供的是失败并列标签，平坦题不识别最小充分证据量；缺少 k=0 对照和干预实验，也无法估计外部检索的因果必要性。", "",
              "## 数据完整性与模型可见文本", "",
              "所有记录按 qid、k 连接。已验证 Bridge=500、Comparison=500、Combined=1000，15000 个唯一 `(qid,k)`；所有 Gold 支持段落/句子索引有效。逐条核对 checkpoint、token journal、预测文件、旧曲线，重算官方 F1/EM，并复现旧分布、并列统计及 45 组固定 k 性能。未发现缺失、重复主键或字段冲突。详见 `validation.json` 的校验项与输入 SHA256。",
              "", "`retrieval.documents` 是服务返回列表；`contexts` 是过滤后传给 QA 的选中段落。实际提示词还调用 `para_to_text(..., 350)`，按 `split(' ')` 截断每段并加标题。本次按原函数语义复原，15000 个 context SHA256 全部匹配。未把原始检索返回但被过滤的内容算作模型证据。上下文统计均只针对当前问题的检索段落；15 个 few-shot 示例是额外提示内容，未当作当前问题的 Gold 检索证据。", "",
              "## 匹配策略与定义", "",
              "- 主匹配使用 HTML 解码、Unicode NFKC、casefold、空白归一化，保留标点、数字与重音。另有仅移除空白的确定性匹配，用于 `Lenin! ` 等排版差异；单独记录为 `whitespace_only`。未使用编辑距离、相似度阈值或语义模糊匹配。",
              "- 段落命中要求规范化正文一致；允许同标题、至少 80 个非空白字符且 12 个词的 Gold 正文前缀，作为截断后仍可识别的段落命中。这不等于完整正文或支持句全部出现。长于 80 字符的完整正文可跨标题确认段落身份。",
              "- 标题相同或去标点后相近但正文不匹配的候选保持 unknown。去标点标题只用于发现歧义，不用于授予命中；例如同名实体、专辑/乐队、标点不同的作品不能合并。",
              "- 支持句单独在实际正文中逐句查找。跨标题必须是完整字面句且至少 40 个非空白字符和 6 个词；更短的无锚点出现保持 unknown。句子可出现在非 Gold 标题下，此时记跨标题支持句命中，但不自动记 Gold 段落命中。",
              "- 每种粒度的类别互斥：全部确定出现=complete；有 unknown 且未完整=unknown；无 unknown 时，确定出现一部分=partial，确定未匹配任何=zero。unknown 即使已有部分命中也单列，其覆盖上下界保存在 CSV/JSON。zero 仅表示没有匹配 Gold 标注文本，不表示缺少任何可能的充分证据。",
              "- 段落覆盖率=每题命中的 Gold 段落比例的均值；句覆盖率=每题命中的 Gold 支持句比例的均值。句子数因题而异，JSON 另存 micro recall；所有比例默认按表内问题数为分母。F1/EM 用 0～1 存储，报告用百分数。", "",
              "匹配方法实际次数（命中对数，跨 k 重复累计，不是题数）：`" + json.dumps(summary["validation"]["literal_match_methods"], ensure_ascii=False) + "`。", "",
              "## A. Oracle 不均衡与检索收益", ""]
    labels = {"all_zero_f1": "全部 F1=0", "flat_positive_f1": "全部 F1 相同且>0", "flat_f1": "F1 完全平坦", "variable_f1": "F1 随 k 变化",
              "k1_unique_best": "k=1 唯一最优", "k1_tied_best": "k=1 并列最优", "k1_not_best": "k=1 非最优", "all_em_zero": "全部 EM=0",
              "positive_f1_no_em": "max F1>0 但全部 EM=0", "all_em_one": "全部 EM=1", "any_f1_drop": "任意相邻 k 出现 F1 下降", "k15_worse_than_k1": "k=15 F1 低于 k=1"}
    lines += [table(["类别", "Bridge /500", "Comparison /500", "Combined /1000"],
                    [[label, *[count_pct(groups[t]["categories"][flag]) for t in GROUPS]] for flag, label in labels.items()]), "",
              "包含关系：全零、正值平坦、变化是一个互斥完备划分；前两类之和是平坦。k=1 唯一最优、并列最优、非最优是另一划分；前两类之和是 Oracle=1。全零属于全部 EM=0，后者还包括 max F1>0 但无 EM 的题。EM=1 必然 F1=1，但 F1=1 未必 EM=1（如词序不同）。不同划分不能直接相加。", "",
              table(["Oracle=1 的构成", *GROUPS], [[labels[f], *[count_pct(groups[t]["oracle1_composition"][f]) for t in GROUPS]]
                    for f in ("all_zero_f1", "flat_positive_f1", "variable_f1", "k1_unique_best", "k1_tied_best", "all_em_zero")]), "",
              table(["筛选后的 Oracle=1", *GROUPS], [[label, *[count_pct(groups[t][key]["k1"]) for t in GROUPS]] for label, key in
                    (("全部问题", "oracle"), ("排除全零", "exclude_all_zero"), ("排除全部平坦（即仅变化题）", "exclude_flat"))]), "",
              "完整标签分布（每格依次为 k=1…15 的数量，分母为 n）：", "",
              table(["组", "筛选", "n", "k=1…15"], [[t, key, groups[t][key]["n"], ", ".join(map(str, groups[t][key]["counts"].values()))]
                    for t in GROUPS for key in ("oracle", "exclude_all_zero", "exclude_flat")]), "",
              "并列最优数量分布（每格依次为并列数 1…15 的题数）：", "",
              table(["组", "多个 k 并列", "并列数 1…15"], [[t, count_pct(groups[t]["tied_best"]), ", ".join(map(str, groups[t]["tied_best_counts"].values()))] for t in GROUPS]), "",
              "检索收益 `delta_f1=max F1−F1(1)`；为最大值收益，天然非负，不代表每次增加 k 都提高 F1：", "",
              table(["delta_f1 区间", *GROUPS], [[b, *[count_pct(groups[t]["delta_f1"]["bins"][b]) for t in GROUPS]] for b in g["delta_f1"]["bins"]]), "",
              "delta 的全部实际取值直方图、分位数与均值在 `summary.json`，逐题原始精度值在 CSV。区间边界0.1/0.3仅在诊断分箱时使用1e-12容差，避免0.8−0.5的浮点尾差被误放入>0.3；原始delta、F1与Oracle的精确比较不变。delta=0 与 Oracle=1 在本规则下等价；“观测的更大 k 没有超过 k=1”和“并列机制选择 1”会同时成立，不能当作两种互斥的因果贡献。", "",
              "## B. Gold 证据覆盖", "", "支持句覆盖（完整/部分/零/未知均为该组题数的比例）：", ""]
    def coverage_rows(unit):
        out = []
        for t in GROUPS:
            for k in FOCUS:
                c = groups[t]["coverage_by_k"][str(k)][unit]
                out.append([t, k, pct(c["macro_recall_lower"]), pct(c["macro_recall_upper"]),
                            *[f"{c['states'][s]['count']} ({pct(c['states'][s]['ratio'])})" for s in STATES]])
        return out
    lines += [table(["组", "k", "覆盖率下界", "上界", "完整", "部分", "零", "未知"], coverage_rows("sentence")), "",
              "Gold 支持段落身份覆盖（不能直接称为 Supporting Facts 完整覆盖）：", "",
              table(["组", "k", "覆盖率下界", "上界", "全部段落", "部分段落", "零段落", "未知"], coverage_rows("paragraph")), "",
              "所有 1000 题均有两个 Gold 支持段落。因此单段模型上下文通常不能同时命中两个 Gold 段落；如果单篇包含其他支持句，句覆盖可能与段落覆盖不同。所有 k 的逐题句数、段落数、unknown、匹配来源见 `per_k_evidence.csv`。", "",
              "全15000对中只有1对最终分类为unknown：Comparison题 `5abc7bc45542993a06baf8ef` 的k7，CASSIOPE卫星与Cassiope植物属同名；k8正确植物段落出现后可确认覆盖。另有标题/正文冲突候选但已有其他可靠命中的情况，合计涉及5题，均未把冲突候选授予命中。重点k表恰好没有k7，因此不能只看表中unknown=0就认为全程无歧义。", "",
              "## C. 证据覆盖与回答正确性", "",
              "以下按实际支持句覆盖分层；行内 n 为性能分母，`k1 EM` 是同一批题在 k=1 的 EM；Oracle 分布以 k:题数列出。其余固定 k 的同类统计完整保存在 JSON。", ""]
    lines += [table(["组", "证据所在 k", "覆盖", "n", "平均 F1", "EM", "k1 EM", "Oracle k:题数"],
        [[t, k, state, c["n"], pct(c["mean_f1"]), pct(c["em"]), pct(c["k1_em"]),
          ", ".join(f"{a}:{b}" for a, b in c["oracle_distribution"]["counts"].items() if b) or "—"]
         for t in GROUPS for k in (1, "oracle") for state in STATES
         for c in [groups[t]["coverage_by_k"][str(k)]["sentence"]["states"][state]]]), "",
        table(["组", "Oracle=1 中 k1 EM=1", "这些正确题：完整/部分/零/未知", "全部 k1 EM=1", "这些正确题：完整/部分/零/未知"],
              [[t, count_pct(groups[t]["oracle1_k1_em1_rate"]),
                "/".join(str(groups[t]["oracle1_k1_em1"]["sentence"]["states"][s]["count"]) for s in STATES),
                f"{groups[t]['all_k1_em1']['n']}/{groups[t]['n']}",
                "/".join(str(groups[t]["all_k1_em1"]["sentence"]["states"][s]["count"]) for s in STATES)] for t in GROUPS]), "",
        "在答案正确样本中的证据类别比例（条件分母明确列为 n）：", "",
        table(["组", "样本选择", "n", "完整", "部分", "零", "未知"],
              [[t, key, groups[t][key]["n"], *[count_pct(groups[t][key]["sentence"]["states"][s]) for s in STATES]]
               for t in GROUPS for key in ("oracle1_k1_em1", "all_k1_em1")]), "",
        "这两种选择的结果应同时阅读。k=1 EM=1 通常已获得最大 F1=1，故按原规则会选择 k=1；若两者集合完全一致，这是评分规则的结果，不能当作独立重复证据。段落粒度同样的分层已保存于 JSON。", "",
        "## D. 类型、答案形式与覆盖增益", "",
        table(["组", "k1 F1", "k1 EM", "Oracle=1", "k1 正确且 Gold 句不完整 /全部", "F1 平坦"],
              [[t, pct(groups[t]["coverage_by_k"]["1"]["mean_f1"]), pct(groups[t]["coverage_by_k"]["1"]["em"]),
                count_pct(groups[t]["oracle"]["k1"]), count_pct(groups[t]["k1_correct_incomplete"]), count_pct(groups[t]["categories"]["flat_f1"])] for t in GROUPS]), "",
        table(["组", "答案类型", "n", "k1 F1", "k1 EM", "Oracle=1", "k1 支持句完整/部分/零/未知"],
              [[t, a, sub["n"], pct(sub["k1"]["mean_f1"]), pct(sub["k1"]["em"]), count_pct(sub["oracle"]["k1"]),
                "/".join(str(sub["k1"]["sentence"]["states"][s]["count"]) for s in STATES)]
               for t in GROUPS for a, sub in groups[t]["answer_types"].items()]), "",
        "Yes/No 的分数被官方规则离散化为正确 1、错误 0，因此一次答对会产生大量 F1=1 并列；其表现不能仅由 50% 猜测基线解释。若 Bridge 的 Yes/No 分层为空，不能作相同答案类型的跨类型因果比较。难度 level 的描述性结果保存在 JSON，未作难度与答案形式匹配后的因果分析。", "",
        "具体而言，Comparison有158道Yes/No（113道k1答对）和342道其他答案（176道k1答对）；Bridge没有Yes/No。Comparison的Oracle=1分别为116/158与251/342，几乎一致，非Yes/No的73.4%也高于Bridge的67.4%。因此Yes/No提高了k1答案EM的混合均值，却不足以解释类型间Oracle=1的差距。Comparison问题常给出两个实体候选，而部分题实际上允许单实体描述直接作答，案例中可看到这些线索；这尚未构成对全部题目的语义标注或因果验证。", "",
        "两类Oracle=1数量相差30题，可描述性分解为：全零项27−154=−127、正值平坦项216−101=+115、变化但k1仍最优项124−82=+42，总和+30。故更高比例既不是由更多失败全零造成，也不能仅用总平坦比例解释（Comparison平坦率反而略低）。", "",
        "同题 k=1→15，按支持句覆盖变化分组（不含 unknown 的行以已确认的句数比较）：", "",
        table(["组", "覆盖变化", "n", "F1 提高", "不变", "下降", "平均 ΔF1", "EM 0→1"],
              [[t, state, x["n"], x["f1_increased"], x["f1_same"], x["f1_decreased"], pct(x["mean_delta_f1"]), x["em_0_to_1"]]
               for t in GROUPS for state, x in groups[t]["transitions"]["1->15"].items()]), "",
        table(["组", "首次完整句覆盖 k>1 的题数", "该 k 对比 k1 F1 提高", "不变", "下降", "从该 k 起曾提高"],
              [[t, *[groups[t]["transitions"]["first_complete_after_k1"][key] for key in
                      ("n", "f1_increased_at_first_complete", "f1_same_at_first_complete", "f1_decreased_at_first_complete", "ever_improved_after_first_complete")]] for t in GROUPS]), "",
        "覆盖增加并不隔离证据的因果作用：同次可能加入干扰段落，模型也可能已经答对。不能把 F1 不变全部归因于“不需要检索”，也不能把 F1 增加全部归因于新命中的 Gold 句。", "",
        table(["组", "曾完整支持句覆盖", "始终未完整 /全部", "Oracle早于首次完整 /曾完整"],
              [[t, count_pct(groups[t]["first_complete_sentence"]["ever_complete"]),
                count_pct(groups[t]["first_complete_sentence"]["never_complete"]),
                count_pct(groups[t]["first_complete_sentence"]["oracle_before_first_complete"])] for t in GROUPS]), "",
        "536道曾完整覆盖题中，420道（78.4%）的答案Oracle早于完整支持句首次到达；在首次完整时，只有125/536（23.3%）较k1提高F1，374题不变、37题下降。答案最优点与完整Gold证据到达点因此有明显错位。", "",
        "## E. 检索一致性、过滤与嵌套", ""]
    audit_labels = {"raw_nested": "原始返回全部 k 两两前缀嵌套", "saved_contexts_nested": "保存 contexts 两两嵌套", "model_contexts_nested": "实际渲染上下文两两嵌套",
                    "raw_k1_equals_k15_first": "原始 k1=原始 k15 首篇", "model_k1_equals_k15_first": "实际 k1=实际 k15 首篇",
                    "any_raw_order_change": "出现原始文档相对顺序变化", "any_model_order_change": "出现实际上下文相对顺序变化",
                    "any_raw_duplicates": "任意 k 出现重复原始文档 ID", "any_context_duplicates": "任意 k 出现重复上下文正文+标题",
                    "any_raw_short": "任意 k 原始返回不足 k", "any_context_short": "任意 k contexts 不足 k", "any_filtered": "任意 k 原始文档被过滤",
                    "any_ambiguous": "任意 k 存在最终unknown", "any_title_body_conflict": "任意 k 有标题/正文冲突候选",
                    "any_runtime_truncation": "任意 k 有350词截断"}
    lines += [table(["检查（按题统计）", *GROUPS], [[label, *[count_pct(groups[t]["retrieval_audit"][key]) for t in GROUPS]] for key, label in audit_labels.items()]), "",
        table(["组", "(qid,k)数", "返回≠contexts 的对数", "遗漏文档数", "其中 >600 词", "350词截断段数", "Gold句仅原始返回可见数"],
              [[t, groups[t]["n"] * 15, sum(groups[t]["n"] - x["retrieval_equals_contexts_records"] for x in groups[t]["retrieval_audit"]["per_k"].values()),
                *[sum(x[key] for x in groups[t]["retrieval_audit"]["per_k"].values()) for key in
                  ("omitted_count", "omitted_over_600_words", "runtime_truncated_count", "raw_only_sentence_hits")]] for t in GROUPS]), "",
        "前缀检查比较所有 105 个小 k→大 k 组合，身份同时考虑 ID/标题/正文（忽略 BM25 浮点 score）；另对原始 contexts、350 词渲染后的上下文分别比较。重复标题与重复 ID/正文不是同一概念，原始列表的重复标题另存 CSV。过滤计数按真实遗漏记录，未将全部遗漏都推定为同一原因。存在非嵌套时需同时参考 `validation.json` 的失败对，不能把跨 k 变化仅解释为追加文档。", "",
        "本轮实际结果：1000题全部通过原始返回、保存上下文、实际渲染上下文的两两嵌套校验；未观察到顺序变化、重复文档ID或返回不足k。519对（涉及66题）发生过滤，共遗漏1090段，遗漏段均未超过600词，因此不是长段阈值这一条件造成。保存上下文均为原始返回的有序子序列。共2题在15对中有运行时截断；其中Bridge题 `5ab3abe9554299753aec597e` 的同一Gold句在k4～15被截断，产生12次原始返回有句而模型没有的差异。该题k8起两个段落均命中，但句子只覆盖3/4，直接说明段落命中不能冒充句覆盖。", "",
        "## F. 代表性案例与定性观察", "",
        f"`cases.jsonl` 共 {len(cases)} 个不同 qid，Bridge/Comparison 各 24；为目的性选例，不代表总体频率。每例保留问题、答案、15 点 F1/EM、所有 k 预测、Gold 支持句、完整 k1 实际正文、较大 k 新证据或额外上下文摘录、匹配详情及可观察/未确认解释。较大 k 摘录最多 4 段，每段 1400 字符，命中的 Gold 句另保留全文；完整 checkpoint 路径可追溯。", "",
        "不可选到的类别（真实空集）：`" + json.dumps(summary["case_selection"]["empty_categories"], ensure_ascii=False) + "`。尤其不能为了展示案例而编造 k1 Gold 完整命中的样本。", ""]
    lines += [table(["qid", "类型", "主类别", "Oracle k", "k1 F1/EM", "k1句覆盖"],
                    [[c["question_id"], c["type"], CASE_LABELS[c["primary_category"]], c["oracle_k"], f"{c['f1_curve'][0]:.3f}/{c['em_curve'][0]}", c["k1_evidence"]["sentence_status"]] for c in cases]), "",
              "对实际文本的重点核读：", "",
              *[f"- **{c['question_id']}（{c['type']}）**：{c['close_reading']}" for c in cases if c["close_reading"]], "",
              "## 局限与最小后续验证（本次未执行）", "",
              "1. **参数知识与提示线索未被识别。** 可先固定本次提示词和解码，对 100～200 题做 k=0 对照，按类型、答案形式、k1正确与覆盖类别分层；保留全部 few-shot 内容并检查样例是否含当前实体/事实线索。k0答对只能证明该提示条件下不需要当前检索输入，仍不能单独区分参数知识、问题线索和猜测。",
              "2. **检索失败与阅读失败需要分开。** 对本次 k15仍缺 Gold 的题，离线标出缺哪条句；后续在 50～100 题中用最小 Gold 句上下文与现有 BM25 对照，分离语料/排名缺失和可见证据未被利用。对于 Gold 仍可能无效的题，需人工验证问题歧义与标注正确性。",
              "3. **证据完整性不是答案所需证据的唯一真值。** Gold 标注可能不是所有等价证明路径；跨标题字面匹配也无法识别改写事实。建议人工复核 30～50 个“k1正确且 Gold不足”样本，记录替代证据是否真的足够，不把词面答案出现视为推理成立。",
              "4. **Dynamic Top-K 的监督局限。** 当前标签混合了失败平坦、成功平坦、短上下文优势和真正检索收益；少数变化题承载了大部分 k>1 学习信号。报告已给出排除全零/平坦的敏感性分布，供评估类不均衡和标签可辨识性；本次未改 Oracle 或 Utility。后续可在独立实验中评估保留并列集合、标注置信度或辅助证据覆盖目标，但这些都应与现有标签明确区分。",
              "5. **可泛化性边界。** 这是 500+500 分层抽样、单模型、单提示词、一次确定性解码和有限 k 的观察结果；Oracle 本身使用 Gold答案选 k，不能直接等同部署时可识别的检索需求。统计未估计随机种子、生成随机性或检索时点波动。", "",
              "## 复现与输出", "",
              "运行：`python3 analysis/hotpot_formal_1000_20261014/evidence_diagnostic/analyze_evidence.py`；仅 Python 标准库，官方评分函数从本地评测源码 AST 加载，无网络依赖。",
              "", "测试：`python3 -m unittest discover -s analysis/hotpot_formal_1000_20261014/evidence_diagnostic -p 'test_*.py' -v`。",
              "", "输出包括 `summary.md`、`summary.json`、`per_question.csv`（1000行）、`per_k_evidence.csv`（15000行）、`cases.jsonl`（48例）、`validation.json`、分析脚本与单元测试。匹配计数与分母由脚本断言检查，旧输入和产物均保持原样。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output == Path(__file__).resolve().parent, "Outputs are restricted to the evidence_diagnostic directory")
    rows, per_k, directories, validation = load_experiment()
    summary = {"schema_version": 1, "analysis_date": "2026-10-17", "timezone": "Asia/Shanghai",
               "oracle_rule": "max official answer F1; smallest k on exact ties (unchanged)",
               "coverage_basis": "actual model contexts after para_to_text split(' ')[:350], verified with context_sha256",
               "state_order": list(STATES), "groups": {group: summarize_group([r for r in rows if group == "combined" or r["type"] == group]) for group in GROUPS},
               "validation": {k: v for k, v in validation.items() if k not in ("nesting_details", "source_sha256")}}
    summary["matching_strategy"] = {"normalization": ["html.unescape", "Unicode NFKC", "casefold", "collapse whitespace"],
        "punctuation_preserved": True, "whitespace_only_fallback": True, "fuzzy_matching": False,
        "same_title_prefix_min_nonspace_characters": 80, "same_title_prefix_min_words": 12,
        "cross_title_sentence_min_nonspace_characters": 40, "cross_title_sentence_min_words": 6,
        "delta_bin_boundary_tolerance": 1e-12, "oracle_tolerance": 0,
        "unknown_handling": "unknown items set a recall interval and take priority over partial/zero in the mutually exclusive status"}
    validation["analysis_script_sha256"] = digest(Path(__file__))
    validation["tests_sha256"] = digest(Path(__file__).with_name("test_analyze_evidence.py"))
    verify_summaries(summary, rows)
    cases, selection = make_cases(rows, directories)
    require(len(cases) == 48 and len({c["question_id"] for c in cases}) == 48, "Expected 48 unique cases")
    require(Counter(c["type"] for c in cases) == {"bridge": 24, "comparison": 24}, "Case type imbalance")
    summary["case_selection"] = selection
    per_question = []
    for r in rows:
        q = {k: v for k, v in r.items() if k not in ("gold", "scores", "per_k")}
        for label, selector in (("k1", 1), ("oracle", r["oracle_k"])):
            p = r["per_k"][selector]
            for unit in ("paragraph", "sentence"):
                for metric in ("hit", "total", "unknown", "status", "recall_lower", "recall_upper"):
                    q[label + "_" + unit + "_" + metric] = p[unit + "_" + metric]
        q["f1_curve"] = [r["scores"][k]["f1"] for k in KS]
        q["em_curve"] = [r["scores"][k]["em"] for k in KS]
        per_question.append(q)
    output.mkdir(exist_ok=True, parents=True)
    write_csv(output / "per_question.csv", per_question)
    write_csv(output / "per_k_evidence.csv", per_k)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    validation["old_statistics_reproduced"] = True
    validation["nesting_checks"] = {"pairs_per_question_per_layer": 105, "layers": ["raw", "saved", "model"],
                                     "total_pair_checks": 315000}
    validation["nesting_failures"] = {qid: detail for qid, detail in validation.pop("nesting_details").items()
                                      if any(not item["nested"] for item in detail.values())}
    for filename, obj in (("summary.json", summary), ("validation.json", validation)):
        (output / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, cases, output)
    print(json.dumps({g: {"n": summary["groups"][g]["n"], "oracle1": summary["groups"][g]["oracle"]["counts"]["1"],
                           "k1_em": summary["groups"][g]["coverage_by_k"]["1"]["em"]} for g in GROUPS}, indent=2))
    print(f"Wrote validated offline diagnostic to {output}")


if __name__ == "__main__":
    main()
