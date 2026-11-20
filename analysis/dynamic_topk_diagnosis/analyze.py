"""Analyze saved HotpotQA Train5000 results. Requires NumPy and Matplotlib."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
EXPERIMENT = ROOT / "experiments/hotpotqa_train_5000_bert"
DATA = EXPERIMENT / "data"
sys.path.insert(0, str(EXPERIMENT / "src"))
from common import config, read_json, require, rows, sha256, text_hash, write_json, write_rows
from evaluate import paired_interval
from soft_oracle import BOUNDARY_ATOL as ATOL

KS = np.arange(1, 16)
EPSILONS = (0, 0.05, 0.1)
ORACLES = ("Oracle ε=0", "Oracle ε=0.05", "Oracle ε=0.1")
SOFT = "Scheme D argmax"
FALLBACK = "Scheme D + fixed k=1 fallback"
BERT = "BERT argmax"
FIXED = [f"Fixed k={k}" for k in KS]
SPLITS = ("train", "validation", "test")


def index(records):
    result = {}
    for row in records:
        qid = row["question_id"]
        require(isinstance(qid, str) and qid and qid not in result, f"Invalid or duplicate ID {qid}")
        result[qid] = row
    return result


def load():
    files = ("curves.jsonl", "labels.jsonl", "queries.jsonl", "splits.json")
    metadata = read_json(DATA / "metadata.json")
    hashes = {name: sha256(DATA / name) for name in files}
    for name, digest in hashes.items():
        require(digest == metadata["artifacts_sha256"][name], f"Fingerprint mismatch: {name}")
    curves, labels, queries = (index(rows(DATA / name)) for name in files[:3])
    split = read_json(DATA / "splits.json")
    all_ids = [qid for role in SPLITS for qid in split[role]]
    cfg = config()
    require(len(set(all_ids)) == len(all_ids) == cfg["expected_questions"], "Split counts or overlap")
    require(set(all_ids) == curves.keys() == labels.keys() == queries.keys(), "Split/source ID coverage")
    require(cfg["k_values"] == KS.tolist(), "Unexpected configured k axis")
    require(all(c["k_values"] == KS.tolist() and c["question_text"] == queries[q]["question_text"]
                and c["type"] == labels[q]["type"] for q, c in curves.items()), "Curve/label/query binding")
    predictions = index(rows(DATA / "bert_predictions.jsonl"))
    require(set(predictions) == set(split["test"] + split["validation"]), "BERT ID coverage")
    for role in ("validation", "test"):
        require(all(predictions[q]["split"] == role for q in split[role]), "BERT split mismatch")
    counts = {role: dict(Counter(labels[q]["label_status"] for q in split[role])) for role in SPLITS}
    for role in SPLITS:
        require(split["counts"][role] == {"all": len(split[role]), "valid": counts[role]["valid"]},
                f"Saved split count mismatch: {role}")
    ids = split["test"]
    evaluation = read_json(EXPERIMENT / "reports/evaluation.json")
    require(text_hash("\n".join(ids)) == evaluation["test_ids_sha256"], "Saved evaluation test ID mismatch")
    hashes.update({name: sha256(DATA / name) for name in ("bert_predictions.jsonl", "test_decisions.jsonl")})
    return ids, [curves[q] for q in ids], [labels[q] for q in ids], [predictions[q] for q in ids], {
        "split_counts": counts, "base_hashes": hashes, "test_ids_sha256": evaluation["test_ids_sha256"],
        "historical_evaluation": evaluation, "label_config": cfg["soft_label"],
        "train_mean_label_entropy": float(np.mean([labels[q]["entropy"] for q in split["train"]
                                                   if labels[q]["label_status"] == "valid"])),
    }


def probability_entropy(p):
    return -(p * np.log(np.maximum(p, 1e-300))).sum(axis=-1)


def matrix(records, field):
    return np.array([r[field] for r in records])


def select_min_cost(tokens, passing):
    # np.argmin resolves equal costs by the smallest k because columns are ordered.
    return np.where(passing, tokens, np.inf).argmin(axis=1)


def select_utility(u, mask):
    return np.where(mask, u, -np.inf).argmax(axis=1)


def performance(f1, tokens, choice, cohort, usage=None):
    a = np.arange(len(f1))[cohort]
    k = choice[cohort]
    result = {
        "n": len(a), "f1": float(f1[a, k].mean()), "context_tokens": float(tokens[a, k].mean()),
        "mean_k": float(KS[k].mean()), "k_counts": np.bincount(k, minlength=len(KS)).tolist(),
    }
    if usage is not None:
        result.update({field: float(values[a, k].mean()) for field, values in usage.items()})
    return result


def group_effect(f_delta, t_delta, mask):
    n = int(mask.sum())
    return {
        "n": n, "f1_delta_mean": float(f_delta[mask].mean()) if n else None,
        "context_delta_mean": float(t_delta[mask].mean()) if n else None,
        "f1_delta_sum": float(f_delta[mask].sum()), "context_delta_sum": int(t_delta[mask].sum()),
    }


def bert_against(f, t, b, target, ceiling, eps, cohort):
    a = np.arange(len(f))
    fd = f[a, b] - f[a, target]
    td = t[a, b] - t[a, target]
    violates = f[a, b] + ATOL < ceiling - eps
    groups = {
        "below_target": b < target, "same_k": b == target, "above_target": b > target,
        "below_and_violate": (b < target) & violates,
        "above_and_violate": (b > target) & violates,
        "violates_quality": violates,
        "feasible_extra_cost": ~violates & (td > 0),
        "feasible_extra_cost_no_f1_gain": ~violates & (td > 0) & (fd <= ATOL),
        "higher_k_quality_loss": (b > target) & (fd < -ATOL),
        "different_k_same_f1": (b != target) & (np.abs(fd) <= ATOL),
        "lower_f1": fd < -ATOL, "higher_f1": fd > ATOL,
    }
    return {key: group_effect(fd, td, mask & cohort) for key, mask in groups.items()}


def confusion(b, target, cohort):
    counts = np.zeros((len(KS), len(KS)), dtype=int)
    np.add.at(counts, (target[cohort], b[cohort]), 1)
    return counts


def bins_summary(confidence, entropy, b, target, ceiling, picked_f1, eps, cohort):
    selected = np.flatnonzero(cohort)
    result = []
    for idx in np.array_split(selected[np.argsort(confidence[selected], kind="stable")], 4):
        result.append({
            "n": len(idx), "confidence_min": float(confidence[idx].min()),
            "confidence_max": float(confidence[idx].max()), "confidence_mean": float(confidence[idx].mean()),
            "entropy_mean": float(entropy[idx].mean()), "k_match_rate": float((b[idx] == target[idx]).mean()),
            "quality_violation_rate": float((picked_f1[idx] + ATOL < ceiling[idx] - eps).mean()),
            "f1_regret_mean": float((ceiling[idx] - picked_f1[idx]).mean()),
        })
    return result


def select_cases(f, t, b, s, q, evidence, valid):
    a = np.arange(len(f))
    chosen = {}

    def pick(label, mask, score, count):
        candidates = sorted(np.flatnonzero(mask), key=lambda i: (-score[i], i))
        chosen.update({int(i): label for i in [i for i in candidates if i not in chosen][:count]})

    peak = f.max(axis=1)
    pick("k=1 严重不足，BERT 仍选 k=1", (b == 0) & (f[:, 0] <= 0.2) & (peak >= 0.9),
         peak - f[:, 0] + 0.01 * evidence[a, q], 2)
    pick("k=1 已答对，BERT 加检索", (b > 0) & (f[:, 0] >= 0.9),
         f[:, 0] - f[a, b] + 0.0001 * (t[a, b] - t[:, 0]), 2)
    pick("软标签与质量 Oracle 的成本差", valid & (s != q), t[a, s] - t[a, q], 2)
    pick("曲线波动，扩大 k 可反复翻转答案", (f.max(axis=1) - f.min(axis=1)) >= 0.9,
         (np.abs(np.diff(f, axis=1)) > ATOL).sum(axis=1), 1)
    pick("证据门槛使原软标签缺失", (~valid) & (f[:, 0] <= 0.2) & (peak >= 0.9),
         t[a, q], 1)
    pick("k=1 已答对，BERT 增加成本但 F1 不变", (b > 0) & (f[:, 0] >= 0.9)
         & (np.abs(f[a, b] - f[:, 0]) <= ATOL), t[a, b] - t[:, 0], 1)
    return chosen


def read_rag(ids, curves, case_indices):
    """Confirm complete-run bindings and scan the saved contexts once per k."""
    source = ROOT / config()["data_source"]
    manifest_path = source / "runs/run_manifest.json"
    manifest = read_json(manifest_path)
    manifest_hash = sha256(manifest_path)
    frozen = read_json(DATA / "metadata.json")["source_outputs"]
    base = source / "runs/predictions" / manifest["set_names"]["full"]
    usage = {field: np.full((len(ids), len(KS)), np.nan) for field in
             ("prompt_tokens", "completion_tokens", "total_tokens")}
    positions = {qid: i for i, qid in enumerate(ids)}
    case_contexts = {i: [] for i in case_indices}
    previous = {}
    changed = np.zeros((len(ids), len(KS)), dtype=bool)
    usage_sources = Counter()
    missed_files = []
    for j, k in enumerate(KS):
        matches = list(base.glob(f"*retrieval_count__{k}___*"))
        if not matches or not (matches[0] / "contexts.jsonl").is_file():
            missed_files.append(int(k))
            for items in case_contexts.values():
                items.append(None)
            continue
        require(len(matches) == 1, f"Ambiguous RAG source for k={k}")
        directory = matches[0]
        marker_path = directory / "complete.json"
        path = directory / "contexts.jsonl"
        marker = read_json(marker_path)
        binding = {
            "manifest_sha256": manifest_hash, "input_sha256": manifest["datasets"]["full"]["sha256"],
            "config_sha256": manifest["configs"][str(k)]["sha256"], "k": int(k),
            "set_name": manifest["set_names"]["full"],
        }
        require(marker["status"] == "complete" and marker["binding"] == binding
                and marker["predict_exit_code"] == 0
                and sha256(marker_path) == frozen[str(k)]["complete_sha256"], f"RAG completion mismatch k={k}")
        digest = hashlib.sha256()
        seen = set()
        with path.open("rb") as f:
            for line in f:
                digest.update(line)
                row = json.loads(line)
                qid = row["qid"]
                if qid not in positions:
                    continue
                i = positions[qid]
                require(qid not in seen and row["k"] == k and row["binding"] == binding
                        and row["prediction"] == curves[i]["predictions"][j]
                        and row["input_question"] == curves[i]["question_text"].strip(),
                        f"RAG row ID/config/answer mismatch k={k} {qid}")
                seen.add(qid)
                api = row["api"]
                usage_sources[api["usage_source"]] += 1
                if api["usage"] is not None:
                    for field in usage:
                        value = api["usage"][field]
                        require(type(value) is int and value >= 0, f"Invalid saved {field} for {qid} k={k}")
                        usage[field][i, j] = value
                    require(usage["prompt_tokens"][i, j] + usage["completion_tokens"][i, j]
                            == usage["total_tokens"][i, j], f"Usage total mismatch for {qid} k={k}")
                reader = row["reader_contexts"]
                if j > 0 and j - 1 == previous.get(i, (-1, None))[0]:
                    old = previous[i][1]
                    changed[i, j] = reader[:len(old)] != old
                previous[i] = (j, reader)
                if i in case_contexts:
                    case_contexts[i].append({
                        "titles": [doc["title"] for doc in row["raw_contexts"]],
                        "reader_context_sha256": [text_hash(doc) for doc in reader],
                        "total_tokens": api["usage"]["total_tokens"] if api["usage"] else None,
                    })
        require(digest.hexdigest() == frozen[str(k)]["contexts_sha256"]
                and digest.hexdigest() == marker["output_hashes"]["contexts.jsonl"], f"RAG context hash mismatch k={k}")
        require(seen == positions.keys(), f"RAG context coverage mismatch k={k}")
        print(f"原始结果对齐 k={k:2d}，测试题 {len(seen)}，已知 API Usage {np.isfinite(usage['total_tokens'][:, j]).sum()}",
              flush=True)
    return usage, case_contexts, {
        "usage_sources": dict(usage_sources), "missing_context_files_k": missed_files,
        "adjacent_context_prefix_changed_questions": int(changed.any(axis=1).sum()),
        "adjacent_context_prefix_changed_pairs": int(changed.sum()),
    }


def compare_labels(f, t, s, q, allowed, evidence_pass, u, valid):
    a = np.arange(len(f))
    with_evidence = select_min_cost(t, allowed & evidence_pass)
    without_evidence = select_utility(u, allowed)
    fd = f[a, s] - f[a, q]
    td = t[a, s] - t[a, q]
    blocked = ~evidence_pass[a, q]
    return {
        "different_k": group_effect(fd, td, valid & (s != q)),
        "oracle_k_blocked_by_evidence": group_effect(fd, td, valid & blocked),
        "different_k_oracle_passes_evidence": group_effect(fd, td, valid & ~blocked & (s != q)),
        "more_context": group_effect(fd, td, valid & (td > 0)),
        "higher_f1": group_effect(fd, td, valid & (fd > ATOL)),
        "same_f1": group_effect(fd, td, valid & (np.abs(fd) <= ATOL)),
        "lower_f1": group_effect(fd, td, valid & (fd < -ATOL)),
        "paired_delta": {"f1": paired_interval(fd[valid]), "context_tokens": paired_interval(td[valid])},
        "four_policies": {
            "quality + min context": performance(f, t, q, valid),
            "quality + evidence + min context": performance(f, t, with_evidence, valid),
            "quality + utility": performance(f, t, without_evidence, valid),
            "quality + evidence + utility (Scheme D)": performance(f, t, s, valid),
        },
    }


def write_metrics(splits):
    with (OUT / "method_metrics.csv").open("w", newline="") as stream:
        fields = ["cohort", "method", "baseline", "n", "f1", "context_tokens", "prompt_tokens",
                  "completion_tokens", "total_tokens", "mean_k", "f1_delta_pp", "context_delta",
                  "context_delta_pct", "total_delta", "total_delta_pct"] + [f"k_{k}_count" for k in KS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cohort, metrics in splits.items():
            for method, result in metrics.items():
                for baseline in FIXED:
                    base = metrics[baseline]
                    row = {**result, "cohort": cohort, "method": method, "baseline": baseline,
                           "f1_delta_pp": 100 * (result["f1"] - base["f1"]),
                           "context_delta": result["context_tokens"] - base["context_tokens"],
                           "context_delta_pct": 100 * (result["context_tokens"] / base["context_tokens"] - 1)}
                    row.update({f"k_{k}_count": v for k, v in zip(KS, row.pop("k_counts"))})
                    if "total_tokens" in result:
                        row.update(total_delta=result["total_tokens"] - base["total_tokens"],
                                   total_delta_pct=100 * (result["total_tokens"] / base["total_tokens"] - 1))
                    writer.writerow(row)


def make_figures(metrics, demand, f, t, cm):
    colors = ["#7a4ba5", "#247d77", "#ba551d"]
    styles = [(BERT, "D", "#b02f41"), (FALLBACK, "^", "#356bb1"), (SOFT, "^", "#356bb1"),
              *zip(ORACLES, ["o", "s", "P"], colors)]

    def save(fig, filename):
        fig.savefig(OUT / filename, dpi=170)
        plt.close(fig)

    def scatter(ax, cohort, field, title, zoom_oracles=False):
        data = metrics[cohort]
        n = data[FIXED[0]]["n"]
        fixed = [data[k] for k in FIXED]
        ax.plot([r[field] for r in fixed], [r["f1"] for r in fixed], ".-", color="#555e6b",
                lw=1, label="Fixed k=1–15")
        for i, r in enumerate(fixed, 1):
            ax.annotate(str(i), (r[field], r["f1"]), xytext=(0, -9 if i % 2 else 5),
                        textcoords="offset points", ha="center", fontsize=7, color="#404a54")
        for name, marker, color in styles:
            if name in data:
                r = data[name]
                label = "Scheme D + k=1 fallback" if name == FALLBACK else name
                ax.scatter(r[field], r["f1"], marker=marker, color=color, label=label,
                           s={"o": 170, "s": 92, "P": 47}.get(marker, 50), alpha=0.85, zorder=3)
        ax.set(xlabel="Mean context tokens" if field == "context_tokens" else "Mean API total tokens",
               ylabel="Mean answer F1", title=f"{title}  (n={n:,})")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
        if zoom_oracles:
            inset = ax.inset_axes([0.27, 0.47, 0.36, 0.36])
            for name, marker, color in styles[-3:]:
                r = data[name]
                inset.scatter(r[field], r["f1"], marker=marker, color=color, alpha=0.85,
                              s={"o": 100, "s": 56, "P": 26}[marker])
            inset.set_title("Oracle points (detail)", fontsize=8)
            inset.margins(0.10)
            inset.ticklabel_format(axis="y", style="plain", useOffset=False)
            inset.tick_params(labelsize=7)
            inset.grid(alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    scatter(axes[0], "all_test", "context_tokens", "All test questions", zoom_oracles=True)
    scatter(axes[1], "valid_test", "context_tokens", "Test questions with valid Scheme D label",
            zoom_oracles=True)
    save(fig, "f1_context_tradeoff.png")
    if "api_complete_test" in metrics:
        fig, ax = plt.subplots(figsize=(8.4, 4.9), layout="constrained")
        scatter(ax, "api_complete_test", "total_tokens", "Test questions with API usage at all 15 k")
        save(fig, "f1_api_total_tradeoff.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    edges = np.linspace(0, 1, 21)
    ax.hist([demand["gain"][demand["nonzero_best"]], demand["gain"][~demand["nonzero_best"]]],
            bins=edges, stacked=True, color=["#356bb1", "#9c9fa5"], edgecolor="white",
            label=["Best F1 > 0", "All k have F1=0"])
    ax.axvline(0.1, color="#ba551d", ls="--", label="Gain = 0.10")
    ax.set(xlabel="Best observed F1 − F1 at k=1", ylabel="Questions",
           title=f"Per-question F1 improvement over k=1  (n={len(f):,})",
           xticks=np.linspace(0, 1, 11))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save(fig, "f1_gain_histogram.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for ax, cohort, names in [(axes[0], "all_test", (BERT, *ORACLES)),
                              (axes[1], "valid_test", (BERT, SOFT, ORACLES[2]))]:
        r = metrics[cohort]
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            ax.bar(KS + width * (i - (len(names) - 1) / 2), r[name]["k_counts"],
                   width, label=name, color=dict((name, color) for name, _, color in styles)[name])
        ax.set(xlabel="Selected k", ylabel="Questions", xticks=KS,
               title=f"{'All test questions' if cohort == 'all_test' else 'Valid Scheme D labels'}"
                     f"  (n={r[BERT]['n']:,})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    save(fig, "selected_k.png")

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    cost = ax.twinx()
    a = ax.plot(KS, f.mean(axis=0), "o-", c="#356bb1", label="Answer F1")
    b = cost.plot(KS, t.mean(axis=0), "s--", c="#ba551d", label="Context tokens")
    ax.set(xlabel="Fixed k", ylabel="Mean answer F1", xticks=KS,
           title=f"Fixed-k quality and context cost  (n={len(f):,})")
    cost.set_ylabel("Mean context tokens")
    ax.legend(a + b, [line.get_label() for line in a + b], loc="lower right")
    ax.grid(alpha=0.2)
    save(fig, "fixed_k_curves.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for ax, (name, counts) in zip(axes.flat, cm.items()):
        ax.imshow(np.log1p(counts), cmap="Blues", vmin=0, vmax=np.log1p(counts.max()))
        for row, col in zip(*np.nonzero(counts)):
            v = counts[row, col]
            ax.text(col, row, str(v), ha="center", va="center", fontsize=7,
                    color="white" if np.log1p(v) > np.log1p(counts.max()) * 0.62 else "#14283d")
        ax.set(xticks=range(len(KS)), xticklabels=KS, yticks=range(len(KS)), yticklabels=KS,
               xlabel="BERT selected k", ylabel="Target k", title=f"{name}  (n={counts.sum():,})")
    save(fig, "bert_confusion.png")


def main():
    ids, curves, labels, predictions, audit = load()
    n = len(ids)
    all_test = np.ones(n, dtype=bool)
    a = np.arange(n)
    f = matrix(curves, "f1_curve")
    t = matrix(curves, "token_cost_curve")
    e = matrix(curves, "evidence_recall_curve")
    peak = f.max(axis=1)
    valid = matrix(labels, "label_status") == "valid"
    probability = matrix(predictions, "probabilities")
    b = matrix(predictions, "argmax_k") - 1
    s = np.array([r["argmax_soft_label_k"] or 1 for r in labels]) - 1
    require(f.shape == t.shape == e.shape == probability.shape == (n, len(KS)), "Invalid curve dimensions")
    require(np.isfinite(f).all() and ((0 <= f) & (f <= 1)).all()
            and ((0 <= e) & (e <= 1)).all() and (t > 0).all(), "Invalid F1, evidence or tokens")
    require(np.isfinite(probability).all() and (probability >= 0).all()
            and np.allclose(probability.sum(axis=1), 1, atol=1e-6, rtol=0)
            and np.array_equal(probability.argmax(axis=1), b), "Invalid BERT probabilities or argmax")

    cfg = audit["label_config"]
    require(cfg["scheme"] == "D" and cfg["epsilon"] == EPSILONS[-1]
            and cfg["evidence_metric"] == "sentence_recall_lower"
            and cfg["cost_metric"] == "context_tokens", "Unexpected label setup")
    passing = [f + ATOL >= peak[:, None] - eps for eps in EPSILONS]
    evidence_pass = e + ATOL >= cfg["tau"]
    mask = passing[-1] & evidence_pass
    cost_norm = t / t.max(axis=1, keepdims=True)
    utility = f - cfg["lambda"] * cost_norm
    require(np.array_equal(mask, matrix(labels, "candidate_mask"))
            and np.array_equal(valid, (peak > 0) & mask.any(axis=1))
            and np.array_equal(s[valid], select_utility(utility, mask)[valid]), "Scheme D label mismatch")
    require(np.allclose(cost_norm, matrix(labels, "normalized_cost_curve"), atol=ATOL, rtol=0),
            "Scheme D cost normalization mismatch")
    valid_labels = np.array([r["soft_label"] for i, r in enumerate(labels) if valid[i]])
    logits = np.where(mask[valid], utility[valid] / cfg["temperature"], -np.inf)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    require(np.allclose(probs, valid_labels, atol=ATOL, rtol=0), "Scheme D softmax mismatch")
    require(all(r["soft_label"] is None and r["argmax_soft_label_k"] is None for i, r in enumerate(labels)
                if not valid[i]), "Invalid label unexpectedly has a choice")

    decisions = index(rows(DATA / "test_decisions.jsonl"))
    require(set(decisions) == set(ids), "Saved decisions ID mismatch")
    for i, qid in enumerate(ids):
        d = decisions[qid]
        require(d["bert_argmax_k"] == KS[b[i]] and d["bert_f1"] == f[i, b[i]]
                and d["bert_context_tokens"] == t[i, b[i]] and d["label_status"] == labels[i]["label_status"],
                f"Saved decision mismatch {qid}")

    oracle = {name: select_min_cost(t, ok) for name, ok in zip(ORACLES, passing)}
    q = oracle[ORACLES[-1]]
    choice = {name: np.full(n, j) for j, name in enumerate(FIXED)}
    choice.update({BERT: b, **oracle, FALLBACK: s, SOFT: s})
    sample_cases = select_cases(f, t, b, s, q, e, valid)
    usage, case_contexts, source_checks = read_rag(ids, curves, sample_cases)
    available = np.isfinite(usage["total_tokens"])
    complete = available.all(axis=1)

    cohorts = {"all_test": all_test, "valid_test": valid}
    if complete.any():
        cohorts["api_complete_test"] = complete
    if (complete & valid).any():
        cohorts["api_complete_valid_test"] = complete & valid
    metrics = {
        name: {method: performance(f, t, k, cohort, usage if name.startswith("api_") else None)
               for method, k in choice.items()
               if method != (FALLBACK if name.endswith("valid_test") else SOFT)}
        for name, cohort in cohorts.items()
    }
    evaluation = audit.pop("historical_evaluation")
    audit["saved_eval_diagnostics"] = {key: evaluation[key] for key in
                                        ("query_specificity", "bert_minus_train_mean", "validation_selected_fixed_k")}
    for old_rows, new_rows in ((evaluation["all_test"], metrics["all_test"]),
                               (evaluation["valid_test"], metrics["valid_test"])):
        for method, old in zip(FIXED + [BERT], [f"fixed_k_{k}" for k in KS] + ["bert"]):
            require(abs(new_rows[method]["f1"] - old_rows[old]["argmax_answer_f1"]) < ATOL
                    and abs(new_rows[method]["context_tokens"] - old_rows[old]["argmax_context_tokens"]) < ATOL,
                    f"Saved aggregate mismatch {method}")
    old_soft = evaluation["valid_test"]["offline_original_soft_oracle"]
    require(abs(metrics["valid_test"][SOFT]["f1"] - old_soft["argmax_answer_f1"]) < ATOL
            and abs(metrics["valid_test"][SOFT]["context_tokens"] - old_soft["argmax_context_tokens"]) < ATOL,
            "Saved Scheme D aggregate mismatch")
    audit["saved_bert_vs_mean_distribution"] = {m: {field: evaluation["all_test"][m][field] for field in
                                                   ("soft_ce", "soft_kl", "expected_answer_f1",
                                                    "expected_context_tokens", "mean_expected_k")}
                                               for m in ("bert", "train_mean_distribution")}
    audit["api_usage"] = {
        **source_checks, "complete_test_questions": int(complete.sum()),
        "known_pairs_per_k": available.sum(axis=0).tolist(),
        "known_pairs": int(available.sum()), "missing_pairs": int((~available).sum()),
        "known_selected_per_method": {name: int(available[a, k].sum()) for name, k in choice.items() if name != SOFT},
    }
    audit["source_anomalies"] = read_json(EXPERIMENT / "reports/label_distribution.json")["anomalies"]
    audit["test_anomalies"] = {
        "answer_extraction_fallback": int(matrix(curves, "answer_fallback_curve").sum()),
        "empty_answer": sum(not answer.strip() for curve in curves for answer in curve["predictions"]),
    }

    gain = peak - f[:, 0]
    nonzero = peak > 0
    diff = np.diff(f, axis=1)
    gain_groups = {
        "best_at_k1_positive": nonzero & (gain <= ATOL),
        "perfect_at_k1": f[:, 0] >= 1 - ATOL,
        "all_zero": ~nonzero,
        "positive_best_le_0.2": nonzero & (peak <= 0.2 + ATOL),
        "strict_gain": gain > ATOL,
        "gain_0_to_0.05": (gain > ATOL) & (gain <= 0.05 + ATOL),
        "gain_0.05_to_0.1": (gain > 0.05 + ATOL) & (gain <= 0.1 + ATOL),
        "gain_gt_0.1": gain > 0.1 + ATOL,
        "gain_ge_0.05": gain + ATOL >= 0.05,
        "any_more_k_worse_than_k1": (f[:, 1:] + ATOL < f[:, [0]]).any(axis=1),
        "k15_worse_than_k1": f[:, -1] + ATOL < f[:, 0],
        "any_adjacent_drop": (diff < -ATOL).any(axis=1),
        "adjacent_rise_and_drop": (diff < -ATOL).any(axis=1) & (diff > ATOL).any(axis=1),
        "all_zero_but_full_evidence_at_some_k": ~nonzero & (e.max(axis=1) >= 1 - ATOL),
        "all_zero_and_no_sentence_hit": ~nonzero & (e.max(axis=1) <= ATOL),
        "any_adjacent_cost_drop": (np.diff(t, axis=1) < 0).any(axis=1),
    }
    for name, ok in zip(ORACLES, passing):
        gain_groups[f"{name}: k1 meets quality and peak > 0"] = nonzero & ok[:, 0]
        gain_groups[f"{name}: k1 misses quality"] = ~ok[:, 0]
    question_type = matrix(curves, "type")
    demands = {
        "counts": {key: int(m.sum()) for key, m in gain_groups.items()},
        "gain_mean": float(gain.mean()),
        "gain_gt_0.1_mean": float(gain[gain_groups["gain_gt_0.1"]].mean()),
        "by_type": {name: {"n": int((question_type == name).sum()),
                           **{key: int(((question_type == name) & m).sum())
                              for key, m in gain_groups.items()}} for name in sorted(set(question_type))},
    }

    effects = {name: bert_against(f, t, b, k, peak, eps, all_test)
               for (name, k), eps in zip(oracle.items(), EPSILONS)}
    effects[SOFT] = bert_against(f, t, b, s, peak, cfg["epsilon"], valid)
    cm = {SOFT: confusion(b, s, valid), **{name: confusion(b, k, all_test) for name, k in oracle.items()}}
    bf = f[a, b]
    bt = t[a, b]
    confidence = probability.max(axis=1)
    entropy = probability_entropy(probability)
    f_delta = bf - f[:, 0]
    t_delta = bt - t[:, 0]
    # Interpolate between the two fixed points to match BERT's observed mean budget.
    mix_k2 = float(t_delta.mean() / (t[:, 1] - t[:, 0]).mean())
    require(0 <= mix_k2 <= 1, "BERT average budget is outside the fixed k=1/2 interval")
    mixed_f1 = (1 - mix_k2) * f[:, 0] + mix_k2 * f[:, 1]
    bert = {
        "effects": effects, "confusion_true_k_rows": {k: v.tolist() for k, v in cm.items()},
        "coverage_quality_oracle_eps_0.1_ge_4": int((q >= 3).sum()),
        "mean_regret_vs_best": float((peak - bf).mean()),
        "k1_selected_when_k1_misses_quality_eps_0.1": int(((b == 0) & ~passing[-1][:, 0]).sum()),
        "equal_budget_fixed_1_2_mix": {
            "k2_probability": mix_k2, "expected_f1": float(mixed_f1.mean()),
            "bert_minus_mix_f1": paired_interval(bf - mixed_f1),
        },
        "probability": {"mean_confidence": float(confidence.mean()), "mean_entropy": float(entropy.mean()),
                        "max_possible_entropy": float(np.log(len(KS))),
                        "mean_probability": probability.mean(axis=0).tolist(),
                        "confidence_quartiles_oracle": bins_summary(
                            confidence, entropy, b, q, peak, bf, 0.1, all_test),
                        "confidence_quartiles_soft_label": bins_summary(
                            confidence, entropy, b, s, peak, bf, 0.1, valid)},
        "vs_fixed1_switch": {name: group_effect(f_delta, t_delta, (b > 0) & m)
                             for name, m in (("improved", f_delta > ATOL), ("tied", np.abs(f_delta) <= ATOL),
                                             ("worse", f_delta < -ATOL))},
        "paired_vs_fixed": {name: {"f1": paired_interval(bf - f[a, choice[name]]),
                                   "context": paired_interval(bt - t[a, choice[name]])}
                            for name in (FIXED[0], FIXED[1], f"Fixed k={evaluation['validation_selected_fixed_k']['k']}")},
    }
    comparison = compare_labels(f, t, s, q, passing[-1], evidence_pass, utility, valid)
    # Holding the actual candidate set fixed separates λ's effect from candidate masking.
    lambda_sweep = {}
    for penalty in (0, 0.05, 0.1, 0.2, 0.5, 1.0):
        k = select_utility(f - penalty * cost_norm, mask)
        lambda_sweep[str(penalty)] = {**performance(f, t, k, valid),
                                      "changed_from_actual": int(((k != s) & valid).sum())}

    epochs = list(rows(EXPERIMENT / "logs/epochs.jsonl"))
    training = {"best_epoch": evaluation["checkpoint_selection"]["epoch"],
                "train_label_entropy": audit["train_mean_label_entropy"],
                "test_valid_label_entropy": float(probability_entropy(valid_labels).mean()),
                "test_valid_mean_max_label_probability": float(valid_labels.max(axis=1).mean()),
                "test_valid_mean_candidate_count": float(mask[valid].sum(axis=1).mean()),
                "epochs": [{**{key: r[key] for key in ("epoch", "train_soft_ce", "validation_soft_ce", "validation_kl")},
                            "train_online_kl_derived": r["train_soft_ce"] - audit["train_mean_label_entropy"]}
                           for r in epochs]}

    cases = []
    for i, kind in sample_cases.items():
        c = curves[i]
        cases.append({
            "question_id": ids[i], "category": kind, "question": c["question_text"], "type": c["type"],
            "gold_answer": c["gold_answer"], "label_status": labels[i]["label_status"],
            "soft_label": labels[i]["soft_label"], "soft_argmax_k": labels[i]["argmax_soft_label_k"],
            "bert_k": int(KS[b[i]]), "bert_confidence": float(confidence[i]),
            "oracles": {name: int(KS[k[i]]) for name, k in oracle.items()},
            "f1_curve": f[i].tolist(), "context_tokens": t[i].tolist(), "sentence_recall": e[i].tolist(),
            "predictions": c["predictions"], "answer_fallback_curve": c["answer_fallback_curve"],
            "contexts": case_contexts[i],
        })

    results = {"candidate_k": KS.tolist(), "audit": audit, "metrics": metrics, "demands": demands,
               "bert": bert, "label_comparison": comparison, "lambda_sweep": lambda_sweep, "training": training}
    write_json(OUT / "summary.json", results)
    write_json(OUT / "cases.json", cases)
    write_rows(OUT / "test_audit.jsonl", (
        {"question_id": qid, "type": curves[i]["type"], "label_status": labels[i]["label_status"],
         "f1": f[i].tolist(), "context_tokens": t[i].tolist(), "best_f1": float(peak[i]),
         "gain_over_k1": float(gain[i]), "soft_argmax_k": labels[i]["argmax_soft_label_k"],
         "bert_argmax_k": int(KS[b[i]]), "bert_confidence": float(confidence[i]),
         "bert_entropy": float(entropy[i]), "max_cost": int(t[i].max()),
         "oracles": {name: int(KS[k[i]]) for name, k in oracle.items()},
         "api_usage_present_by_k": available[i].tolist(),
         "api_total_tokens_by_k": [int(value) if np.isfinite(value) else None
                                   for value in usage["total_tokens"][i]]}
        for i, qid in enumerate(ids)))
    write_metrics(metrics)
    with (OUT / "confusion.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target", "target_k", "bert_k", "count"])
        for name, counts in cm.items():
            for i, k in enumerate(KS):
                writer.writerows((name, k, j, count) for j, count in zip(KS, counts[i]))
    make_figures(metrics, {"gain": gain, "nonzero_best": nonzero}, f, t, cm)
    from report import write_report

    write_report(OUT, results, cases)
    m = metrics["all_test"]
    print(f"测试集 {n} 题｜k=1 达到正的最优 F1 {int(gain_groups['best_at_k1_positive'].sum())} 题"
          f"｜提高 k 可提升 F1 超过 0.10 有 {int(gain_groups['gain_gt_0.1'].sum())} 题"
          f"｜全零 {int((~nonzero).sum())} 题")
    for method in (FIXED[0], BERT, FALLBACK, *ORACLES):
        print(f"{method:32s} F1={m[method]['f1']:.5f}  Context Tokens={m[method]['context_tokens']:.2f}")
    print(f"质量 Oracle ε=0.10 下，BERT 有 {effects[ORACLES[-1]]['violates_quality']['n']} 题低于质量门槛，"
          f"{effects[ORACLES[-1]]['feasible_extra_cost_no_f1_gain']['n']} 题额外消耗 Token 且 F1 无收益。")
    sv, qv = metrics["valid_test"][SOFT], metrics["valid_test"][ORACLES[-1]]
    print(f"有效标签 {sv['n']} 题｜Scheme D 比质量 Oracle 平均多用 "
          f"{sv['context_tokens'] - qv['context_tokens']:.2f} 个 Context Token，"
          f"其中 {comparison['oracle_k_blocked_by_evidence']['n']} 题的 Oracle k 被证据门槛排除。")
    print(f"保留 k 频数的置换检验 p={audit['saved_eval_diagnostics']['query_specificity']['permutation_p_greater_argmax_f1']:.3f}，"
          f"BERT 逐题分配的回答收益证据较弱。API Total Token 统一在 {int(complete.sum())} 题上比较。")
    print(f"报告 {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
