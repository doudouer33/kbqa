"""Materialize Scheme B, maximum-F1 hard labels for the frozen HotpotQA split."""

import csv

import numpy as np

from analyze_labels import select_k
from training_common import CLASSES, HEAD_CLASSES, entrypoint, label_counts, load_data, write_jsonl
from prepare_data import provenance, require
from analysis.hotpot_formal_runner.storage import atomic_json  # noqa: E402


def build(paths):
    datasets, sources = load_data(paths, ("train", "val", "test"))
    labels, reports = [], {}
    for split, (rows, cohort) in datasets.items():
        ks = select_k(cohort.f1, cohort.tokens, CLASSES, paths.selection["tie_atol"])
        zero = cohort.f1.max(axis=1) == 0
        require(np.all(ks[zero] == 0), f"All-zero questions failed to select k=0 in {split}")
        records = []
        for row, k, all_zero in zip(rows, ks.tolist(), zero.tolist()):
            records.append({"qid": row["qid"], "question": row["question"], "split": split, "optimal_k": k,
                            "stage1_label": int(k == 10), "stage2_label": k if k != 10 else None,
                            "flat5_label": CLASSES.index(k), "all_zero_f1": all_zero})
        labels.extend(records)
        if split in ("train", "val"):
            reports[split] = {"sample_count": len(records), "all_zero_f1_count": int(zero.sum()),
                              "class_counts": {head: label_counts(records, head).tolist() for head in HEAD_CLASSES}}
    oracle = paths.label_analysis / "label_distribution.csv"
    with oracle.open(encoding="utf-8", newline="") as stream:
        reference = {(r["split"], int(r["label"])): int(r["count"]) for r in csv.DictReader(stream)
                     if r["scheme"] == "B" and r["objective"] == "max_f1" and r["stage"] == "final_k"
                     and r["group_by"] == "overall" and r["population"] == "all"}
    for split, report in reports.items():
        expected = [reference[split, k] for k in CLASSES]
        require(report["class_counts"] == {"flat5": expected, "stage1": [sum(expected[:4]), expected[4]],
                                           "stage2": expected[:4]}, f"Labels differ from frozen Oracle counts for {split}")
    write_jsonl(paths.labels, labels)
    atomic_json(paths.training / "label_summary.json", {"sources": {**sources, "oracle_label_distribution": provenance(oracle),
                "labels": provenance(paths.labels)}, "scheme": "B", "objective": "max_f1", "candidates": CLASSES,
                "tie_atol": paths.selection["tie_atol"], "total_labels": len(labels), "splits": reports})
    print("Labels verified against Oracle-Candidate:", reports, flush=True)


if __name__ == "__main__":
    entrypoint(build, __doc__)
