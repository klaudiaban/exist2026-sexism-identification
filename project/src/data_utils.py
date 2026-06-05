"""
Load JSON datasets and generate hard / soft labels for every task.
"""
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List

from config import (
    MEMES_TRAIN_JSON, MEMES_TEST_JSON,
    TASK1_LABELS, TASK2_LABELS, TASK3_LABELS,
)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Hard label helpers ───────────────────────────────────────────────────────

def hard_binary(votes: List[str]) -> str:
    """Majority vote for YES/NO."""
    c = Counter(votes)
    return c.most_common(1)[0][0]


def hard_type(votes: List[str]) -> str:
    """Majority vote for DIRECT/JUDGEMENTAL/NO (raw '-' treated as 'NO')."""
    mapped = ["NO" if v == "-" else v for v in votes]
    c = Counter(mapped)
    return c.most_common(1)[0][0]


def hard_categories(per_annotator: List[List[str]]) -> List[str]:
    """
    Multi-label majority vote.
    Each annotator supplies a list of categories (or ['-'] = non-sexist).
    Returns all categories voted by >50 % of annotators.
    If no category passes threshold, falls back to the top-1 category.
    """
    n = len(per_annotator)
    counts: Counter = Counter()
    for ann_labels in per_annotator:
        for lbl in ann_labels:
            lbl = "NO" if lbl == "-" else lbl
            counts[lbl] += 1

    majority = [lbl for lbl, cnt in counts.items() if cnt / n > 0.5]
    if majority:
        return majority
    return [counts.most_common(1)[0][0]]


# ─── Soft label helpers ───────────────────────────────────────────────────────

def soft_binary(votes: List[str]) -> dict:
    n = len(votes)
    c = Counter(votes)
    return {"YES": round(c.get("YES", 0) / n, 6),
            "NO":  round(c.get("NO",  0) / n, 6)}


def soft_type(votes: List[str]) -> dict:
    n = len(votes)
    mapped = ["NO" if v == "-" else v for v in votes]
    c = Counter(mapped)
    return {lbl: round(c.get(lbl, 0) / n, 6) for lbl in TASK2_LABELS}


def soft_categories(per_annotator: List[List[str]]) -> dict:
    """Per-label fraction (independent binary probs; sum can exceed 1)."""
    n = len(per_annotator)
    counts: Counter = Counter()
    for ann_labels in per_annotator:
        for lbl in ann_labels:
            lbl = "NO" if lbl == "-" else lbl
            counts[lbl] += 1
    return {lbl: round(counts.get(lbl, 0) / n, 6) for lbl in TASK3_LABELS}


# ─── Dataset records ──────────────────────────────────────────────────────────

def build_memes_records(data: dict, is_test: bool = False) -> List[dict]:
    """
    Returns a list of dicts with all info needed for feature extraction + labels.
    """
    records = []
    for sample_id, v in data.items():
        rec = {
            "id":         sample_id,
            "lang":       v["lang"],
            "text":       v.get("text", "") or "",
            "image_path": str(Path(v["path_memes"])),
            "sensorial":  v.get("sensorial", {}),
            "split":      v.get("split", ""),
        }
        if not is_test:
            rec["hard_t21"] = hard_binary(v["labels_task2_1"])
            rec["soft_t21"] = soft_binary(v["labels_task2_1"])
            rec["hard_t22"] = hard_type(v["labels_task2_2"])
            rec["soft_t22"] = soft_type(v["labels_task2_2"])
            rec["hard_t23"] = hard_categories(v["labels_task2_3"])
            rec["soft_t23"] = soft_categories(v["labels_task2_3"])
        records.append(rec)
    return records


# ─── Label matrices for sklearn ───────────────────────────────────────────────

def records_to_label_matrix(records: List[dict], task_key: str) -> list:
    """Extract a flat list of labels from records for a given task key."""
    return [r[task_key] for r in records]


def categories_to_binary_matrix(records: List[dict], task_key: str, label_set: list):
    """
    Convert multi-label hard labels to a 2-D binary matrix (n_samples x n_classes).
    Ignores 'NO' because non-sexist items are handled by task2_1 first.
    """
    import numpy as np
    classes = [l for l in label_set if l != "NO"]
    mat = np.zeros((len(records), len(classes)), dtype=int)
    for i, rec in enumerate(records):
        for lbl in rec[task_key]:
            if lbl in classes:
                mat[i, classes.index(lbl)] = 1
    return mat, classes


if __name__ == "__main__":
    recs = build_memes_records(load_json(MEMES_TRAIN_JSON))
    print(f"Memes train records: {len(recs)}")
    print("Example record keys:", list(recs[0].keys()))
    print("hard_t21:", recs[0]["hard_t21"])
    print("soft_t21:", recs[0]["soft_t21"])
    print("hard_t23:", recs[0]["hard_t23"])
    print("soft_t23:", recs[0]["soft_t23"])
