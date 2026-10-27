"""Metrics per test rung and per effect class.

All functions take prediction rows (dicts with at least `label` and `score`; group and breakdown
functions also read `source_id`, `n_chunks`, `undeclared`, `position`, `mutation`).
Metrics are on the mismatch class; plain accuracy is not reported (~27% mismatches).
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def binary_metrics(labels, scores, threshold: float = 0.5) -> dict:
    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) >= threshold
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pr_auc = float(average_precision_score(y, scores)) if y.any() else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc,
            "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "n": len(y)}


def best_f1_threshold(labels, scores) -> float:
    """The score cut-off with the highest F1 (tune on val only)."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(labels, dtype=bool), scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def group_accuracy(rows: list[dict], threshold: float = 0.5) -> float:
    """Share of source tools where every pair is right (mutant flagged, twin and rewrite cleared)."""
    ok: dict[str, bool] = defaultdict(lambda: True)
    for r in rows:
        ok[r["source_id"]] &= (r["score"] >= threshold) == bool(r["label"])
    return sum(ok.values()) / len(ok)


def chunk_bucket(r: dict) -> str:
    return "3+" if r["n_chunks"] >= 3 else str(r["n_chunks"])


def effect_key(r: dict) -> str | None:
    return r["undeclared"][0] if r["undeclared"] else None


BREAKDOWNS = {
    "chunks": chunk_bucket,                                    # precision matters here: leak check
    "effect": effect_key,                                      # recall per hidden effect
    "position": lambda r: r.get("position") if r["label"] else None,
    "mutation": lambda r: r["mutation"] if r["label"] else None,  # inject vs weaken
}


def breakdown(rows: list[dict], key, threshold: float = 0.5) -> dict:
    """Precision / recall / counts per group; rows whose key is None are left out."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = key(r)
        if k is not None:
            groups[k].append(r)
    out = {}
    for k in sorted(groups):
        g = groups[k]
        m = binary_metrics([r["label"] for r in g], [r["score"] for r in g], threshold)
        out[k] = {"precision": m["precision"], "recall": m["recall"], "n": m["n"],
                  "n_mismatch": sum(bool(r["label"]) for r in g)}
    return out


def report(rows: list[dict], threshold: float) -> dict:
    """Everything section 5 asks for, at one threshold."""
    labels, scores = [r["label"] for r in rows], [r["score"] for r in rows]
    return {
        **binary_metrics(labels, scores, threshold),
        "group_accuracy": group_accuracy(rows, threshold),
        "breakdowns": {name: breakdown(rows, key, threshold) for name, key in BREAKDOWNS.items()},
    }
