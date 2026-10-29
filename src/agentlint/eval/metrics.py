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


def effect_key(mutation: str):
    """Hidden effect of one mismatch type, so inject and weaken recall aren't mixed."""
    return lambda r: r["undeclared"][0] if r["undeclared"] and r["mutation"] == mutation else None


BREAKDOWNS = {
    "chunks": chunk_bucket,                                    # honest FP rate here: leak check
    "effect": effect_key("inject_code"),                       # recall per hidden effect
    "effect_weaken": effect_key("weaken_description"),
    "position": lambda r: r.get("position"),                   # inject recall + twin FP rate
    "mutation": lambda r: r["mutation"] or "original",         # every pair type
}


def breakdown(rows: list[dict], key, threshold: float = 0.5) -> dict:
    """Counts, recall and honest FP rate per group; rows whose key is None are left out.

    Recall is None for a group with no mismatches, fp_rate is None for a group with no honest
    pairs, and precision is None unless the group has both (it would mean nothing otherwise).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = key(r)
        if k is not None:
            groups[k].append(r)
    out = {}
    for k in sorted(groups):
        g = groups[k]
        m = binary_metrics([r["label"] for r in g], [r["score"] for r in g], threshold)
        n_mismatch = sum(bool(r["label"]) for r in g)
        n_honest = len(g) - n_mismatch
        out[k] = {"precision": m["precision"] if n_honest and n_mismatch else None,
                  "recall": m["recall"] if n_mismatch else None,
                  "fp_rate": m["fp"] / n_honest if n_honest else None,
                  "n": m["n"], "n_mismatch": n_mismatch, "n_honest": n_honest}
    return out


def report(rows: list[dict], threshold: float) -> dict:
    """Everything section 5 asks for, at one threshold."""
    labels, scores = [r["label"] for r in rows], [r["score"] for r in rows]
    return {
        **binary_metrics(labels, scores, threshold),
        "group_accuracy": group_accuracy(rows, threshold),
        "breakdowns": {name: breakdown(rows, key, threshold) for name, key in BREAKDOWNS.items()},
    }
