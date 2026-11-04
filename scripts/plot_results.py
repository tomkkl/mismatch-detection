"""Charts for docs/results-binary.md: training curves, val vs test, errors by pair type.

Usage: python scripts/plot_results.py [--out docs/figures]   (needs the `plots` extra)

Reads only committed files: checkpoints/{binary,binary_5ep,binary_frozen}/metrics.json,
checkpoints/binary/val_predictions.jsonl and checkpoints/binary/eval/full-test/. Writes one SVG
per chart and prints the numbers the write-up quotes.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, PercentFormatter
from sklearn.metrics import precision_recall_curve

from agentlint.eval.metrics import report

CKPT = Path("checkpoints")
RUNS = {  # checkpoint dir -> legend label
    "binary": "fine-tuned, 10 epochs",
    "binary_5ep": "fine-tuned, 5 epochs",
    "binary_frozen": "frozen encoder + head",
}
MAIN = 0.5
MUTATIONS = [  # (mutation key, label, mismatch?)
    ("inject_code", "inject", True),
    ("weaken_description", "weaken", True),
    ("original", "original", False),
    ("declare_effect", "twin", False),
    ("honest_rewrite", "rewrite", False),
]

# Reference palette (dataviz skill), categorical slots in fixed order, plus ink and grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
    "svg.hashsalt": "agentlint",  # stable ids, so regenerating doesn't churn the diff
})


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK_2)
    ax.plot([x], [y], "o", color=color, markersize=5, markeredgecolor="white",
            markeredgewidth=1.5)


def loss_scale(train_path: Path) -> float:
    """Mean pair weight under train.py's pos_weight (honest / mismatch count in train).

    BCEWithLogitsLoss(pos_weight=w) averages w*loss over mismatches and loss over honest pairs
    without dividing by the weights, so the logged loss runs ~1.47x plain BCE. Dividing by the
    mean weight puts it back on the plain-BCE scale: ln 2 = 0.69 for a model that says 0.5.
    """
    labels = [bool(r["mismatch"]) for r in load_rows(train_path)]
    n, n_pos = len(labels), sum(labels)
    pos_weight = (n - n_pos) / n_pos
    return (n - n_pos + pos_weight * n_pos) / n


def training_curves(hist: dict, scale: float, out: Path):
    for name, key, ylabel, title in [
        ("train_loss", "train_loss", f"train loss (weighted BCE / {scale:.2f})",
         "Train loss by epoch, on the plain-BCE scale"),
        ("val_pr_auc", "pr_auc", "val PR-AUC", "Val PR-AUC by epoch"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for (ckpt, label), color in zip(RUNS.items(), SERIES):
            ep = [h["epoch"] for h in hist[ckpt]]
            ys = [h["train_loss"] / scale if key == "train_loss" else h["val_at_0.5"][key]
                  for h in hist[ckpt]]
            ax.plot(ep, ys, color=color, label=label)
            end_label(ax, ep[-1], ys[-1], f"{ys[-1]:.2f}", color)
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.7, 10.9)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if key == "pr_auc":
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(0, 0.8)
            ax.axhline(np.log(2), color=INK_2, linewidth=1, linestyle="--")
            ax.annotate("ln 2 = 0.69: every score 0.5", (10.9, np.log(2)), xytext=(0, 4),
                        textcoords="offset points", ha="right", fontsize=8, color=INK_2)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.legend(loc="lower right" if key == "pr_auc" else "lower left")
        save(fig, out, name)


def headline(val: dict, test: dict, out: Path):
    keys = [("pr_auc", "PR-AUC"), ("precision", "precision"), ("recall", "recall"),
            ("f1", "F1"), ("group_accuracy", "group acc")]
    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (label, m, color) in enumerate([("val", val, SERIES[0]), ("test", test, SERIES[1])]):
        ys = [m[k] for k, _ in keys]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), ys, w, color=color, label=label)
        ax.bar_label(bars, labels=[f"{y:.2f}" for y in ys], padding=2, fontsize=8, color=INK_2)
    ax.set_xticks(x, [k[1] for k in keys])
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", visible=False)
    ax.set_title("Val vs test, fine-tuned 10 epochs (epoch 9), cut-off 0.5", loc="left",
                 fontsize=11, color=INK)
    ax.legend(loc="upper right", ncols=2, bbox_to_anchor=(1, 1.12))
    save(fig, out, "val_vs_test")


def counts(rows: list[dict], mutation: str, threshold: float) -> tuple[int, int]:
    sel = [r for r in rows if (r["mutation"] or "original") == mutation]
    return sum(r["score"] >= threshold for r in sel), len(sel)


def by_pair_type(val_rows: list[dict], test_rows: list[dict], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), width_ratios=[2, 3])
    for ax, want_mismatch, title in [(axes[0], True, "Mismatches caught (recall)"),
                                     (axes[1], False, "Honest pairs flagged (FP rate)")]:
        muts = [m for m in MUTATIONS if m[2] == want_mismatch]
        x = np.arange(len(muts))
        w = 0.36
        for i, (rows, label, color) in enumerate([(val_rows, "val", SERIES[0]),
                                                  (test_rows, "test", SERIES[1])]):
            cs = [counts(rows, m, MAIN) for m, _, _ in muts]
            ys = [k / n for k, n in cs]
            bars = ax.bar(x + (i - 0.5) * (w + 0.02), ys, w, color=color, label=label)
            ax.bar_label(bars, labels=[f"{k}/{n}" for k, n in cs], padding=2, fontsize=8,
                         color=INK_2)
        ax.set_xticks(x, [m[1] for m in muts])
        ax.set_ylim(0, 1.15 if want_mismatch else 0.2)
        ax.set_yticks(np.linspace(0, 1, 6) if want_mismatch else np.linspace(0, 0.2, 5))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[1].legend(loc="upper right")
    fig.suptitle("By pair type, cut-off 0.5", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, out, "by_pair_type")


def score_strip(rows: list[dict], out: Path):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    names = []
    for i, (label, want, color) in enumerate([("honest", False, SERIES[0]),
                                              ("mismatch", True, SERIES[1])]):
        s = np.clip([r["score"] for r in rows if bool(r["label"]) == want], 1e-4, 1 - 1e-4)
        y = i + rng.uniform(-0.25, 0.25, len(s))
        ax.scatter(s, y, s=14, color=color, alpha=0.75, linewidths=0.6, edgecolors="white")
        names.append(f"{label} ({len(s)})")
    ax.axvline(MAIN, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate("cut-off 0.5", (MAIN, 1.45), xytext=(3, 0), textcoords="offset points",
                fontsize=8, color=INK_2, va="center")
    ax.set_xscale("logit")
    ax.set_xlim(3e-4, 1 - 3e-4)
    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.6, 1.65)
    ax.set_yticks([0, 1], names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("model score (logit scale)")
    ax.set_title("Test scores: nearly all at 0 or 1", loc="left", fontsize=11, color=INK)
    save(fig, out, "test_scores")


def pr_curves(val_rows: list[dict], test_rows: list[dict], val_auc: float, test_auc: float,
              out: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for label, rows, auc, color in [("val", val_rows, val_auc, SERIES[0]),
                                    ("test", test_rows, test_auc, SERIES[1])]:
        p, r, _ = precision_recall_curve([bool(x["label"]) for x in rows],
                                         [x["score"] for x in rows])
        ax.step(r, p, where="post", color=color, label=f"{label} (PR-AUC {auc:.3f})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall, fine-tuned 10 epochs", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower left")
    save(fig, out, "pr_curves")


def summary(name: str, rows: list[dict], m: dict, saved: float):
    print(f"== {name}: n {m['n']} | PR-AUC {m['pr_auc']:.3f}")
    for thr in (MAIN, saved):
        r = report(rows, thr)
        print(f"  @{thr:.4g} P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} "
              f"group {r['group_accuracy']:.3f} tp {r['tp']} fp {r['fp']} fn {r['fn']} | "
              + ", ".join(f"{lab} {k}/{n}" for (mut, lab, _), (k, n) in
                          zip(MUTATIONS, [counts(rows, mu, thr) for mu, _, _ in MUTATIONS])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    hist = {c: json.loads((CKPT / c / "metrics.json").read_text())["history"] for c in RUNS}
    saved = json.loads((CKPT / "binary" / "threshold.json").read_text())["threshold"]
    val_rows = load_rows(CKPT / "binary" / "val_predictions.jsonl")
    test_rows = load_rows(CKPT / "binary" / "eval" / "full-test" / "predictions.jsonl")
    test = json.loads((CKPT / "binary" / "eval" / "full-test" / "metrics.json").read_text())
    val_main = report(val_rows, MAIN)

    scale = loss_scale(Path("data/splits/full/train.jsonl"))
    training_curves(hist, scale, out)
    headline(val_main, test["at_0.5"], out)
    by_pair_type(val_rows, test_rows, out)
    score_strip(test_rows, out)
    pr_curves(val_rows, test_rows, val_main["pr_auc"], test["at_0.5"]["pr_auc"], out)

    for c in RUNS:
        best = max(hist[c], key=lambda h: h["val_at_0.5"]["pr_auc"])
        print(f"{c}: loss {hist[c][0]['train_loss']:.3f} -> {hist[c][-1]['train_loss']:.3f} "
              f"(/{scale:.3f}: {hist[c][0]['train_loss'] / scale:.3f} -> "
              f"{hist[c][-1]['train_loss'] / scale:.3f}) | "
              f"best epoch {best['epoch']} val PR-AUC {best['val_at_0.5']['pr_auc']:.3f} | "
              "by epoch " + " ".join(f"{h['val_at_0.5']['pr_auc']:.3f}" for h in hist[c]))
    summary("val", val_rows, val_main, saved)
    summary("test", test_rows, test["at_0.5"], saved)
    print(f"wrote {len(list(out.glob('*.svg')))} charts to {out}")


if __name__ == "__main__":
    main()
