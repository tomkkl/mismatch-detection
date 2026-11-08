import pytest

from agentlint.eval.metrics import (best_f1_threshold, binary_metrics, bootstrap_pr_auc, breakdown,
                                    chunk_bucket, group_accuracy, report)


def _row(source: str, label: bool, score: float, n_chunks: int = 1, effect: str | None = None,
         mutation: str | None = None, position: str | None = None) -> dict:
    return {"source_id": source, "label": label, "score": score, "n_chunks": n_chunks,
            "undeclared": [effect] if effect else [], "mutation": mutation, "position": position}


ROWS = [
    # tool a: all right
    _row("a", False, 0.1), _row("a", True, 0.9, 2, "destroy", "inject_code", "start"),
    _row("a", False, 0.2, 2, mutation="declare_effect"),
    # tool b: mutant caught, but the twin is wrongly flagged -> whole group wrong
    _row("b", True, 0.8, 3, "spend", "inject_code", "end"),
    _row("b", False, 0.7, 3, mutation="declare_effect", position="end"),
    # tool c: weakened description missed
    _row("c", False, 0.3), _row("c", True, 0.4, 1, "modify", "weaken_description"),
]


def test_binary_metrics():
    m = binary_metrics([r["label"] for r in ROWS], [r["score"] for r in ROWS], 0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert 0 < m["pr_auc"] <= 1


def test_best_f1_threshold():
    labels, scores = [False, False, True, True], [0.1, 0.3, 0.35, 0.9]
    t = best_f1_threshold(labels, scores)
    assert binary_metrics(labels, scores, t)["f1"] == 1.0


def test_group_accuracy_needs_every_pair_right():
    assert group_accuracy(ROWS, 0.5) == pytest.approx(1 / 3)


def test_breakdowns():
    chunks = breakdown(ROWS, chunk_bucket, 0.5)
    assert {k: v["n"] for k, v in chunks.items()} == {"1": 3, "2": 2, "3+": 2}
    assert chunks["3+"]["precision"] == 0.5
    assert chunks["3+"]["fp_rate"] == 1.0 and chunks["2"]["fp_rate"] == 0.0
    full = report(ROWS, 0.5)["breakdowns"]
    assert full["mutation"] == {
        "declare_effect": {"precision": None, "recall": None, "fp_rate": 0.5,
                           "n": 2, "n_mismatch": 0, "n_honest": 2},
        "inject_code": {"precision": None, "recall": 1.0, "fp_rate": None,
                        "n": 2, "n_mismatch": 2, "n_honest": 0},
        "original": {"precision": None, "recall": None, "fp_rate": 0.0,
                     "n": 2, "n_mismatch": 0, "n_honest": 2},
        "weaken_description": {"precision": None, "recall": 0.0, "fp_rate": None,
                               "n": 1, "n_mismatch": 1, "n_honest": 0},
    }
    assert set(full["effect"]) == {"destroy", "spend"}
    assert set(full["effect_weaken"]) == {"modify"}
    assert full["position"]["start"]["n"] == 1 and full["position"]["end"]["fp_rate"] == 1.0


def test_bootstrap_pr_auc_brackets_the_point_estimate():
    lo, hi = bootstrap_pr_auc(ROWS, n=500)
    point = report(ROWS, 0.5)["pr_auc"]
    assert 0 <= lo <= point <= hi <= 1
    assert bootstrap_pr_auc(ROWS, n=500) == (lo, hi)  # seeded


def test_multilabel_report_per_effect_macro_and_top_effect():
    from agentlint.eval.metrics import SCORED_EFFECTS, multilabel_report
    base = {e: 0.01 for e in SCORED_EFFECTS} | {"read": 0.9}  # read is never scored
    def row(pid, undeclared, hot):
        scores = dict(base) | {e: 0.95 for e in hot}
        return {"pair_id": pid, "source_id": pid.split("/")[0], "label": bool(undeclared),
                "undeclared": undeclared, "scores": scores,
                "score": max(scores[e] for e in SCORED_EFFECTS), "n_chunks": 1,
                "mutation": "inject_code" if undeclared else None, "position": None}
    rows = [row("a/m", ["spend"], ["spend"]), row("b/m", ["destroy"], ["modify"]),
            row("c/h", [], []), row("d/h", [], [])]
    m = multilabel_report(rows, 0.5)
    assert set(m["per_effect"]) == {"spend", "destroy"}  # effects with positives only
    assert m["per_effect"]["spend"]["f1"] == 1.0 and m["per_effect"]["destroy"]["recall"] == 0
    assert m["top_effect_right"] == 0.5  # both flagged at pair level, one names the right effect
    assert m["pair"]["recall"] == 1.0 and m["pair"]["fp"] == 0


def test_pos_weights_per_effect():
    from agentlint.model.train import pos_weights
    from agentlint.schema import Effect, ToolPair
    mk = lambda i, und: ToolPair(pair_id=f"p{i}", source_id=f"s{i}", name="f", params={},
                                 description="d", code="x", mismatch=bool(und), undeclared=und)
    pairs = [mk(0, [Effect.SPEND]), mk(1, []), mk(2, []), mk(3, [])]
    w = pos_weights(pairs, "multilabel").tolist()
    assert w[list(Effect).index(Effect.SPEND)] == 3.0 and w[list(Effect).index(Effect.READ)] == 1.0
    assert pos_weights(pairs, "binary").tolist() == [3.0]
