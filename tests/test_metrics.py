import pytest

from agentlint.eval.metrics import (best_f1_threshold, binary_metrics, breakdown, chunk_bucket,
                                    group_accuracy, report)


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
    _row("b", False, 0.7, 3, mutation="declare_effect"),
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
    full = report(ROWS, 0.5)["breakdowns"]
    assert full["mutation"] == {
        "inject_code": {"precision": 1.0, "recall": 1.0, "n": 2, "n_mismatch": 2},
        "weaken_description": {"precision": 0.0, "recall": 0.0, "n": 1, "n_mismatch": 1},
    }
    assert set(full["effect"]) == {"destroy", "spend", "modify"}
    assert set(full["position"]) == {"start", "end"}
