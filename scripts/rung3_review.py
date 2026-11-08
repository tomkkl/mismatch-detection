"""Rung 3 hand review: data for the review page, and decisions back into tools.jsonl.

  python scripts/rung3_review.py page <out.json>
      candidates.jsonl -> the compact list the review page (docs/review/index.html) loads.
  python scripts/rung3_review.py apply <decisions.json>
      decisions exported from the page's shared store ({source_id: {review, declared, note,
      reviewer, at}}) -> data/raw/rung3/tools.jsonl, with meta.review / meta.declared set.
      Every candidate is written; build.py (`honest_from`) uses only review == "ok".
"""

import json
import sys
from collections import Counter
from pathlib import Path

CANDIDATES = Path("data/raw/rung3/candidates.jsonl")
TOOLS = Path("data/raw/rung3/tools.jsonl")
EFFECTS = {"read", "modify", "destroy", "relocate", "send_outward", "execute", "spend",
           "change_authority"}


def _rows() -> list[dict]:
    return [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _draft(d: dict | None, model: str) -> dict:
    d = d or {}
    if "error" in d:
        return {"error": d["error"], "model": model}
    return {"honest": d.get("honest"), "declared": d.get("declared", []),
            "domain": d.get("domain"), "reason": d.get("reason", ""), "model": model}


def page(out: str) -> None:
    tools = []
    for r in _rows():
        m, s = r["meta"], r["meta"]["source"]
        d2 = m.get("draft2")
        tools.append({
            "id": r["source_id"], "name": r["name"], "fw": m["framework"], "repo": s["repo"],
            "path": s["path"], "url": s["url"], "license": s["license"],
            "desc": r["description"], "params": r["params"], "code": r["code"],
            "d1": _draft(m.get("draft"), "Sonnet 5.5"),
            "d2": _draft(d2, "Opus 5.5") if d2 else None,
        })
    Path(out).write_text(json.dumps(tools, separators=(",", ":")), encoding="utf-8")
    print(f"{len(tools)} tools -> {out}")


def apply(decisions_path: str) -> None:
    decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    rows, counts, bad = _rows(), Counter(), []
    for r in rows:
        d = decisions.get(r["source_id"])
        if not d:
            r["meta"]["review"] = None
            counts["not reviewed"] += 1
            continue
        if d["review"] not in ("ok", "drop") or not set(d.get("declared", [])) <= EFFECTS:
            bad.append(r["source_id"])
            continue
        r["meta"]["review"] = d["review"]
        r["meta"]["review_note"] = d.get("note", "")
        r["meta"]["reviewed_at"] = d.get("at")
        if d["review"] == "ok":
            r["meta"]["declared"] = sorted(set(d["declared"]))
        counts[d["review"]] += 1
    if bad:
        raise SystemExit(f"malformed decisions for {bad}; fix them on the page first")
    TOOLS.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{dict(counts)} -> {TOOLS}")


if __name__ == "__main__":
    {"page": page, "apply": apply}[sys.argv[1]](sys.argv[2])
