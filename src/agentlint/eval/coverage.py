"""Coverage comparison: what Agent Audit catches, what our model catches, and the union.

Cases are the rung 1 test pairs (capability mismatches + honest pairs) and the conventional
bug set (data/bugs.py). Agent Audit "catches" a case with any non-suppressed finding; our
model flags a case at the main 0.5 cut-off. Reported per group, so the write-up can say
which kinds each tool covers and how often each raises a false alarm on honest pairs.
"""

import json
from collections import Counter
from pathlib import Path

MAIN_THRESHOLD = 0.5


def group(pair: dict) -> str:
    bug = (pair.get("meta") or {}).get("bug_type")
    if bug:
        return f"bug: {bug}"
    if pair["mismatch"]:
        return "mismatch: inject" if pair["mutation"] == "inject_code" else "mismatch: weaken"
    return f"honest: {pair['mutation'] or 'original'}"


def _jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def report(pairs: list[dict], findings: dict[str, list[dict]], scores: dict[str, float],
           threshold: float = MAIN_THRESHOLD) -> dict:
    groups: dict[str, list[dict]] = {}
    for p in pairs:
        groups.setdefault(group(p), []).append(p)
    out = {}
    for name in sorted(groups):
        ps = groups[name]
        aa = [bool(findings.get(p["pair_id"])) for p in ps]
        model = [scores.get(p["pair_id"], 0.0) >= threshold for p in ps]
        missing = sum(p["pair_id"] not in scores for p in ps)
        rules = Counter(r for p in ps for r in {f["rule_id"] for f in findings.get(p["pair_id"], [])})
        n = len(ps)
        out[name] = {
            "n": n,
            "agent_audit": sum(aa), "model": sum(model),
            "either": sum(a or m for a, m in zip(aa, model)),
            "both": sum(a and m for a, m in zip(aa, model)),
            "agent_audit_rate": round(sum(aa) / n, 3), "model_rate": round(sum(model) / n, 3),
            "either_rate": round(sum(a or m for a, m in zip(aa, model)) / n, 3),
            "top_rules": dict(rules.most_common(5)),
            "unscored_by_model": missing,
        }
    return out


def markdown(table: dict) -> str:
    lines = ["| Group | n | Agent Audit | Model @0.5 | Either | Both | Top Agent Audit rules |",
             "|---|---|---|---|---|---|---|"]
    for name, r in table.items():
        rules = ", ".join(f"{k} ({v})" for k, v in r["top_rules"].items()) or "none"
        unscored = r["unscored_by_model"] == r["n"]
        model = "not scored" if unscored else f"{r['model']} ({r['model_rate']:.0%})"
        either = "-" if unscored else r["either"]
        both = "-" if unscored else r["both"]
        lines.append(f"| {name} | {r['n']} | {r['agent_audit']} ({r['agent_audit_rate']:.0%}) "
                     f"| {model} | {either} | {both} | {rules} |")
    return "\n".join(lines) + "\n"


def run(cfg: dict) -> dict:
    a = cfg["audit"]
    pairs = [p for path in a["pairs"] for p in _jsonl(path)]
    findings = json.loads((Path(a["out"]) / "findings.json").read_text(encoding="utf-8"))
    scores = {r["pair_id"]: r["score"] for path in cfg["model_predictions"]
              if Path(path).exists() for r in _jsonl(path)}
    table = report(pairs, findings, scores)
    out = Path(a["out"])
    (out / "coverage.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (out / "coverage.md").write_text(markdown(table), encoding="utf-8")
    return table
