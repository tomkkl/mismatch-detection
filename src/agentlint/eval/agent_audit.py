"""Run Agent Audit (AA) on a test set, once, and score it like our model.

Design: `docs/agent-audit-comparison.md` (sections 5-7). Steps:

1. Wrap each pair as one small file, `probes/aa_run/<tag>/<pair_id, / -> __>/service/tools/ops.py`:
   the pair's code, with `@tool` added if the function has no tool decorator, and the
   description as the function's docstring. Plus one positive-control file.
2. One `agent-audit scan <dir> --format json --min-tier suppressed` call for the whole set.
3. Findings -> per-pair verdicts (four rules fixed in advance), rows in our predictions.jsonl
   schema, the same metrics as our model (`metrics.report`), a code-identity check and, when
   our model's predictions are given, a coverage table.

Writes `runs/agent_audit/<tag>/{findings.json, predictions_<rule>.jsonl, metrics.json}` and
refuses to overwrite a result: each set is scanned once.
"""

import ast
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_curve

from agentlint.eval.metrics import BREAKDOWNS, binary_metrics, bootstrap_pr_auc, breakdown, group_accuracy
from agentlint.model.train import load_pairs
from agentlint.schema import ToolPair

AA_VERSION = "0.19.2"
PROBE_ROOT = Path("probes/aa_run")
RESULTS_ROOT = Path("runs/agent_audit")
CONTROL_DIR = "_positive_control"
FILE_IN_PAIR_DIR = Path("service/tools/ops.py")

# Fixed 2026-11-11, before any run (design note section 6).
CAPABILITY_RULES = frozenset({
    "AGENT-001", "AGENT-002", "AGENT-014", "AGENT-017", "AGENT-035",
    "AGENT-037", "AGENT-041", "AGENT-047", "AGENT-053",
})
SEVERITY_RANK = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}
BINARY_RULES = ("main", "capability", "any")
VERDICT_RULES = BINARY_RULES + ("ordinal",)
TOOL_DECORATOR_HINT = "tool"   # AA matches any decorator whose name contains "tool"

CONTROL_SOURCE = '''import subprocess
from langchain_core.tools import tool


@tool
def run_report(user_input: str, cmd: str) -> str:
    """Evaluates an expression and runs a shell command."""
    subprocess.run(cmd, shell=True)
    return str(eval(user_input))
'''


# ---------- 1. wrapping ----------

def pair_dir_name(pair_id: str) -> str:
    return pair_id.replace("/", "__")


def _decorator_name(d: ast.expr) -> str:
    return ast.unparse(d.func if isinstance(d, ast.Call) else d)


def wrap_pair(pair: ToolPair) -> str:
    """The pair's code as an AA-recognizable tool: description as docstring, `@tool` if needed."""
    tree = ast.parse(pair.code)
    fns = [n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == pair.name]
    if len(fns) != 1:
        raise ValueError(f"expected one function named {pair.name!r}, found {len(fns)}")
    fn = fns[0]
    if ast.get_docstring(fn) is not None:
        raise ValueError("function already has a docstring")
    fn.body.insert(0, ast.Expr(ast.Constant(pair.description)))
    if not any(TOOL_DECORATOR_HINT in _decorator_name(d) for d in fn.decorator_list):
        fn.decorator_list.insert(0, ast.Name("tool"))
        imported = any(isinstance(n, ast.ImportFrom) and any(a.name == "tool" and a.asname is None
                                                             for a in n.names)
                       for n in tree.body)
        if not imported:
            tree.body.insert(0, ast.ImportFrom("langchain_core.tools", [ast.alias("tool")], 0))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def write_set(pairs: list[ToolPair], root: Path) -> dict[str, str]:
    """Write one file per pair plus the control under `root` (wiped first). Returns {pair_id: error}."""
    if root.exists():
        shutil.rmtree(root)
    failures = {}
    for p in pairs:
        try:
            src = wrap_pair(p)
        except (SyntaxError, ValueError) as e:
            failures[p.pair_id] = f"{type(e).__name__}: {e}"
            continue
        path = root / pair_dir_name(p.pair_id) / FILE_IN_PAIR_DIR
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src, encoding="utf-8")
    control = root / CONTROL_DIR / FILE_IN_PAIR_DIR
    control.parent.mkdir(parents=True, exist_ok=True)
    control.write_text(CONTROL_SOURCE, encoding="utf-8")
    return failures


# ---------- 2. scanning ----------

def run_scan(root: Path, out_json: Path) -> dict:
    """One AA scan of `root`. AA exits 1 when it has findings, so success = valid JSON output."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "agent_audit", "scan", str(root.resolve()), "--format", "json",
           "--min-tier", "suppressed", "--no-color", "-o", str(out_json.resolve())]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1) or not out_json.exists():
        raise RuntimeError(f"agent-audit failed (exit {proc.returncode}): {proc.stderr[-2000:]}")
    scan = json.loads(out_json.read_text(encoding="utf-8"))
    if scan.get("version") != AA_VERSION:
        raise RuntimeError(f"expected Agent Audit {AA_VERSION}, got {scan.get('version')}")
    return scan


def findings_by_dir(scan: dict, root: Path) -> dict[str, list[dict]]:
    """Group findings by the top-level folder under `root` (one folder per pair)."""
    root = root.resolve()
    out: dict[str, list[dict]] = defaultdict(list)
    for f in scan["findings"]:
        rel = Path(f["location"]["file_path"]).resolve().relative_to(root)
        out[rel.parts[0]].append(f)
    return out


# ---------- 3. verdicts and metrics ----------

def _active(f: dict) -> bool:
    return not f.get("suppressed") and f.get("tier") != "SUPPRESSED"


def verdict_scores(findings: list[dict]) -> dict[str, int]:
    """The four verdict rules of design note section 6, for one pair's findings."""
    active = [f for f in findings if _active(f)]
    return {
        "main": int(any(f["tier"] in ("BLOCK", "WARN") for f in active)),
        "capability": int(any(f["rule_id"] in CAPABILITY_RULES for f in active)),
        "any": int(bool(findings)),
        "ordinal": max((SEVERITY_RANK.get(f["severity"], 0) for f in active), default=0),
    }


def _row(p: ToolPair, findings: list[dict], model_row: dict | None) -> dict:
    row = {
        "pair_id": p.pair_id, "source_id": p.source_id, "label": p.mismatch,
        "undeclared": [e.value for e in p.undeclared], "mutation": p.mutation,
        "position": p.meta.get("position"),
        "conventional_bug": bool(p.meta.get("conventional_bug")), "bug_kind": p.meta.get("bug_kind"),
        "rules": sorted(f["rule_id"] for f in findings),
        "verdicts": verdict_scores(findings),
    }
    if model_row is not None:
        row["n_chunks"] = model_row["n_chunks"]
        row["model_score"] = model_row["score"]
    return row


def _report(rows: list[dict], threshold: float) -> dict:
    """`metrics.report`, minus the chunk breakdown when chunk counts are unknown."""
    has_chunks = all("n_chunks" in r for r in rows)
    return {
        **binary_metrics([r["label"] for r in rows], [r["score"] for r in rows], threshold),
        "group_accuracy": group_accuracy(rows, threshold),
        "breakdowns": {name: breakdown(rows, key, threshold) for name, key in BREAKDOWNS.items()
                       if name != "chunks" or has_chunks},
    }


def with_score(rows: list[dict], rule: str) -> list[dict]:
    return [{**{k: v for k, v in r.items() if k != "verdicts"}, "score": r["verdicts"][rule]}
            for r in rows]


def code_identity(rows: list[dict], code_of: dict[str, str]) -> dict:
    """Groups of pairs with identical code in one tool; how often AA's output differs inside one."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["source_id"], code_of[r["pair_id"]])].append(r)
    multi = [g for g in groups.values() if len(g) > 1]
    out = {"n_groups": len(multi), "n_pairs": sum(len(g) for g in multi),
           "mixed_labels": sum(len({r["label"] for r in g}) > 1 for g in multi),
           "findings_differ": sum(len({tuple(r["rules"]) for r in g}) > 1 for g in multi)}
    for rule in VERDICT_RULES:
        out[f"verdict_differs_{rule}"] = sum(len({r["verdicts"][rule] for r in g}) > 1 for g in multi)
    return out


def coverage(rows: list[dict], rule: str, model_threshold: float = 0.5) -> dict:
    """Who catches what: our model at `model_threshold` vs AA under one binary verdict rule."""
    cats = {
        "mismatches": [r for r in rows if r["label"]],
        "conventional_bugs": [r for r in rows if r["conventional_bug"]],
        "honest_flagged": [r for r in rows if not r["label"] and not r["conventional_bug"]],
    }
    out = {}
    for name, g in cats.items():
        ours = [r["model_score"] >= model_threshold for r in g]
        aa = [bool(r["verdicts"][rule]) for r in g]
        out[name] = {"n": len(g),
                     "ours_only": sum(o and not a for o, a in zip(ours, aa)),
                     "aa_only": sum(a and not o for o, a in zip(ours, aa)),
                     "both": sum(o and a for o, a in zip(ours, aa)),
                     "neither": sum(not o and not a for o, a in zip(ours, aa)),
                     "union": sum(o or a for o, a in zip(ours, aa))}
    return out


def load_model_rows(paths: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["pair_id"] in out:
                raise ValueError(f"pair {r['pair_id']} appears in more than one model predictions file")
            out[r["pair_id"]] = r
    return out


def aa_score(pairs_path: str, tag: str, model_preds: list[str] | None = None, force: bool = False,
             limit: int | None = None, probe_root: Path = PROBE_ROOT,
             results_root: Path = RESULTS_ROOT) -> dict:
    out_dir = results_root / tag
    if (out_dir / "metrics.json").exists() and not force:
        raise FileExistsError(f"{out_dir} already has results; each set is scanned once "
                              "(pass --force only to redo a run that failed or was wrong)")
    pairs = load_pairs(Path(pairs_path), limit)
    if len({p.pair_id for p in pairs}) != len(pairs):
        raise ValueError("duplicate pair_id in the pairs file")
    model_rows = load_model_rows(model_preds) if model_preds else {}
    if model_preds:
        missing = [p.pair_id for p in pairs if p.pair_id not in model_rows]
        if missing:
            raise ValueError(f"{len(missing)} pairs have no model prediction, e.g. {missing[:3]}")

    root = probe_root / tag
    failures = write_set(pairs, root)
    scan = run_scan(root, out_dir / "findings.json")
    by_dir = findings_by_dir(scan, root)

    control = by_dir.pop(CONTROL_DIR, [])
    if not any(_active(f) and f["tier"] == "BLOCK" for f in control):
        raise RuntimeError("positive control did not fire; this run is invalid")
    known = {pair_dir_name(p.pair_id) for p in pairs}
    stray = sorted(set(by_dir) - known)
    if stray:
        raise RuntimeError(f"findings outside any pair folder: {stray[:5]}")
    n_files = len(pairs) - len(failures) + 1
    if scan.get("scanned_files") != n_files:
        failures["_scan"] = f"AA scanned {scan.get('scanned_files')} files, expected {n_files}"

    rows = [_row(p, by_dir.get(pair_dir_name(p.pair_id), []), model_rows.get(p.pair_id))
            for p in pairs if p.pair_id not in failures]
    code_of = {p.pair_id: p.code for p in pairs}

    per_rule = {}
    for rule in BINARY_RULES:
        scored = with_score(rows, rule)
        per_rule[rule] = _report(scored, 0.5)
        if model_rows:
            per_rule[rule]["coverage_vs_model_at_0.5"] = coverage(rows, rule)
    ordinal = with_score(rows, "ordinal")
    labels, scores = [r["label"] for r in ordinal], [r["score"] for r in ordinal]
    prec, rec, thr = precision_recall_curve(labels, scores)
    per_rule["ordinal"] = {"pr_auc": binary_metrics(labels, scores)["pr_auc"],
                           "pr_auc_ci95": list(bootstrap_pr_auc(ordinal)),
                           "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist(),
                                        "thresholds": thr.tolist()}}

    rule_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for rid in set(r["rules"]):
            rule_counts[rid] += 1
    result = {
        "pairs": str(pairs_path), "tag": tag, "n": len(rows), "aa_version": scan["version"],
        "scan_summary": scan.get("summary"), "scanned_files": scan.get("scanned_files"),
        "failures": failures, "control_rules": sorted(f["rule_id"] for f in control),
        "capability_rules": sorted(CAPABILITY_RULES), "model_predictions": model_preds or [],
        "pairs_with_rule": dict(sorted(rule_counts.items())),
        "code_identity": code_identity(rows, code_of), "verdicts": per_rule,
    }
    for rule in VERDICT_RULES:
        with open(out_dir / f"predictions_{rule}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in with_score(rows, rule))
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{pairs_path} | {len(rows)} pairs | Agent Audit {scan['version']} | "
          f"{len(failures)} failures | control: {', '.join(result['control_rules'])}")
    for rule in BINARY_RULES:
        m = per_rule[rule]
        print(f"{rule:10s} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
              f"group acc {m['group_accuracy']:.3f} (tp {m['tp']} fp {m['fp']} fn {m['fn']})")
    ci = per_rule["ordinal"]["pr_auc_ci95"]
    print(f"ordinal    PR-AUC {per_rule['ordinal']['pr_auc']:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})")
    ci_check = result["code_identity"]
    print(f"code identity: {ci_check['n_groups']} groups, findings differ in "
          f"{ci_check['findings_differ']}, main verdict differs in {ci_check['verdict_differs_main']}")
    print(f"saved to {out_dir}")
    return result
