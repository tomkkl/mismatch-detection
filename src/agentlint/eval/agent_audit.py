"""Run Agent Audit (v0.19.2) on the same cases as our model and compare coverage.

Each pair becomes one file, `<root>/case_NNNN/<name>_service/tools/<name>.py`, with its
description put back as the tool function's docstring (our stored code has docstrings
stripped; a real tool ships with one). Agent Audit scans the whole root once; a case counts
as caught if it has **any** non-suppressed finding (decided 2026-11-07), at any severity or
tier. Per-rule counts are kept so generic rules (e.g. "no error handling") can be told apart.

Gotchas:
- Agent Audit suppresses findings under test-looking paths (/tests/, /fixtures/, /examples/,
  /samples/, /mocks/ ...). Cases are written under neutral paths only.
- Always include a positive control (eval / shell=True). A silent run without a firing
  control proves nothing; `run` refuses a result where no case fired at all.
"""

import ast
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from agentlint.data.verify import find_function
from agentlint.schema import ToolPair


def with_docstring(pair: ToolPair) -> str:
    """The pair's code with its description as the tool function's docstring."""
    tree = ast.parse(pair.code)
    fn = find_function(tree, pair.name)
    if fn is not None:
        fn.body.insert(0, ast.Expr(ast.Constant(pair.description)))
    return ast.unparse(tree) + "\n"


def write_cases(pairs: list[ToolPair], root: Path) -> dict[str, str]:
    """Write one neutral-path file per pair. Returns {relative file path: pair_id}."""
    if root.exists():
        shutil.rmtree(root)
    cases = {}
    for i, p in enumerate(pairs):
        rel = Path(f"case_{i:04d}") / f"{p.name}_service" / "tools" / f"{p.name}.py"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(with_docstring(p), encoding="utf-8")
        cases[rel.as_posix()] = p.pair_id
    (root / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    return cases


def scan(root: Path, out_json: Path, exe: str = "agent-audit") -> dict:
    """Run Agent Audit on root, every severity and tier, JSON output."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "scan", str(root), "--format", "json", "--severity", "info",
           "--min-tier", "info", "--no-color", "--output", str(out_json)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not out_json.exists():  # exit code is non-zero whenever there are findings
        raise RuntimeError(f"agent-audit wrote no output (exit {proc.returncode}): "
                           f"{proc.stderr[-2000:]}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def findings_by_pair(result: dict, cases: dict[str, str], root: Path) -> dict[str, list[dict]]:
    """{pair_id: [non-suppressed findings]} for every case, including ones with none."""
    root_posix = root.as_posix().rstrip("/") + "/"
    out: dict[str, list[dict]] = {pid: [] for pid in cases.values()}
    for f in result.get("findings", []):
        if f.get("suppressed"):
            continue
        path = f["location"]["file_path"].replace("\\", "/")
        rel = path.split(root_posix, 1)[-1] if root_posix in path else path
        pid = cases.get(rel)
        if pid is None:  # path printed relative to something else: match on the case folder
            pid = next((v for k, v in cases.items() if path.endswith(k)), None)
        if pid is not None:
            out[pid].append({"rule_id": f["rule_id"], "severity": f["severity"],
                             "tier": f.get("tier"), "line": f["location"].get("start_line"),
                             "title": f.get("title")})
    return out


def run(pairs: list[ToolPair], root: Path, out_dir: Path, exe: str = "agent-audit") -> dict:
    """Write cases, scan, and save {pair_id: findings} to out_dir/findings.json."""
    cases = write_cases(pairs, root)
    result = scan(root, out_dir / "agent_audit_raw.json", exe)
    by_pair = findings_by_pair(result, cases, root)
    if not any(by_pair.values()):
        raise RuntimeError("Agent Audit found nothing on any case: no positive control "
                           "fired, so this run proves nothing (check paths and version)")
    (out_dir / "findings.json").write_text(json.dumps(by_pair, indent=1), encoding="utf-8")
    rules = defaultdict(int)
    for fs in by_pair.values():
        for r in {f["rule_id"] for f in fs}:
            rules[r] += 1
    return {"cases": len(cases), "caught": sum(bool(v) for v in by_pair.values()),
            "cases_per_rule": dict(sorted(rules.items())),
            "version": result.get("version")}
