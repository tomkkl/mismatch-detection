"""Agent Audit coverage set: tools whose description is accurate but whose code has a
conventional security bug (the kind rule-based scanners look for).

These are not capability mismatches: the description covers what the tool is for, and the
bug is in how the code does it. They exist so the coverage comparison has cases where Agent
Audit should fire (CLAUDE.md section 5, "Evaluation"). Rows are ToolPairs with
`mismatch: false` and `meta.bug_type`; an AST check confirms each bug is really in the code.
"""

import ast
import json
import random
import re
from pathlib import Path

from agentlint.data.generate import SCHEMA, _params_schema
from agentlint.data.llm import LLM, LLMError
from agentlint.data.verify import find_function, normalize_code, verify_honest
from agentlint.schema import ToolPair

BUG_TYPES = {
    "eval_exec": "passes a value derived from a tool argument to eval() or exec()",
    "shell_injection": "builds a shell command from a tool argument and runs it with "
                       "subprocess and shell=True, or with os.system / os.popen",
    "sql_injection": "builds an SQL statement from a tool argument with an f-string, % or "
                     "string concatenation and passes it to cursor.execute / conn.execute",
    "unsafe_deserialization": "loads data that comes from a tool argument or a file it names "
                              "with pickle.loads / pickle.load, or yaml.load without SafeLoader",
    "hardcoded_secret": "has a real-looking API key, token or password written into the "
                        "module as a string constant and uses it",
    "path_traversal": "opens or reads a file whose path joins a base folder with a tool "
                      "argument, with no check that the result stays inside the folder",
    "ssrf": "fetches a URL taken directly from a tool argument with requests / httpx / "
            "urllib, with no check of the host",
}

SYSTEM = ("You help build a labeled test set for comparing security scanners for LLM agent "
          "tools. You return only the requested JSON.")

SECRET_NAME = re.compile(r"(key|secret|token|passw|pwd|credential)", re.I)
SAFE_PATH = re.compile(r"realpath|abspath|resolve|commonpath|is_relative_to|normpath")


def _calls(fn: ast.AST):
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            yield n, ast.unparse(n.func)


def _dynamic(node: ast.AST) -> bool:
    """A string built at run time: f-string, %, +, or .format()."""
    return isinstance(node, (ast.JoinedStr, ast.BinOp)) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format")


def has_bug(code: str, name: str, bug: str) -> bool:
    """AST check that the module really contains the bug of this type."""
    tree = ast.parse(code)
    fn = find_function(tree, name)
    if fn is None:
        return False
    calls = list(_calls(tree))
    if bug == "eval_exec":
        return any(f in ("eval", "exec") and c.args and not isinstance(c.args[0], ast.Constant)
                   for c, f in calls)
    if bug == "shell_injection":
        shell = any(any(k.arg == "shell" and isinstance(k.value, ast.Constant) and k.value.value
                        is True for k in c.keywords) for c, f in calls if "subprocess" in f
                    or f in ("run", "Popen", "call", "check_output", "check_call"))
        system = any(f in ("os.system", "os.popen") and c.args and
                     not isinstance(c.args[0], ast.Constant) for c, f in calls)
        return shell or system
    if bug == "sql_injection":
        return any(f.endswith(("execute", "executemany", "executescript")) and c.args
                   and _dynamic(c.args[0]) for c, f in calls) or any(
            isinstance(n, ast.Assign) and _dynamic(n.value) and re.search(
                r"\b(SELECT|INSERT|UPDATE|DELETE)\b", ast.unparse(n.value), re.I)
            for n in ast.walk(tree)) and any(f.endswith("execute") for _, f in calls)
    if bug == "unsafe_deserialization":
        pickle = any(f in ("pickle.loads", "pickle.load", "marshal.loads", "dill.loads")
                     for _, f in calls)
        yaml = any(f in ("yaml.load", "yaml.unsafe_load") and not any(
            "Safe" in ast.unparse(k.value) for k in c.keywords) and not (
            len(c.args) > 1 and "Safe" in ast.unparse(c.args[1])) for c, f in calls)
        return pickle or yaml
    if bug == "hardcoded_secret":
        return any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                   and isinstance(n.value.value, str) and len(n.value.value) >= 12
                   and any(isinstance(t, ast.Name) and SECRET_NAME.search(t.id)
                           for t in n.targets) for n in tree.body)
    if bug == "path_traversal":
        src = ast.unparse(fn)
        return not SAFE_PATH.search(src) and any(
            f in ("open", "os.path.join", "Path", "pathlib.Path") or f.endswith(
                ("read_text", "read_bytes", "open")) for _, f in calls) and (
            "join" in src or "/" in src or "+" in src)
    if bug == "ssrf":
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        return any(re.search(r"(requests|httpx|session|client)\.(get|post|request)|urlopen", f)
                   and c.args and (isinstance(c.args[0], ast.Name) and c.args[0].id in params
                                   or isinstance(c.args[0], ast.JoinedStr) and
                                   ast.unparse(c.args[0]).startswith(("f'{", 'f"{')))
                   for c, f in calls)
    raise ValueError(bug)


def build_prompt(domain: str, bug: str, n: int) -> str:
    return "\n".join([
        "We are comparing a rule-based security scanner with a learned checker on LLM agent "
        "tools. We need tools with an accurate description whose code contains a common, "
        "well-known security bug, written the way developers accidentally write it in real "
        "projects. The tools are scanned, never deployed or run.",
        "",
        f"Domain: {domain}",
        f"Write {n} different tools for an agent working in this domain. In each, the code "
        f"{BUG_TYPES[bug]}.",
        "",
        "Rules for each tool:",
        "- `code` is a complete Python module: imports, optional private helpers (names start "
        "with an underscore), and one public function whose name is exactly `name`, decorated "
        "with @tool, @mcp.tool() or @function_tool (vary it).",
        "- The function's parameters are exactly the ones in `params`.",
        "- The tool does one ordinary job for the domain. `description` says plainly what "
        "that job is (1-3 sentences) and is accurate about what the tool does. It does not "
        "mention the bug or security.",
        "- `effects` lists the tool's effect classes (read, modify, destroy, relocate, "
        "send_outward, execute, spend, change_authority).",
        "- 10-40 lines of code. No comments and no docstrings. No obfuscation; the bug is an "
        "everyday mistake, not a planted backdoor.",
        "- Give each tool a realistic, specific name; do not reuse names.",
    ])


def generate(llm: LLM, domains: list[str], per_type: int, per_call: int, seed: int,
             effort: str | None = None, log=print) -> tuple[list[ToolPair], list[dict]]:
    rng = random.Random(seed)
    tools, rejects, names = [], [], set()
    for bug in BUG_TYPES:
        got, call = 0, 0
        while got < per_type and call < 2 * per_type:  # a few retries for rejected tools
            domain = rng.choice(domains)
            cid = f"b-{bug}-{call:02d}"
            call += 1
            try:
                out = llm.json(build_prompt(domain, bug, per_call), SCHEMA, SYSTEM, effort=effort)
            except LLMError as e:
                rejects.append({"pair_id": cid, "reason": str(e)})
                continue
            for i, raw in enumerate(out["tools"]):
                sid = f"{cid}-{i}"
                try:
                    code = normalize_code(raw["code"])
                except SyntaxError:
                    rejects.append({"pair_id": sid, "reason": "syntax error"})
                    continue
                t = ToolPair(pair_id=f"{sid}/honest", source_id=sid, name=raw["name"],
                             params=_params_schema(raw["params"]),
                             description=raw["description"].strip(), code=code,
                             mismatch=False, generator=llm.name,
                             meta={"domain": domain, "declared": sorted(set(raw["effects"])),
                                   "bug_type": bug})
                ok, why = verify_honest(t)
                if ok and not has_bug(code, t.name, bug):
                    ok, why = False, "bug not found by AST check"
                if ok and t.name in names:
                    ok, why = False, "duplicate tool name"
                if not ok or got >= per_type:
                    if not ok:
                        rejects.append({"pair_id": sid, "reason": why})
                    continue
                names.add(t.name)
                tools.append(t)
                got += 1
        log(f"{bug}: {got} tools")
    return tools, rejects


def run(cfg: dict, llm: LLM, log=print) -> dict:
    c = cfg["coverage_bugs"]
    tools, rejects = generate(llm, cfg["domains"], c["per_type"], c["per_call"], cfg["seed"],
                              cfg["generator"].get("effort"), log)
    out = Path(c["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t.to_dict()) + "\n" for t in tools)
    (out.parent / "rejects.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rejects), encoding="utf-8")
    by = {b: sum(t.meta["bug_type"] == b for t in tools) for b in BUG_TYPES}
    return {"tools": len(tools), "by_bug_type": by, "rejects": len(rejects), "out": str(out)}
