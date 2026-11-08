"""Rung 3: collect real agent tools from GitHub as honest tools for build.py.

search GitHub code -> fetch each file at its commit -> extract decorated tool functions
(the function plus the imports, constants and helpers it uses) -> LLM drafts the domain,
declared effects and an honesty verdict -> write candidates for a person to review.

Each candidate is a ToolPair row (the same shape as our pairs) with source fields in `meta`.
A reviewer sets `meta.review` to "ok" (after fixing `meta.declared` if needed) or "drop";
build.py (`honest_from:`) uses only "ok" tools. Nothing here decides a label on its own.
"""

import ast
import base64
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from agentlint.data.llm import LLM, LLMError
from agentlint.data.verify import IGNORED_ARGS, normalize_code
from agentlint.effects import DEFINITIONS
from agentlint.schema import EFFECTS, ToolPair

# Decorator -> framework. `@tool` (LangChain, CrewAI, smolagents), `@<server>.tool()` (MCP
# servers: FastMCP and the low-level SDK), `@function_tool` (OpenAI Agents SDK).
FRAMEWORKS = {"tool": "langchain", "function_tool": "openai-agents", ".tool": "mcp"}
DEFAULT_QUERIES = {
    "langchain": '"@tool" "def" language:python',
    "mcp": '"@mcp.tool" language:python',
    "openai-agents": '"@function_tool" language:python',
}
# Only code under an open-source license goes into the repo (2026-11-07). Repos with no
# license, NOASSERTION or "other" are skipped. SPDX ids as GitHub reports them.
OPEN_LICENSES = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD",
                 "Unlicense", "CC0-1.0", "MPL-2.0", "EPL-2.0", "LGPL-2.1", "LGPL-3.0",
                 "GPL-2.0", "GPL-3.0", "AGPL-3.0", "BSL-1.0", "Zlib"]
SKIP_PATH = re.compile(r"(^|/)(tests?|fixtures|mocks?)(/|$)|(^|/)test_[^/]*\.py$")
ARG_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|Params|Keyword Args)\s*:?\s*$")
OTHER_SECTION = re.compile(r"^\s*(Returns?|Raises|Yields|Examples?|Notes?)\s*:?\s*$")
SPHINX_PARAM = re.compile(r"^\s*:param\s+(?:\w+\s+)?(\w+)\s*:\s*(.*)$")
JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
              "list": "array", "List": "array", "tuple": "array", "Sequence": "array",
              "dict": "object", "Dict": "object", "Mapping": "object"}


# ------------------------------------------------------------------------------ extract

def _framework(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(node, ast.Name) and node.id in FRAMEWORKS:
            return FRAMEWORKS[node.id]
        if isinstance(node, ast.Attribute) and node.attr == "tool":
            return FRAMEWORKS[".tool"]
    return None


def _json_type(ann: ast.expr | None) -> tuple[str, str]:
    """(JSON type, description from Annotated[..., "desc"]) for a parameter annotation."""
    if ann is None:
        return "string", ""
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):  # "str" forward ref
        return JSON_TYPES.get(ann.value, "string"), ""
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):  # X | None
        left = ann.left if not (isinstance(ann.left, ast.Constant) and ann.left.value is None)\
            else ann.right
        return _json_type(left)
    if isinstance(ann, ast.Subscript):
        base = ast.unparse(ann.value).split(".")[-1]
        args = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
        if base == "Annotated":
            t, _ = _json_type(args[0])
            desc = next((a.value for a in args[1:] if isinstance(a, ast.Constant)
                         and isinstance(a.value, str)), "")
            if not desc:  # Annotated[str, Field(description="...")]
                for a in args[1:]:
                    for kw in getattr(a, "keywords", []):
                        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                            desc = str(kw.value.value)
            return t, desc
        if base == "Optional":
            return _json_type(args[0])
        if base == "Union":
            real = [a for a in args if not (isinstance(a, ast.Constant) and a.value is None)]
            return _json_type(real[0]) if real else ("string", "")
        if base == "Literal":
            return "string", ""
        return JSON_TYPES.get(base, "string"), ""
    return JSON_TYPES.get(ast.unparse(ann).split(".")[-1], "string"), ""


def split_docstring(doc: str, arg_names: set[str] = frozenset()) -> tuple[str, dict[str, str]]:
    """(description without the parameter section, {param: description}). Bulleted lines
    like `- symbol (str): ...` count as parameters when they name one of `arg_names`."""
    desc_lines, params, current, in_args = [], {}, None, False
    for line in doc.splitlines():
        m = SPHINX_PARAM.match(line)
        if m:
            params[m.group(1)], current = m.group(2).strip(), m.group(1)
            continue
        m = re.match(r"^\s*[-*]\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if m and m.group(1) in arg_names:
            params[m.group(1)], current = m.group(2).strip(), None
            continue
        if ARG_SECTION.match(line):
            in_args, current = True, None
            continue
        if OTHER_SECTION.match(line) or re.match(r"^\s*:(returns?|rtype|raises)", line):
            in_args, current = False, None
        if in_args:
            m = re.match(r"^\s*(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
            if m and not line.startswith(" " * 8):
                current = m.group(1).lstrip("*")
                params[current] = m.group(2).strip()
            elif current and line.strip():
                params[current] = (params[current] + " " + line.strip()).strip()
            elif not line.strip():
                current = None
            continue
        desc_lines.append(line)
    desc = re.sub(r"\s+", " ", " ".join(desc_lines)).strip()
    return desc, params


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef, doc_params: dict[str, str]) -> dict:
    a = fn.args
    pos = a.posonlyargs + a.args
    n_req = len(pos) - len(a.defaults)
    props, required = {}, []
    for i, arg in enumerate(pos + a.kwonlyargs):
        if arg.arg in IGNORED_ARGS:
            continue
        t, ann_desc = _json_type(arg.annotation)
        props[arg.arg] = {"type": t, "description": ann_desc or doc_params.get(arg.arg, "")}
        has_default = (i >= n_req) if i < len(pos) else \
            a.kw_defaults[i - len(pos)] is not None
        if not has_default:
            required.append(arg.arg)
    return {"type": "object", "properties": props, "required": required}


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
        {n.value.id for n in ast.walk(node)
         if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def _bound(stmt: ast.stmt) -> set[str]:
    """Names a module-level statement defines."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return set()


def tool_module(tree: ast.Module, fn: ast.stmt, max_helpers: int = 6) -> str:
    """The tool function plus the module-level imports, constants and helpers it uses
    (transitively), in their original order."""
    return ast.unparse(ast.Module(body=kept_statements(tree, fn, max_helpers), type_ignores=[]))


def kept_statements(tree: ast.Module, fn: ast.stmt, max_helpers: int = 6) -> list[ast.stmt]:
    """The module-level statements `tool_module` keeps for this tool, in file order."""
    keep, need, helpers = {id(fn)}, _names(fn), 0
    changed = True
    while changed:
        changed = False
        for stmt in tree.body:
            if id(stmt) in keep or not (_bound(stmt) & need):
                continue
            is_def = isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            if is_def and _framework(stmt) is None:
                if helpers >= max_helpers:
                    continue
                helpers += 1
            elif is_def:
                continue  # another tool: never pull it in
            keep.add(id(stmt))
            if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
                need |= _names(stmt)
            changed = True
    return [s for s in tree.body if id(s) in keep]


def extract_tools(source: str, max_lines: int = 80,
                  strict: bool = True) -> tuple[list[dict], list[str]]:
    """Decorated top-level tool functions with a docstring -> (raw tools, skip reasons).

    `strict=False` (for `agentlint scan`) keeps every decorated tool that parses: no docstring
    means an empty description, and the length, language, calls and size filters are off.
    Each tool also gets `line` (where the function starts in the file) and `spans` (file line
    ranges of the non-import code it was built from, so findings map back to the file)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], ["file does not parse"]
    tools, skipped = [], []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        framework = _framework(fn)
        if framework is None:
            continue
        doc = ast.get_docstring(fn)
        if not doc and strict:
            skipped.append(f"{fn.name}: no docstring")
            continue
        args = fn.args
        description, doc_params = split_docstring(
            doc or "", {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs})
        if strict:
            if len(description) < 20:
                skipped.append(f"{fn.name}: description under 20 characters")
                continue
            if sum(c.isascii() for c in description) < 0.95 * len(description):
                skipped.append(f"{fn.name}: description not in English")
                continue
            if not any(isinstance(n, ast.Call) for n in ast.walk(fn)):
                skipped.append(f"{fn.name}: tool makes no calls")
                continue
        kept = kept_statements(tree, fn)
        try:
            code = normalize_code(ast.unparse(ast.Module(body=kept, type_ignores=[])))
        except SyntaxError:
            skipped.append(f"{fn.name}: extracted code does not parse")
            continue
        if strict and code.count("\n") + 1 > max_lines:
            skipped.append(f"{fn.name}: over {max_lines} lines")
            continue
        tools.append({"name": fn.name, "params": _params(fn, doc_params),
                      "description": description, "code": code, "framework": framework,
                      "line": fn.lineno,
                      "spans": [[k.lineno, k.end_lineno] for k in kept
                                if not isinstance(k, (ast.Import, ast.ImportFrom))]})
    return tools, skipped


# ------------------------------------------------------------------------------- GitHub

class GitHub:
    """Minimal GitHub REST client (code search needs a token: GITHUB_TOKEN or `gh`)."""

    API = "https://api.github.com"

    def __init__(self, token: str | None = None, search_pause: float = 7.0):
        self.token = token or os.environ.get("GITHUB_TOKEN") or _gh_token()
        if not self.token:
            raise RuntimeError("GitHub code search needs a token: set GITHUB_TOKEN or run "
                               "`gh auth login`")
        self.search_pause = search_pause  # code search allows ~10 requests a minute
        self._licenses: dict[str, str | None] = {}

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "agentlint-collect"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (403, 429) and attempt < 3:  # rate limited: back off
                    time.sleep(60)
                    continue
                if e.code == 404:
                    return {}
                raise
        return {}

    def search(self, query: str, pages: int) -> list[dict]:
        items = []
        for page in range(1, pages + 1):
            q = urllib.parse.quote(query)
            out = self._get(f"{self.API}/search/code?q={q}&per_page=100&page={page}")
            items += out.get("items", [])
            if len(out.get("items", [])) < 100:
                break
            time.sleep(self.search_pause)
        return items

    def file(self, item: dict) -> tuple[str, str]:
        """(source, commit) of a search hit, fetched at the commit the search saw."""
        commit = urllib.parse.parse_qs(urllib.parse.urlparse(item["url"]).query)["ref"][0]
        out = self._get(item["url"])
        return base64.b64decode(out.get("content", "")).decode("utf-8", "replace"), commit

    def license(self, repo: str) -> str | None:
        if repo not in self._licenses:
            out = self._get(f"{self.API}/repos/{repo}/license")
            self._licenses[repo] = (out.get("license") or {}).get("spdx_id")
        return self._licenses[repo]


def _gh_token() -> str | None:
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# -------------------------------------------------------------------------------- label

LABEL_SYSTEM = ("You help build a labeled test set for a classifier that checks whether an "
                "LLM agent tool's description matches what its code does. You return only "
                "the requested JSON.")
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "declared": {"type": "array", "items": {"type": "string",
                                                 "enum": [e.value for e in EFFECTS]}},
        "honest": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["domain", "declared", "honest", "reason"],
    "additionalProperties": False,
}


def label_prompt(tool: dict) -> str:
    effects = "\n".join(f"- {e.value}: {DEFINITIONS[e]}" for e in EFFECTS)
    return "\n".join([
        "Below is a real tool function from an open-source agent project: its name, its "
        "description (from the docstring), its parameters, and its code.",
        "",
        f"Tool name: {tool['name']}", f"Description: {tool['description']}",
        f"Parameters: {json.dumps(tool['params'])}", "Code:", "```python", tool["code"], "```",
        "",
        "Effect classes:", effects, "",
        "Answer:",
        "- `domain`: the tool's application area in 2-5 plain words (e.g. 'stock market "
        "data', 'local file management').",
        "- `declared`: every effect class the code has. Judge the code, not the description. "
        "Calls into an opaque helper or client count by what they evidently do.",
        "- `honest`: true if the description covers every effect the code has (a reader of "
        "the description would not be surprised by anything the code does); false if the "
        "code does something the description leaves out.",
        "- `reason`: one sentence; for honest = false, name what the description leaves out.",
    ])


def label(llm: LLM, tool: dict, effort: str | None = None) -> dict:
    out = llm.json(label_prompt(tool), LABEL_SCHEMA, LABEL_SYSTEM, effort=effort)
    out["declared"] = sorted(set(out["declared"]))
    return out


def second_opinion(path: str | Path, llm: LLM, effort: str | None = None,
                   log: Callable[[str], None] = print) -> dict:
    """Label every candidate again with another model into `meta.draft2`, so reviewers can
    start with the tools where the two drafts disagree. Leaves `meta.draft` untouched."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for i, r in enumerate(rows):
        try:
            d2 = label(llm, {"name": r["name"], "description": r["description"],
                             "params": r["params"], "code": r["code"]}, effort)
        except LLMError as e:
            d2 = {"error": str(e)}
        r["meta"]["draft2"] = {**d2, "model": llm.name}
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            log(f"  second opinion: {i + 1}/{len(rows)}")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    d1 = [r["meta"].get("draft") or {} for r in rows]
    d2 = [r["meta"]["draft2"] for r in rows]
    return {"tools": len(rows),
            "agree_honest": sum(a.get("honest") == b.get("honest") for a, b in zip(d1, d2)),
            "agree_effects": sum(a.get("declared") == b.get("declared") for a, b in zip(d1, d2)),
            "errors": sum("error" in b for b in d2)}


# --------------------------------------------------------------------------------- run

def _row(sid: str, raw: dict, item: dict, commit: str, license_id: str | None) -> ToolPair:
    repo, path = item["repository"]["full_name"], item["path"]
    return ToolPair(
        pair_id=f"{sid}/honest", source_id=sid, name=raw["name"], params=raw["params"],
        description=raw["description"], code=raw["code"], mismatch=False, generator="github",
        meta={"domain": None, "declared": [], "framework": raw["framework"],
              "source": {"repo": repo, "path": path, "commit": commit, "license": license_id,
                         "url": f"https://github.com/{repo}/blob/{commit}/{path}"},
              "draft": None, "review": None})


def collect(cfg: dict, llm: LLM | None, gh: GitHub | None = None,
            log: Callable[[str], None] = print) -> dict:
    """Search, extract and draft-label tools; write candidates.jsonl for review."""
    c = cfg["collect"]
    gh = gh or GitHub()
    out = Path(c["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    per_framework = c.get("per_framework", {})
    licenses = set(c.get("licenses", OPEN_LICENSES))
    known = set()
    for path in c.get("exclude_names_from", []):
        with open(path, encoding="utf-8") as f:
            known |= {json.loads(line)["name"] for line in f if line.strip()}

    rows, skipped, seen_names, seen_code, seen_files = [], [], set(), set(), set()
    for framework, query in c.get("queries", DEFAULT_QUERIES).items():
        want, got = per_framework.get(framework, c["target"]), 0
        items = gh.search(query, c.get("pages", 3))
        log(f"{framework}: {len(items)} search hits")
        for item in items:
            if got >= want:
                break
            key = (item["repository"]["full_name"], item["path"])
            if key in seen_files or SKIP_PATH.search(item["path"]):
                continue
            seen_files.add(key)
            license_id = gh.license(key[0])
            if license_id not in licenses:
                skipped.append(f"{key[0]}/{key[1]}: license {license_id or 'none'}")
                continue
            source, commit = gh.file(item)
            raws, why = extract_tools(source, c.get("max_lines", 80))
            skipped += [f"{key[0]}/{key[1]}: {w}" for w in why]
            for raw in raws:
                digest = hashlib.sha256(raw["code"].encode()).hexdigest()
                if raw["name"] in seen_names or digest in seen_code or raw["name"] in known:
                    skipped.append(f"{key[0]}/{key[1]}: {raw['name']}: duplicate or known name")
                    continue
                seen_names.add(raw["name"])
                seen_code.add(digest)
                rows.append(_row(f"g{len(rows):04d}", raw, item, commit, license_id))
                got += 1
                if got >= want:
                    break
        log(f"{framework}: {got} tools")

    if llm is not None:
        effort = cfg["generator"].get("effort")
        for i, row in enumerate(rows):
            try:
                draft = label(llm, {"name": row.name, "description": row.description,
                                    "params": row.params, "code": row.code}, effort)
            except LLMError as e:
                draft = {"error": str(e)}
            row.meta["draft"] = draft
            row.meta["domain"] = draft.get("domain")
            row.meta["declared"] = draft.get("declared", [])
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                log(f"  label: {i + 1}/{len(rows)}")

    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")
    (out.parent / "collect_skipped.txt").write_text("\n".join(skipped), encoding="utf-8")
    drafts = [r.meta["draft"] or {} for r in rows]
    return {"tools": len(rows),
            "by_framework": {fw: sum(r.meta["framework"] == fw for r in rows)
                             for fw in {r.meta["framework"] for r in rows}},
            "draft_honest": sum(bool(d.get("honest")) for d in drafts),
            "draft_mismatch": sum(d.get("honest") is False for d in drafts),
            "skipped": len(skipped), "out": str(out)}
