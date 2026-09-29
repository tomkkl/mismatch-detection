"""Normalization and AST label checks.

Every tool's code goes through `normalize_code` (parse, drop docstrings, unparse), so comments
and formatting can never carry the label. Real code must get the same step at scan time.

A pair is kept only if its label is backed by the code:
- injected mutant: a new operation is present in the mutant and absent in the honest version,
  little else changed, and the tool name / params do not reveal the effect;
- honest twin: same code as the mutant, description now declares the effect;
- weakened description: code unchanged, the effect was declared before and is not now.
"""

import ast
import difflib

from agentlint.effects import code_effects, text_mentions
from agentlint.schema import Effect, ToolPair

IGNORED_ARGS = {"self", "cls", "ctx", "context"}


def normalize_code(code: str) -> str:
    """Parse, remove docstrings, and print back. Raises SyntaxError on bad code."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
                if not body and not isinstance(node, ast.Module):
                    body.append(ast.Pass())
    return ast.unparse(tree)


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def function_args(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    return {n for n in names if n not in IGNORED_ARGS}


def interface_text(tool: ToolPair) -> str:
    """Name plus parameter names and descriptions: the parts that must not leak an effect."""
    parts = [tool.name]
    for pname, spec in tool.params.get("properties", {}).items():
        parts += [pname, spec.get("description", "")]
    return " ".join(parts)


def _length_ok(new: str, old: str, bounds: tuple[float, float]) -> bool:
    ratio = len(new) / max(len(old), 1)
    return bounds[0] <= ratio <= bounds[1]


# ---------------------------------------------------------------------------- honest

def verify_honest(tool: ToolPair) -> tuple[bool, str]:
    try:
        tree = ast.parse(tool.code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"
    fn = find_function(tree, tool.name)
    if fn is None:
        return False, "tool function not found"
    declared = set(tool.params.get("properties", {}))
    if function_args(fn) != declared:
        return False, "params do not match function signature"
    if not tool.meta.get("declared"):
        return False, "no declared effects"
    if not any(isinstance(n, ast.Call) for n in ast.walk(fn)):
        return False, "tool makes no calls"

    # Soft signals, recorded for later analysis.
    seen = set(code_effects(tree))
    declared_fx = {Effect(e) for e in tool.meta["declared"]}
    tool.meta["checks"] = {
        "lexicon_missing": sorted(e.value for e in declared_fx - seen),
        "lexicon_extra": sorted(e.value for e in seen - declared_fx),
        "desc_missing": sorted(e.value for e in declared_fx
                               if e != Effect.READ and not text_mentions(tool.description, e)),
    }
    return True, "ok"


# ---------------------------------------------------------------------------- inject

def diff_lines(old: str, new: str) -> tuple[list[int], int]:
    """Return (1-indexed lines added in `new`, number of `old` lines deleted or replaced)."""
    a, b = old.splitlines(), new.splitlines()
    added, changed = [], 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            added += range(j1 + 1, j2 + 1)
        if op in ("delete", "replace"):
            changed += i2 - i1
    return added, changed


def _new_ops(honest_tree: ast.AST, mutant_tree: ast.AST, lines: set[int]) -> list[ast.AST]:
    """Calls or attribute/key assignments on `lines` of the mutant that the honest code lacks."""
    def ops(tree):
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                yield n
            elif isinstance(n, (ast.Assign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in targets):
                    yield n

    known = {ast.dump(n) for n in ops(honest_tree)}
    return [n for n in ops(mutant_tree) if n.lineno in lines and ast.dump(n) not in known]


def verify_inject(honest: ToolPair, mutant: ToolPair, effect: Effect, max_changed: int = 2,
                  max_added: int = 15) -> tuple[bool, str]:
    """Check an injected mutant. On success fills `mutant.injected_lines` and soft checks."""
    try:
        h_tree, m_tree = ast.parse(honest.code), ast.parse(mutant.code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"
    fn = find_function(m_tree, mutant.name)
    if fn is None:
        return False, "tool function not found"
    if function_args(fn) != function_args(find_function(h_tree, honest.name)):
        return False, "signature changed"
    if mutant.description != honest.description or mutant.params != honest.params:
        return False, "interface changed"
    if text_mentions(interface_text(mutant), effect):
        return False, "name/params reveal effect"

    added, changed = diff_lines(honest.code, mutant.code)
    if not added:
        return False, "nothing added"
    if changed > max_changed:
        return False, f"rewrote {changed} existing lines"
    if len(added) > max_added:
        return False, f"added {len(added)} lines"
    new_ops = _new_ops(h_tree, m_tree, set(added))
    if not new_ops:
        return False, "no new operation on added lines"

    mutant.injected_lines = added
    lexicon = code_effects(m_tree).get(effect, [])
    mutant.meta.setdefault("checks", {})["lexicon_agrees"] = bool(set(lexicon) & set(added))
    return True, "ok"


# ---------------------------------------------------------------------------- twin / weaken

def verify_twin(mutant: ToolPair, twin: ToolPair, effect: Effect,
                length_bounds=(0.6, 1.8)) -> tuple[bool, str]:
    """The model quotes the phrase that declares the effect; it must be new to the description.

    The word lexicon is too narrow to require here ("looks up", "mode 444"), so it is recorded
    as a soft signal only.
    """
    if twin.code != mutant.code:
        return False, "code changed"
    phrase = twin.meta.get("declaring_phrase", "").lower()
    if len(phrase) < 3 or phrase not in twin.description.lower():
        return False, "declaring phrase missing from description"
    if phrase in mutant.description.lower():
        return False, "declaring phrase was already in the original description"
    if not _length_ok(twin.description, mutant.description, length_bounds):
        return False, "description length out of range"
    twin.meta["checks"] = {"lexicon_declares": text_mentions(phrase, effect)}
    return True, "ok"


def verify_length_match(mutant: ToolPair, matched: ToolPair, twin: ToolPair, effect: Effect,
                        tol: float = 0.15) -> tuple[bool, str]:
    """The mismatch description, rewritten to its twin's length, must still not declare the
    added effect."""
    if matched.code != mutant.code:
        return False, "code changed"
    if text_mentions(matched.description, effect):
        return False, "rewrite mentions effect"
    phrase = twin.meta.get("declaring_phrase", "").lower()
    if phrase and phrase in matched.description.lower():
        return False, "rewrite contains the twin's declaring phrase"
    if not _length_ok(matched.description, twin.description, (1 - tol, 1 + tol)):
        return False, "rewrite length not matched to twin"
    return True, "ok"


def verify_weaken(honest: ToolPair, weak: ToolPair, effect: Effect,
                  length_bounds=(0.6, 1.8)) -> tuple[bool, str]:
    if weak.code != honest.code:
        return False, "code changed"
    if not text_mentions(honest.description, effect):
        return False, "effect was not declared in the original description"
    if text_mentions(weak.description, effect):
        return False, "description still mentions effect"
    if text_mentions(interface_text(weak), effect):
        return False, "name/params reveal effect"
    if not _length_ok(weak.description, honest.description, length_bounds):
        return False, "description length out of range"
    return True, "ok"