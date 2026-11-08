"""`agentlint scan`: check the agent tools in Python files against their own descriptions.

Finds decorated tools (`@tool`, `@<server>.tool()`, `@function_tool`) with the rung 3
extractor, scores each with a trained checkpoint, and reports a verdict at 0.5. For a
flagged tool it lists the effects that are likely hidden, with file line numbers:
- binary checkpoint: effects the word list sees in the tool's code (`effects.py`) that the
  name, parameters and description never mention;
- multi-label checkpoint: effects the model scores at 0.5 or above.
The model says whether; the word list says where (CLAUDE.md section 5).
"""

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentlint.data.collect import extract_tools
from agentlint.data.verify import interface_text
from agentlint.effects import code_effects, text_mentions
from agentlint.schema import Effect, ToolPair

SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", "site-packages", ".tox"}
MAIN_THRESHOLD = 0.5


@dataclass
class Finding:
    effect: str
    lines: list[int]          # line numbers in the scanned file


@dataclass
class ToolResult:
    file: str
    line: int
    name: str
    framework: str
    score: float | None
    flagged: bool | None
    hidden: list[Finding] = field(default_factory=list)
    note: str = ""            # why a tool could not be scored, or a caveat


def python_files(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_file() and p.suffix == ".py":
            out.append(p)
        elif p.is_dir():
            out += [f for f in sorted(p.rglob("*.py"))
                    if not SKIP_DIRS & set(f.relative_to(p).parts)]
    return out


def find_tools(path: Path) -> tuple[list[ToolPair], list[dict], list[str]]:
    """Tools in one file as ToolPairs (label unknown), their file positions, skip notes."""
    source = path.read_text(encoding="utf-8", errors="replace")
    raws, skipped = extract_tools(source, strict=False)
    pairs = []
    for raw in raws:
        sid = f"{path.as_posix()}:{raw['line']}"
        pairs.append(ToolPair(pair_id=f"{sid}/{raw['name']}", source_id=sid, name=raw["name"],
                              params=raw["params"], description=raw["description"],
                              code=raw["code"], mismatch=False,
                              meta={"framework": raw["framework"]}))
    return pairs, raws, [f"{path.as_posix()}: {s}" for s in skipped]


def file_effect_lines(source: str, spans: list[list[int]]) -> dict[Effect, list[int]]:
    """Word-list effects in the original file, restricted to the tool's own code spans."""
    try:
        fx = code_effects(ast.parse(source))
    except SyntaxError:
        return {}
    inside = lambda n: any(a <= n <= b for a, b in spans)
    return {e: [n for n in lines if inside(n)] for e, lines in fx.items()
            if any(inside(n) for n in lines)}


def undeclared_by_words(pair: ToolPair, effect_lines: dict[Effect, list[int]]) -> list[Finding]:
    """Effects the code has that the interface (name, params, description) never mentions.
    `read` is left out: an undeclared read changes nothing (CLAUDE.md section 5)."""
    iface = interface_text(pair) + " " + pair.description
    return [Finding(e.value, lines) for e, lines in effect_lines.items()
            if e is not Effect.READ and not text_mentions(iface, e)]


class Model:
    """A trained checkpoint (`<dir>/best`, `config.yaml`), as score.py loads it."""

    def __init__(self, checkpoint: str | Path, batch_size: int = 8):
        import yaml
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from agentlint.model.train import pick_device
        checkpoint = Path(checkpoint)
        self.cfg = yaml.safe_load(open(checkpoint / "config.yaml"))
        self.task = self.cfg["task"]
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint / "best")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint / "best").to(self.device)
        self.batch_size = batch_size

    def score(self, pairs: list[ToolPair]) -> dict[str, dict]:
        """{pair_id: prediction row}; pairs whose interface is too long for the model are
        left out (the caller reports them)."""
        from torch.utils.data import DataLoader

        from agentlint.model.dataset import PairDataset, encode_pair
        from agentlint.model.train import predict
        ok = []
        for p in pairs:
            try:
                encode_pair(p, self.tokenizer, self.cfg["max_length"], self.cfg["chunk_stride"])
                ok.append(p)
            except ValueError:
                pass
        if not ok:
            return {}
        ds = PairDataset(ok, self.tokenizer, max_length=self.cfg["max_length"],
                         stride=self.cfg["chunk_stride"], task=self.task)
        loader = DataLoader(ds, batch_size=self.batch_size, collate_fn=ds.collate)
        return {r["pair_id"]: r for r in predict(self.model, loader, {p.pair_id: p for p in ok},
                                                 self.device)}


def scan(paths: list[str], model, threshold: float = MAIN_THRESHOLD) -> tuple[list[ToolResult], list[str]]:
    """Scan files/folders with a loaded `Model` (or anything with `.task` and `.score`)."""
    results, notes = [], []
    for path in python_files(paths):
        pairs, raws, skipped = find_tools(path)
        notes += skipped
        if not pairs:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        rows = model.score(pairs)
        for pair, raw in zip(pairs, raws):
            res = ToolResult(file=path.as_posix(), line=raw["line"], name=pair.name,
                             framework=raw["framework"], score=None, flagged=None)
            row = rows.get(pair.pair_id)
            if row is None:
                res.note = "interface too long for the model (description over ~380 tokens)"
                results.append(res)
                continue
            res.score = round(row["score"], 4)
            res.flagged = row["score"] >= threshold
            lines = file_effect_lines(source, raw["spans"])
            if not pair.description:
                res.note = "no docstring: the tool is checked against its name and parameters only"
            if res.flagged and model.task == "multilabel":
                res.hidden = [Finding(e, lines.get(Effect(e), []))
                              for e, s in sorted(row["scores"].items(), key=lambda x: -x[1])
                              if e != "read" and s >= threshold]
            elif res.flagged:
                res.hidden = undeclared_by_words(pair, lines)
                if not res.hidden:
                    res.note = (res.note + "; " if res.note else "") + \
                        "the word list found no unmentioned effect: read the code"
            results.append(res)
    return results, notes


def format_table(results: list[ToolResult], notes: list[str]) -> str:
    out = []
    flagged = [r for r in results if r.flagged]
    for r in results:
        verdict = "?" if r.flagged is None else ("MISMATCH" if r.flagged else "ok")
        score = "  -  " if r.score is None else f"{r.score:.3f}"
        out.append(f"{verdict:<9} {score}  {r.file}:{r.line}  {r.name}")
        for f in r.hidden:
            where = ", ".join(str(n) for n in f.lines) or "line not found"
            out.append(f"{'':<17}hidden {f.effect}: line {where}")
        if r.note:
            out.append(f"{'':<17}note: {r.note}")
    out.append(f"\n{len(results)} tools, {len(flagged)} flagged as doing more than their "
               f"description says.")
    if notes:
        out.append(f"{len(notes)} skipped: " + "; ".join(notes[:5]) +
                   (" ..." if len(notes) > 5 else ""))
    return "\n".join(out)


def to_json(results: list[ToolResult], notes: list[str]) -> str:
    return json.dumps({"tools": [asdict(r) for r in results], "skipped": notes}, indent=2)
