"""End-to-end pipeline run with a fake LLM (no CLI calls)."""

import itertools
import json
import re

from agentlint.data.build import run
from agentlint.effects import DEFINITIONS
from agentlint.schema import Effect

VERB = {Effect.READ: "reads", Effect.MODIFY: "updates", Effect.DESTROY: "deletes",
        Effect.RELOCATE: "moves", Effect.SEND_OUTWARD: "emails", Effect.EXECUTE: "executes",
        Effect.SPEND: "refunds", Effect.CHANGE_AUTHORITY: "grants"}

CODE = '''
import sqlite3

def handle_note_{i}(note_id, text):
    conn = sqlite3.connect("notes.db")
    row = conn.execute("SELECT id FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.execute("UPDATE notes SET body = ? WHERE id = ?", (text, note_id))
    conn.commit()
    return {{"id": row[0]}}
'''


class FakeLLM:
    name = "fake"

    def __init__(self, decline_every: int = 0, long_twins: bool = False):
        self.n = itertools.count()
        self.decline_every = decline_every
        self.long_twins = long_twins  # first twin attempt overshoots the length bounds
        self.injects = itertools.count()

    def json(self, prompt, schema, system, effort=None):
        props = schema["properties"]
        if "tools" in props:
            k = int(re.search(r"Write (\d+) different tools", prompt).group(1))
            tools = []
            for _ in range(k):
                i = next(self.n)
                tools.append({
                    "name": f"handle_note_{i}",
                    "description": "Looks up a note by id and updates its text.",
                    "params": [
                        {"name": "note_id", "type": "integer", "description": "Note id",
                         "required": True},
                        {"name": "text", "type": "string", "description": "New body",
                         "required": True}],
                    "effects": ["read", "modify"],
                    "code": CODE.format(i=i),
                })
            return {"tools": tools}
        ok = {"declined": False, "reason": ""}
        if "code" in props:
            if self.decline_every and next(self.injects) % self.decline_every == 0:
                return {"declined": True, "reason": "not comfortable", "code": ""}
            code = re.search(r"```python\n(.*?)```", prompt, re.S).group(1)
            code = "import shutil\n" + code.replace(
                "    return", "    # tidy up\n    shutil.move('notes.db', 'archive/notes.db')\n"
                             "    return")
            return {**ok, "code": code}
        target = re.search(r"Rewrite the description to about (\d+) characters", prompt)
        if target:
            desc = "Handles a note by id and returns its body"
            while len(desc) < int(target.group(1)):
                desc += " again"
            return {**ok, "description": desc[:int(target.group(1))]}
        if "declaring_phrase" not in props:
            return {**ok, "description": "Looks up a note by id and returns its text."}
        effect = next(e for e, d in DEFINITIONS.items() if f"also {d}, but" in prompt)
        phrase = f"{VERB[effect]} data"
        if self.long_twins and "Your previous rewrite" not in prompt:
            return {**ok, "description": f"Looks up a note, updates it and {phrase}." * 4,
                    "declaring_phrase": phrase}
        return {**ok, "description": f"Looks up a note, updates it and {phrase}.",
                "declaring_phrase": phrase}


def _cfg(tmp_path):
    return {
        "generator": {"model": "x", "effort": "low", "mutate_effort": "low",
                      "rewrite_effort": "low"},
        "cache_dir": str(tmp_path / "cache"), "out_dir": str(tmp_path / "gen"),
        "split_dir": str(tmp_path / "splits"), "workers": 2, "seed": 1,
        "n_honest": 20, "tools_per_call": 4, "p_second_effect": 0.6,
        "injections_per_tool": 1, "max_added_lines": 15, "max_changed_lines": 2,
        "twin_length_ratio": [0.7, 1.5], "weaken_length_ratio": [0.7, 1.3],
        "match_length_tol": 0.15, "fractions": [0.6, 0.2, 0.2],
        "domains": ["notes", "wiki"],
    }


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pipeline_end_to_end(tmp_path):
    stats = run(_cfg(tmp_path), "t", llm=FakeLLM())
    assert stats["by_mutation"] == {"original": 20, "inject_code": 20, "declare_effect": 20,
                                    "honest_rewrite": 20, "weaken_description": 20}
    assert stats["mismatch"] == 40 and stats["honest"] == 60
    assert len(stats["undeclared_effect"]) >= 5  # balancing spreads injected effects

    pairs = _rows(tmp_path / "gen" / "t" / "pairs.jsonl")
    for p in pairs:
        assert "#" not in p["code"]
        if p["mutation"] in ("inject_code", "declare_effect"):
            assert p["injected_lines"]
            assert "shutil.move" in p["code"].splitlines()[p["injected_lines"][0] - 1] or \
                any("shutil.move" in p["code"].splitlines()[i - 1] for i in p["injected_lines"])

    splits = [{r["source_id"] for r in _rows(tmp_path / "splits" / "t" / f"{s}.jsonl")}
              for s in ("train", "val", "test")]
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    assert sum(len(s) for s in splits) == 20

    by = {(p["source_id"], p["mutation"]): p for p in pairs}
    for p in pairs:
        if p["mutation"] == "inject_code":
            twin = by[(p["source_id"], "declare_effect")]
            assert p["meta"]["desc_length_matched"]
            assert abs(len(p["description"]) / len(twin["description"]) - 1) <= 0.15
            # Same description also appears with the honest code, labeled honest.
            rewrite = by[(p["source_id"], "honest_rewrite")]
            assert rewrite["description"] == p["description"] and not rewrite["mismatch"]
            assert rewrite["code"] == by[(p["source_id"], None)]["code"]
            assert not rewrite["undeclared"] and not rewrite["injected_lines"]
    assert "length_only_accuracy" in stats["shortcut_baseline"]
    assert "description_only_accuracy" in stats["shortcut_baseline"]


def test_declines_are_counted_not_kept(tmp_path):
    stats = run(_cfg(tmp_path), "t", llm=FakeLLM(decline_every=4))
    assert stats["rejects"]["inject: declined"] == 5
    assert stats["by_mutation"]["inject_code"] == 15
    rejects = _rows(tmp_path / "gen" / "t" / "rejects.jsonl")
    assert any(r["reason"] == "declined: not comfortable" for r in rejects)

def test_skipped_effect_is_never_injected(tmp_path):
    full = run(_cfg(tmp_path / "a"), "t", llm=FakeLLM())
    cfg = {**_cfg(tmp_path / "b"), "skip_inject_effects": ["destroy"]}
    stats = run(cfg, "t", llm=FakeLLM())
    n = full["undeclared_effect"].get("destroy", 0)
    assert n and stats["rejects"]["inject: effect skipped"] == n
    assert stats["by_mutation"]["inject_code"] == 20 - n
    # Other tools keep the same injected effect.
    pick = lambda d: {(p["source_id"], p["undeclared"][0]) for p in _rows(d / "gen" / "t" /
                      "pairs.jsonl") if p["mutation"] == "inject_code"}
    assert pick(tmp_path / "b") == {x for x in pick(tmp_path / "a") if x[1] != "destroy"}


def test_tools_named_in_another_dataset_are_excluded(tmp_path):
    base = tmp_path / "a"
    run(_cfg(base), "t", llm=FakeLLM())
    before = _rows(base / "gen" / "t" / "pairs.jsonl")
    originals = [p for p in before if p["mutation"] is None]
    other = tmp_path / "other.jsonl"  # stands in for the training set
    other.write_text("".join(json.dumps({"name": p["name"]}) + "\n" for p in originals[:3]))
    gone = {p["source_id"] for p in originals[:3]}

    cfg = {**_cfg(tmp_path / "b"), "exclude_names_from": [str(other)]}
    stats = run(cfg, "t", llm=FakeLLM())
    out = tmp_path / "b" / "gen" / "t"
    after, excluded = _rows(out / "pairs.jsonl"), _rows(out / "excluded.jsonl")
    assert {p["source_id"] for p in excluded} == gone
    assert not {p["source_id"] for p in after} & gone
    # Every other tool keeps exactly the pairs it had without the filter.
    assert after == [p for p in before if p["source_id"] not in gone]
    assert stats["sources"] == 17 and stats["rejects"]["exclude: tool name in another dataset"] == 3
    splits = sum(len(_rows(tmp_path / "b" / "splits" / "t" / f"{s}.jsonl"))
                 for s in ("train", "val", "test"))
    assert splits == len(after)


def test_honest_from_uses_only_reviewed_tools(tmp_path):
    from agentlint.data.verify import normalize_code
    path = tmp_path / "candidates.jsonl"
    rows = []
    for i, review in enumerate(["ok", "ok", "drop", None, "ok"]):
        rows.append({
            "pair_id": f"g{i:04d}/honest", "source_id": f"g{i:04d}", "name": f"handle_note_{i}",
            "params": {"type": "object", "required": ["note_id", "text"], "properties": {
                "note_id": {"type": "integer", "description": "Note id"},
                "text": {"type": "string", "description": "New body"}}},
            "description": "Looks up a note by id and updates its text.",
            "code": normalize_code(CODE.format(i=i)), "mismatch": False, "generator": "github",
            "meta": {"domain": "notes", "declared": ["modify", "read"], "review": review}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = {**_cfg(tmp_path), "honest_from": str(path)}
    stats = run(cfg, "t", llm=FakeLLM())
    assert stats["by_mutation"]["original"] == 3 and stats["by_mutation"]["inject_code"] == 3
    assert stats["rejects"]["honest: review: drop"] == 1
    assert stats["rejects"]["honest: review: not reviewed"] == 1
    pairs = _rows(tmp_path / "gen" / "t" / "pairs.jsonl")
    assert {p["source_id"] for p in pairs} == {"g0000", "g0001", "g0004"}


def test_twin_length_miss_is_retried(tmp_path):
    stats = run(_cfg(tmp_path), "t", llm=FakeLLM(long_twins=True))
    assert stats["by_mutation"]["declare_effect"] == 20
    assert stats["twin_retried"] == {True: 20}
    assert not any(k.startswith("twin") for k in stats["rejects"])


def test_weaken_only_with_prefix_and_third_effect(tmp_path):
    cfg = {**_cfg(tmp_path), "weaken_only": True, "id_prefix": "w", "p_second_effect": 1.0,
           "p_third_effect": 1.0, "fractions": [1.0, 0.0, 0.0]}
    stats = run(cfg, "t", llm=FakeLLM())
    assert set(stats["by_mutation"]) == {"original", "weaken_description"}
    pairs = _rows(tmp_path / "gen" / "t" / "pairs.jsonl")
    assert all(p["source_id"].startswith("w") for p in pairs)
    assert len(_rows(tmp_path / "splits" / "t" / "train.jsonl")) == len(pairs)


def test_plan_calls_defaults_unchanged():
    from agentlint.data.generate import plan_calls
    a = plan_calls(12, 4, ["x", "y"], 0.6, 566)
    b = plan_calls(12, 4, ["x", "y"], 0.6, 566, p_third=0.0, prefix="c")
    assert [(p.call_id, [s.effects for s in p.specs]) for p in a] == \
        [(p.call_id, [s.effects for s in p.specs]) for p in b]
    c = plan_calls(12, 4, ["x", "y"], 1.0, 1, p_third=1.0, prefix="w")
    assert all(len(s.effects) == 3 for p in c for s in p.specs) and c[0].call_id == "w0000"
