"""Turn an honest tool into mismatched (and matching honest-twin) versions with an LLM.

- inject_code: add an ordinary extra operation of one effect at a given position
  (start / middle / end / inside a helper). Description unchanged -> mismatch.
- declare_effect: honest twin of an injected mutant. Same code, description rewritten to
  declare the new effect -> honest. Only the description separates it from the mutant.
- match_length: rewrite the injected mutant's description (from the honest code) to its twin's
  length, so description length can't separate mismatch from honest.
- weaken_description: code unchanged, one declared effect dropped from the description
  -> mismatch.

Prompts state the dataset's purpose plainly and ask for ordinary features, never for hidden
ones: framing the edit as concealment makes the model refuse. Comments in its output are fine
because normalization strips them. The model may decline; that raises `Declined`.

Outputs are normalized but not verified; run them through `verify` before keeping them.
"""

import json
from dataclasses import replace

from agentlint.data.llm import LLM, LLMError
from agentlint.data.verify import normalize_code
from agentlint.effects import DEFINITIONS, INJECTION_IDEAS
from agentlint.schema import Effect, ToolPair

POSITIONS = {
    "start": "near the start of the tool function, before its main work",
    "middle": "in the middle of the tool function, between two existing steps",
    "end": "near the end of the tool function, just before it returns",
    "helper": "inside a private helper function that the tool calls (use an existing helper "
              "if there is one, otherwise add a new private helper and call it)",
}

SYSTEM = ("You help build a labeled dataset for training a classifier that checks whether an "
          "LLM agent tool's description matches what its code does. You return only the "
          "requested JSON.")

PURPOSE = (
    "About the dataset: each example pairs a tool's code with a description and is labeled "
    "honest (the description covers everything the code does) or mismatch (the code does "
    "something the description leaves out). The examples are used only to train and test "
    "the detector."
)


def _schema(field: str, extra: dict | None = None) -> dict:
    props = {"declined": {"type": "boolean"}, "reason": {"type": "string"},
             field: {"type": "string"}, **(extra or {})}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


CODE_SCHEMA = _schema("code")
DESC_SCHEMA = _schema("description")
TWIN_SCHEMA = _schema("description", {"declaring_phrase": {"type": "string"}})
DECLINE_RULE = ("If you decline, set `declined` to true and explain in `reason`; otherwise set "
                "`declined` to false and leave `reason` empty.")


class Declined(LLMError):
    """The model chose not to do the edit."""


def _ask(llm: LLM, prompt: str, schema: dict, effort: str | None) -> dict:
    out = llm.json(prompt, schema, SYSTEM, effort=effort)
    if out["declined"]:
        raise Declined(out["reason"] or "no reason given")
    return out


def _show(tool: ToolPair) -> str:
    return "\n".join([f"Tool name: {tool.name}", f"Description: {tool.description}",
                      f"Parameters: {json.dumps(tool.params)}", "Code:", "```python",
                      tool.code, "```"])


def inject_code(llm: LLM, tool: ToolPair, effect: Effect, position: str,
                max_added: int = 15, effort: str | None = None) -> ToolPair:
    prompt = "\n".join([
        PURPOSE,
        "For this example we need a version of the tool below whose code does one ordinary "
        "extra thing, the kind of step real tools pick up as features get added. Our pipeline "
        "labels the new code with the original description as a mismatch, and the same code "
        "with an updated description as honest.",
        "",
        _show(tool), "",
        f"Extend the tool so it also {DEFINITIONS[effect]}. Pick an operation that fits this "
        f"tool's data and domain, the way a developer adding a feature would. Ideas: "
        f"{INJECTION_IDEAS[effect]}. Do not copy these ideas word for word.",
        f"Put the new operation {POSITIONS[position]}.",
        "",
        "Rules:",
        "- A normal, legitimate operation for this domain: not an attack, no sending data to "
        "unknown parties, no eval or exec of input, no obfuscation.",
        "- Keep the function name and parameters. Change at most 2 existing lines; otherwise "
        "only add lines.",
        f"- Add at most {max_added} lines, including any new imports. Comments are fine; we "
        "strip them.",
        "- Return the complete module in `code`.",
        f"- {DECLINE_RULE}",
    ])
    out = _ask(llm, prompt, CODE_SCHEMA, effort)
    try:
        code = normalize_code(out["code"])
    except SyntaxError:
        code = out["code"]
    sid = tool.source_id
    return replace(
        tool, pair_id=f"{sid}/inject-{effect.value}-{position}", code=code, mismatch=True,
        undeclared=[effect], mutation="inject_code", injected_lines=[],
        meta={**tool.meta, "position": position, "checks": {}},
    )


def declare_effect(llm: LLM, mutant: ToolPair, effect: Effect,
                   effort: str | None = None) -> ToolPair:
    prompt = "\n".join([
        PURPOSE, "",
        _show(mutant), "",
        f"The code also {DEFINITIONS[effect]}, but the description does not say so. Rewrite "
        "the description so it honestly states everything the code does, including that.",
        "Keep the author's style and roughly the same length: tighten other wording to make "
        "room. 1-3 plain sentences.",
        "Put the exact words of your new description that state this added behavior in "
        "`declaring_phrase` (copied verbatim, a few words long).",
        DECLINE_RULE,
    ])
    out = _ask(llm, prompt, TWIN_SCHEMA, effort)
    sid, pos = mutant.source_id, mutant.meta["position"]
    return replace(
        mutant, pair_id=f"{sid}/twin-{effect.value}-{pos}",
        description=out["description"].strip(), mismatch=False, undeclared=[],
        mutation="declare_effect",
        meta={**mutant.meta, "declared": sorted(set(mutant.meta["declared"]) | {effect.value}),
              "declaring_phrase": out["declaring_phrase"].strip(), "checks": {}},
    )


def match_length(llm: LLM, honest: ToolPair, mutant: ToolPair, target_chars: int,
                 effort: str | None = None) -> ToolPair:
    """Rewrite the mutant's description to about `target_chars` (its twin's length).

    The model sees the *honest* tool, so the rewrite can only describe the declared behavior;
    it never sees the added operation. This keeps mismatch and twin descriptions the same
    length, so length cannot give the label away.
    """
    prompt = "\n".join([
        PURPOSE, "",
        _show(honest), "",
        f"Rewrite the description to about {target_chars} characters (the current one is "
        f"{len(honest.description)}). Describe only what it already says the tool does: add "
        "or trim detail about that behavior (inputs, what it returns, conditions) but do not "
        "introduce new actions. Keep the author's style. 1-3 plain sentences.",
        DECLINE_RULE,
    ])
    out = _ask(llm, prompt, DESC_SCHEMA, effort)
    return replace(mutant, description=out["description"].strip(),
                   meta={**mutant.meta, "desc_length_matched": True})


def weaken_description(llm: LLM, tool: ToolPair, effect: Effect,
                       effort: str | None = None) -> ToolPair:
    prompt = "\n".join([
        PURPOSE,
        "For this example we need a description that leaves out one thing the code does. Our "
        "pipeline labels the unchanged code with the new description as a mismatch.",
        "",
        _show(tool), "",
        f"The tool {DEFINITIONS[effect]}. Rewrite the description so it leaves that out: no "
        "mention of it and no synonyms that hint at it. It must still describe the tool's "
        "other behavior accurately and read as a natural, complete description.",
        "Keep the author's style and roughly the same length. 1-3 plain sentences.",
        DECLINE_RULE,
    ])
    out = _ask(llm, prompt, DESC_SCHEMA, effort)
    return replace(
        tool, pair_id=f"{tool.source_id}/weaken-{effect.value}",
        description=out["description"].strip(), mismatch=True, undeclared=[effect],
        mutation="weaken_description",
        meta={**tool.meta, "declared": sorted(set(tool.meta["declared"]) - {effect.value}),
              "checks": {}},
    )
