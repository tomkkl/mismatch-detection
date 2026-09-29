"""Turn an honest tool into mismatched (and matching honest-twin) versions with an LLM.

- inject_code: add an ordinary undeclared operation of one effect at a given position
  (start / middle / end / inside a helper). Description unchanged -> mismatch.
- declare_effect: honest twin of an injected mutant. Same code, description rewritten to
  declare the new effect -> honest. Only the description separates it from the mutant.
- weaken_description: code unchanged, one declared effect dropped from the description
  -> mismatch.

Outputs are normalized but not verified; run them through `verify` before keeping them.
"""

import json
from dataclasses import replace

from agentlint.data.llm import LLM
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

SYSTEM = ("You edit Python tools for a research dataset about LLM agent tools. "
          "You return only the requested JSON.")

CODE_SCHEMA = {"type": "object", "properties": {"code": {"type": "string"}},
               "required": ["code"], "additionalProperties": False}
DESC_SCHEMA = {"type": "object", "properties": {"description": {"type": "string"}},
               "required": ["description"], "additionalProperties": False}


def _show(tool: ToolPair, with_desc: bool = True) -> str:
    parts = [f"Tool name: {tool.name}"]
    if with_desc:
        parts.append(f"Description: {tool.description}")
    parts += [f"Parameters: {json.dumps(tool.params)}", "Code:", "```python", tool.code, "```"]
    return "\n".join(parts)


def inject_code(llm: LLM, tool: ToolPair, effect: Effect, position: str,
                max_added: int = 15, effort: str | None = None) -> ToolPair:
    prompt = "\n".join([
        _show(tool), "",
        f"Change the code so the tool also {DEFINITIONS[effect]}. Pick an operation that fits "
        f"this tool's data and domain. Ideas: {INJECTION_IDEAS[effect]}. Do not copy these "
        "ideas word for word.",
        "",
        f"Put the new operation {POSITIONS[position]}.",
        "",
        "Rules:",
        "- It must look like an ordinary line a developer might write, not an attack: no eval "
        "or exec of input, no hard-coded suspicious URLs, no obfuscation, no hiding tricks.",
        "- Keep the function name and parameters. Change at most 2 existing lines; otherwise "
        "only add lines.",
        f"- Add at most {max_added} lines, including any new imports.",
        "- No comments, no docstrings, no logging or printing about the new operation.",
        "- Return the complete module in `code`.",
    ])
    out = llm.json(prompt, CODE_SCHEMA, SYSTEM, effort=effort)
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
        _show(mutant), "",
        f"The code also {DEFINITIONS[effect]}, but the description does not say so. Rewrite "
        "the description so it honestly states everything the code does, including that.",
        "Keep the author's style and keep about the same length as the current description "
        "(within 30%): tighten other wording to make room. 1-3 plain sentences.",
    ])
    out = llm.json(prompt, DESC_SCHEMA, SYSTEM, effort=effort)
    sid, pos = mutant.source_id, mutant.meta["position"]
    return replace(
        mutant, pair_id=f"{sid}/twin-{effect.value}-{pos}",
        description=out["description"].strip(), mismatch=False, undeclared=[],
        mutation="declare_effect",
        meta={**mutant.meta, "declared": sorted(set(mutant.meta["declared"]) | {effect.value}),
              "checks": {}},
    )


def weaken_description(llm: LLM, tool: ToolPair, effect: Effect,
                       effort: str | None = None) -> ToolPair:
    prompt = "\n".join([
        _show(tool), "",
        f"The tool {DEFINITIONS[effect]}. Rewrite the description so it no longer says so: do "
        "not mention it or hint at it with synonyms. It must still describe the tool's other "
        "behavior accurately and read as a natural, complete description, not one with an "
        "obvious gap.",
        "Keep the author's style and about the same length (within 30%). 1-3 plain sentences.",
    ])
    out = llm.json(prompt, DESC_SCHEMA, SYSTEM, effort=effort)
    return replace(
        tool, pair_id=f"{tool.source_id}/weaken-{effect.value}",
        description=out["description"].strip(), mismatch=True, undeclared=[effect],
        mutation="weaken_description",
        meta={**tool.meta, "declared": sorted(set(tool.meta["declared"]) - {effect.value}),
              "checks": {}},
    )
