"""Generate honest tools (name + params + description + code) with an LLM.

Each call asks for a few tools in one domain. Every tool gets its own spec (declared effects,
length, decorator, code habits) so domains, naming and length vary. Code is normalized
(no comments, no docstrings) before it is stored.
"""

import random
from dataclasses import dataclass

from agentlint.data.llm import LLM
from agentlint.data.verify import normalize_code
from agentlint.effects import DEFINITIONS
from agentlint.schema import EFFECTS, Effect, ToolPair

LENGTHS = {"short": "10-20", "medium": "20-40", "long": "40-80"}
DECORATORS = ["none", "@tool", "@mcp.tool()", "@function_tool"]
HABITS = [
    "validates inputs and returns early on bad input",
    "puts part of the work in a small private helper function",
    "uses a context manager",
    "builds and returns a dict result",
    "straight-line code with few branches",
    "uses a small dataclass for the result",
    "short local variable names",
    "long descriptive local variable names",
]

SYSTEM = (
    "You write realistic Python implementations of tools for LLM agents, like the ones found "
    "in open-source agent projects (LangChain tools, MCP servers, OpenAI Agents SDK function "
    "tools). You return only the requested JSON."
)

SCHEMA = {
    "type": "object",
    "properties": {"tools": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "params": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string",
                             "enum": ["string", "integer", "number", "boolean", "array",
                                      "object"]},
                    "description": {"type": "string"},
                    "required": {"type": "boolean"},
                },
                "required": ["name", "type", "description", "required"],
                "additionalProperties": False,
            }},
            "effects": {"type": "array", "items": {"type": "string",
                                                   "enum": [e.value for e in EFFECTS]}},
            "code": {"type": "string"},
        },
        "required": ["name", "description", "params", "effects", "code"],
        "additionalProperties": False,
    }}},
    "required": ["tools"],
    "additionalProperties": False,
}


@dataclass
class ToolSpec:
    effects: list[Effect]
    length: str
    decorator: str
    habit: str


@dataclass
class CallPlan:
    call_id: str
    domain: str
    specs: list[ToolSpec]


def plan_calls(n_tools: int, per_call: int, domains: list[str], p_second: float,
               seed: int) -> list[CallPlan]:
    rng = random.Random(seed)
    order = list(domains)
    rng.shuffle(order)
    plans, made = [], 0
    while made < n_tools:
        k = min(per_call, n_tools - made)
        specs = []
        for _ in range(k):
            first = rng.choice(EFFECTS)
            effects = [first]
            if rng.random() < p_second:
                effects.append(rng.choice([e for e in EFFECTS if e != first]))
            specs.append(ToolSpec(effects, rng.choice(list(LENGTHS)), rng.choice(DECORATORS),
                                  rng.choice(HABITS)))
        plans.append(CallPlan(f"c{len(plans):04d}", order[len(plans) % len(order)], specs))
        made += k
    return plans


def build_prompt(plan: CallPlan) -> str:
    lines = [f"Domain: {plan.domain}", "", f"Write {len(plan.specs)} different tools for an "
             "agent working in this domain. Tool specs:"]
    for i, s in enumerate(plan.specs, 1):
        deco = "no decorator" if s.decorator == "none" else f"decorated with {s.decorator}"
        lines.append(f"{i}. effects: {', '.join(e.value for e in s.effects)}; about "
                     f"{LENGTHS[s.length]} lines of code; {deco}; code style: {s.habit}")
    lines += ["", "Effect meanings:"]
    lines += [f"- {e.value}: {d}" for e, d in DEFINITIONS.items()]
    lines += [
        "",
        "Rules for each tool:",
        "- `code` is a complete Python module: imports, optional private helpers (names start "
        "with an underscore), and one public function whose name is exactly `name`.",
        "- The public function's parameters are exactly the ones in `params` (a leading "
        "`self` or `ctx` is allowed).",
        "- The code performs exactly the listed effects and no others. Set `effects` to that "
        "list.",
        "- Use ordinary libraries and clients a real project would use (sqlite3, pathlib, "
        "shutil, requests, boto3, smtplib, subprocess, stripe, or a module-level client for an "
        "internal service). The code does not need to run.",
        "- No comments and no docstrings anywhere in the code.",
        "- `description` is what the author would write for the agent: 1-3 plain sentences "
        "that state every effect the tool has, in the domain's own words. Never mention "
        "effects the tool does not have. Do not use the effect labels above as words.",
        "- Give each tool a realistic, specific name; do not reuse names across tools.",
    ]
    return "\n".join(lines)


def _params_schema(params: list[dict]) -> dict:
    return {
        "type": "object",
        "properties": {p["name"]: {"type": p["type"], "description": p["description"]}
                       for p in params},
        "required": [p["name"] for p in params if p["required"]],
    }


def generate_batch(llm: LLM, plan: CallPlan, effort: str | None = None) -> list[ToolPair]:
    """One LLM call -> honest tools (normalized, not yet verified)."""
    out = llm.json(build_prompt(plan), SCHEMA, SYSTEM, effort=effort)
    tools = []
    for i, (raw, spec) in enumerate(zip(out["tools"], plan.specs)):
        try:
            code = normalize_code(raw["code"])
        except SyntaxError:
            code = raw["code"]  # verify_honest rejects it
        sid = f"{plan.call_id}-{i}"
        tools.append(ToolPair(
            pair_id=f"{sid}/honest", source_id=sid, name=raw["name"],
            params=_params_schema(raw["params"]), description=raw["description"].strip(),
            code=code, mismatch=False, generator=llm.name,
            meta={"domain": plan.domain, "declared": sorted(set(raw["effects"])),
                  "requested": [e.value for e in spec.effects], "length": spec.length,
                  "decorator": spec.decorator, "habit": spec.habit},
        ))
    return tools
