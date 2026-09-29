"""Shared data types."""

from dataclasses import dataclass, field
from enum import Enum


class Effect(str, Enum):
    READ = "read"
    MODIFY = "modify"
    DESTROY = "destroy"
    RELOCATE = "relocate"
    SEND_OUTWARD = "send_outward"
    EXECUTE = "execute"
    SPEND = "spend"
    CHANGE_AUTHORITY = "change_authority"


EFFECTS: list[Effect] = list(Effect)


@dataclass
class ToolPair:
    """One (interface, implementation) example.

    `source_id` groups an honest tool with all its mutants so they land in the same split.
    """

    pair_id: str
    source_id: str
    name: str
    params: dict
    description: str
    code: str
    mismatch: bool
    undeclared: list[Effect] = field(default_factory=list)
    mutation: str | None = None          # None | "inject_code" | "declare_effect" | "weaken_description"
    injected_lines: list[int] = field(default_factory=list)
    generator: str | None = None         # which LLM produced it (test rung 2)
    meta: dict = field(default_factory=dict)  # domain, position, declared effects, check results

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["undeclared"] = [e.value for e in self.undeclared]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ToolPair":
        d = dict(d)
        d["undeclared"] = [Effect(e) for e in d.get("undeclared", [])]
        return cls(**d)
