"""The model says *whether*; the AST says *where*.

Find the lines in a flagged tool whose operations match the predicted undeclared effect.
"""

from agentlint.effects import code_effects
from agentlint.schema import Effect


def locate(code: str, effect: Effect) -> list[int]:
    return code_effects(code).get(effect, [])
