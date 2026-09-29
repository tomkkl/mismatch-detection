"""Tokenize (interface, code) as one cross-encoder input.

Inputs over 512 tokens are chunked (interface repeated in every chunk) instead of truncated;
the model max-pools chunk logits. Truncation can cut the lines that carry the label.
"""

from agentlint.schema import ToolPair


def format_interface(pair: ToolPair) -> str:
    """Name + params + description as the first segment."""
    raise NotImplementedError


class PairDataset:
    def __init__(self, pairs: list[ToolPair], tokenizer, max_length: int = 512, stride: int = 128):
        raise NotImplementedError
