"""Train / val / test split grouped by `source_id`. Never split randomly."""

import random
from collections import defaultdict

from agentlint.schema import ToolPair


def group_split(
    pairs: list[ToolPair], fractions=(0.8, 0.1, 0.1), seed: int = 566
) -> tuple[list[ToolPair], list[ToolPair], list[ToolPair]]:
    groups: dict[str, list[ToolPair]] = defaultdict(list)
    for p in pairs:
        groups[p.source_id].append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_train = int(len(keys) * fractions[0])
    n_val = int(len(keys) * fractions[1])
    parts = (keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:])
    return tuple([p for k in part for p in groups[k]] for part in parts)
