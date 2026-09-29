from agentlint.data.split import group_split
from agentlint.schema import ToolPair


def _pair(i: int, source: str) -> ToolPair:
    return ToolPair(
        pair_id=str(i), source_id=source, name="f", params={}, description="", code="",
        mismatch=bool(i % 2),
    )


def test_mutants_stay_with_their_source():
    pairs = [_pair(i, f"src{i // 3}") for i in range(60)]
    splits = group_split(pairs)
    seen = [{p.source_id for p in s} for s in splits]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert sum(len(s) for s in splits) == len(pairs)
