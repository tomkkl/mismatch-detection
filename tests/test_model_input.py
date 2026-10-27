import pytest
import torch

from agentlint.model.classifier import max_pool_by_pair
from agentlint.model.dataset import (PairDataset, collapse_whitespace, encode_pair,
                                     format_interface, params_compact)
from agentlint.schema import Effect, ToolPair

CODE = '''
def list_project_files(folder):
    items = os.listdir(folder)
    return sorted(items)
'''

PARAMS = {
    "type": "object",
    "properties": {"folder": {"type": "string", "description": "Project folder."}},
    "required": ["folder"],
}


def _pair(code: str = CODE, i: int = 0, mismatch: bool = False) -> ToolPair:
    return ToolPair(
        pair_id=str(i), source_id="s", name="list_project_files", params=PARAMS,
        description="List the files in a project folder.", code=code, mismatch=mismatch,
        undeclared=[Effect.RELOCATE] if mismatch else [],
    )


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    except OSError:
        pytest.skip("tokenizer not available offline")


def test_collapse_whitespace():
    out = collapse_whitespace(CODE)
    assert "\n" not in out and "  " not in out
    assert out.startswith("def list_project_files(folder): items")


def test_interface_is_compact():
    assert params_compact(PARAMS) == "folder (string, required): Project folder."
    assert format_interface(_pair()).split("\n") == [
        "list_project_files", "folder (string, required): Project folder.",
        "List the files in a project folder.",
    ]


def test_short_pair_is_one_chunk(tok):
    assert len(encode_pair(_pair(), tok)) == 1


def test_long_pair_chunks_overlap_and_cover_the_code(tok):
    long_code = "\n".join(f"x{i} = shutil.move(src_{i}, dst_{i})" for i in range(200))
    pair = _pair(long_code)
    chunks = encode_pair(pair, tok, max_length=512, stride=128)
    assert len(chunks) >= 3
    interface = tok(format_interface(pair), add_special_tokens=False)["input_ids"]
    code_parts = []
    for c in chunks:
        assert len(c) <= 512
        assert c[0] == tok.cls_token_id and c[1:1 + len(interface)] == interface
        code_parts.append(c[len(interface) + 3:-1])  # after [CLS] interface [SEP][SEP], before [SEP]
    for a, b in zip(code_parts, code_parts[1:]):
        assert a[-128:] == b[:128]
    rebuilt = code_parts[0] + [t for part in code_parts[1:] for t in part[128:]]
    assert rebuilt == tok(collapse_whitespace(long_code), add_special_tokens=False)["input_ids"]


def test_interface_too_long_is_rejected(tok):
    pair = _pair()
    pair.description = "word " * 450
    with pytest.raises(ValueError):
        encode_pair(pair, tok)


def test_collate_maps_chunks_to_pairs(tok):
    long_code = "\n".join(f"x{i} = shutil.move(src_{i}, dst_{i})" for i in range(200))
    ds = PairDataset([_pair(), _pair(long_code, 1, mismatch=True)], tok)
    batch = ds.collate([ds[0], ds[1]])
    n_long = len(ds[1]["chunks"])
    assert batch["pair_index"].tolist() == [0] + [1] * n_long
    assert batch["input_ids"].shape[0] == 1 + n_long
    assert batch["attention_mask"][0].sum() == len(ds[0]["chunks"][0])
    assert batch["labels"].tolist() == [[0.0], [1.0]]

    multi = PairDataset([_pair(long_code, 1, mismatch=True)], tok, task="multilabel")
    assert multi[0]["labels"].sum() == 1


def test_max_pool_by_pair():
    logits = torch.tensor([[0.1, -1.0], [2.0, -3.0], [-0.5, 0.75], [1.5, 0.25]])
    pooled = max_pool_by_pair(logits, torch.tensor([0, 0, 1, 1]), 2)
    assert pooled.tolist() == [[2.0, -1.0], [1.5, 0.75]]


def test_chunk_layout_matches_the_tokenizer(tok):
    pair = _pair()
    expected = tok(format_interface(pair), collapse_whitespace(pair.code))["input_ids"]
    assert encode_pair(pair, tok) == [expected]
