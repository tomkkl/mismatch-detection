"""Tokenize (interface, code) as one cross-encoder input.

Code is whitespace-collapsed (the tokenizer spends one token per indentation space). Inputs
still over 512 tokens are chunked, with the interface repeated in every chunk and `stride`
code tokens shared between neighbouring chunks, instead of truncated: truncation can cut the
lines that carry the label. The model max-pools chunk logits per pair
(`classifier.max_pool_by_pair`).

At scan time, real code goes through `data.verify.normalize_code` and then the same
`collapse_whitespace` here.
"""

import re

import torch

from agentlint.schema import EFFECTS, ToolPair

MIN_CODE_TOKENS = 128  # refuse interfaces that leave less room than this for code per chunk


def collapse_whitespace(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def params_compact(params: dict) -> str:
    """One `name (type, required): description` entry per parameter, joined with '; '."""
    required = set(params.get("required", []))
    return "; ".join(
        f"{k} ({v.get('type', '?')}{', required' if k in required else ''}): {v.get('description', '')}"
        for k, v in params.get("properties", {}).items()
    )


def format_interface(pair: ToolPair) -> str:
    """Name + params + description as the first segment."""
    return f"{pair.name}\n{params_compact(pair.params)}\n{pair.description}"


def encode_pair(pair: ToolPair, tokenizer, max_length: int = 512, stride: int = 128) -> list[list[int]]:
    """Input ids of each chunk: `[CLS] interface [SEP] code slice [SEP]`."""
    # Windows are built by hand: the tokenizer's own overflow (return_overflowing_tokens) stops
    # after one extra chunk for pairs in transformers 5.x and silently drops the rest of the code.
    interface = tokenizer(format_interface(pair), add_special_tokens=False)["input_ids"]
    code = tokenizer(collapse_whitespace(pair.code), add_special_tokens=False)["input_ids"]
    room = max_length - len(interface) - tokenizer.num_special_tokens_to_add(pair=True)
    if room < max(MIN_CODE_TOKENS, stride + 1):
        raise ValueError(f"{pair.pair_id}: interface is {len(interface)} tokens, leaving {room} for code")
    starts = range(0, max(len(code) - stride, 1), room - stride)
    cls, sep = [tokenizer.cls_token_id], [tokenizer.sep_token_id]
    return [cls + interface + sep + sep + code[s:s + room] + sep for s in starts]  # RoBERTa pair layout


def label_vector(pair: ToolPair, task: str) -> torch.Tensor:
    if task == "binary":
        return torch.tensor([float(pair.mismatch)])
    return torch.tensor([float(e in pair.undeclared) for e in EFFECTS])


class PairDataset(torch.utils.data.Dataset):
    """One item per pair (not per chunk), so batches and the loss stay per pair."""

    def __init__(self, pairs: list[ToolPair], tokenizer, max_length: int = 512, stride: int = 128,
                 task: str = "binary"):
        self.items = [
            {"pair_id": p.pair_id, "chunks": encode_pair(p, tokenizer, max_length, stride),
             "labels": label_vector(p, task)}
            for p in pairs
        ]
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]

    def collate(self, items: list[dict]) -> dict:
        return collate_chunks(items, self.pad_token_id)


def collate_chunks(items: list[dict], pad_token_id: int) -> dict:
    """Flatten every pair's chunks into one padded batch; `pair_index[c]` = pair of chunk c."""
    chunks = [c for it in items for c in it["chunks"]]
    width = max(len(c) for c in chunks)
    input_ids = torch.full((len(chunks), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(chunks), width), dtype=torch.long)
    for row, c in enumerate(chunks):
        input_ids[row, : len(c)] = torch.tensor(c)
        attention_mask[row, : len(c)] = 1
    pair_index = torch.tensor([i for i, it in enumerate(items) for _ in it["chunks"]])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pair_index": pair_index,
        "labels": torch.stack([it["labels"] for it in items]),
        "pair_ids": [it["pair_id"] for it in items],
    }
