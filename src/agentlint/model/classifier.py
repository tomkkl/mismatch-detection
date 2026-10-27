"""GraphCodeBERT cross-encoder with a classification head on [CLS].

binary: 1 logit (mismatch yes/no). multilabel: one sigmoid logit per effect class.
A pair's logits are the max over its chunks' logits; the loss is computed on the pooled logits.
"""

import torch
from transformers import AutoModelForSequenceClassification

from agentlint.schema import EFFECTS

MODEL_NAME = "microsoft/graphcodebert-base"


def num_labels(task: str) -> int:
    return 1 if task == "binary" else len(EFFECTS)


def build_model(task: str = "binary", model_name: str = MODEL_NAME):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels(task))


def max_pool_by_pair(chunk_logits: torch.Tensor, pair_index: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """[n_chunks, n_labels] -> [n_pairs, n_labels], max over each pair's chunks, per label."""
    index = pair_index.unsqueeze(1).expand_as(chunk_logits)
    out = chunk_logits.new_full((n_pairs, chunk_logits.size(1)), float("-inf"))
    return out.scatter_reduce(0, index, chunk_logits, reduce="amax")
