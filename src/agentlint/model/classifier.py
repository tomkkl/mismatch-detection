"""GraphCodeBERT cross-encoder with a classification head on [CLS].

binary: 1 logit (mismatch yes/no). multilabel: one sigmoid logit per effect class.
"""

from agentlint.schema import EFFECTS

MODEL_NAME = "microsoft/graphcodebert-base"


def num_labels(task: str) -> int:
    return 1 if task == "binary" else len(EFFECTS)


def build_model(task: str = "binary", model_name: str = MODEL_NAME):
    raise NotImplementedError
