"""Measure cross-encoder input lengths against GraphCodeBERT's 512-token limit.

Usage: python scripts/token_lengths.py [pairs.jsonl] [--model microsoft/graphcodebert-base]

Reports, for several ways of building the input, the length distribution and the share of
pairs over the limit (overall and per label). Lengths include the special tokens of a pair
encoding: tok(first_segment, code).
"""

import argparse
import json
import re

import numpy as np
from transformers import AutoTokenizer
from transformers.utils import logging

LIMIT = 512


def params_compact(params: dict) -> str:
    """One `name (type, required): description` entry per parameter, joined with '; '."""
    required = set(params.get("required", []))
    return "; ".join(
        f"{k} ({v.get('type', '?')}{', required' if k in required else ''}): {v.get('description', '')}"
        for k, v in params.get("properties", {}).items()
    )


def one_space_indent(code: str) -> str:
    """Shrink each 4-space indentation level to a single space."""
    return "\n".join(
        re.sub(r"^( {4})+", lambda m: " " * (len(m.group(0)) // 4), line) for line in code.split("\n")
    )


def collapse_whitespace(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def interface(r: dict) -> str:
    return f"{r['name']}\n{params_compact(r['params'])}\n{r['description']}"


VARIANTS = {
    "desc + code": lambda r: (r["description"], r["code"]),
    "name + params(compact) + desc + code": lambda r: (interface(r), r["code"]),
    "name + params(JSON) + desc + code": lambda r: (
        f"{r['name']}\n{json.dumps(r['params'])}\n{r['description']}",
        r["code"],
    ),
    "desc + code, 1-space indent": lambda r: (r["description"], one_space_indent(r["code"])),
    "desc + code, whitespace collapsed": lambda r: (r["description"], collapse_whitespace(r["code"])),
    "name + params(compact) + desc + code, whitespace collapsed": lambda r: (
        interface(r),
        collapse_whitespace(r["code"]),
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="?", default="data/generated/full/pairs.jsonl")
    ap.add_argument("--model", default="microsoft/graphcodebert-base")
    args = ap.parse_args()

    logging.set_verbosity_error()  # silence the "longer than 512" warning; that's what we measure
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.pairs) as f:
        rows = [json.loads(line) for line in f]
    mismatch = np.array([r["mismatch"] for r in rows])

    code = np.array([len(tok(r["code"], add_special_tokens=False)["input_ids"]) for r in rows])
    ws = np.array(
        [sum(t.strip("Ġ") == "" or t == "Ċ" for t in tok.tokenize(r["code"])) for r in rows]
    )
    print(f"{len(rows)} pairs, {mismatch.sum()} mismatches ({mismatch.mean():.1%})")
    print(
        f"code alone: median {np.median(code):.0f}, p95 {np.percentile(code, 95):.0f}, max {code.max()}; "
        f"whitespace-only tokens: median {np.median(ws / code):.1%} of code tokens\n"
    )

    print(f"{'input':<60} {'median':>6} {'p95':>5} {'max':>5} {'>512':>6} {'mism.':>6} {'honest':>6}")
    for name, build in VARIANTS.items():
        lengths = np.array([len(tok(*build(r))["input_ids"]) for r in rows])
        over = lengths > LIMIT
        print(
            f"{name:<60} {np.median(lengths):>6.0f} {np.percentile(lengths, 95):>5.0f} {lengths.max():>5} "
            f"{over.mean():>6.1%} {over[mismatch].mean():>6.1%} {over[~mismatch].mean():>6.1%}"
        )


if __name__ == "__main__":
    main()
