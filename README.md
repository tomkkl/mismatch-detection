# AgentLint

Detect LLM agent tools whose **implementation does more than their declared interface says**
(name + parameter schema + description). CSCI 566, Fall 2026.

See `docs/agentlint-preproposal.md` for the proposal and `CLAUDE.md` for current decisions.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,baseline]"
pytest
```

## Layout

```
src/agentlint/
  schema.py          effect classes, ToolPair record
  effects.py         effect definitions + word lists (label checks, localization)
  data/
    llm.py           claude CLI backend (plan usage) with on-disk cache
    generate.py      LLM-generated honest tools
    mutate.py        make mismatches (inject code / honest twin / weaken description)
    verify.py        code normalization (AST round-trip) + AST label checks
    split.py         group split by source (never random)
    build.py         pipeline: generate -> verify -> mutate -> split
  model/
    dataset.py       tokenize description + code, chunk long inputs
    classifier.py    GraphCodeBERT cross-encoder
    train.py         fine-tuning loop
  eval/
    metrics.py       per-rung and per-class metrics
    agent_audit.py   run the Agent Audit baseline, compare coverage
  localize.py        AST: which lines carry the undeclared effect
  cli.py             `agentlint` entry point
configs/             training configs (binary, multilabel)
data/                raw, generated, splits (git-ignored)
probes/              Agent Audit probe scripts; generated cases go in probes/aa_run/
notebooks/           Colab training notebooks
scripts/             one-off utilities
tests/               unit tests for our own code
docs/                proposal and write-ups
```

## Building the dataset

Generation runs through the `claude` CLI (log in with `claude` first), so it uses your Claude
plan's usage, not the paid API. Settings are in `configs/datagen.yaml`.

```
agentlint data build configs/datagen.yaml --tag pilot --n-honest 40
```

Output goes to `data/generated/<tag>/` (`pairs.jsonl`, `rejects.jsonl`, `stats.json`) and
`data/splits/<tag>/`. If a usage limit stops the run, rerun the same command later: finished
LLM calls are cached in `data/generated/llm_cache/` and reused.

## Test ladder (report each separately)

1. held-out generated pairs
2. pairs from a different LLM, same recipe
3. real GitHub tools with an injected capability
4. (optional) real mismatches found in the wild
