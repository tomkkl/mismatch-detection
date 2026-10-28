# AgentLint (CSCI 566 course project)

**Goal:** detect LLM agent tools whose **code does more than their interface says** (name +
parameter schema + description).

Example: a tool described as "List the files in a project folder" that also moves the whole
directory. The code is fine on its own and the description is fine on its own. Only the pair is
wrong.

Team of 6, ~10 weeks, Fall 2026. The course requires a trained or fine-tuned model.

**Repo:** https://github.com/tomkkl/mismatch-detection (branch `main`).

This file is read by Claude Code at the start of every session, and it is also the team's
shared notes. Keep it current: when a decision is made or a milestone lands, update it in the
same commit.

---

## 1. Getting started

### Setup

```
git clone https://github.com/tomkkl/mismatch-detection.git
cd mismatch-detection
python3 -m venv .venv
# macOS / Linux:
.venv/bin/pip install -e ".[dev,baseline]"
# Windows:
.venv\Scripts\pip install -e ".[dev,baseline]"
pytest
```

- Python 3.10+ (developed on 3.13). `baseline` installs Agent Audit v0.19.2.
- Training runs on USC CARC or Colab (a T4 is enough for this model size).
- macOS: if `agentlint` suddenly can't be imported, see "Gotchas".

### Getting the dataset

The dataset is committed to the repo (~25 MB), so you get it with `git clone`:

- `data/splits/full/` (`train.jsonl`, `val.jsonl`, `test.jsonl`): what you need to train.
- `data/generated/full/`: `pairs.jsonl` (all pairs), `stats.json` (counts and leak
  baselines), `rejects.jsonl` (what was filtered and why).
- `data/generated/llm_cache/`: cached LLM answers. With it, rerunning the build after a
  pipeline change (filters, matching, splits) makes zero LLM calls. New runs with a different
  seed or model still need new calls.

Other outputs (pilots, run logs, new tags) stay git-ignored. To commit a new dataset tag, add a
`!data/...` line to `.gitignore`.

### Rebuilding the dataset

```
agentlint data build configs/datagen.yaml --tag full --n-honest 800
```

Generation runs through the `claude` CLI (log in with `claude` first), so it uses a Claude
plan's usage, not the paid API. If a usage limit stops the run, rerun the same command later:
finished calls are cached in `data/generated/llm_cache/` and reused.

---

## 2. Where we are and what's next (updated 2026-10-26)

**Done**
- Agent Audit probes: it misses capability mismatches entirely (see section 8).
- Data generation pipeline (`src/agentlint/data/`), with tests.
- Training dataset `full`: **2,572 pairs from 763 source tools, 689 mismatches (26.8%)**.
  Splits: train 2,062 / val 259 / test 251. Balanced across the 7 injected effects (91–105
  each) and the 4 injection positions. Full breakdown in section 3.
- Token lengths measured (2026-10-23): about half the pairs are over 512 tokens (section 3).
- **Long inputs decided (2026-10-26): collapse whitespace, then chunk + max-pool** (section 5,
  "Input"). Implemented in `src/agentlint/model/dataset.py` (`encode_pair`, `PairDataset`,
  `collate_chunks`) and `classifier.py` (`build_model`, `max_pool_by_pair`), with tests. On
  `full`: 61.0% of pairs fit in 1 chunk, 33.8% need 2, 5.2% need 3+.

**Open issues (not blocking)**
- Few weakened mismatches (129, ~5% of pairs): 288 tools had no effect that could be removed
  from the description.
- The word-list effect checker (`effects.py`) misses SQL-table forms of effects (payment rows,
  role tables, insert-then-delete) and even `stripe.PaymentIntent.create`. Labels are fine
  (hand-checked), but fix it before using `localize.py` for line numbers in reports.
- The description-only baseline is ~1 point above majority (74.2% vs 73.2%). Small; watch it.
- **Chunk count goes with the label:** 48.9% of mismatches need 2+ chunks vs 35.4% of honest
  pairs. With max-pooling, more chunks also means more chances for a high score. Report val by
  chunk count (1 / 2 / 3+) and check honest multi-chunk pairs aren't over-flagged.

### What's next

Steps 2 and 3 can run at the same time with different people.

1. ~~Decide how to handle long inputs~~ (done 2026-10-26).
2. **Train the first model (binary).** Code is written but not run yet:
   `src/agentlint/model/train.py` (weighted BCE on the max-pooled logit, best epoch by val
   PR-AUC, best-F1 threshold saved), `src/agentlint/eval/metrics.py` (P / R / F1, PR-AUC,
   group accuracy, breakdowns by chunks / effect / position / mutation, tested in
   `tests/test_metrics.py`), configs `binary.yaml` and `binary_frozen.yaml`, and
   `notebooks/train_binary.ipynb`. CARC scripts are in `scripts/carc/` (2026-10-28):
   `bash scripts/carc/setup.sh` once on a login node, then
   `sbatch -A <account> scripts/carc/train.slurm <config>`. Not yet run on CARC. Next: smoke
   test (`runs/smoke.yaml`, written by setup), then both configs.
   Done when we have val precision / recall / F1, PR-AUC, group accuracy and the chunk-count
   breakdown for both.
3. **Build the other test sets.**
   - Rung 2: a fresh-seed Opus set, ~100–150 tools (`--tag rung2-opus`, new `seed`, never 566).
   - Rung 3: collect real tool functions (LangChain / OpenAI Agents tools, Python MCP servers)
     and inject with the same pipeline. The collection format is still an open design question.
   - Agent Audit coverage set: our mismatches + some conventional bugs (eval, `shell=True`).
4. **Evaluation code** in `src/agentlint/eval/`: metrics and breakdowns are in `metrics.py`;
   still needed is a script that scores a saved checkpoint on a test set. Then score `test` and rungs 2–3 **once**, and run Agent Audit on the coverage set.
5. **Line numbers in reports:** fix the word-list gaps (see open issues) so `localize.py` points
   at the right lines.

**If time allows:** data-flow input via tree-sitter (ablation); more weakened mismatches; a
narrow "returns undeclared sensitive data" class to replace general `read`.

---

## 3. The training dataset (`full`)

**2,572 description + code pairs built from 763 made-up agent tools**, each labeled honest or
mismatch. 689 pairs (26.8%) are mismatches. Built 2026-10-13/14 with Claude Sonnet 5.5, seed
566, config `configs/datagen.yaml`. Files: `data/generated/full/pairs.jsonl` (everything),
`data/splits/full/{train,val,test}.jsonl`. Section 5 explains *why* it's built this way.

### Where the pairs come from

1. Sonnet wrote 763 honest tools: a Python function plus a description that matches it. They
   cover 42 domains (online store orders, payroll, Kubernetes, crypto wallets…). 445 tools
   declare two effects and 318 declare one; all 8 effect classes appear about equally as
   declared effects (142–165 tools each).
2. From each honest tool the pipeline builds up to 5 pairs. Here is a real one, a tool that
   runs an attendee import script:

| Pair (`mutation`) | Code | Description | Label |
|---|---|---|---|
| original (`null`) | runs the import script | "Runs the attendee import script for an event using a CSV file of guests." | honest |
| `inject_code` | same, plus `os.remove(csv_path)` after a successful import | "Runs the attendee import script… Returns the exit code and the last part of the log." | **mismatch** (deletes a file it doesn't mention) |
| `declare_effect` (twin) | same as `inject_code` | "Imports guests from a CSV…, returns the log tail, and **deletes the CSV file** if the import succeeds." | honest |
| `honest_rewrite` | original code | same as `inject_code`'s description | honest |
| `weaken_description` | original code | a declared effect quietly removed from the description | **mismatch** (not every tool gets one) |

The same code appears with both labels, and so does the same description. **Only the pair
decides the label**, so the model can't win by spotting "longer code", "new import" or
"padded description". It has to compare the two.

The `inject_code` description is rewritten from the *honest* code (so it never mentions the
added operation) and length-matched to within ±15% of the twin's.

### Counts

| Pair type | Count | Label |
|---|---|---|
| original | 763 | honest |
| `inject_code` | 560 | mismatch |
| `declare_effect` (twin) | 560 | honest |
| `honest_rewrite` | 560 | honest |
| `weaken_description` | 129 | mismatch |

Not every tool gets all five pairs:
- **466 tools:** original + inject, twin and rewrite (4 pairs).
- **94 tools:** all 5 pairs.
- **35 tools:** original + weaken.
- **168 tools:** only the original. 95 of them drew `read` as their injection target (now
  skipped); the rest failed a quality check along the way (see `rejects.jsonl`).

### Mismatch types

- **Injected (560):** 7 effect classes, 91–105 each (modify, send outward, execute, change
  authority, spend, relocate, destroy). No `read`. Spread evenly over 4 positions: start,
  middle, end, and inside a helper function.
- **Weakened (129):** skewed toward modify (31) and execute (24). This type is the thin spot in
  the dataset.

### Sizes

- Code: median 25 lines, max 60. Comments and docstrings are stripped.
- Descriptions: median 200 characters, max 352.
### Token lengths (GraphCodeBERT tokenizer, all 2,572 pairs, measured 2026-10-23)

| Input | Median | p95 | Max | Over 512 |
|---|---|---|---|---|
| Code alone | 466 | 879 | 1,111 | — |
| Description + code | 508 | 932 | 1,172 | **49.6%** |
| Name + params (compact) + description + code | 562 | 994 | 1,211 | 58.4% |
| Name + params (raw JSON) + description + code | 610 | 1,050 | 1,252 | 65.9% |
| Description + code, indentation cut to 1 space per level | 438 | 763 | 1,022 | 36.4% |
| Description + code, all whitespace collapsed | 405 | 715 | 974 | 29.0% |
| Name + params (compact) + description + code, whitespace collapsed | 459 | 784 | 1,019 | 39.0% |

- Code is long in tokens for ~25 lines because the tokenizer makes **one token per indentation
  space**: ~21% of code tokens are whitespace.
- **Mismatches are longer:** over 512 for 60.5% of mismatches vs 45.6% of honest pairs
  (description + code). Truncation would hit mismatches harder.
- Lengths include the special tokens of a pair encoding (`tok(first_segment, code)`).
  "Compact" params = `name (type, required): description` per parameter, joined with `; `.
- **Chunks per pair** under the chosen input (last row above, chunks of 512 with a 128-token
  overlap): 1 chunk 61.0%, 2 chunks 33.8%, 3+ chunks 5.2%. Mismatches: 51.1 / 39.9 / 9.0%;
  honest: 64.6 / 31.5 / 3.9%.
- Rerun with `python scripts/token_lengths.py [pairs.jsonl]`.

### Splits (by tool, so no tool spans two splits)

| Split | Pairs | Tools | Mismatch |
|---|---|---|---|
| train | 2,062 | 610 | 26.7% |
| val | 259 | 76 | 28.2% |
| test | 251 | 77 | 26.3% |

### Shortcut check

Guessing "honest" every time scores 73.2%. A classifier that only sees description length
scores 72.6%, and one that only sees description words scores 74.2%. Neither does better than
guessing, so the labels can't be read off the surface. (`stats.json`, `shortcut_baseline`.)

### Fields in each row

| Field | Meaning |
|---|---|
| `pair_id` | unique id, e.g. `c0002-1/inject-relocate-start` |
| `source_id` | the honest tool this pair came from; groups a tool's pairs (group accuracy) |
| `name`, `params`, `description` | the tool's interface (params is a JSON schema) |
| `code` | normalized Python source |
| `mismatch` | the label (`true` = mismatch) |
| `undeclared` | which effect is hidden, e.g. `["destroy"]`; empty when honest |
| `mutation` | pair type (`null` for the original) |
| `injected_lines` | line numbers of the hidden operation (inject, twin, weaken) |
| `generator` | which LLM made it |
| `meta` | domain, declared effects, injection position, soft-check results |

### Quality checks done by hand

- 33 injections that the word-list checker didn't recognize: none had the wrong effect.
- All 70 `read` injections: only 8 were real mismatches, so `read` injections were dropped.
- Pilot hand checks of a few pairs and weakened descriptions: quality on par with Opus.

---

## 4. Working agreement: ask before architecture decisions

This applies to Claude Code, and is a good rule for the team too.

Stop and ask before deciding any of the following. Do not implement first and explain after.

- adding, removing or restructuring modules, packages or the directory layout
- data formats and schemas (dataset fields, label format, split strategy)
- model choices: base model, classification head, loss, input construction, max length / chunking
- evaluation metrics or what counts as a test set
- adding a new dependency
- anything that changes a decision in section 5

How to ask: give 2–3 options with one or two lines of tradeoff each, say which you recommend and
why, then wait for an answer. Keep it short and plain.

No approval needed for routine work: implementing a design that is already agreed, bug fixes,
tests, refactors inside one module, docs. When unsure whether something is architectural, ask.

---

## 5. Current decisions

### Model

- **One detector:** a **GraphCodeBERT cross-encoder** (`microsoft/graphcodebert-base`).
  Description and code go in together as one input; a classification head on `[CLS]` gives
  the verdict. (Earlier two-tier / trace-auditing designs were dropped.)
- **Head (decided 2026-10-26):** GraphCodeBERT is released as an encoder only, with no
  classifier. `build_model` adds transformers' standard RoBERTa classification head, randomly
  initialized and **trained by us**: `[CLS]` (768) → dropout → dense 768→768 → tanh →
  dropout → linear 768→1 (or 768→8 for multi-label). Two layers, not one: the same head is
  used in the fine-tuned model and in the frozen "before" baseline, so the only difference
  between the two runs is whether the encoder is fine-tuned. A single linear layer would make
  the baseline weaker and flatter fine-tuning.
- Start **binary** (mismatch yes/no), then **multi-label** (one sigmoid output per effect class).
- Start **text-only**. Adding data flow (tree-sitter) is a later ablation.
- **Input (decided 2026-10-26):** first segment = name + compact params + description
  (`format_interface`); second segment = code with all whitespace collapsed to single spaces
  (`collapse_whitespace`). If that's over 512 tokens, the code is split into overlapping
  windows (128 tokens shared between neighbours), each a full input
  `<s> interface </s></s> code slice </s>` with the interface repeated. Each chunk gets a logit;
  the pair's logit is the **max over its chunks** (per label for multi-label), and the loss is
  computed on that pooled logit (multiple-instance learning: a pair is a mismatch if any chunk
  shows an undeclared operation). Never truncate.
- The model says *whether*; it cannot say *where*. Line numbers in reports come from the AST
  (`localize.py`).

### Training (decided 2026-10-26)

- **Where:** USC CARC (SLURM job script still to write). Colab notebook as a fallback. Never a
  laptop (a full run froze a Mac).
- **Loss:** BCE with `pos_weight` = honest / mismatch count in train (~2.7), on the max-pooled
  pair logit.
- **Batching:** 8 pairs per step (each pair brings 1–3+ chunks), gradient accumulation 2
  (effective 16 pairs), fp16 on GPU. AdamW, lr 2e-5, weight decay 0.01, 10% linear warmup,
  grad clip 1.0, 5 epochs, seed 566.
- **Model selection:** keep the epoch with the best val PR-AUC. The threshold with the best F1
  on val is saved (`threshold.json`) and reused unchanged when scoring `test`. `train.py` never
  reads `test`.
- **"Before" baseline** (`binary_frozen.yaml`): encoder frozen, only the 2-layer head trains
  (see "Head" above), lr 1e-3, 10 epochs.

### Evaluation (decided 2026-10-05)

- **Test sets:** rungs 1–3 of the test ladder (section 6), each reported separately, plus the
  Agent Audit coverage set.
- Rung 1 is `data/splits/full/test.jsonl`. Tune on `val`; score `test` **once**, at the end.
- **Metrics per test set:** precision / recall / F1 on the mismatch class, and PR-AUC. Not plain
  accuracy (only ~27% of pairs are mismatches).
- **Breakdowns:** recall per effect class, per injection position, and inject vs weaken; plus
  precision and recall by chunk count (1 / 2 / 3+), to catch a "more chunks means mismatch"
  leak.
- **Group accuracy:** share of source tools where *every* pair is right (mutant flagged, twin
  and rewrite cleared). The hardest number to get by shortcut.
- **Before vs after fine-tuning:** "before" = frozen GraphCodeBERT + the same 2-layer head,
  trained (an untrained head is just random). In writing: "a frozen GraphCodeBERT encoder with a
  trained 2-layer classification head", not "linear head".
- **Agent Audit** (the professor's tool) is the only baseline. Frame it as coverage: run both on
  the same set (our mismatches + some conventional bugs like eval / shell injection) and report
  what each catches and the union.

### Dataset (decided 2026-09-29, updated since)

What the dataset contains is in section 3. This lists **why each piece exists:**
- **Injection by LLM rewrite:** the LLM adds one *ordinary* operation of the target effect
  (move, update, refund; never eval or attacker URLs) at a target position (start / middle /
  end / helper). An AST check rejects bad outputs. Templates were rejected because the model
  would just learn the snippet bank.
- **Honest twin:** without it, "the code got longer" would predict the label.
- **Length-matched mismatch description** (within ±15% of the twin): otherwise twins are
  always longer and length leaks the label (pilot: a length-only classifier hit 72% vs 61%
  majority).
- **Honest rewrite pair** (2026-10-05): without it, the padded rewritten descriptions only ever
  appeared as mismatches, and the description alone separated them (78% vs 50% chance).
- `stats.json` reports a **length-only** and a **description-only** (tf-idf) baseline on every
  run. Both should stay near the majority rate.
- **Normalization by AST round-trip:** parse, drop docstrings, `ast.unparse`. Removes comments
  and makes formatting uniform. The model input then collapses all whitespace
  (`model/dataset.py`). Both steps must run on real code at scan time: `normalize_code`, then
  `collapse_whitespace`.
- **Split by source tool:** a tool and all its variants share a split; duplicate tool names are
  dropped so near-copies can't cross splits. Never split randomly.
- **No `read` injections** (2026-10-14, `skip_inject_effects: [read]` in
  `configs/datagen.yaml`): an undeclared read changes nothing, and only matters if the data
  leaves, which `send outward` covers. A hand check of all 70 read injections found only 8 real
  mismatches. Honest tools can still *declare* `read`. Known blind spot: a tool that returns
  more data to the agent than it says. The skip happens after the effect is picked, so other
  tools' targets (and cache keys) don't change, but ~1/8 of tools get no injected mutant.
- **Generator: Claude Sonnet 5.5** (`claude-sonnet-5-5`) via `claude -p --json-schema`, on plan
  usage. Switched from Opus on 2026-10-05 (similar quality, less than half the cost). Results
  are cached on disk, keyed by model.

---

## 6. Effect classes and test ladder

**Effect classes:** read · modify · destroy · relocate · send outward · execute ·
spend/transact · change authority

The set is small and closed; the code that produces each effect is unbounded. That is the
project's core argument for learning over rules.

**Test ladder** (report each separately):
1. Held-out generated pairs (sanity check).
2. Pairs from a *different* LLM, same recipe (tests whether the model memorized the generator's
   style). Use a different `seed`: the same seed reuses the same tool requests (domain +
   effects), so the old Opus pilot's 40 tools mirror the first 40 training tools and must not be
   scored.
3. Real GitHub tools with a capability we inject (real code, known labels).

(Rung 4, real mismatches found in the wild, was dropped for time on 2026-10-05.)

**Data hygiene** (the model will cheat otherwise): no comments on injected lines; vary
injection position, domains, naming and length; keep a tool and its mutants in the same split;
the AST check confirms the added operation is in the mismatch code, not in the honest code, and
that the name/params don't give it away.

---

## 7. Files

Package code is in `src/agentlint/` (full layout in `README.md`). Tests: `pytest`.

- `src/agentlint/data/`: dataset pipeline (`build.py` runs generate → verify → inject + twin →
  match → weaken → split)
- `scripts/token_lengths.py`: input lengths vs the 512-token limit (section 3)
- `src/agentlint/effects.py`: effect definitions and the word lists for the soft checks
- `src/agentlint/model/`: input construction, model, training loop (`train.py`)
- `src/agentlint/eval/`: `metrics.py` (all section 5 metrics and breakdowns); test-set
  scoring not written yet
- `configs/datagen.yaml`: dataset settings; `configs/binary.yaml`, `binary_frozen.yaml`
  (before-fine-tuning baseline), `multilabel.yaml`: training
- `notebooks/train_binary.ipynb`: Colab notebook that trains both binary models
- `docs/agentlint-preproposal.md`: current proposal
- `probes/aa_run/`: generated Agent Audit cases (keep the path neutral; see Gotchas)

---

## 8. Agent Audit: verified facts (v0.19.2, commit 1009782)

Use these exact claims in writing. Earlier, looser versions were wrong.

- Detection is entirely rule-based. The paper (Zhang, Nian & Zhao, 2026, arXiv 2603.22853)
  names "learning-based vulnerability detection" as future work.
- The docstring IS extracted (`scanners/python_scanner.py:1128` and `:1160`,
  `ast.get_docstring`) into `ToolDefinition.description`, but never used for detection. No rule
  compares the description to inferred permissions.
- Description *content* is analyzed only when it comes from an MCP config file (AGENT-056,
  `scanners/mcp_config_scanner.py:1332`) or a live MCP server (`inspect` subcommand,
  `scanners/mcp_inspector.py`). It checks whether a description is *malicious*, not *accurate*.
- Permission inference is a fixed dict of ~20 call names (`python_scanner.py:61–81`). `'open'`
  maps to FILE_READ regardless of mode. `Path.rename` and `conn.execute` are not in it;
  `shutil.move` is (FILE_WRITE).
- AGENT-002 only counts (`rules/engine.py:808`, `:835`): >15 tools or >5 high-risk tools.
- AGENT-003 (data exfiltration chain) is declared in YAML but **not implemented**: nothing reads
  `operation_chain`, and `_evaluate_rule` only handles patterns of type python_ast /
  function_call / regex.
- The secret scanner reads whole-file text, so a credential inside a docstring would still be
  flagged. Do not write "never reads the docstring."

**Probe results:** in both rounds, a known-vulnerable control gave 4 findings (3 critical) and a
capability mismatch gave 0. In round 2, a minimal pair (byte-identical except 4 lines that
cancel an order and write a refund row) gave indistinguishable output. A hostile docstring
("ignore all previous instructions…") gave 0.

---

## 9. Gotchas

- **macOS: `No module named 'agentlint'`** even though `pip list` shows it. macOS sets the
  `hidden` flag on the editable-install `.pth` file, and Python 3.13 skips hidden `.pth` files.
  Fix: `chflags nohidden .venv/lib/python3.13/site-packages/*.pth` (check with `ls -lO`).
  pytest still passes because it finds `src/` on its own, so it won't catch this.
- **Agent Audit suppresses findings in test-looking paths.** Any path containing `/fixtures/`,
  `/tests/`, `/examples/`, `/demos/`, `/samples/`, `/mocks/` etc. is downgraded to SUPPRESSED
  (`analysis/context_classifier.py`). Put probe files under neutral names
  (e.g. `aa_run/<case>/order_service/tools/orders.py`).
- **Always include a positive control** (eval / `shell=True`) in any Agent Audit run. A silent
  result without a firing control proves nothing.
- **512-token limit.** About half of `full` is over it (section 3). Never truncate: truncation
  can cut the exact lines that carry the label, and mismatches are longer than honest pairs.
  Indentation is expensive (one token per space), hence `collapse_whitespace` + chunking.
- **Don't use the tokenizer's own chunking.** In transformers 5.x,
  `return_overflowing_tokens=True` on a (text, code) pair stops after one extra chunk and
  silently drops the rest of the code (3,200 code tokens came back as 512 + 142). transformers
  5 also removed `build_inputs_with_special_tokens`. `encode_pair` builds the windows and the
  RoBERTa pair layout by hand; a test checks it matches `tok(a, b)` for short pairs.
- **Don't train on a laptop.** A full `agentlint train` run froze a Mac (2026-10-26):
  ~2,000 pairs, up to 3+ chunks of 512 each, through a 125M-parameter encoder. Train on USC CARC or Colab
  (`notebooks/train_binary.ipynb`). Locally, set `limit_train` / `limit_val` in a copy of the
  config for a quick check.
- **Generation prompts must not sound like concealment.** Asking for an undeclared operation
  with "no comments, no logging, must look ordinary" got 23/40 refusals (read as planting a
  backdoor). Stating the detector-training purpose and asking for an ordinary feature got 0/40.
  Comments in LLM output are harmless: normalization strips them.
- **Don't trust the word-list check as a label check.** `meta.checks.lexicon_agrees = false`
  usually means the word list is missing a pattern, not that the mutant is wrong (0 of 33
  hand-checked had the wrong effect).

---

## 10. Writing style for proposal text

Keep prose simple and plain. Short sentences, concrete examples, no jargon where a plain word
works.

---

## 11. Progress log

Newest first. Add an entry when a milestone lands or a plan changes.

- **2026-10-26 (later):** Wrote the binary training loop, the metrics module (with tests),
  the frozen-encoder baseline config and the Colab notebook. Not run yet: a local training run
  froze the Mac, so training moves to USC CARC. Kept transformers' standard 2-layer RoBERTa head
  (trained from scratch by us) for both runs instead of a single linear layer (section 5,
  "Head").

- **2026-10-26:** Team picked option 2 for long inputs: collapse whitespace, chunk (512, 128
  overlap, interface repeated), max-pool chunk logits, loss on the pooled logit. Implemented
  input construction and pooling in `model/dataset.py` and `model/classifier.py`, with tests;
  smoke-tested a forward pass on val. On `full`, 39% of pairs need 2+ chunks (48.9% of
  mismatches vs 35.4% of honest pairs), so chunk count is now a required val breakdown.

- **2026-10-23:** Measured token lengths on `full` with the GraphCodeBERT tokenizer:
  description + code is over 512 for 49.6% of pairs (median 508, max 1,172), 29% after
  collapsing whitespace. Mismatches run longer than honest pairs. Opened a team decision on
  long inputs (section 2); recommended collapsing whitespace + chunk and max-pool.
- **2026-10-15:** Moved the repo to the team's GitHub (`tomkkl/mismatch-detection`). Reorganized
  this file for the team.
- **2026-10-14:** Dropped `read` injections (section 5). Rebuilt `full` from cache with 0 new
  calls: 2,572 pairs, 689 mismatches; 560 inject / 560 twin / 560 rewrite / 129 weakened.
  Baselines: length 72.6%, description-only 74.2% vs majority 73.2%.
- **2026-10-13:** Full Sonnet run finished: 763 tools, 2,782 pairs, 744 LLM calls, $5.91 at list
  price. Hand-checked 33 injections the word list didn't confirm: none had the wrong effect.
  Checked all 70 `read` injections: 8 real, 45 weak, 17 effectively honest. Cause: the read
  hint in `effects.py` asked for a read "the tool does not need to mention".
- **2026-10-05:** Re-ran the pilot on macOS. Found a description-only leak (78% vs 50%) and
  fixed it with the honest rewrite pair. Added one retry for twins that miss the length bounds
  (9/10 recovered). Switched the generator from Opus to Sonnet 5.5: pilot quality on par, $1.84
  vs $4.10. Decided the evaluation plan; dropped rung 4.
- **2026-09-29:** Pipeline built. First pilot on Windows: 23/40 refusals and a length leak
  (72% vs 61%). Fixed with honest prompt framing and length-matched mismatch descriptions.
