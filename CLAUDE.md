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
- Training runs on Colab (a T4 is enough for this model size).
- macOS: if `agentlint` suddenly can't be imported, see "Gotchas".

### Getting the dataset

`data/` is git-ignored (too large; share via Drive). Ask Dennis for a copy of:

- `data/splits/full/` (`train.jsonl`, `val.jsonl`, `test.jsonl`): what you need to train.
- `data/generated/full/`: `pairs.jsonl` (all pairs), `stats.json` (counts and leak
  baselines), `rejects.jsonl` (what was filtered and why).
- `data/generated/llm_cache/`: only needed to rebuild or extend the dataset without paying for
  the LLM calls again.

### Rebuilding the dataset

```
agentlint data build configs/datagen.yaml --tag full --n-honest 800
```

Generation runs through the `claude` CLI (log in with `claude` first), so it uses a Claude
plan's usage, not the paid API. If a usage limit stops the run, rerun the same command later:
finished calls are cached in `data/generated/llm_cache/` and reused.

---

## 2. Where we are (updated 2026-10-14)

**Done**
- Agent Audit probes: it misses capability mismatches entirely (see section 7).
- Data generation pipeline (`src/agentlint/data/`), with tests.
- Training dataset `full`: **2,572 pairs from 763 source tools, 689 mismatches (26.8%)**.
  Splits: train 2,062 / val 259 / test 251. Balanced across the 7 injected effects (91–105
  each) and the 4 injection positions.

**Next, roughly in order**
1. **Measure token lengths** of description + code with the GraphCodeBERT tokenizer on `full`.
   The limit is 512 tokens; plan to chunk + max-pool rather than truncate.
2. **Binary training.** `src/agentlint/model/{dataset,classifier,train}.py`, a training config
   in `configs/`, a Colab notebook in `notebooks/`. Use a weighted loss (mismatch is ~27%).
   Also train the "before fine-tuning" baseline (frozen encoder + linear head).
   Current state: these files are stubs (5–17 lines each); `configs/binary.yaml` and
   `multilabel.yaml` exist but haven't been checked; `notebooks/` is empty.
3. **Test rung 2:** a fresh-seed Opus set, ~100–150 tools, same recipe
   (`--tag rung2-opus`, change `generator.model` and `seed` in the config). Must not reuse
   seed 566 (see section 5).
4. **Test rung 3:** collect real tool functions (LangChain / OpenAI Agents tools, Python MCP
   servers), normalize them, inject with the same pipeline. Nothing built yet; the collection
   format is still an open design question.
5. **Evaluation code** in `src/agentlint/eval/` (metrics in section 4, including group
   accuracy) and the Agent Audit coverage run.

**Open issues (not blocking)**
- Few weakened mismatches (129, ~5% of pairs): 288 tools had no effect that could be removed
  from the description.
- The word-list effect checker (`effects.py`) misses SQL-table forms of effects (payment rows,
  role tables, insert-then-delete) and even `stripe.PaymentIntent.create`. Labels are fine
  (hand-checked), but fix it before using `localize.py` for line numbers in reports.
- The description-only baseline is ~1 point above majority (74.2% vs 73.2%). Small; watch it.

---

## 3. Working agreement: ask before architecture decisions

This applies to Claude Code, and is a good rule for the team too.

Stop and ask before deciding any of the following. Do not implement first and explain after.

- adding, removing or restructuring modules, packages or the directory layout
- data formats and schemas (dataset fields, label format, split strategy)
- model choices: base model, classification head, loss, input construction, max length / chunking
- evaluation metrics or what counts as a test set
- adding a new dependency
- anything that changes a decision in section 4

How to ask: give 2–3 options with one or two lines of tradeoff each, say which you recommend and
why, then wait for an answer. Keep it short and plain.

No approval needed for routine work: implementing a design that is already agreed, bug fixes,
tests, refactors inside one module, docs. When unsure whether something is architectural, ask.

---

## 4. Current decisions

### Model

- **One detector:** a **GraphCodeBERT cross-encoder** (`microsoft/graphcodebert-base`).
  Description and code go in together as one input; a classification head on `[CLS]` gives
  the verdict. (Earlier two-tier / trace-auditing designs were dropped.)
- Start **binary** (mismatch yes/no), then **multi-label** (one sigmoid output per effect class).
- Start **text-only**. Adding data flow (tree-sitter) is a later ablation.
- The model says *whether*; it cannot say *where*. Line numbers in reports come from the AST
  (`localize.py`).

### Evaluation (decided 2026-10-05)

- **Test sets:** rungs 1–3 of the test ladder (section 5), each reported separately, plus the
  Agent Audit coverage set.
- Rung 1 is `data/splits/full/test.jsonl`. Tune on `val`; score `test` **once**, at the end.
- **Metrics per test set:** precision / recall / F1 on the mismatch class, and PR-AUC. Not plain
  accuracy (only ~27% of pairs are mismatches).
- **Breakdowns:** recall per effect class, per injection position, and inject vs weaken.
- **Group accuracy:** share of source tools where *every* pair is right (mutant flagged, twin
  and rewrite cleared). The hardest number to get by shortcut.
- **Before vs after fine-tuning:** "before" = frozen GraphCodeBERT + a trained linear head (an
  untrained head is just random).
- **Agent Audit** (the professor's tool) is the only baseline. Frame it as coverage: run both on
  the same set (our mismatches + some conventional bugs like eval / shell injection) and report
  what each catches and the union.

### Dataset (decided 2026-09-29, updated since)

**What one source tool turns into.** Each honest tool produces up to 5 pairs:

| `mutation` | Code | Description | Label |
|---|---|---|---|
| `null` (original) | honest | original | honest |
| `inject_code` | honest + one added operation | rewritten from the honest code, length-matched to the twin | **mismatch** |
| `declare_effect` (twin) | same as `inject_code` | rewritten to declare the added operation | honest |
| `honest_rewrite` | honest | same as `inject_code` | honest |
| `weaken_description` | honest | one declared effect removed | **mismatch** |

So the same code shows up with honest and mismatch labels, and so does the same description.
**Only the pair decides the label.** That is the point: the model can't win by spotting
"longer code", "new import" or "padded description".

Key fields per row: `pair_id`, `source_id`, `name`, `params`, `description`, `code`,
`mismatch`, `undeclared` (effect list), `mutation`, `injected_lines`, `meta` (domain, position,
checks).

**Why each piece exists:**
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
  and makes formatting uniform. The same step must run on real code at scan time.
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

## 5. Effect classes and test ladder

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

## 6. Files

Package code is in `src/agentlint/` (full layout in `README.md`). Tests: `pytest`.

- `src/agentlint/data/`: dataset pipeline (`build.py` runs generate → verify → inject + twin →
  match → weaken → split)
- `src/agentlint/effects.py`: effect definitions and the word lists for the soft checks
- `src/agentlint/model/`, `src/agentlint/eval/`: training and evaluation (stubs so far)
- `configs/datagen.yaml`: dataset settings; `configs/binary.yaml`, `multilabel.yaml`: training
- `docs/agentlint-preproposal.md`: current proposal
- `probes/probe_agent_audit_v2.py`: round 1 Agent Audit probe (5 cases, filesystem)
- `probes/probe_round2.py`: round 2 probe (minimal pair, database)
- `probes/demo_agent_audit.py`: team demo (control / honest / tool that secretly moves its
  directory)
- `probes/aa_run/`: generated Agent Audit cases (keep the path neutral; see Gotchas)

---

## 7. Agent Audit: verified facts (v0.19.2, commit 1009782)

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

## 8. Gotchas

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
- **512-token limit.** Measure the length distribution early; chunk + max-pool instead of
  truncating (truncation can cut the exact lines that carry the label).
- **Generation prompts must not sound like concealment.** Asking for an undeclared operation
  with "no comments, no logging, must look ordinary" got 23/40 refusals (read as planting a
  backdoor). Stating the detector-training purpose and asking for an ordinary feature got 0/40.
  Comments in LLM output are harmless: normalization strips them.
- **Don't trust the word-list check as a label check.** `meta.checks.lexicon_agrees = false`
  usually means the word list is missing a pattern, not that the mutant is wrong (0 of 33
  hand-checked had the wrong effect).

---

## 9. Writing style for proposal text

Keep prose simple and plain. Short sentences, concrete examples, no jargon where a plain word
works.

---

## 10. Progress log

Newest first. Add an entry when a milestone lands or a plan changes.

- **2026-10-15:** Moved the repo to the team's GitHub (`tomkkl/mismatch-detection`). Reorganized
  this file for the team.
- **2026-10-14:** Dropped `read` injections (section 4). Rebuilt `full` from cache with 0 new
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
