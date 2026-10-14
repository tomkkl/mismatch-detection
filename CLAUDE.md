# AgentLint — CSCI 566 course project

Detect LLM agent tools whose **implementation does more than their declared interface says**
(name + parameter schema + description). Example: a tool described as "List the files in a
project folder" that also moves the whole directory. The code is fine on its own and the
description is fine on its own; only the pair is wrong.

Team of 6, ~10 weeks, Fall 2026. The course requires a trained/fine-tuned model.

## Working agreement: ask before architecture decisions

Stop and ask the user before deciding any of the following. Do not implement first and explain after.

- adding, removing or restructuring modules, packages or the directory layout
- data formats and schemas (dataset fields, label format, split strategy)
- model choices: base model, classification head, loss, input construction, max length / chunking
- evaluation metrics or what counts as a test set
- adding a new dependency
- anything that changes a decision under "Current decisions" below

How to ask: give 2–3 options with one or two lines of tradeoff each, say which you recommend and
why, then wait for an answer. Keep it short and plain.

No approval needed for routine work: implementing a design that is already agreed, bug fixes,
tests, refactors inside one module, docs. When unsure whether something is architectural, ask.

When a decision is made, update "Current decisions" in this file as part of the same change.

## Current decisions

- Single detector (earlier two-tier / trace-auditing designs were dropped).
- Core model: **GraphCodeBERT cross-encoder** (`microsoft/graphcodebert-base`). Description and
  code go in together as one input; a classification head on `[CLS]` outputs the verdict.
  - Start **binary** (mismatch yes/no), then move to **multi-label** (one sigmoid head per
    effect class).
  - Start **text-only** (no data-flow graph input). Adding data flow via tree-sitter is a later
    ablation.
- Benchmark against **Agent Audit** only (the professor's tool). Frame as coverage: run both on
  the same test set (mismatches + some conventional bugs like eval/shell injection) and report
  what each catches and the union. Also report GraphCodeBERT before vs after fine-tuning.
- **Evaluation** (decided 2026-10-05):
  - Test sets = rungs 1–3 of the test ladder (see Data plan), each reported separately, plus
    the Agent Audit coverage set. Rung 4 (real mismatches in the wild) is **dropped** for time.
  - Rung 1 is `data/splits/<tag>/test.jsonl`. Tune on `val`; score `test` once at the end.
  - Metrics per test set: precision / recall / F1 on the mismatch class, and PR-AUC. Not plain
    accuracy (mismatch is only ~30% of pairs).
  - Breakdowns: recall per effect class, per injection position, and inject vs weaken.
  - **Group accuracy**: share of source tools where every pair is right (mutant flagged, twin
    and rewrite cleared). The hardest number to get by shortcut.
  - "Before fine-tuning" = frozen GraphCodeBERT encoder + trained linear head (an untrained
    head is just random).
- Dataset construction (decided 2026-09-29):
  - **Injection by LLM rewrite**: the LLM adds one ordinary op of the target effect at the target
    position, no comments. The AST check filters bad outputs. (Templates rejected: the model
    would learn the snippet bank.)
  - **Honest twin** for every injected mutant: same mutant code, description rewritten to declare
    the new effect, labeled honest. Only the description decides the label, so the model can't
    cheat on "longer code / new import = mismatch".
  - **Length-matched mismatch description**: the injected mutant's description is also rewritten
    (from the *honest* code, so the model never sees the added op) to within ±15% of its twin's
    length. Otherwise twins are always longer and length leaks the label (pilot: a length-only
    classifier hit 72% vs 61% majority). `stats.json` reports a length-only baseline every run.
  - **Honest rewrite pair** (decided 2026-10-05): the honest code + the mutant's length-matched
    description, labeled honest (no LLM call). Without it, originals are short and matched
    rewrites are padded, so the description alone separated original vs mutant (length 78%,
    words 76%, chance 50%) even though the overall length baseline looked fine. Now every
    mismatch description also appears as honest. `stats.json` also reports a description-only
    (tf-idf) baseline.
  - **Normalization by AST round-trip**: parse, drop docstrings, `ast.unparse`. Removes all
    comments and makes formatting uniform. The same step must run on real code at scan time.
  - **Split by source tool**: an honest tool, its mutants and its twins share a split; duplicate
    tool names are dropped so near-copies can't cross splits.
  - **Generator: Claude Sonnet 5.5 (`claude-sonnet-5-5`) via the `claude` CLI**
    (`claude -p --json-schema`), which runs on the user's plan usage, not the paid API. Switched
    from Opus on 2026-10-05 at the user's request. Results are cached on disk (keyed by model)
    so runs resume across usage windows. Pilot (~40 honest tools) first, then the full run.
    The Opus pilot outputs remain in the cache; Opus is a natural "different LLM" for test rung 2.
  - **No `read` injections** (decided 2026-10-14): `skip_inject_effects: [read]` in
    `configs/datagen.yaml`. An undeclared read changes nothing; it only matters when the data
    leaves, and that is covered by `send outward`. A hand check of all 70 read injections found
    only 8 real mismatches. Honest tools may still *declare* `read`; weaken never targets it.
    Known blind spot: a tool that returns more data to the agent than it says. If wanted later,
    define a narrower class ("returns undeclared sensitive data") rather than general read.
    The skip happens *after* the effect is picked, so other tools' targets (and cache keys) don't
    change. Side effect: ~1/8 of tools get no injected mutant (95 in `full`).

## Where we are (2026-10-14)

**Done:** generation pipeline; full training dataset `full` (see the 2026-10-14 log entry).
`data/generated/full/pairs.jsonl`, `stats.json`, `rejects.jsonl`; splits in `data/splits/full/`.

**Next, roughly in order:**
1. **Measure token lengths** of description + code with the GraphCodeBERT tokenizer on `full`
   (see Gotchas: 512-token limit; chunk + max-pool rather than truncate).
2. **Training (binary first)**: `src/agentlint/model/{dataset,classifier,train}.py`,
   `configs/` training config, Colab notebook in `notebooks/`. Use a weighted loss (mismatch
   is ~27%). Also train the "before fine-tuning" baseline (frozen encoder + linear head).
   As of 2026-10-14 these are stubs (5–17 lines each; `eval/` too), and `configs/binary.yaml` /
   `multilabel.yaml` exist but are unchecked. `notebooks/` is empty. Ask before model-design choices
   not covered by "Current decisions" (e.g. max length, chunking, loss weights).
3. **Test rung 2**: fresh-seed Opus set, ~100–150 tools, same recipe
   (`--tag rung2-opus`, set `generator.model` and a new `seed`). Must not reuse seed 566.
   With a fresh seed nothing is cached, so the skip-after-pick wastes ~1/8 of tools; fine, or
   bump `n_honest` to compensate.
4. **Test rung 3**: collect real tool functions (LangChain / OpenAI Agents tools, Python MCP
   servers), normalize, inject with the same pipeline. Nothing built yet; collection format is
   an open design question (ask).
5. **Evaluation code** (`src/agentlint/eval/metrics.py`) for the metrics under "Evaluation",
   including group accuracy; Agent Audit coverage set + `eval/agent_audit.py`.

**Open issues (not blocking):**
- Weakened mismatches are few (129, ~5% of pairs): 288 tools had no weakenable effect.
- Lexicon soft check (`effects.py`) misses SQL-table forms of effects (payment rows, role
  tables, insert-select-then-delete) and `stripe.PaymentIntent.create`. Labels are fine (hand
  check), but fix before using `localize.py` for line numbers in reports.
- Description-only baseline is ~1 point above majority (74.2% vs 73.2%). Small; watch it.

## Progress log

Newest first. Add an entry when a milestone lands or a plan changes.

- **2026-10-14** — Dropped `read` injections (see decisions). Rebuilt `full` from cache
  (0 new calls): 763 sources, **2,572 pairs, 689 mismatch (26.8%)**: 560 inject, 560 twin,
  560 honest rewrite, 129 weakened. Splits: train 2,062 / val 259 / test 251. Effects 91–105
  each across the 7 remaining classes; positions 137–142 each. Baselines: length 72.6%,
  description-only 74.2% vs majority 73.2%.
  Also found why the venv kept losing `agentlint`: see the hidden `.pth` gotcha.

- **2026-10-13** — Full Sonnet run finished (`--tag full`, resumed after a usage limit; had to
  `pip install -e . --no-deps` because the venv's editable install was stale). 763 source
  tools, 2,782 pairs (759 mismatch, 27%): 630 inject, 630 twin, 630 honest rewrite, 129
  weakened. Splits: train 2,224 / val 274 / test 284. Shortcut baselines at majority
  (length 72.3%, description-only 73.7% vs 72.7%). 744 calls, $5.91 list. Weakened is only
  129 (4.6%): 288 tools had no weakenable effect. Undeclared effects balanced (70–105 each;
  `read` lowest). 153/630 injections not confirmed by the lexicon check.
  Hand check of 33 of those 153 (spend 12, authority 9, relocate 7, other 5): 0 had the wrong
  effect; 30 clearly right. The lexicon just misses SQL-table versions of the effects (payment
  rows, role tables, insert-select-then-delete) and even `stripe.PaymentIntent.create`.
  Weak spot is `read`: 2/2 sampled were trivial metadata reads (object size, DB size) inside
  tools that already read that data, arguably honest. One relocate had a junk dead line.
  Then checked all 70 `read` injections: 8 clear (returns data the description doesn't cover:
  customer email, payment history, a manager's email), 45 weak (reads a related field and
  logs/returns it), 17 effectively honest (validation, duplicate checks, unused reads). Cause:
  the read injection hint in `effects.py:33` literally asks for a read "the tool does not need
  to mention". Weaken never targets `read`. (Resolved 2026-10-14: read injections dropped.)

- **2026-10-05** — Re-ran the 40-tool pilot on macOS (Windows cache didn't carry over).
  0 refusals; 108 pairs; length baseline 65.8% vs 63.9% majority. But a per-contrast check
  found a description-only leak (original vs mutant: length 78%, words 76%). Fixed with the
  honest rewrite pair (see decisions). Rebuilt pilot: 137 pairs, length 70.8% / words 71.5%
  vs 71.5% majority. Main yield loss: 10/39 twins rejected for length (each drops 4 pairs);
  only 10 weakened mismatches (13 tools had no weakenable effect).
  Added one retry for twins that miss the length bounds (the retry prompt states the allowed
  range): 9/10 recovered, pilot now 164 pairs (48 mismatch), baselines still at majority
  (length 70.1%, words 70.7% vs 70.7%). Started the full 800-tool run (`--tag full`,
  projected ~3,300 pairs).
  Open: weakened mismatches are few (~6% of pairs); mismatch share ~29%, plan a weighted loss.
  Then switched the generator to Sonnet 5.5 (user request); stopped the Opus full run at
  20/200 generate calls. Sonnet pilot (`--tag pilot-sonnet`): 160 pairs (48 mismatch),
  0 refusals, 8/36 twins needed the retry; length 68.1% / words 70.0% vs 70.0% majority;
  original+rewrite vs mutant at majority (66.7%). Hand check of 8 pairs + 3 weakened: quality
  on par with Opus. 151 calls, $1.84 list (vs $4.10 for Opus). Started the full Sonnet run
  (`--tag full`).
  Evaluation plan decided (see Current decisions): rungs 1–3 + Agent Audit coverage set; rung 4
  dropped for time. Still to build: rung 2 (fresh-seed Opus set) and rung 3 (real GitHub tools).
- **2026-09-29** — Pipeline built (generate → verify → inject + twin → match → weaken →
  split). First pilot on Windows: 23/40 refusals and a length leak (72% vs 61%); fixed by
  honest prompt framing and length-matched mismatch descriptions.

## Effect classes

read · modify · destroy · relocate · send outward · execute · spend/transact · change authority

The effect set is small and closed; the code that produces it is unbounded. That is the
project's core argument for learning over rules.

## Data plan

- **Train:** LLM-generated honest tools + mismatched versions (target 2,000–5,000 pairs), balanced
  across effect classes.
- **Test ladder, report each separately:**
  1. held-out generated pairs (sanity)
  2. pairs from a *different* LLM, same recipe (tests generator-style memorization). Use a
     different `seed` from the training run: the same seed reuses the same tool requests
     (domain + effects), so the Opus pilot's 40 tools mirror the first 40 Sonnet train tools
     and must not be scored. Proposed: fresh-seed Opus set, ~100–150 tools (not yet generated).
  3. real GitHub tools with a capability we inject (real code, known labels). Not built yet:
     collect real tool functions (LangChain / OpenAI Agents tools, Python MCP servers),
     normalize, inject with the same pipeline.
  (Rung 4, real mismatches found in the wild, dropped for time on 2026-10-05.)
- Mismatches in both directions: add undeclared code, OR weaken the description with code unchanged.
- **Hygiene (the model will cheat otherwise):**
  - no comments on injected lines; strip or randomize comments everywhere
  - vary injection position (start / middle / end / inside a helper)
  - vary domains, naming, length
  - keep an honest tool and all its mutants in the SAME split; split by source/repo, never randomly
  - auto-verify labels with an AST check: injected op present in mismatch, absent in honest;
    name/params do not reveal it
- Injected code must be *ordinary* operations (move, update, refund), never obviously malicious
  code like eval or attacker URLs.

## Verified facts about Agent Audit (agent-audit v0.19.2, commit 1009782)

Use these exact claims in writing — earlier looser versions were wrong.

- Detection is entirely rule-based. Paper (Zhang, Nian & Zhao, 2026, arXiv 2603.22853) names
  "learning-based vulnerability detection" as future work.
- The docstring IS extracted (`scanners/python_scanner.py:1128` and `:1160`, `ast.get_docstring`)
  into `ToolDefinition.description`, but never used for detection. No rule compares description
  to inferred permissions.
- Description *content* is analyzed only when it comes from an MCP config file (AGENT-056,
  emitted at `scanners/mcp_config_scanner.py:1332`) or a live MCP server (`inspect` subcommand,
  `scanners/mcp_inspector.py`). It checks whether a description is *malicious*, not *accurate*.
- Permission inference is a fixed dict of ~20 call names (`python_scanner.py:61–81`).
  `'open'` maps to FILE_READ regardless of mode. `Path.rename`, `conn.execute` are not in it.
  `shutil.move` is (FILE_WRITE).
- AGENT-002 is implemented as counting only (`rules/engine.py:808`, `:835`): >15 tools or >5
  high-risk tools.
- AGENT-003 (data exfiltration chain) is declared in YAML but **not implemented** — nothing reads
  `operation_chain`; `_evaluate_rule` only handles `patterns` of type python_ast / function_call /
  regex.
- The secret scanner reads whole-file text, so a credential inside a docstring would still be
  flagged. Do not write "never reads the docstring."

## Probe results so far

Both rounds: known-vulnerable control → 4 findings (3 critical); capability mismatch → 0.
Round 2 minimal pair (byte-identical except 4 lines that cancel an order and write a refund row)
gave indistinguishable output. A hostile docstring ("ignore all previous instructions…") → 0.

## Gotchas

- **`ModuleNotFoundError: No module named 'agentlint'` on macOS** even though `pip list` shows
  it: macOS sets the `hidden` file flag on the editable-install `.pth` file, and Python 3.13
  skips hidden `.pth` files. Fix: `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`
  (check with `ls -lO`). Reinstalling also works until the flag comes back. pytest still passes
  because it finds `src/` on its own, so it doesn't catch this.

- **Agent Audit suppresses findings in test-looking paths.** Any path containing
  `/fixtures/`, `/tests/`, `/examples/`, `/demos/`, `/samples/`, `/mocks/` etc. is downgraded to
  SUPPRESSED (`analysis/context_classifier.py`). Put probe files under neutral names
  (e.g. `aa_run/<case>/order_service/tools/orders.py`).
- Always include a positive control (eval / shell=True) in any Agent Audit run; a silent result
  without a firing control proves nothing.
- GraphCodeBERT max input is 512 tokens. Measure the length distribution early; chunk + max-pool
  instead of truncating (truncation can cut the exact lines that carry the label).
- The model cannot localize. Model says *whether*; AST says *where* (line numbers in the report).
- **Generation prompts must not sound like concealment.** Asking Opus for an undeclared op with
  "no comments, no logging, must look ordinary" got 23/40 refusals (it read it as planting a
  backdoor). Stating the detector-training purpose and asking for an ordinary feature got 0/40.
  Comments in LLM output are harmless: normalization strips them.

## Environment

Windows, PyCharm, project at `C:\Users\denni\PycharmProjects\agent-auditor`, `.venv`,
`pip install -e ".[dev,baseline]"` (includes agent-audit v0.19.2). Training on Colab T4 is fine
for this model size.

Also used on macOS (Python 3.13): `python3 -m venv .venv && .venv/bin/pip install -e ".[dev,baseline]"`.
`data/generated/` (including `llm_cache/`) is git-ignored, so it does not move between machines;
copy it by hand to resume a run elsewhere.

## Files

Package code is in `src/agentlint/` (layout in `README.md`). Tests: `pytest`.

- `docs/agentlint-preproposal.md` — current proposal
- `probes/probe_agent_audit_v2.py` — round 1 probe (5 cases, filesystem)
- `probes/probe_round2.py` — round 2 probe (minimal pair, database)
- `probes/demo_agent_audit.py` — team demo (control / honest / tool that secretly moves its directory)
- `probes/aa_run/` — generated Agent Audit cases (keep path neutral; see Gotchas)

## Writing style for proposal text

Keep prose simple and plain. Short sentences, concrete examples, no jargon where a plain word works.
