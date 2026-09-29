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
- Dataset construction (decided 2026-09-29):
  - **Injection by LLM rewrite**: the LLM adds one ordinary op of the target effect at the target
    position, no comments. The AST check filters bad outputs. (Templates rejected: the model
    would learn the snippet bank.)
  - **Honest twin** for every injected mutant: same mutant code, description rewritten to declare
    the new effect, labeled honest. Only the description decides the label, so the model can't
    cheat on "longer code / new import = mismatch".
  - **Normalization by AST round-trip**: parse, drop docstrings, `ast.unparse`. Removes all
    comments and makes formatting uniform. The same step must run on real code at scan time.
  - **Split by source tool**: an honest tool, its mutants and its twins share a split; duplicate
    tool names are dropped so near-copies can't cross splits.
  - **Generator: Claude Opus via the `claude` CLI** (`claude -p --json-schema`), which runs on the
    user's plan usage, not the paid API. Results are cached on disk so runs resume across usage
    windows. Pilot (~40 honest tools) first, then decide on the full run.

## Effect classes

read · modify · destroy · relocate · send outward · execute · spend/transact · change authority

The effect set is small and closed; the code that produces it is unbounded. That is the
project's core argument for learning over rules.

## Data plan

- **Train:** LLM-generated honest tools + mismatched versions (target 2,000–5,000 pairs), balanced
  across effect classes.
- **Test ladder, report each separately:**
  1. held-out generated pairs (sanity)
  2. pairs from a *different* LLM, same recipe (tests generator-style memorization)
  3. real GitHub tools with a capability we inject (real code, known labels)
  4. optional: real mismatches found in the wild, hand-labeled
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

- **Agent Audit suppresses findings in test-looking paths.** Any path containing
  `/fixtures/`, `/tests/`, `/examples/`, `/demos/`, `/samples/`, `/mocks/` etc. is downgraded to
  SUPPRESSED (`analysis/context_classifier.py`). Put probe files under neutral names
  (e.g. `aa_run/<case>/order_service/tools/orders.py`).
- Always include a positive control (eval / shell=True) in any Agent Audit run; a silent result
  without a firing control proves nothing.
- GraphCodeBERT max input is 512 tokens. Measure the length distribution early; chunk + max-pool
  instead of truncating (truncation can cut the exact lines that carry the label).
- The model cannot localize. Model says *whether*; AST says *where* (line numbers in the report).

## Environment

Windows, PyCharm, project at `C:\Users\denni\PycharmProjects\agent-auditor`, `.venv`,
`pip install -e ".[dev,baseline]"` (includes agent-audit v0.19.2). Training on Colab T4 is fine
for this model size.

## Files

Package code is in `src/agentlint/` (layout in `README.md`). Tests: `pytest`.

- `docs/agentlint-preproposal.md` — current proposal
- `probes/probe_agent_audit_v2.py` — round 1 probe (5 cases, filesystem)
- `probes/probe_round2.py` — round 2 probe (minimal pair, database)
- `probes/demo_agent_audit.py` — team demo (control / honest / tool that secretly moves its directory)
- `probes/aa_run/` — generated Agent Audit cases (keep path neutral; see Gotchas)

## Writing style for proposal text

Keep prose simple and plain. Short sentences, concrete examples, no jargon where a plain word works.
