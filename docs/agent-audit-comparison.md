# Design note: Agent Audit vs our model on every test set

Status: **decisions settled 2026-11-11** (section 9); implementation in progress. Drafted
2026-11-10, with a 12-pair probe (section 3).

## 1. Question

Agent Audit (AA, v0.19.2, commit 1009782) is our only baseline. We want to know two things:

1. **On our test sets, does AA catch mismatches** (code that does more than its name, params and
   description say)? We expect not: AA never compares the description with what the code does
   (CLAUDE.md section 8). Pairs that share code should get the same AA verdict whatever their
   label.
2. **What does each tool catch that the other misses?** AA is built for conventional bugs
   (eval, `shell=True`, SQL injection, secrets). Our model is not. So we report coverage: what
   ours catches, what AA catches, and the union. This is the framing agreed in section 5.

This is not a leaderboard. AA was not designed for our task, and the write-up should say so.

## 2. What AA can and cannot see

Verified facts (CLAUDE.md section 8):

- Detection is rule-based. It reads the docstring into `ToolDefinition.description` but no
  rule uses it for detection. No rule compares the description with inferred permissions.
- Permissions come from a fixed table of about 20 call names. `open` always maps to FILE_READ;
  `Path.rename` and `conn.execute` are not in it; `shutil.move` is.
- AGENT-003 (data exfiltration chain) is declared but not implemented.
- Findings in test-looking paths (`/tests/`, `/fixtures/`, `/examples/`, ...) are downgraded
  to SUPPRESSED. Our generated files must use neutral paths.

Consequence: for AA, an injected mutant and its honest twin are **the same input** (same code,
different docstring). Weakened pairs and rewrites are the same input as the original. So on
these pairs AA cannot do better than chance by construction. We will still measure it, and
check that claim directly (section 6, "code-identity check").

## 3. Probe (done while drafting, not committed)

12 pairs from rung 1 `test`, wrapped as in section 5 and scanned with
`agent-audit scan <dir> --format json --min-tier suppressed`:

| Rule | Severity / tier | Fired on |
|---|---|---|
| AGENT-022 "No error handling in tool execution" | medium / BLOCK | 5 of 12 pairs, honest and mismatch alike |
| AGENT-026 "LangChain tool input not sanitized" | critical / BLOCK | both pairs of `c0020-2` (original and its weakened pair) |
| AGENT-034 "Tool function without input validation" | high / INFO | 1 honest original (`c0067-3`) |
| AGENT-047 "Subprocess execution without sandbox" | medium / INFO | same honest original |

The inject and twin of `c0068-2` got identical findings. Two lessons:

- The scan works on our wrapped code, and its output follows the code, not the label.
- **How we turn findings into a yes/no verdict matters a lot.** Generic rules such as
  AGENT-022 fire on many tools, whatever they do. A rule like "any finding = flagged" would flag
  a large share of honest pairs. So we fix several verdict rules in advance (section 6).

## 4. Sets

The four scored sets, plus a coverage set still to build. AA runs on each **once**.

| Set | File | Pairs | Tools | Mismatches |
|---|---|---|---|---|
| Rung 1 `test` | `data/splits/full/test.jsonl` | 251 | 77 | 66 |
| Rung 2 | `data/generated/rung2-opus/pairs.jsonl` | 503 | 139 | 144 |
| Rung 3 (b) Opus | `data/generated/github-opus/pairs_43.jsonl` | 138 | 43 | 33 |
| Rung 3 (a) GPT templates | `data/github_tools_43_pairs.jsonl` | 215 | 43 | 79 |
| Coverage set | `data/coverage/pairs.jsonl` | 128 | 50 | 34 (+25 conventional bugs) |

Rung 3 uses the same 43-tool files our model scored, so both tools see exactly the same pairs.

**Coverage set (built 2026-11-10, `scripts/build_coverage_set.py`).** Its job is to give AA
a fair chance at the bugs it is designed for, so the coverage table has something in both
columns. 128 pairs from 50 tools:

- **25 conventional-bug tools** (`data/coverage/conventional_bugs.jsonl`), hand-written by
  Claude, 5 per kind: `eval` / `exec` on an argument; `shell=True` or `os.system` with a
  formatted string; `pickle.loads` / `pickle.load` / unsafe `yaml.load`; SQL built with
  f-strings, `%` or `+`; a hard-coded API key, token or password. Domains and decorators
  (`@tool`, `@mcp.tool()`, `@function_tool`, none) vary like `full`. Each description honestly
  states every effect, so these are **not** mismatches under our definition: the bug is in
  *how* the code does what it says. Edge case: an `eval` of a user formula is described as
  "evaluates the expression", not as running code; we count that as declared.
  Secrets are generic random strings, with no real provider prefixes (so GitHub push
  protection doesn't block them, and nothing looks like a real key).
- **25 tools sampled from rung 1 `test`, with all their pairs** (103 pairs, 34 mismatches):
  all 12 tools that have a weakened pair, plus 13 inject-only tools (seed 2028). That gives
  22 inject, 12 weakened, 25 original, 22 twin, 22 rewrite.
- Labels: `mismatch` as usual; `meta.conventional_bug` and `meta.bug_kind` (in `meta` because
  `ToolPair` has a fixed set of fields).
- **Checks in the build script:** the code parses; the function and its arguments match the
  params schema; an AST pattern per kind confirms the bug is in the code. The word-list effect
  check found nothing undeclared, apart from `conn.close()`, which the word list reads as
  `destroy` (a known gap).
- **Hand check:** not needed for the whole set (labels are simple, and the AST check covers
  "is the bug there"). An optional 15-minute skim of the 25 tools by one teammate is enough.

**Our model on the set:** the rung 1 rows reuse their saved scores
(`checkpoints/binary/eval/full-test/`), so they are not rescored. The 25 bug tools need one
CARC run:
`sbatch -A yzhao010_1531 --gres=gpu:a100:1 scripts/carc/score.slurm checkpoints/binary data/coverage/conventional_bugs.jsonl --tag coverage-bugs`.

## 5. Turning a pair into AA input

AA scans files on disk, so each pair becomes one small Python file:

```
probes/aa_run/<set>/<pair_id with / replaced by __>/service/tools/ops.py
```

- `probes/aa_run/` is already git-ignored, and the path has no test-looking parts.
- Build the file from the pair's normalized `code`: parse it, add `@tool` (unless it already
  has a tool decorator; many pairs carry `@tool`, `@mcp.tool()` or `@function_tool`) to the function
  named `name`, and insert the `description` as that function's docstring. Add
  `from langchain_core.tools import tool` at the top. `@tool` is what AA uses to recognize an
  agent tool (`TOOL_DECORATORS` in `scanners/python_scanner.py`). The params schema is not
  added; AA does not use it.
- Scan the whole `<set>` folder in one call with `--format json --min-tier suppressed`, then
  map findings back to pairs by file path.
- If a pair fails to parse or AA fails on its file, record it with a reason. Never drop it
  silently. (Rung 3 functions sometimes lack their original imports; AA only needs the AST,
  so this should be rare.)
- **Positive control in every run:** one extra file with `eval(user_input)` and
  `subprocess.run(cmd, shell=True)` inside a `@tool`. If it does not fire, the run is invalid.

## 6. Verdicts and metrics

AA gives findings (rule, severity, tier), not a score. We turn them into a per-pair verdict
with rules **fixed before any run and never changed after seeing results.** We report all of
them:

| Verdict rule | Pair is flagged if its file has... |
|---|---|
| **Main** | any non-suppressed finding at tier BLOCK or WARN |
| **Capability rules only** | a finding from a fixed list of rules about dangerous operations or permissions (list below, fixed 2026-11-11) |
| **Any finding** | any finding, including INFO tier |
| **Ordinal score** | score = highest severity found: none 0, info 1, low 2, medium 3, high 4, critical 5. Used for PR-AUC and a PR curve only |

Capability list (**fixed 2026-11-11**, before any run): AGENT-001 (command injection),
AGENT-002 (excessive permissions), AGENT-014 (overly permissive tool access), AGENT-017 and
AGENT-035 (unsandboxed / unrestricted code execution), AGENT-037 (side-effect operation
without human approval), AGENT-041 (SQL injection), AGENT-047 (subprocess without sandbox),
AGENT-053 (self-modification). Generic hygiene rules such as AGENT-022 (error handling) and
AGENT-034 (input validation) are left out. Some rules (e.g. AGENT-047) are implemented in
Python rather than YAML, so the list is checked against the code, not only
`rules/builtin/*.yaml`. Checked 2026-11-11 against the installed 0.19.2: all nine exist;
AGENT-047 has no YAML entry and is emitted by `scanners/privilege_scanner.py`.

**Metrics,** the same as for our model (CLAUDE.md section 5), per set and per verdict rule:

- precision / recall / F1 on the mismatch class, PR-AUC (ordinal score) with the tool
  bootstrap interval, group accuracy;
- by pair type (inject, weaken, original, twin, rewrite) and by effect class.

AA predictions are written in our `predictions.jsonl` schema (`score` = 0/1 verdict or the
ordinal score), so `agentlint.eval.metrics.report` and `bootstrap_pr_auc` work unchanged.

**Code-identity check.** For every group of pairs with identical code (inject and twin;
original, rewrite and weaken), count how often AA's verdict differs within the group. We
expect 0%. This is the cleanest evidence for "AA cannot see the mismatch", stronger than any
recall number.

**Coverage table** (per set, at the main verdict and our 0.5 cut-off):

| | caught by ours only | by AA only | by both | by neither | union |
|---|---|---|---|---|---|
| Mismatches | | | | | |
| Conventional bugs (coverage set) | | | | | |
| Honest pairs flagged | | | | | |

## 7. Implementation (approved 2026-11-11)

- New module `src/agentlint/eval/agent_audit.py` with three steps: write the wrapped files,
  run `agent-audit scan` once per set, turn findings into verdicts and metrics.
- CLI: `agentlint aa-score <pairs.jsonl> --tag <tag>`.
- Output: `runs/agent_audit/<tag>/{findings.json,predictions.jsonl,metrics.json}`, plus the
  AA version in `metrics.json`. Like `score.py`, it refuses to overwrite a result
  (`--force` only to redo a broken run).
- Runs locally: AA is static analysis, no GPU, so no CARC job.
- Tests: wrapping keeps the code and puts the description in the docstring; the positive
  control fires; each verdict rule on a small fixture of findings; the path mapping.
- AA is already a dependency (`baseline` extra), pinned at 0.19.2.

## 8. Ground rules

- Each set is scanned **once** with the pinned AA version. Results are committed.
- Verdict rules and the capability list are written into this note before the first run.
- Nothing about our model changes because of these results (no retraining, no new cut-off).
- Once AA has run on rung 3, the rung 3 numbers in `docs/results-binary.md` stop being
  provisional.

## 9. Decisions (all settled 2026-11-11)

1. ~~Capability rule list~~ **decided 2026-11-11:** the draft list in section 6, unchanged.
2. ~~Coverage set~~ **decided and built 2026-11-10** (section 4): 25 hand-written bug tools
   plus 25 rung 1 tools with all their pairs.
3. ~~Rung 3 context~~ **decided 2026-11-11:** scan only the wrapped function, same as rungs
   1–2. Whole repos are out of scope (provenance is incomplete).
4. ~~Where results go~~ **decided 2026-11-11:** a separate `docs/results-agent-audit.md`,
   linked from `docs/results-binary.md`.

## 10. Order of work

1. ~~Team settles section 9~~ (done 2026-11-11).
2. Write `agent_audit.py` and its tests; dry run on 12 pairs.
3. ~~Build the coverage set~~ (done 2026-11-10); score its 25 bug tools once with our model on
   CARC.
4. Run AA once on each of the five sets.
5. Write up: metrics tables, code-identity check, coverage table; drop "provisional" from
   rung 3.
