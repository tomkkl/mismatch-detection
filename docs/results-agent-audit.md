# Agent Audit vs our detector on the test ladder

Written 2026-11-12; coverage set (section 7) added 2026-11-18. Agent Audit (AA)
0.19.2, run once per set with `agentlint aa-score` (design: `docs/agent-audit-comparison.md`).
Our model's numbers are the saved ones from `docs/results-binary.md` (`checkpoints/binary`,
cut-off 0.5); nothing was rescored for this write-up.

## Summary

- **AA cannot see a capability mismatch.** In every set, pairs with identical code get
  identical AA findings, whether the description is honest or not: 0 differences in 496
  same-code groups. Given an injected mismatch and its honest twin (same code, the
  description declares the extra operation), AA ties all 245 times. Our model scores the
  mismatch higher in all 245.
- **AA's recall is not detection.** It flags 38–55% of all pairs, mainly through generic
  rules (no error handling, no input validation) that fire on honest code just as often. Its
  recall on injections equals its false-alarm rate on their twins, exactly, in all four sets.
- **Ranking is at or barely above chance:** AA's PR-AUC is within 0.04 of the share of
  mismatches in each set (0.275 vs 0.263 on rung 1; 0.319 vs 0.286 on rung 2; 0.270 vs 0.239
  on rung 3 Opus; 0.401 vs 0.367 on rung 3 GPT). On rung 2 and rung 3 GPT the interval sits
  just above chance, for reasons that have nothing to do with descriptions (section 5). Ours
  is 0.85–0.96.
- **At its main verdict, AA's F1 is 0.35–0.47 with group accuracy 0.00–0.19.** Our model has
  F1 0.70–0.93 and group accuracy 0.28–0.91.
- **This is a statement about scope, not quality.** AA is built to find dangerous code
  (injection, unsafe execution, secrets). On the coverage set (section 7) its bug-specific
  rules catch 13 of 25 conventional bugs (all 10 eval / shell, 1/5 SQL, 2/5 secrets, 0/5
  unsafe pickle / yaml), and our model catches none, as it should: they are not mismatches.
  The two tools cover disjoint problems.

## 1. Setup

| | |
|---|---|
| Tool | Agent Audit 0.19.2 (`baseline` extra), static, rule-based, local (no GPU) |
| Input | each pair written as one Python file: the code with the description as its docstring and a `@tool` decorator if it had none, under a neutral path (AA suppresses test-looking paths) |
| Runs | one `agent-audit scan` per set, once (2026-11-12). A known-vulnerable control file is added to every scan; it fired every time (AGENT-001 / 017 / 035), and no file failed |
| Sets | rung 1 `test` (251 pairs), rung 2 Opus (503), rung 3 Opus on 43 GitHub tools (138), rung 3 GPT templates on the same 43 tools (215); coverage set (128, run 2026-11-18). Same files our model scored |
| Our model | `checkpoints/binary`, at 0.5, saved predictions |

## 2. Verdict rules (fixed before any run)

AA gives findings, not a score, so a pair's verdict comes from rules written down before the
first run (note, section 6):

| Rule | A pair is flagged if its file has... |
|---|---|
| **Main** | any non-suppressed finding at tier BLOCK or WARN |
| **Capability** | a non-suppressed finding from 9 rules about dangerous operations or permissions (AGENT-001, 002, 014, 017, 035, 037, 041, 047, 053) |
| **Any** | any finding, suppressed included |
| **Ordinal** | score = highest severity (none 0 … critical 5); used for PR-AUC only |

## 3. Headline numbers

| Set | Pairs (mismatch) | Tool | P | R | F1 | Group acc | PR-AUC (95% CI) |
|---|---|---|---|---|---|---|---|
| rung 1 `test` | 251 (66) | **ours** | 0.97 | 0.89 | **0.93** | **0.91** | **0.962** (0.928–0.995) |
| | | AA main | 0.30 | 0.62 | 0.40 | 0.16 | 0.275 (0.252–0.299) |
| rung 2 Opus | 503 (144) | **ours** | 0.96 | 0.90 | **0.93** | **0.86** | **0.962** (0.942–0.981) |
| | | AA main | 0.33 | 0.61 | 0.43 | 0.09 | 0.319 (0.298–0.340) |
| rung 3 (b) Opus | 138 (33) | **ours** | 0.70 | 0.91 | **0.79** | **0.74** | **0.942** (0.886–0.986) |
| | | AA main | 0.29 | 0.45 | 0.35 | 0.19 | 0.270 (0.239–0.314) |
| rung 3 (a) GPT | 215 (79) | **ours** | 0.85 | 0.59 | **0.70** | **0.28** | **0.853** (0.811–0.898) |
| | | AA main | 0.41 | 0.54 | 0.47 | 0.00 | 0.401 (0.382–0.418) |

Share of mismatches (PR-AUC of a random scorer): 0.263, 0.286, 0.239, 0.367.

The other verdict rules don't change the picture:

| Set | AA capability P / R / F1 | AA any P / R / F1 |
|---|---|---|
| rung 1 | 0.21 / 0.06 / 0.09 | 0.30 / 0.64 / 0.40 |
| rung 2 | 0.31 / 0.18 / 0.23 | 0.33 / 0.70 / 0.45 |
| rung 3 Opus | 0.31 / 0.12 / 0.17 | 0.29 / 0.45 / 0.35 |
| rung 3 GPT | 0.44 / 0.28 / 0.34 | 0.41 / 0.57 / 0.48 |

Rung 3 GPT keeps all 215 pairs as scored, as in `docs/results-binary.md`.

## 4. Same code, same verdict

This is the cleanest evidence. Every set has groups of pairs that share code exactly:
injection + honest twin, and original + honest rewrite + weakened description. Within a group
only the description, and so sometimes the label, changes.

| Set | Same-code groups | Groups with both labels | AA findings differ | AA verdict differs (any rule) |
|---|---|---|---|---|
| rung 1 | 111 | 66 | 0 | 0 |
| rung 2 | 229 | 144 | 0 | 0 |
| rung 3 Opus | 63 | 33 | 0 | 0 |
| rung 3 GPT | 93 | 79 | 0 | 0 |

Paired comparison, mismatch vs an honest pair with the same code (AA by ordinal score, ours by
model score):

| Pairs compared | Ours: mismatch higher / tie / lower | AA: higher / tie / lower |
|---|---|---|
| injection vs its twin (245) | **245 / 0 / 0** | 0 / 245 / 0 |
| weakened vs original or rewrite (141) | **121 / 1 / 19** | 0 / 141 / 0 |

AA's results depend only on the code, so it cannot separate a pair whose description hides an
operation from the same code with an honest description. Our model separates every injection
from its twin. Of its 19 losses on weakened pairs, 17 are on rung 3 GPT, the known weak spot
(`docs/results-binary.md`, section 8).

## 5. Where AA's recall comes from

AA flags at the same rate across pair types that share code (main verdict):

| Set | Inject recall | Twin flagged | Weaken recall | Original flagged | Rewrite flagged |
|---|---|---|---|---|---|
| rung 1 | 0.61 (54) | 0.61 (54) | 0.67 (12) | 0.48 (77) | 0.50 (54) |
| rung 2 | 0.57 (110) | 0.57 (110) | 0.74 (34) | 0.47 (139) | 0.47 (110) |
| rung 3 Opus | 0.42 (31) | 0.42 (31) | 1.00 (2) | 0.33 (43) | 0.32 (31) |
| rung 3 GPT | 0.70 (50) | 0.70 (50) | 0.28 (29) | 0.33 (43) | 0.33 (43) |

- Inject recall equals the twin false-alarm rate in every set, because they are the same
  code. Weakened pairs share code with their original; the rates differ only because tools
  with a weakened pair are a subset.
- Injected code is flagged a bit more than original code (0.61 vs 0.48 on rung 1) because it is
  longer and does more, so generic rules have more to fire on. The honest twin gets the same
  boost.
- The rules that fire most are generic hygiene checks: AGENT-022 "No Error Handling in Tool
  Execution" (rung 1: 77 pairs; rung 2: 174), AGENT-034 "Tool Function Without Input
  Validation" (54; 144) and AGENT-026 "LangChain Tool Input Not Sanitized" (61; 80). They fire
  on honest and mismatched code alike.
- **AA beats chance on two sets, by a small margin:** rung 2 (PR-AUC 0.319 vs 0.286,
  interval 0.298–0.340) and rung 3 GPT (0.401 vs 0.367, interval 0.382–0.418). On rung 2 it
  is the boost above: injected code draws more findings than original code. On rung 3 GPT the
  templated injections often add code that AA treats as dangerous (AGENT-035
  "Unrestricted Code Execution" on 43 pairs), so injected code is flagged at 0.70 vs 0.33 for
  originals. The honest twins carry the same code and are flagged at 0.70 too. The lift comes
  from what the templates add to the code, not from reading descriptions.

## 6. Coverage: what each tool catches

At AA's main verdict and our 0.5 cut-off:

| Set | Mismatches | Ours only | AA only | Both | Neither | Union |
|---|---|---|---|---|---|---|
| rung 1 | 66 | 22 | 4 | 37 | 3 | 63 |
| rung 2 | 144 | 51 | 9 | 79 | 5 | 139 |
| rung 3 Opus | 33 | 16 | 1 | 14 | 2 | 31 |
| rung 3 GPT | 79 | 13 | 9 | 34 | 23 | 56 |

| Set | Honest pairs | Flagged by ours only | AA only | Both |
|---|---|---|---|---|
| rung 1 | 185 | 1 | 96 | 1 |
| rung 2 | 359 | 2 | 176 | 4 |
| rung 3 Opus | 105 | 10 | 34 | 3 |
| rung 3 GPT | 136 | 4 | 59 | 4 |

Adding AA to our model catches 23 more mismatches across the four sets (4 + 9 + 1 + 9), but
adds 365 honest false alarms (96 + 176 + 34 + 59). Given section 4, these extra catches are
coincidence: AA flags the honest twin of each of them too.

## 7. Coverage set: conventional bugs

128 pairs from 50 tools: 25 hand-written tools with a conventional bug (eval / exec,
`shell=True`, unsafe pickle / yaml, string-built SQL, hard-coded secret; 5 of each, honest
descriptions) plus 25 rung 1 tools with all their pairs (103 pairs, 34 mismatches). This is
where AA should do well and our model should not: the bugs are not mismatches. Our model
scored the 25 bug tools once (CARC job 14323275, commit aa448cf; the first try, job 14066569,
crashed on the all-honest set and saved nothing); the rung 1 rows reuse their saved scores.
AA ran once (2026-11-18): 0 failures, control fired, 0 differences in the 47 same-code groups.

**Our model flags none of the 25 bug tools** (highest score 0.0025). That is the right answer
under our definition, and it means our model gives no protection against these bugs.

**AA on the 25 bug tools, by kind.** "Bug rule" = a finding whose rule names the bug itself:
AGENT-001 / 017 / 035 for eval and shell, AGENT-041 for SQL injection, AGENT-004 for
hard-coded credentials. On the 103 rung 1 pairs they fire on one tool only: `c0094-2`
hard-codes a placeholder Stripe key, so AGENT-004 flags all 4 of its pairs (same code). That
is a real secret finding, unrelated to the tool's injected mismatch.

| Bug kind | Tools | AA main verdict | AA bug rule | Ours at 0.5 |
|---|---|---|---|---|
| eval / exec | 5 | 5 | **5** (AGENT-001, 017, 035) | 0 |
| `shell=True` / `os.system` | 5 | 5 | **5** (AGENT-035; AGENT-001 on 4) | 0 |
| unsafe pickle / yaml | 5 | 4 | **0** (no rule fires) | 0 |
| string-built SQL | 5 | 4 | **1** (AGENT-041) | 0 |
| hard-coded secret | 5 | 3 | **2** (AGENT-004) | 0 |
| **Total** | 25 | 21 | **13** | 0 |

- AA's main verdict flags 21/25, but 8 of those come only from the generic rules (AGENT-034
  input validation, AGENT-022 error handling), which also flag 42 of the 69 honest rung 1
  pairs in this set (61%). Counting only bug rules, AA catches 13/25: all code and shell
  execution, but 1/5 SQL, 2/5 secrets and no unsafe deserialization.
- On the capability-rules verdict (section 2), AA catches 11/25 bug tools (eval and shell,
  plus 1 SQL) and flags none of the 103 rung 1 pairs, mismatches included (0/34).

**Coverage table** (AA main verdict, our 0.5 cut-off):

| Coverage set | Pairs | Ours only | AA only | Both | Neither | Union |
|---|---|---|---|---|---|---|
| Mismatches | 34 | 9 | 4 | 18 | 3 | 31 |
| Conventional bugs | 25 | 0 | 21 | 0 | 4 | 21 |
| Honest pairs flagged | 69 | 0 | 42 | 0 | 27 | 42 |

With AA's capability-rules verdict instead, the rows are: mismatches 27 / 0 / 0 / 7 (ours
only / AA only / both / neither); bugs 0 / 11 / 0 / 14; honest flagged 0 / 0 / 0. So the two
tools catch **disjoint** things: the only overlap is `c0094-2`'s injected pair, which our
model flags for the mismatch (0.999) and AA flags for the hard-coded key in all 4 pairs.
Our model alone catches 27/34 mismatches with 0/69 false alarms here.

## 8. Takeaways

- The two tools answer different questions. AA asks "is this code dangerous?"; we ask "does
  this code do more than its interface says?". On our question AA is at or barely above
  chance, and the same-code check shows why: it never looks at the description.
- AA's flags on our sets are mostly generic hygiene findings that fire on about half of all
  honest tools, so they can't serve as a mismatch signal.
- The coverage set confirms complementary coverage: AA catches conventional bugs (best on
  code and shell execution, weak on SQL, secrets and deserialization), our model catches
  mismatches; they overlap only by coincidence (one tool with a hard-coded key). Running both is the
  sensible setup.
- The rung 3 model numbers in `docs/results-binary.md` are now final (AA has run on rung 3).

## Files

- AA results: `runs/agent_audit/{full-test,rung2-opus,rung3-opus,rung3-gpt,coverage}/`
  (`findings.json`, `metrics.json`, `predictions_{main,capability,any,ordinal}.jsonl`).
- Our model: `checkpoints/binary/eval/{full-test,rung2-opus-pairs,rung3-opus,rung3-gpt,coverage-bugs}/`
  (job log `runs/slurm-agentlint-score-14323275.out`).
- Code: `src/agentlint/eval/agent_audit.py`, `agentlint aa-score`.
- Design and fixed rules: `docs/agent-audit-comparison.md`. Earlier probes: CLAUDE.md section 8.
