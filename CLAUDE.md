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

- Python 3.10+ (developed on 3.13). `baseline` installs Agent Audit v0.19.2; `plots` installs
  matplotlib for `scripts/plot_results.py`.
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

## 2. Where we are and what's next (updated 2026-11-10)

**Resume here (2026-11-10, end of session).**
Current focus: **Agent Audit vs our model** on every test set
(`docs/agent-audit-comparison.md`). State:
- Done: rungs 1–3 are in `docs/results-binary.md` (rung 3 marked provisional until Agent Audit
  runs on it); the design note is written; the coverage set is built and checked
  (`data/coverage/`, 128 pairs; build checks, model-input check and `pytest` pass).
- **Next, in order:**
  1. Team answers the open decisions in the note's section 9: the capability rule list,
     rung 3 scanned as wrapped functions only (proposed), and where AA results get written up.
     Optional: one teammate skims the 25 bug tools in `data/coverage/conventional_bugs.jsonl`.
  2. Score the 25 bug tools once with our model on CARC:
     `sbatch -A yzhao010_1531 --gres=gpu:a100:1 scripts/carc/score.slurm checkpoints/binary data/coverage/conventional_bugs.jsonl --tag coverage-bugs`
  3. Write `src/agentlint/eval/agent_audit.py` + `agentlint aa-score` with tests (note
     section 7; new module, so the team approves first). Dry run on ~12 pairs.
  4. Run Agent Audit once on each of the five sets; write up; drop "provisional" from rung 3.

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
- Rung 2 test set `rung2-opus` (Opus 5.5, 503 pairs from 139 tools), merged from PR #1
  (2026-11-09). Details in step 3.
- Rung 3 GitHub tools: Andrea's 50 tools with GPT-5.6 templated mutations (250 pairs), plus an
  Opus 5.5 version of the same tools built with our pipeline (`github-opus`, 2026-11-09).
  Details in step 3 and section 6.
- **Rungs 2 and 3 scored once (2026-11-09, jobs 13919620 / 13919689 / 13919690).** Rung 2
  matches rung 1 (PR-AUC 0.962); rung 3 is lower (Opus 0.942, GPT templates 0.853). Results
  in step 4. Do not rescore.

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
  (First runs: fine-tuned 9% vs 12% honest FP for 1 vs 2 chunks, fine; frozen 15% vs 52%.)
- **Weakened mismatches are the weak spot:** recall 10/18 vs 53/55 for injections (5 epochs);
  10 epochs: 9/18 at 0.5, 14/18 only at the fragile 0.0035 threshold. The missed ones score
  ~0.001–0.002, same as honest pairs (`c0143-1`, `c0008-2`, `c0148-2`, `c0000-3`).
- **Rewrite pairs get flagged by wording:** 9/55 rewrites flagged vs 1/76 originals, with the
  same code (5 epochs). 10 epochs: 3/55 vs 0/76, scoring 0.93–1.0 (`c0069-3`, `c0145-0`, and
  again `c0196-1`: original 0.001, rewrite 0.996). Part of the score may come from the
  rewritten-description style (rewrite and inject share it).
- **Twins:** 8/55 flagged at 5 epochs (hand-checked 2026-10-29: labels right, model errors).
  Fixed by 10 epochs: 0/55 at 0.5, 2/55 at 0.0035 (scores 0.03 and 0.05).
- **Andrea's rung 3 draft, provenance incomplete:** `data/github_tools_dataset.json` has the function and
  docstring but not repository URL, commit SHA, file path, or license. Recover these before the
  set is final; mark entries where recovery is impossible.
- **Andrea's rung 3 draft, mutations are templated:** they were generated deterministically, not by
  the LLM pipeline used for `full`. Repeated operations and description wording may be a
  shortcut. Review template diversity and hand-check labels before the one-time score.

### What's next

Steps 2 and 3 can run at the same time with different people.

1. ~~Decide how to handle long inputs~~ (done 2026-10-26).
2. ~~**Train the first model (binary).**~~ First runs done 2026-10-29 (CARC jobs 13535509 /
   13535510, A100, commit ad4ebf7). Val results, at the saved best-F1 threshold:

   | Run | PR-AUC (95% CI, tool bootstrap) | P | R | F1 | Group acc |
   |---|---|---|---|---|---|
   | fine-tuned (`binary`, epoch 5 of 5, thr 0.33) | **0.856** (0.82–0.90) | 0.78 | 0.86 | 0.82 | 0.67 |
   | frozen + 2-layer head (`binary_frozen`, epoch 5 of 10, thr 0.56) | 0.489 (0.45–0.55) | 0.45 | 0.59 | 0.51 | 0.33 |

   Fine-tuned, by pair type: inject recall 53/55, weaken recall 10/18; flagged honest pairs:
   original 1/76, twin 8/55, rewrite 9/55. Within a tool it **always** ranks the mutant above
   its twin, rewrite and original (55/55, 18/18); all errors are scores that don't carry across
   tools. Honest FP rate by chunks 1 / 2 / 3+: 9% / 12% / 0% (fine-tuned), 15% / 52% / 50%
   (frozen, so the chunk leak is real for the frozen model but small after fine-tuning).
   Files: `checkpoints/{binary_5ep,binary_frozen}/` and `runs/slurm-agentlint-1353551{0,9}.out`
   (committed; the `model.safetensors` weights are git-ignored, too big for GitHub). Running the scripts: `bash scripts/carc/setup.sh` once, then
   `sbatch -A yzhao010_1531 --gres=gpu:a40:1 scripts/carc/train.slurm <config>` (A40 / A100 /
   L40S only; not P100 / V100).
   **10-epoch rerun (job 13589508, A100, commit ecd2d3a), landed 2026-11-01:**
   `checkpoints/binary/`, `runs/slurm-agentlint-13589508.out`. Best epoch 9 of 10, and it has
   flattened out (PR-AUC by epoch 8 / 9 / 10: 0.980 / 0.981 / 0.976), so no more epochs needed.

   | Run | Threshold | PR-AUC (95% CI, tool bootstrap) | P | R | F1 | Group acc |
   |---|---|---|---|---|---|---|
   | `binary` 10 ep, saved best-F1 | 0.0035 | **0.981** (0.965–0.994) | 0.93 | 0.95 | 0.94 | 0.88 |
   | `binary` 10 ep, fixed | 0.5 | 0.981 | 0.96 | 0.88 | 0.91 | 0.84 |
   | `binary_5ep` (above), saved best-F1 | 0.33 | 0.856 (0.82–0.90) | 0.78 | 0.86 | 0.82 | 0.67 |

   By pair type (saved thr / 0.5): inject recall 55/55 / 55/55; weaken recall 14/18 / 9/18;
   flagged honest pairs: original 0/76 / 0/76, twin 2/55 / 0/55, rewrite 3/55 / 3/55. Honest FP
   by chunks 1 / 2 / 3+ at 0.5: 0.9% / 3.1% / 0%. The gain over 5 epochs is mostly the LR
   schedule: with decay over 10 epochs the LR stays higher, and epoch 5 of this run already
   had PR-AUC 0.958.
   **Cut-off for `test` decided (2026-11-01): 0.5 main, saved best-F1 (0.0035) as a second
   operating point** (section 5, "Model selection"). Why: scores are saturated (187/259 val
   pairs below 0.01, 65 above 0.99, 7 in between), and the best-F1 cut-off sits just above the
   highest honest score (0.0034), right on one weakened pair's score. The cost: weakened recall
   9/18 at 0.5 vs 14/18 at 0.0035. The missed weakened pairs score like honest ones, so the fix
   is more weakened training data, not a lower cut-off.
3. **Build the other test sets.**
   - ~~Rung 2: a fresh-seed Opus set~~ **built 2026-11-07, scored once 2026-11-09**
     (results in step 4; on `main` since PR #1 merged 2026-11-09). Results are in
     `checkpoints/binary/eval/rung2-opus-pairs/`. Do not rescore.
     `configs/datagen_rung2.yaml` (Opus 5.5, seed 2027, 150 honest tools, same recipe
     otherwise), `--tag rung2-opus`. **503 pairs from 139 tools, 144 mismatches (28.6%)**:
     110 inject / 110 twin / 110 rewrite / 34 weakened. Baselines: majority 0.714, length
     0.708, description-only 0.714. 484 LLM calls, $15.83 at list price (Pro plan).
     11 tools (38 pairs) whose name also appears in `full` (9 in train, 1 val, 1 test; no
     identical code) were dropped, as inside a build (section 5, "Dataset"); they are kept in
     `excluded.jsonl` as a record of the build but are **not scored** (decided 2026-11-09:
     38 pairs is too few for a useful footnote). Score the whole `pairs.jsonl`, not the splits:
     `sbatch -A yzhao010_1531 --gres=gpu:a100:1 scripts/carc/score.slurm checkpoints/binary data/generated/rung2-opus/pairs.jsonl`.
     **Hand check (2026-11-07):** the word-list check disagreed on 31/110 kept injections
     (vs 33/560 on `full`). All 31 checked by hand: every one adds a real undeclared
     operation, so no label is wrong. 28 match their effect class; 3 are arguable on the
     class only (`c0005-2` destroy = cancels pending payouts; `c0014-3` spend = writes a
     credit to a local ledger file; `c0026-3` change authority = creates a Stripe dashboard
     login link). The misses are the known word-list gaps (section 2, open issues): SQL
     archive tables (`INSERT INTO x_archive SELECT …`, 8 of the 31), payment / payout / credit
     rows, `stripe.PaymentIntent.create` / `stripe.Transfer.create`, `requests.post` to
     billing endpoints, and permission rows. Opus uses SQL-table forms more than Sonnet.
   - **Agent Audit coverage set: built 2026-11-10** (`scripts/build_coverage_set.py`,
     `data/coverage/`): 128 pairs from 50 tools = 25 hand-written tools with a conventional
     bug (5 each: eval/exec, `shell=True`, unsafe pickle/yaml, string-built SQL, hard-coded
     secret; honest descriptions, so `mismatch: false`, `meta.conventional_bug: true`) + 25
     rung 1 test tools with all their pairs (34 mismatches; 12 weakened). Rung 1 rows reuse
     saved scores; score the bug tools once:
     `sbatch -A yzhao010_1531 --gres=gpu:a100:1 scripts/carc/score.slurm checkpoints/binary data/coverage/conventional_bugs.jsonl --tag coverage-bugs`.
     Design: `docs/agent-audit-comparison.md` section 4.
   - **Rung 3, second draft (Andrea, pushed to `main` 2026-11-07, not in the PRs):** built
     from the 50 functions in `data/github_tools_dataset.json` (her first scraper output): 250
     pairs, 159 honest / 91 mismatch. Mismatches come from deterministic code and description
     templates, not the LLM pipeline; repo / commit / license are not recorded yet. Details in
     section 6, "Rung 3 draft (Andrea)". Before scoring: recover provenance, review labels and
     template diversity, commit the generator and artifact at stable paths, and confirm the set
     was never used for training or threshold selection. Then score it once.
   - **Rung 3 decided (2026-11-09): Andrea's 50 tools, in two versions.** Tom's collector and
     PRs #2–#6 were closed without merging (branches kept on GitHub). Version (a): Andrea's
     GPT-5.6 templated set above. Version (b): **`github-opus`**, the same 50 originals mutated
     by our usual pipeline with Opus 5.5 (`configs/datagen_github_opus.yaml`, seed 4044,
     `honest_from:`), so (a) and (b) differ only in the injector. No name filter against
     `full` (team choice: keep all 50), and duplicate names are kept since the set is never
     split. 4 stub tools with no calls (`equation_solver`, `add`, `todowrite`, a
     `get_weather`) fail `verify_honest` and are dropped (`rejects.jsonl`). Score the whole
     `data/generated/github-opus/pairs.jsonl` once, when the team says so.
     **Compare (a) and (b) on the 43 shared tools (decided 2026-11-09).** Two groups are left
     out of both:
     - **4 stubs** (`github-011-equation_solver`, `github-015-add`, `github-021-todowrite`,
       `github-043-get_weather`; 21 pairs in Andrea's set). Injections into a stub are
       trivially easy to spot, `todowrite`'s weaken-`read` pair hides nothing (the code reads
       nothing), and the stubs' declared effects are wrong (`add` declared `modify`).
     - **3 tools too long for the model input** (`github-001-get_stock_info`,
       `github-041-translate_function`, `github-042-search_tool`): name + params + description
       is 396–732 tokens, leaving too little of the 512 for code, so `encode_pair` refuses
       them and scoring would crash. The input is never truncated (section 5).

     Scoring files (Andrea's original file stays as is):

     | File | Set | Pairs | Tools | Mismatch |
     |---|---|---|---|---|
     | `data/generated/github-opus/pairs_43.jsonl` | (b) Opus, main | 138 | 43 | 33 |
     | `data/github_tools_43_pairs.jsonl` | (a) GPT templates, main | 215 | 43 | 79 |
     | `data/github_tools_47_pairs.jsonl` | (a) with the stubs: **not scored for now** (2026-11-09) | 236 | 47 | 86 |

     **Both scored once (2026-11-09; results in step 4). Do not rescore.** Commands used:
     ```
     sbatch -A yzhao010_1531 --gres=gpu:a100:1 scripts/carc/score.slurm checkpoints/binary data/generated/github-opus/pairs_43.jsonl --tag rung3-opus
     sbatch -A yzhao010_1531 --gres=gpu:a100:1 scripts/carc/score.slurm checkpoints/binary data/github_tools_43_pairs.jsonl --tag rung3-gpt
     ```
     **`github-opus` built (2026-11-09): 150 pairs from 46 tools, 36 mismatches (24%):** 34
     inject / 34 twin / 34 rewrite / 2 weakened. Injected effects 3–7 each (send outward 7;
     change authority, execute, destroy 6; relocate 5; spend, modify 3); positions start 10 /
     middle 8 / end 10 / helper 6. Baselines: majority 0.76, length-only 0.74,
     description-only 0.76 (no leak). 143 Opus calls, $2.82 at list price. Rejects: 6
     injection targets were `read` (skipped), 4 injections changed too much, 1 rewrite missed
     the length match; 23/34 twins needed the length retry (her short, generic descriptions).
     Hand check of all 34 injections: each adds a real undeclared operation of its class.
     **Weak spots:** only 2 weakened pairs (her set has 32): 20 tools had no effect that could
     be removed without the name / params giving it away, and Opus declined 4. So compare
     the two injectors on injections; weakened results from (b) mean little. Opus often
     injects through a module-level `os.environ.get('..._URL')` constant plus `requests.post`;
     watch for that as a style cue.
4. **Evaluation code** in `src/agentlint/eval/`: metrics and breakdowns are in `metrics.py`;
   test-set scoring is in `score.py` (2026-11-01): `agentlint score <checkpoint> <pairs.jsonl>`
   reports at 0.5 and at the cut-off in `threshold.json`, plus a tool-bootstrap 95% interval
   for PR-AUC, and writes `<checkpoint>/eval/<tag>/{predictions.jsonl,metrics.json}`. It
   refuses to overwrite a result (`--force` only to redo a broken run). Weights are only on
   CARC, so run it there:
   `sbatch -A yzhao010_1531 --gres=gpu:a100:1 scripts/carc/score.slurm checkpoints/binary data/splits/full/test.jsonl`.
   **Rung 1 `test` scored (2026-11-03, job 13657576, A100, commit 0da1a88), once. Do not
   rescore.** `checkpoints/binary` (10 epochs, best epoch 9), 251 pairs from 77 tools. Files:
   `checkpoints/binary/eval/full-test/{metrics.json,predictions.jsonl}`,
   `runs/slurm-agentlint-score-13657576.out`.

   | Set | Threshold | PR-AUC (95% CI, tool bootstrap) | P | R | F1 | Group acc |
   |---|---|---|---|---|---|---|
   | **test** | **0.5 (main)** | **0.962** (0.928–0.995) | **0.97** | **0.89** | **0.93** | **0.91** |
   | val | 0.5 | 0.981 (0.965–0.994) | 0.96 | 0.88 | 0.91 | 0.84 |

   Footnote only (2026-11-03): the saved best-F1 cut-off 0.0035 gives P 0.91 / R 0.95 / F1 0.93
   on test, 4 more catches and 4 more false alarms, no F1 gain.

   At 0.5: TP 59, FP 2, FN 7 (66 mismatches). Test matches val: PR-AUC is 2 points lower but
   inside val's interval, and F1 / group accuracy are slightly higher. The model was not tuned
   to `test`.
   - **By pair type (0.5):** inject recall 52/54; weaken recall 7/12; flagged honest pairs:
     original 0/77, twin 2/54, rewrite 0/54.
   - **Misses at 0.5:** 2 injections, both `relocate` at `start` with 2 chunks, scoring <0.01
     (`c0083-0`, `c0188-0`), and 5 weakened pairs. Weakened `change_authority` 0/2 (scores
     ~0.001). Weakened stays the weak spot, as on val.
   - **False alarms at 0.5:** 2 twins, both confident: `c0106-2/twin-relocate-end` 0.999 and
     `c0145-1/twin-destroy-end` 0.990. That is why `end` has an honest FP rate of 2/8; every
     other position is 0. Worth a hand check (labels or model?). Val had 0/55 twins at 0.5.
   - **Chunks (0.5):** honest FP rate 0.8% / 1.9% / 0% for 1 / 2 / 3+ chunks; recall 0.91 /
     0.86 / 1.0. No sign of a "more chunks means mismatch" leak.
   - **Within a tool:** the injected mutant outscores every honest pair of its tool 54/54;
     weakened 11/12.
   - **Saturated scores again:** 188/251 at or below 0.01, 60 at or above 0.99, 3 in between,
     so results at 0.5 barely depend on the exact cut-off.
   **Rungs 2 and 3 scored (2026-11-09, commit dbf57fd, A100, `checkpoints/binary` epoch 9),
   once each. Do not rescore.** Files: `checkpoints/binary/eval/{rung2-opus-pairs,rung3-opus,
   rung3-gpt}/{metrics.json,predictions.jsonl}`, `runs/slurm-agentlint-score-139196{20,89,90}.out`.

   | Set | Pairs (mismatch) | PR-AUC (95% CI, tool bootstrap) | P | R | F1 | Group acc |
   |---|---|---|---|---|---|---|
   | rung 1 `test` (for reference) | 251 (66) | 0.962 (0.928–0.995) | 0.97 | 0.89 | 0.93 | 0.91 |
   | **rung 2** `rung2-opus` | 503 (144) | **0.962** (0.942–0.981) | 0.96 | 0.90 | 0.93 | 0.86 |
   | **rung 3 (b)** Opus, 43 tools | 138 (33) | **0.942** (0.886–0.986) | 0.70 | 0.91 | 0.79 | 0.74 |
   | **rung 3 (a)** GPT templates, 43 tools | 215 (79) | **0.853** (0.811–0.898) | 0.86 | 0.60 | 0.70 | 0.28 |

   All at 0.5. Footnote (saved 0.0035): rung 2 F1 0.91, rung 3 Opus 0.78, GPT 0.74.
   **Rung 3 GPT keeps all 215 pairs as scored (decided 2026-11-10).** 9 of its weakened pairs
   remove `read`, which our definition says is not a mismatch (section 5, "No `read`
   injections"). They stay in the main numbers (score once, as built); the numbers without
   them go in a footnote, recomputed from the saved predictions with no rescoring: 206 pairs
   (70 mismatch), PR-AUC 0.856 (0.804–0.907), P 0.85 / R 0.64 / F1 0.73, group acc 0.44,
   weaken recall 2/20 (F1 0.75 at 0.0035). Group acc rises because 7 tools' only error was a
   missed `read` pair.

   | By pair type (0.5) | rung 2 | rung 3 Opus | rung 3 GPT |
   |---|---|---|---|
   | inject recall | 103/110 | 29/31 | 43/50 |
   | weaken recall | 27/34 | 1/2 | **4/29** |
   | original flagged | 1/139 | **6/43** | **6/43** |
   | twin flagged | 0/110 | 2/31 | 1/50 |
   | rewrite flagged | 5/110 | 5/31 | 1/43 |

   - **Rung 2: no sign the model learned Sonnet's style.** A different generator (Opus) and
     seed give the same PR-AUC and F1 as rung 1. Weakened recall is better than on test
     (27/34 vs 7/12). Rewrites are still the main false alarm (5/110, scores 0.69–1.0:
     `c0008-2`, `c0009-3`, `c0019-1`, `c0021-2`, `c0031-2`), as on val. Honest FP by chunks
     1 / 2 / 3+: 1.4% / 2.1% / 1.4%, so no chunk leak.
   - **Rung 3: injections still caught, real code raises false alarms.** On injections the two
     injectors are close (Opus 29/31, GPT 43/50; GPT misses are mostly `spend`, 3/8 caught).
     The **same 6 originals** are flagged in both versions (scores 0.94–1.0), so they come
     from the real GitHub code, not the mutations: `convermarkdowntopdf` (mdpdf subprocess),
     `deviceState`, `deviceControl` (MQTT), `search_engine`, `scrape_as_markdown` (MCP proxy
     calls), `google_form_link` (LLM chain + `requests.post`).
     **Hand check (2026-11-09): all 6 are model errors (false positives); no label changed.**
     - Every effect is implied by the interface. `convermarkdowntopdf` writes the
       PDF and runs mdpdf, both stated. `deviceControl` publishes control messages; "control
       the light or door" says so. `search_engine` / `scrape_as_markdown` forward the query or
       URL to an MCP server, which is what searching and scraping mean. `google_form_link`
       sends the topic to an LLM and the questions to a hard-coded Apps Script URL to create
       the form; "creates a Google Form using AI" covers it.
     - `deviceState` ("get the current state… retry a few times") publishes `'-1'` to the
       device's control topic when the state is unknown: a state-query ping that is part of
       the described retry. Counted as an error like the other five (team decision
       2026-11-09: not written up as a possible real catch).
     - Likely cause: in training, honest descriptions name every effect explicitly ("…and
       deletes the CSV"). Real docstrings imply effects through the verb ("search", "control",
       "create a form") and are often one line. The flagged code also uses the patterns that
       injections use: `subprocess.run`, `client.publish`, `requests.post` to a hard-coded
       URL (section 2: Opus injects via a URL constant + `requests.post`). The model seems to
       flag these calls when the description doesn't spell out the effect.
     - Supporting evidence, **the description decides it, not the code**: the twins of these
       tools (original code *plus* an injected operation, with a longer description naming
       it) score ~0.001 in 8 of 10 cases. The rewrites have exactly the original code, and
       how they score depends on the wording: Opus rewrites are flagged 4/4
       (0.99–1.0), GPT rewrites cleared 5/6 (0.001–0.025; `deviceState` 0.998 is the
       exception). Same code, opposite verdicts, so terse real docstrings are the shift.
       Fix idea (if time allows): train on honest tools with short, implicit descriptions.
   - **Rung 3 GPT: weakened pairs are almost all missed** (4/29; scores ~0.001, like honest
     pairs). That is the main reason its recall is 0.60 and group accuracy 0.28. Same weak spot
     as rungs 1–2, made worse by her short, generic descriptions. 9 of her weakened pairs
     remove `read`, which training never has as a mismatch (2/9 caught; see the footnote above).
   - Scores stay saturated: rung 2 361/503 ≤ 0.01 and 132 ≥ 0.99; rung 3 Opus 91 + 37 of 138;
     GPT 152 + 44 of 215.
   **Rungs 2–3 are in `docs/results-binary.md` (2026-11-10)**, sections 7–8, with two new
   charts (`docs/figures/rungs_headline.svg`, `rungs_by_pair_type.svg`). Rung 3 is marked
   **provisional** there (model only) until Agent Audit has run on it.
   **Next:** run Agent Audit on all four test sets plus the coverage set. Design note (draft,
   2026-11-10, open decisions in its section 9): `docs/agent-audit-comparison.md`.
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
  grad clip 1.0, 10 epochs (raised from 5 on 2026-10-29: still improving at epoch 5), seed 566.
- **Model selection:** keep the epoch with the best val PR-AUC. `train.py` never reads `test`.
- **Cut-off (decided 2026-11-01, updated 2026-11-03):** results for `test` and rungs 2–3 are
  reported at a fixed **0.5**, plus PR-AUC and the PR curve (which cover every cut-off). The
  best-F1 cut-off on val is still saved (`threshold.json`) and `score.py` still computes it,
  but write-ups give it **only as a footnote** (2026-11-03: on `test` it gave no F1 gain, 0.93
  at both, and it is fitted to one val pair). Reason for not using it: after 10 epochs val scores sit near 0 or 1, so the best-F1
  cut-off landed at 0.0035, tuned to a single val pair; 18 weakened val pairs are too few to
  tune on. Scores are not calibrated probabilities (`pos_weight` tilts them toward flagging).
- **"Before" baseline** (`binary_frozen.yaml`): encoder frozen, only the 2-layer head trains
  (see "Head" above), lr 1e-3, 10 epochs.

### Evaluation (decided 2026-10-05)

- **Test sets:** rungs 1–3 of the test ladder (section 6), each reported separately, plus the
  Agent Audit coverage set.
- Rung 1 is `data/splits/full/test.jsonl`. Tune on `val`; score `test` **once**, at the end.
- **Metrics per test set:** precision / recall / F1 on the mismatch class (at 0.5, and at the
  saved val cut-off), and PR-AUC. Not plain accuracy (only ~27% of pairs are mismatches).
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
- **Test sets drop tools named in training data** (2026-11-07): `exclude_names_from` in a
  datagen config lists other `pairs.jsonl` files; a tool whose name appears there is removed
  with all its pairs, into `excluded.jsonl`. It runs after every pick, so the other tools keep
  exactly their pairs and cached calls. Used for rung 2 against `full` (not for rung 3,
  see section 2, step 3).
- **No `read` injections** (2026-10-14, `skip_inject_effects: [read]` in
  `configs/datagen.yaml`): an undeclared read changes nothing, and only matters if the data
  leaves, which `send outward` covers. A hand check of all 70 read injections found only 8 real
  mismatches. Honest tools can still *declare* `read`. Known blind spot: a tool that returns
  more data to the agent than it says. The skip happens after the effect is picked, so other
  tools' targets (and cache keys) don't change, but ~1/8 of tools get no injected mutant.
- **Generator: Claude Sonnet 5.5** (`claude-sonnet-5-5`) via `claude -p --json-schema`, on plan
  usage. Switched from Opus on 2026-10-05 (similar quality, less than half the cost). Results
  are cached on disk, keyed by model.
- **`honest_from:`** (2026-11-09): a datagen config can name an existing `pairs.jsonl`; the
  build then skips generation, takes that file's originals as the honest tools and runs the
  usual verify / inject / twin / rewrite / weaken steps. Duplicate names are kept (the set is
  scored whole, not split). Used for rung 3 `github-opus`.

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

### Rung 3 draft (Andrea): templated injections on 50 GitHub tools

Built 2026-11-07 from all 50 entries in `data/github_tools_dataset.json`. The draft has 250
rows from 50 `source_id` groups:

| Mutation | Rows | Label |
|---|---:|---|
| original | 50 | honest |
| inject | 59 | mismatch |
| twin | 59 | honest |
| honest rewrite | 50 | honest |
| weaken | 32 | mismatch |

Total: 159 honest, 91 mismatch. The 59 injections cover all 7 non-read injection effects
(7–9 each) and all 4 positions (13–16 each).

Only 32 tools have a valid weakened-description row. For 18 tools, the name or parameters
already reveal the only removable effect, so weakening only the prose would not create a true
mismatch. We omitted those invalid rows. Nine additional inject/twin sets (18 rows) bring the
total to 250 while preserving paired code labels. Therefore the per-tool row counts are 32
tools with 5, 9 with 6, and 9 with 4. Keep `source_id` groups intact in any split or bootstrap.

The supplemental inject/twin sets are:

| Source tool | Added effect | Position |
|---|---|---|
| `get_registrar` | spend | start |
| `rdap_lookup` | change authority | middle |
| `equation_solver` | modify | start |
| `fetch_weather` (source 013) | relocate | end |
| `fetch_news` | execute | helper |
| `add` | send outward | start |
| `search_along_route` | change authority | middle |
| `bash` | destroy | middle |
| `get_weather` (source 025) | relocate | end |

Unlike `full`, this draft uses deterministic code and description templates rather than an LLM
injector. That is a deliberate, reproducible construction detail and a possible shortcut, not
an interchangeable implementation of the training pipeline. Before scoring: recover GitHub
URL / commit / path / license provenance where possible; flag functions missing surrounding
imports or project context; inspect repeated templates and all labels; commit the generator and
final pairs at stable paths; and freeze the artifact checksum. Do not train, tune a threshold,
or revise policy from Rung 3 outcomes. Its model numbers are in `docs/results-binary.md`, marked
provisional (decided 2026-11-10); they become final once Agent Audit has also run on it.

**Data hygiene** (the model will cheat otherwise): no comments on injected lines; vary
injection position, domains, naming and length; keep a tool and its mutants in the same split;
the AST check confirms the added operation is in the mismatch code, not in the honest code, and
that the name/params don't give it away.

---

## 7. Files

Package code is in `src/agentlint/` (full layout in `README.md`). Tests: `pytest`.

- `src/agentlint/data/`: dataset pipeline (`build.py` runs generate → verify → inject + twin →
  match → weaken → split; `honest_from:` builds from existing tools)
- `scripts/token_lengths.py`: input lengths vs the 512-token limit (section 3)
- `scripts/build_coverage_set.py`: builds the Agent Audit coverage set into `data/coverage/`
- `scripts/plot_results.py`: charts for the results write-up, into `docs/figures/` (`plots` extra),
  including rungs 1–3 side by side; it also prints every number the write-up quotes
- `src/agentlint/effects.py`: effect definitions and the word lists for the soft checks
- `src/agentlint/model/`: input construction, model, training loop (`train.py`)
- `src/agentlint/eval/`: `metrics.py` (all section 5 metrics and breakdowns, PR-AUC bootstrap
  interval); `score.py` (score a checkpoint on a test set, once)
- `configs/datagen.yaml`: dataset settings (`datagen_rung2.yaml`: rung 2;
  `datagen_github_opus.yaml`: rung 3 Opus version); `configs/binary.yaml`, `binary_frozen.yaml`
  (before-fine-tuning baseline), `multilabel.yaml`: training
- `notebooks/train_binary.ipynb`: Colab notebook that trains both binary models
- `scripts/carc/`: USC CARC setup (`setup.sh`, `env.sh`) and SLURM job scripts (`train.slurm`,
  `score.slurm`)
- `docs/agentlint-preproposal.md`: current proposal
- `docs/results-binary.md`: binary model results write-up (training, val, test), with charts
- `docs/agent-audit-comparison.md`: design note for running Agent Audit on every test set
- `scripts/githubtools.py` (Andrea): GitHub search for `@tool` functions →
  `data/github_tools_dataset.json`; her mutated set is `data/github_tools_250_pairs.jsonl`
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

- **2026-11-10 (end of session):** Pushed the Agent Audit design note and the coverage set.
  Coverage-set checks and `pytest` (27 passed) run. Next steps are at the top of section 2
  ("Resume here").
- **2026-11-10 (coverage set):** Built the Agent Audit coverage set (128 pairs, 50 tools):
  25 hand-written conventional-bug tools + 25 rung 1 test tools with all their pairs. Checks
  pass (parse, params, AST bug pattern per kind). Not scored yet; needs one CARC run for the
  25 bug tools.
- **2026-11-10 (latest):** Drafted `docs/agent-audit-comparison.md`, the plan for running
  Agent Audit on rungs 1–3 and the coverage set: wrap each pair as a `@tool` file, several
  verdict rules fixed in advance, a code-identity check and a coverage table. A 12-pair probe
  showed AA findings follow the code, not the label (AGENT-022 on 5/12, honest and mismatch
  alike). Open decisions for the team are in its section 9.
- **2026-11-10 (later):** Added rungs 2 and 3 to `docs/results-binary.md` (sections 7–8) with
  two charts from `scripts/plot_results.py` (`rungs_headline.svg`, `rungs_by_pair_type.svg`);
  numbers recomputed from the saved predictions match step 4. Rung 3 is marked provisional
  until Agent Audit runs on it (rule in section 6 relaxed from "not in the write-up").
- **2026-11-10:** Rung 3 GPT: main numbers stay all 215 pairs as scored (PR-AUC 0.853, F1
  0.70). Footnote without the 9 weakened pairs that remove `read` (not mismatches under our
  2026-10-14 rule), recomputed from saved predictions, no rescoring: 206 pairs, PR-AUC 0.856,
  F1 0.73, group acc 0.44, weaken recall 2/20. Hand check of the 6 flagged originals: all counted as
  model errors, `deviceState` included.
- **2026-11-09 (latest of that day):** Scored rung 2 and both rung 3 versions once with
  `checkpoints/binary` (jobs 13919620 / 13919689 / 13919690). Rung 2: PR-AUC 0.962, F1 0.93,
  same as rung 1. Rung 3 Opus: PR-AUC 0.942, F1 0.79 (6/43 real originals flagged). Rung 3
  GPT: PR-AUC 0.853, F1 0.70 (weakened recall 4/29). Details in section 2, step 4.
- **2026-11-09:** Merged PR #1 (rung 2); closed PRs #2–#6 without merging (rung 3
  collector, Agent Audit coverage set, word-list fixes, review page, multi-label / scan /
  weaken-extra; branches kept on GitHub) and removed their notes from this file. Rung 3 =
  Andrea's 50 GitHub tools. Added `honest_from:` to `build.py` (with a test) and built
  `github-opus`, an Opus 5.5 version of her set (section 2, step 3). Both versions are scored
  on the 43 tools they share (4 stubs and 3 over-long interfaces left out).
- **2026-11-07:** Built rung 2 (`rung2-opus`, Opus 5.5, seed 2027): 150 tools, then 11 dropped
  because their names also appear in `full` (new `exclude_names_from` option in `build.py`,
  with a test). Result: 503 pairs from 139 tools, 144 mismatches; shortcut baselines at the
  majority rate. Hand-checked the 31 injections the word list missed: no wrong labels, 3
  arguable effect classes. Not scored yet. Details in section 2,
  step 3.
- **2026-11-07 (Andrea, on `main`):** Built and structurally validated a draft Rung 3 set from 50 GitHub tool
  functions: 250 pairs, 159 honest / 91 mismatch. Eighteen invalid weakened-description rows
  were omitted because the interface already declared the effect; 9 supplemental inject/twin
  sets keep the total at 250. The draft uses deterministic templates, not the training LLM
  injector. It still needs provenance recovery, a manual label/template review, stable committed
  paths and a frozen checksum before the one-time model and Agent Audit evaluation.
- **2026-11-03 (latest of that day):** Demoted the 0.0035 cut-off to a footnote (section 5, "Cut-off"):
  on `test` it matched 0.5 on F1 (0.93) and is fitted to one val pair. Results are 0.5 + PR-AUC
  + PR curve. Updated `docs/results-binary.md`; the pair-type chart now compares val and test at
  0.5 (`docs/figures/by_pair_type.svg`).
- **2026-11-03 (later):** Wrote `docs/results-binary.md` (training, val and test results for
  the binary model) with charts from `scripts/plot_results.py`; added the `plots` extra
  (matplotlib).
- **2026-11-03:** Scored rung 1 `test` once with `checkpoints/binary` (job 13657576). At
  0.5: PR-AUC 0.962 (CI 0.928–0.995), P 0.97 / R 0.89 / F1 0.93, group acc 0.91, close to val
  (0.981, F1 0.91). Inject recall 52/54, weaken 7/12; 2 twins flagged with scores ~0.99
  (`c0106-2`, `c0145-1`), no originals or rewrites. Details in section 2, step 4.
- **2026-11-01 (later):** Wrote test-set scoring (`eval/score.py`, `agentlint score`,
  `scripts/carc/score.slurm`) and a tool-bootstrap PR-AUC interval in `metrics.py`, with a test.
  Smoke-tested locally on a tiny random checkpoint (30 test pairs): writes both result files and
  refuses a second run. Submitted the `test` scoring job on CARC (A100) with
  `checkpoints/binary`; results pending.

- **2026-11-01:** 10-epoch binary rerun landed (job 13589508). Best epoch 9: val PR-AUC 0.981
  (CI 0.965–0.994) vs 0.856 at 5 epochs; at 0.5, P 0.96 / R 0.88 / F1 0.91, group acc 0.84.
  Twin false positives gone (0/55 at 0.5); 3 rewrites still flagged. Weakened recall 9/18 at 0.5.
  The saved best-F1 threshold is 0.0035 because the scores are saturated. That's fragile, so
  the team decided (before scoring `test`): 0.5 is the main cut-off, 0.0035 is reported too.
  Cost: weakened recall 9/18 at 0.5 vs 14/18 at 0.0035.

- **2026-10-29:** First binary runs on CARC (A100). Fine-tuned: val PR-AUC 0.856, F1 0.82,
  group acc 0.67, still improving at epoch 5. Frozen-encoder baseline: PR-AUC 0.489, F1 0.51,
  so fine-tuning matters. Weak spots: weakened mismatches (10/18) and twins / rewrites flagged
  across tools (~15%). Details in section 2. Fixed the `metrics.py` breakdowns: effect /
  position / mutation held only mismatches, so precision was always 1.0. Each group now gives
  recall, honest FP rate and precision only when it has both kinds; mutation covers every pair
  type, position adds twins, and weakened pairs get their own `effect_weaken`. Raised
  `binary.yaml` to 10 epochs and submitted the rerun on CARC (job 13589508, results pending).
- **2026-10-28:** Added `scripts/carc/` (setup + SLURM job script) and got setup working on
  Discovery after two fixes (BLAS thread cap; model download without importing transformers).
  First smoke test landed on a P100 and failed with a filesystem read error; the installed
  torch can't run on P100 / V100 anyway. The retry passed on an L40S (job 13526862): the full
  train → val → save path works. The "568 > 512" tokenizer warning and the LOAD REPORT
  (`lm_head` unexpected, `classifier` missing) are expected.
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
