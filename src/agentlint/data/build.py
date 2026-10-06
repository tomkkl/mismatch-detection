"""Dataset pipeline: generate -> verify -> inject + twin + rewrite -> weaken -> split.

Writes to data/generated/<tag>/ (pairs.jsonl, rejects.jsonl, stats.json) and
data/splits/<tag>/ (train/val/test.jsonl). LLM answers are cached, so rerunning the same
command after a usage-limit stop resumes instead of starting over.
"""

import json
import random
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import yaml

from agentlint.data.generate import generate_batch, plan_calls
from agentlint.data.llm import LLM, ClaudeCLI, LLMError, UsageLimitError
from agentlint.data.mutate import (POSITIONS, Declined, declare_effect, honest_rewrite,
                                   inject_code, match_length, weaken_description)
from agentlint.data.split import group_split
from agentlint.data.verify import (interface_text, verify_honest, verify_inject,
                                   verify_length_match, verify_twin, verify_weaken)
from agentlint.effects import code_effects, text_mentions
from agentlint.localize import locate
from agentlint.schema import EFFECTS, Effect, ToolPair


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parallel(fn: Callable, items: list, workers: int, label: str) -> list[tuple]:
    """Run fn over items. Returns (item, result, error) in input order.

    Per-item LLM errors are returned; a usage-limit error stops everything.
    """
    results: list[tuple | None] = [None] * len(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = (items[i], fut.result(), None)
            except UsageLimitError:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Declined as e:
                results[i] = (items[i], None, f"declined: {e}")
            except LLMError as e:
                results[i] = (items[i], None, str(e))
            done += 1
            if done % 10 == 0 or done == len(items):
                print(f"  {label}: {done}/{len(items)}", flush=True)
    return results  # type: ignore[return-value]


def _pick(candidates: Iterable, counts: Counter, rng: random.Random):
    """Least-used candidate so far, ties broken at random."""
    candidates = list(candidates)
    low = min(counts[c] for c in candidates)
    return rng.choice([c for c in candidates if counts[c] == low])


def run(cfg: dict, tag: str, llm: LLM | None = None) -> dict:
    gen = cfg["generator"]
    if llm is None:
        llm = ClaudeCLI(model=gen["model"], effort=gen["effort"], cache_dir=cfg["cache_dir"])
    workers, seed = cfg["workers"], cfg["seed"]
    rejects: list[dict] = []

    def reject(stage: str, pair_id: str, reason: str):
        rejects.append({"stage": stage, "pair_id": pair_id, "reason": reason})

    # 1. honest tools
    plans = plan_calls(cfg["n_honest"], cfg["tools_per_call"], cfg["domains"],
                       cfg["p_second_effect"], seed)
    print(f"generate: {len(plans)} calls for {cfg['n_honest']} honest tools", flush=True)
    honest: list[ToolPair] = []
    for plan, tools, err in _parallel(lambda p: generate_batch(llm, p, gen["effort"]), plans,
                                      workers, "generate"):
        if err:
            reject("generate", plan.call_id, err)
            continue
        honest += tools

    kept, names = [], set()
    for t in sorted(honest, key=lambda t: t.source_id):
        ok, why = verify_honest(t)
        if ok and t.name in names:
            ok, why = False, "duplicate tool name"
        if not ok:
            reject("honest", t.pair_id, why)
            continue
        names.add(t.name)
        kept.append(t)
    honest = kept

    # 2. injected mutants + honest twins
    effect_counts, pos_counts, jobs = Counter(), Counter(), []
    for t in honest:
        rng = random.Random(f"{seed}:{t.source_id}")
        present = set(code_effects(t.code))
        iface = interface_text(t)
        cands = [e for e in EFFECTS if e.value not in t.meta["declared"]
                 and e not in present and not text_mentions(iface, e)]
        for _ in range(cfg["injections_per_tool"]):
            if not cands:
                reject("inject", t.pair_id, "no injectable effect")
                break
            e = _pick(cands, effect_counts, rng)
            cands.remove(e)
            p = _pick(POSITIONS, pos_counts, rng)
            effect_counts[e] += 1
            pos_counts[p] += 1
            jobs.append((t, e, p))

    by_source = {t.source_id: t for t in honest}
    print(f"inject: {len(jobs)} mutants", flush=True)
    mutants = []
    for (t, e, p), m, err in _parallel(
            lambda j: inject_code(llm, j[0], j[1], j[2], cfg["max_added_lines"],
                                  gen["mutate_effort"]), jobs, workers, "inject"):
        if err:
            reject("inject", f"{t.source_id}/inject-{e.value}-{p}", err)
            continue
        ok, why = verify_inject(t, m, e, cfg["max_changed_lines"], cfg["max_added_lines"])
        if not ok:
            reject("inject", m.pair_id, why)
            continue
        mutants.append(m)

    print(f"twin: {len(mutants)} descriptions", flush=True)
    with_twin = []
    for m, tw, err in _parallel(
            lambda m: declare_effect(llm, m, m.undeclared[0], gen["rewrite_effort"]), mutants,
            workers, "twin"):
        if err:
            reject("twin", m.pair_id, err)
            continue
        ok, why = verify_twin(m, tw, m.undeclared[0], tuple(cfg["twin_length_ratio"]))
        if not ok:
            reject("twin", tw.pair_id, why)
            continue
        with_twin.append((m, tw))

    # Rewrite each mismatch description to its twin's length; drop the pair if that fails,
    # since an unmatched pair is exactly the length shortcut this step removes.
    print(f"match: {len(with_twin)} descriptions", flush=True)
    mutants, twins = [], []
    for (m, tw), mm, err in _parallel(
            lambda j: match_length(llm, by_source[j[0].source_id], j[0],
                                   len(j[1].description), gen["rewrite_effort"]),
            with_twin, workers, "match"):
        if err:
            reject("match", m.pair_id, err)
            continue
        ok, why = verify_length_match(m, mm, tw, m.undeclared[0], cfg["match_length_tol"])
        if not ok:
            reject("match", m.pair_id, why)
            continue
        mutants.append(mm)
        twins.append(tw)
    rewrites = [honest_rewrite(by_source[m.source_id], m) for m in mutants]

    # 3. weakened descriptions
    weak_counts, wjobs = Counter(), []
    for t in honest:
        declared = [Effect(e) for e in t.meta["declared"]]
        if len(declared) < 2:
            continue
        iface = interface_text(t)
        cands = [e for e in declared if e != Effect.READ and text_mentions(t.description, e)
                 and not text_mentions(iface, e)]
        if not cands:
            reject("weaken", t.pair_id, "no weakenable effect")
            continue
        e = _pick(cands, weak_counts, random.Random(f"{seed}:{t.source_id}:weaken"))
        weak_counts[e] += 1
        wjobs.append((t, e))

    print(f"weaken: {len(wjobs)} descriptions", flush=True)
    weakened = []
    for (t, e), w, err in _parallel(
            lambda j: weaken_description(llm, j[0], j[1], gen["rewrite_effort"]), wjobs,
            workers, "weaken"):
        if err:
            reject("weaken", f"{t.source_id}/weaken-{e.value}", err)
            continue
        ok, why = verify_weaken(t, w, e, tuple(cfg["weaken_length_ratio"]))
        if not ok:
            reject("weaken", w.pair_id, why)
            continue
        w.injected_lines = locate(w.code, e)
        weakened.append(w)

    # 4. write
    pairs = sorted(honest + mutants + twins + rewrites + weakened, key=lambda p: p.pair_id)
    out = Path(cfg["out_dir"]) / tag
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "pairs.jsonl", [p.to_dict() for p in pairs])
    _write_jsonl(out / "rejects.jsonl", rejects)

    split_dir = Path(cfg["split_dir"]) / tag
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "val", "test"),
                          group_split(pairs, tuple(cfg["fractions"]), seed)):
        _write_jsonl(split_dir / f"{name}.jsonl", [p.to_dict() for p in part])

    stats = _stats(pairs, rejects, llm)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 1) if xs else 0.0


def _length_baseline(pairs: list[ToolPair]) -> dict:
    """Shortcut check: how well do description length + code line count, or the description
    words alone, predict the label? Both should stay near the majority-class rate, since the
    label depends on description and code together. Folds are grouped by source."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    y = np.array([p.mismatch for p in pairs])
    groups = [p.source_id for p in pairs]
    n_folds = min(5, len(set(groups)))
    majority = round(float(max(y.mean(), 1 - y.mean())), 3) if len(y) else 0.0
    if n_folds < 2 or len(set(y)) < 2:
        return {"majority": majority, "length_only_accuracy": None,
                "description_only_accuracy": None}
    cv = GroupKFold(n_folds)
    X = np.array([[len(p.description), p.code.count("\n") + 1] for p in pairs], float)
    length = cross_val_score(LogisticRegression(max_iter=1000), X, y, groups=groups, cv=cv)
    words = cross_val_score(
        make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000)),
        [p.description for p in pairs], y, groups=groups, cv=cv)
    return {"majority": majority, "length_only_accuracy": round(float(length.mean()), 3),
            "description_only_accuracy": round(float(words.mean()), 3)}


def _stats(pairs: list[ToolPair], rejects: list[dict], llm: LLM) -> dict:
    by_label = {lab: [p for p in pairs if p.mismatch == lab] for lab in (False, True)}
    return {
        "pairs": len(pairs),
        "sources": len({p.source_id for p in pairs}),
        "mismatch": len(by_label[True]),
        "honest": len(by_label[False]),
        "by_mutation": dict(Counter(p.mutation or "original" for p in pairs)),
        "undeclared_effect": dict(Counter(p.undeclared[0].value for p in by_label[True])),
        "inject_position": dict(Counter(p.meta["position"] for p in pairs
                                        if p.mutation == "inject_code")),
        "domains": len({p.meta["domain"] for p in pairs}),
        "lexicon_agrees_on_inject": dict(Counter(
            p.meta["checks"].get("lexicon_agrees") for p in pairs if p.mutation == "inject_code")),
        # Shortcut watch: these should look alike for both labels.
        "mean_desc_chars": {"honest": _mean([len(p.description) for p in by_label[False]]),
                            "mismatch": _mean([len(p.description) for p in by_label[True]])},
        "mean_code_lines": {"honest": _mean([p.code.count("\n") + 1 for p in by_label[False]]),
                            "mismatch": _mean([p.code.count("\n") + 1 for p in by_label[True]])},
        "shortcut_baseline": _length_baseline(pairs),
        "lexicon_declares_on_twin": dict(Counter(
            p.meta["checks"].get("lexicon_declares") for p in pairs
            if p.mutation == "declare_effect")),
        "rejects": dict(Counter(
            f"{r['stage']}: {'declined' if r['reason'].startswith('declined') else r['reason'][:60]}"
            for r in rejects)),
        "llm": {"name": llm.name, "calls": getattr(llm, "calls", None),
                "cache_hits": getattr(llm, "cache_hits", None),
                "list_price_usd": round(getattr(llm, "cost_usd", 0.0), 2)},
    }
