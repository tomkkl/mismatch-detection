"""Score a saved checkpoint on a test set (rung 1 `test`, rungs 2–3), once.

Reads `<checkpoint>/best/` (model + tokenizer), `config.yaml` (input settings) and
`threshold.json` (best-F1 cut-off on val). Reports at 0.5, the main cut-off, and at the saved
cut-off (CLAUDE.md section 5), plus a tool-bootstrap 95% interval for PR-AUC.

Writes to `<checkpoint>/eval/<tag>/`: predictions.jsonl, metrics.json. Refuses to overwrite an
existing result: each test set is scored once, after tuning on val is done.
"""

import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agentlint.eval.metrics import bootstrap_pr_auc, report
from agentlint.model.dataset import PairDataset
from agentlint.model.train import load_pairs, pick_device, predict

MAIN_THRESHOLD = 0.5


def default_tag(pairs_path: Path) -> str:
    """data/splits/full/test.jsonl -> full-test; data/generated/rung2-opus/pairs.jsonl -> rung2-opus-pairs."""
    return f"{pairs_path.parent.name}-{pairs_path.stem}"


def out_dir_for(checkpoint: Path, pairs_path: Path, tag: str | None = None) -> Path:
    return checkpoint / "eval" / (tag or default_tag(pairs_path))


def score(checkpoint: str, pairs_path: str, tag: str | None = None, force: bool = False,
          batch_size: int | None = None) -> dict:
    checkpoint, pairs_path = Path(checkpoint), Path(pairs_path)
    out_dir = out_dir_for(checkpoint, pairs_path, tag)
    if (out_dir / "metrics.json").exists() and not force:
        raise FileExistsError(f"{out_dir} already has results; test sets are scored once "
                              "(pass --force only to redo a run that failed or was wrong)")

    cfg = yaml.safe_load(open(checkpoint / "config.yaml"))
    if cfg["task"] != "binary":
        raise NotImplementedError("only task: binary is implemented so far")
    saved = json.loads((checkpoint / "threshold.json").read_text())
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "best")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint / "best").to(device)
    pairs = load_pairs(pairs_path)
    ds = PairDataset(pairs, tokenizer, max_length=cfg["max_length"], stride=cfg["chunk_stride"],
                     task=cfg["task"])
    loader = DataLoader(ds, batch_size=batch_size or cfg["batch_size"], collate_fn=ds.collate)
    rows = predict(model, loader, {p.pair_id: p for p in pairs}, device)

    ci = bootstrap_pr_auc(rows)
    result = {
        "checkpoint": str(checkpoint), "pairs": str(pairs_path), "n": len(rows),
        "checkpoint_epoch": saved["epoch"], "pr_auc_ci95": list(ci),
        "at_0.5": report(rows, MAIN_THRESHOLD),
        "at_saved_threshold": report(rows, saved["threshold"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"{pairs_path} | {len(rows)} pairs | checkpoint {checkpoint} (epoch {saved['epoch']})")
    print(f"PR-AUC {result['at_0.5']['pr_auc']:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})")
    for name, t in [("at_0.5", MAIN_THRESHOLD), ("at_saved_threshold", saved["threshold"])]:
        m = result[name]
        mut = m["breakdowns"]["mutation"]
        print(f"@{t:.4g} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
              f"group acc {m['group_accuracy']:.3f} | "
              + " ".join(f"{k} {'R' if v['recall'] is not None else 'FP'} "
                         f"{v['recall'] if v['recall'] is not None else v['fp_rate']:.2f}"
                         for k, v in mut.items()))
    print(f"saved to {out_dir}")
    return result
