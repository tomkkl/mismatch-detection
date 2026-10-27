"""Fine-tune on a config from configs/*.yaml.

Each pair's logit is the max over its chunks (`max_pool_by_pair`); the weighted BCE loss is on
that pooled logit. `freeze_encoder: true` trains only the head: the "before fine-tuning"
baseline. The epoch with the best val PR-AUC is kept. `test` is never read here.

Writes to `output_dir`: best/ (model + tokenizer), metrics.json, val_predictions.jsonl,
threshold.json (best-F1 threshold on val, for scoring test later), config.yaml.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from agentlint.eval.metrics import best_f1_threshold, report
from agentlint.model.classifier import build_model, max_pool_by_pair
from agentlint.model.dataset import PairDataset
from agentlint.schema import ToolPair


def load_pairs(path: Path, limit: int | None = None) -> list[ToolPair]:
    with open(path) as f:
        pairs = [ToolPair.from_dict(json.loads(line)) for line in f]
    return pairs[:limit] if limit else pairs


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def freeze_encoder(model) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier.")


def pooled_logits(model, batch: dict, device: torch.device) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
    return max_pool_by_pair(out.logits.float(), batch["pair_index"].to(device), len(batch["pair_ids"]))


@torch.no_grad()
def predict(model, loader: DataLoader, pairs: dict[str, ToolPair], device: torch.device) -> list[dict]:
    """One row per pair: score = sigmoid(pooled logit), plus the fields the breakdowns need."""
    model.eval()
    rows = []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            scores = torch.sigmoid(pooled_logits(model, batch, device)).squeeze(1).tolist()
        n_chunks = torch.bincount(batch["pair_index"], minlength=len(batch["pair_ids"])).tolist()
        for pid, score, n in zip(batch["pair_ids"], scores, n_chunks):
            p = pairs[pid]
            rows.append({
                "pair_id": pid, "source_id": p.source_id, "score": score, "label": p.mismatch,
                "n_chunks": n, "undeclared": [e.value for e in p.undeclared],
                "mutation": p.mutation, "position": p.meta.get("position"),
            })
    return rows


def train(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path))
    if cfg["task"] != "binary":
        raise NotImplementedError("only task: binary is implemented so far")
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir, out_dir = Path(cfg["data_dir"]), Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_pairs = load_pairs(data_dir / "train.jsonl", cfg.get("limit_train"))
    val_pairs = load_pairs(data_dir / "val.jsonl", cfg.get("limit_val"))
    ds_args = dict(max_length=cfg["max_length"], stride=cfg["chunk_stride"], task=cfg["task"])
    train_ds = PairDataset(train_pairs, tokenizer, **ds_args)
    val_ds = PairDataset(val_pairs, tokenizer, **ds_args)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=train_ds.collate, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=val_ds.collate)
    val_by_id = {p.pair_id: p for p in val_pairs}

    model = build_model(cfg["task"], cfg["model_name"])
    if cfg.get("freeze_encoder"):
        freeze_encoder(model)
    model.to(device)

    n_pos = sum(p.mismatch for p in train_pairs)
    pos_weight = torch.tensor([(len(train_pairs) - n_pos) / n_pos], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    accum = cfg.get("grad_accum", 1)
    total_steps = cfg["epochs"] * -(-len(train_loader) // accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.get("warmup", 0.0) * total_steps), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print(f"device {device} | train {len(train_pairs)} pairs / {sum(len(i['chunks']) for i in train_ds.items)} "
          f"chunks | val {len(val_pairs)} | pos_weight {pos_weight.item():.2f} | "
          f"trainable params {sum(p.numel() for p in trainable):,}")

    history, best = [], None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n_batches = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = pooled_logits(model, batch, device)
            loss = loss_fn(logits, batch["labels"].to(device))
            scaler.scale(loss / accum).backward()
            total, n_batches = total + loss.item(), n_batches + 1
            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        rows = predict(model, val_loader, val_by_id, device)
        labels, scores = [r["label"] for r in rows], [r["score"] for r in rows]
        threshold = best_f1_threshold(labels, scores)
        entry = {"epoch": epoch, "train_loss": total / n_batches,
                 "val_at_0.5": report(rows, 0.5), "val_at_best_f1": report(rows, threshold)}
        history.append(entry)
        m, t = entry["val_at_0.5"], entry["val_at_best_f1"]
        print(f"epoch {epoch} | loss {entry['train_loss']:.4f} | PR-AUC {m['pr_auc']:.3f} | "
              f"@0.5 P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} | "
              f"@{threshold:.2f} F1 {t['f1']:.3f} group acc {t['group_accuracy']:.3f}")

        if best is None or m["pr_auc"] > best["val_at_0.5"]["pr_auc"]:
            best = entry
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
            with open(out_dir / "val_predictions.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            (out_dir / "threshold.json").write_text(json.dumps({"threshold": threshold, "epoch": epoch}))

    result = {"best_epoch": best["epoch"], "best": best, "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"best epoch {best['epoch']}: PR-AUC {best['val_at_0.5']['pr_auc']:.3f}, saved to {out_dir}")
    return result
