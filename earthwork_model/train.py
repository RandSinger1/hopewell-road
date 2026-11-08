"""
train.py — Train the earthwork Mask R-CNN on a processor dataset.

    python -m earthwork_model.train --dataset data/earthworks/dataset/default \
        [--out data/earthworks/models/default] [--epochs 40] [--device cuda] [--resume]

Writes to --out:
    last.pt / best.pt   checkpoints (best = highest validation AP50)
    metrics.csv         per-epoch train loss and validation AP50 / recall
    train_state.pt      model + optimizer + LR schedule + AMP scaler + RNG after the
                        last finished epoch (~3x a checkpoint); --resume continues
                        from it, so an interrupted run (e.g. a Colab session that
                        timed out) loses at most one epoch.  Delete it when done.

Training needs no GDAL / PDAL: on a machine without them (Colab) only this
package, earthwork_processor/{features,augment}.py and parameters.py are
needed — see earthwork_model/colab_bundle.py and colab_train.ipynb.

A GPU is strongly recommended; on CPU an epoch over a few thousand chips
takes hours.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

import parameters
from earthwork_model.dataset import ChipDataset, collate, load_manifest
from earthwork_model.model import build_model, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def average_precision(preds: list[dict], targets: list[dict], iou: float = 0.5) -> tuple[float, float]:
    """Class-agnostic box AP and max recall at an IoU threshold (VOC-style,
    all-point interpolation)."""
    scored, n_gt = [], 0
    for p, t in zip(preds, targets):
        n_gt += len(t["boxes"])
        if not len(p["boxes"]):
            continue
        order = p["scores"].argsort(descending=True)
        taken = torch.zeros(len(t["boxes"]), dtype=torch.bool)
        ious = box_iou(p["boxes"][order], t["boxes"]) if len(t["boxes"]) else None
        for k, j in enumerate(order):
            hit = False
            if ious is not None:
                cand = ious[k].clone()
                cand[taken] = 0
                best = int(cand.argmax())
                if cand[best] >= iou:
                    taken[best] = hit = True
            scored.append((float(p["scores"][j]), hit))
    if not n_gt:
        return float("nan"), float("nan")
    scored.sort(key=lambda s: -s[0])
    tp = np.cumsum([h for _, h in scored])
    fp = np.cumsum([not h for _, h in scored])
    recall = tp / n_gt if len(scored) else np.zeros(1)
    precision = tp / np.maximum(tp + fp, 1) if len(scored) else np.zeros(1)
    # Precision envelope, integrated over recall
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return ap, float(recall[-1]) if len(scored) else 0.0


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    preds, targets = [], []
    for images, tgts in loader:
        out = model([im.to(device) for im in images])
        preds += [{k: v.cpu() for k, v in o.items() if k in ("boxes", "scores")} for o in out]
        targets += [{"boxes": t["boxes"]} for t in tgts]
    return average_precision(preds, targets)


STATE_FILE = "train_state.pt"


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu())
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda"]])


def _save_atomic(obj, path: Path) -> None:
    """Write then rename: a session killed mid-save (Colab) never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(dataset_dir: Path, out_dir: Path, epochs: int, batch_size: int, lr: float,
          device: str, workers: int, pretrained: bool, resume: bool = False) -> Path:
    seed_everything(parameters.EW_SEED)
    manifest = load_manifest(dataset_dir)
    train_ds = ChipDataset(dataset_dir, "train", augment=True)
    val_ds = ChipDataset(dataset_dir, "val")
    if not len(train_ds):
        raise SystemExit("No training chips in the dataset")
    pin = device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                          collate_fn=collate, drop_last=len(train_ds) > batch_size, pin_memory=pin,
                          persistent_workers=workers > 0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=workers, collate_fn=collate,
                        pin_memory=pin)

    model = build_model(len(manifest["classes"]), len(train_ds.channels), manifest["chip_px"],
                        pretrained=pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=parameters.EW_WEIGHT_DECAY)
    steps = epochs * max(len(train_dl), 1)
    warmup = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))))
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    meta = {k: manifest[k] for k in ("feature_spec", "classes", "chip_px", "channel_mean", "channel_std")}
    meta["dataset"] = str(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -math.inf
    start = 1
    run_cfg = {"dataset": dataset_dir.name, "epochs": epochs, "batch_size": batch_size, "lr": lr,
               "steps_per_epoch": len(train_dl)}
    state_path = out_dir / STATE_FILE
    if resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        if st["config"] != run_cfg:
            raise SystemExit(f"{state_path} was saved by a different run:\n  saved {st['config']}\n"
                             f"  now   {run_cfg}\nUse the same --dataset/--epochs/--batch-size/--lr, "
                             f"or delete it to start over")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        _set_rng_state(st["rng"])
        start, best_score = st["epoch"] + 1, st["best_score"]
        print(f"Resuming from {state_path}: epoch {st['epoch']} done, best score {best_score:.3f}")
        if start > epochs:
            print("All epochs already done")
            return out_dir / "best.pt"
    elif resume:
        print(f"No {state_path} yet - starting from epoch 1")
    resuming = start > 1
    with open(out_dir / "metrics.csv", "a" if resuming else "w", newline="") as fh:
        log = csv.writer(fh)
        if not resuming:
            log.writerow(["epoch", "train_loss", "val_ap50", "val_recall"])
        for epoch in range(start, epochs + 1):
            model.train()
            total, n = 0.0, 0
            for images, tgts in train_dl:
                images = [im.to(device) for im in images]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
                with torch.autocast("cuda", enabled=use_amp):
                    loss = sum(model(images, tgts).values())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item()
                n += 1
            ap, rec = evaluate(model, val_dl, device) if len(val_ds) else (float("nan"), float("nan"))
            log.writerow([epoch, total / max(n, 1), ap, rec])
            fh.flush()
            print(f"epoch {epoch:3d}  loss {total / max(n, 1):.4f}  val AP50 {ap:.3f}  recall {rec:.3f}",
                  flush=True)
            save_checkpoint(out_dir / "last.pt", model, {**meta, "epoch": epoch, "val_ap50": ap})
            # No validation sites → fall back to the lowest training loss
            score = ap if not math.isnan(ap) else -total / max(n, 1)
            if score > best_score:
                best_score = score
                save_checkpoint(out_dir / "best.pt", model, {**meta, "epoch": epoch, "val_ap50": ap})
            # Last: a state file implies its epoch's metrics row and checkpoints are written
            _save_atomic({"config": run_cfg, "epoch": epoch, "best_score": best_score,
                          "model": model.state_dict(), "optimizer": opt.state_dict(),
                          "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                          "rng": _rng_state()}, state_path)
    return out_dir / "best.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=parameters.EW_ROOT / "dataset" / "default")
    ap.add_argument("--out", type=Path, default=None, help="Default: data/earthworks/models/<dataset name>")
    ap.add_argument("--epochs", type=int, default=parameters.EW_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=parameters.EW_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=parameters.EW_LEARNING_RATE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    ap.add_argument("--no-pretrained", action="store_true", help="Train from scratch (no COCO weights)")
    ap.add_argument("--resume", action="store_true",
                    help=f"Continue from --out/{STATE_FILE} if present (same dataset/epochs/batch/lr)")
    args = ap.parse_args()
    out = args.out or parameters.EW_ROOT / "models" / args.dataset.name
    best = train(args.dataset, out, args.epochs, args.batch_size, args.lr, args.device,
                 args.workers, pretrained=parameters.EW_PRETRAINED and not args.no_pretrained,
                 resume=args.resume)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
