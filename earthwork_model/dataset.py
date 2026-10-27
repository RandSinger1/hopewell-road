"""
dataset.py — torch Dataset over earthwork_processor chips.

Each item is (image (C, N, N) float32, target) in torchvision detection
format: target = {boxes (K, 4) xyxy, labels (K,) 1-based, masks (K, N, N) uint8}.
Chips with no earthwork have K = 0, which Mask R-CNN treats as pure background.

Training adds exact 8-fold dihedral augmentation (rot90 × flip) on top of
the processor's arbitrary-angle chips, using features.rot90 / flip so the
dx/dy vector channels rotate with the image.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from earthwork_processor import features
from earthwork_processor.augment import rasterize


def load_manifest(dataset_dir: Path | str) -> dict:
    return json.loads((Path(dataset_dir) / "manifest.json").read_text())


class ChipDataset(Dataset):
    def __init__(self, dataset_dir: Path | str, split: str, augment: bool = False):
        self.root = Path(dataset_dir)
        self.manifest = load_manifest(self.root)
        self.chips = [c for c in self.manifest["chips"] if c["split"] == split]
        self.channels = tuple(self.manifest["feature_spec"]["channels"])
        self.mean = np.asarray(self.manifest["channel_mean"], np.float32)[:, None, None]
        self.std = np.asarray(self.manifest["channel_std"], np.float32)[:, None, None]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, i: int):
        d = np.load(self.root / self.chips[i]["file"])
        feats = d["feats"].astype(np.float32)
        valid = d["valid"]
        size = feats.shape[-1]
        polys = np.split(d["poly_xy"], np.cumsum(d["poly_len"])[:-1]) if len(d["poly_len"]) else []
        masks = np.stack([rasterize(p, size) for p in polys]) if polys else np.zeros((0, size, size), bool)

        if self.augment:
            # torch's RNG, not numpy's: DataLoader reseeds it per worker
            k = int(torch.randint(4, ()))
            feats = features.rot90(feats, k, self.channels)
            masks = np.rot90(masks, k, axes=(1, 2))
            valid = np.rot90(valid, k)
            if torch.rand(()) < 0.5:
                feats = features.flip(feats, "lr", self.channels)
                masks, valid = masks[:, :, ::-1], valid[:, ::-1]

        feats = np.where(valid[None], (feats - self.mean) / self.std, 0.0).astype(np.float32)
        return torch.from_numpy(feats), make_target(masks, d["poly_cls"])


def make_target(masks: np.ndarray, classes: np.ndarray) -> dict:
    keep = masks.reshape(len(masks), -1).any(axis=1) if len(masks) else np.zeros(0, bool)
    masks, classes = np.ascontiguousarray(masks[keep]), classes[keep]
    boxes = np.zeros((len(masks), 4), np.float32)
    for j, m in enumerate(masks):
        rows, cols = np.nonzero(m)
        boxes[j] = cols.min(), rows.min(), cols.max() + 1, rows.max() + 1
    return {"boxes": torch.from_numpy(boxes),
            "labels": torch.as_tensor(classes + 1, dtype=torch.int64),
            "masks": torch.from_numpy(masks.astype(np.uint8))}


def collate(batch):
    return tuple(zip(*batch))
