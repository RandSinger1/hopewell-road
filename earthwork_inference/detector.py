"""
detector.py — Sliding-window earthwork detection over a GroundGrid.

    det = Detector("data/earthworks/models/default/best.pt")
    candidates = det.detect(grid)          # list[Candidate], work-CRS outlines

Features are computed once over the whole grid with the checkpoint's own
FeatureSpec (never parameters.py, which may have changed since training),
then cut into overlapping chips.  Detections from overlapping chips are
merged by greedy suppression on intersection-over-*smaller*-box, which also
removes the truncated copy of an earthwork cut by a chip edge (plain IoU NMS
keeps it, because a fragment's IoU with the whole is small).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

import parameters
from earthwork_model.model import load_checkpoint
from earthwork_processor.features import FeatureSpec, compute_features
from lidar.ground_grid import GroundGrid


@dataclass
class Candidate:
    outline: Polygon          # work CRS (metres)
    score: float
    label: str
    area_m2: float


def _starts(n: int, chip: int, stride: int) -> list[int]:
    if n <= chip:
        return [0]
    s = list(range(0, n - chip + 1, stride))
    if s[-1] != n - chip:
        s.append(n - chip)
    return s


def suppress(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    """Greedy suppression by intersection / area of the smaller box."""
    order = np.argsort(-scores)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in order:
        if keep:
            k = np.asarray(keep)
            iw = np.minimum(boxes[k, 2], boxes[i, 2]) - np.maximum(boxes[k, 0], boxes[i, 0])
            ih = np.minimum(boxes[k, 3], boxes[i, 3]) - np.maximum(boxes[k, 1], boxes[i, 1])
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            if (inter / np.minimum(area[k], area[i]).clip(1e-9) > thresh).any():
                continue
        keep.append(int(i))
    return keep


def mask_outline(mask: np.ndarray) -> np.ndarray | None:
    """Largest external contour of a boolean mask as (K, 2) edge-convention (col, row)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(np.float64)
    return c + 0.5 if len(c) >= 3 else None      # pixel centres → edge coords


class Detector:
    def __init__(self, checkpoint: Path | str, device: str | None = None, batch_size: int = 4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.spec = FeatureSpec.from_dict(self.meta["feature_spec"])
        self.chip = int(self.meta["chip_px"])
        self.classes = list(self.meta["classes"])
        self.mean = np.asarray(self.meta["channel_mean"], np.float32)[:, None, None]
        self.std = np.asarray(self.meta["channel_std"], np.float32)[:, None, None]
        self.batch_size = batch_size

    @torch.no_grad()
    def detect(self, grid: GroundGrid, stride: int = parameters.EW_SCAN_STRIDE_PX,
               score_threshold: float = parameters.EW_SCORE_THRESHOLD,
               overlap_thresh: float = parameters.EW_NMS_IOU,
               min_area_m2: float = parameters.EW_MIN_CANDIDATE_M2) -> list[Candidate]:
        if abs(grid.pixel_m - self.spec.pixel_m) > 1e-6:
            raise ValueError(f"grid is {grid.pixel_m} m/px, model expects {self.spec.pixel_m}")
        feats, valid = compute_features(grid.z, self.spec)
        feats = np.where(valid[None], (feats - self.mean) / self.std, 0.0).astype(np.float32)
        h, w = valid.shape
        if h < self.chip or w < self.chip:          # pad small grids up to one chip
            ph, pw = max(0, self.chip - h), max(0, self.chip - w)
            feats = np.pad(feats, ((0, 0), (0, ph), (0, pw)))
            valid = np.pad(valid, ((0, ph), (0, pw)))

        windows = [(r, c) for r in _starts(feats.shape[1], self.chip, stride)
                   for c in _starts(feats.shape[2], self.chip, stride)
                   if valid[r:r + self.chip, c:c + self.chip].mean() > 0.05]
        boxes, scores, labels, masks = [], [], [], []
        for b in range(0, len(windows), self.batch_size):
            batch = windows[b:b + self.batch_size]
            images = [torch.from_numpy(feats[:, r:r + self.chip, c:c + self.chip]).to(self.device)
                      for r, c in batch]
            for (r, c), out in zip(batch, self.model(images)):
                for k in torch.nonzero(out["scores"] >= score_threshold).flatten().tolist():
                    m = (out["masks"][k, 0] > 0.5).cpu().numpy()
                    if not m.any():
                        continue
                    rows, cols = np.nonzero(m)
                    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
                    boxes.append([c + c0, r + r0, c + c1, r + r1])
                    masks.append((r + r0, c + c0, m[r0:r1, c0:c1]))
                    scores.append(float(out["scores"][k]))
                    labels.append(int(out["labels"][k]))
        if not boxes:
            return []

        found = []
        for i in suppress(np.asarray(boxes, float), np.asarray(scores), overlap_thresh):
            r0, c0, m = masks[i]
            ring = mask_outline(m)
            if ring is None:
                continue
            x, y = grid.to_world(ring[:, 0] + c0, ring[:, 1] + r0)
            poly = Polygon(np.column_stack([x, y])).buffer(0)
            if poly.is_empty or poly.area < min_area_m2:
                continue
            if poly.geom_type != "Polygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            found.append(Candidate(poly, scores[i], self.classes[labels[i] - 1], float(poly.area)))
        return found
