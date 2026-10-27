"""
model.py — Mask R-CNN (ResNet-50-FPN v2) adapted to LiDAR feature chips.

Why Mask R-CNN: the labels are hand-drawn outlines, and the scan has to save
outlines, so an instance-*segmentation* detector uses all the label
information and returns a polygon per detection — a box alone fits a
diagonal wall or a circle badly.  ResNet-50-FPN is a well-understood
backbone whose FPN levels span 4–64 px strides, matching mounds (~10 m) up to
enclosures (~400 m) at 1 m pixels.

Adaptations from the stock COCO model:
  * ``min_size = max_size = chip_px`` — torchvision otherwise resizes every
    image to 800 px, which would silently change the metres-per-pixel scale;
  * ``image_mean = 0, image_std = 1`` — inputs are already standardised with
    dataset channel statistics (see dataset.py), not ImageNet RGB statistics;
  * a first conv for any channel count: pretrained RGB filters are averaged
    and tiled, so 3-channel (dx, dy, magnitude) input keeps them as-is;
  * all backbone stages trainable — gradients are far from natural images.

Checkpoints carry the feature spec, classes and channel statistics, so
inference can never silently use different preprocessing from training.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ARCH = "maskrcnn_resnet50_fpn_v2"


def _adapt_first_conv(model: nn.Module, in_channels: int) -> None:
    old = model.backbone.body.conv1
    if old.in_channels == in_channels:
        return
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        mean = old.weight.mean(dim=1, keepdim=True)              # (out, 1, k, k)
        new.weight.copy_(mean.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
    model.backbone.body.conv1 = new


def build_model(num_classes: int, in_channels: int, chip_px: int, pretrained: bool = True) -> nn.Module:
    """``num_classes`` excludes background."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=None, trainable_backbone_layers=5 if pretrained else None,
        min_size=chip_px, max_size=chip_px,
        image_mean=[0.0] * in_channels, image_std=[1.0] * in_channels,
        box_detections_per_img=100,
    )
    n = num_classes + 1
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, n)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, n)
    _adapt_first_conv(model, in_channels)
    return model


def save_checkpoint(path: Path | str, model: nn.Module, meta: dict) -> None:
    """``meta`` must hold feature_spec, classes, chip_px, channel_mean, channel_std."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": ARCH, "state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: Path | str, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("arch") != ARCH:
        raise ValueError(f"{path}: unsupported architecture {ckpt.get('arch')!r}")
    model = build_model(len(ckpt["classes"]), len(ckpt["feature_spec"]["channels"]),
                        ckpt["chip_px"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
