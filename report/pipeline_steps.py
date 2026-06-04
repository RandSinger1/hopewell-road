"""
pipeline_steps.py  —  Step-by-step processing pipeline figure.

Produces a 2-row × 5-column figure showing every stage from raw DTM to
detected parallel wall pairs.

Row 1 — Pre-processing
  [1] Raw DTM (hillshaded)
  [2] Median-smoothed DTM
  [3] Local Relief Model (LRM)
  [4] Multi-scale Frangi ridge response
  [5] Sky-View Factor

Row 2 — Detection
  [6] Binary ridge threshold
  [7] Morphological closing
  [8] Medial-axis skeleton
  [9] Labelled skeleton components
  [10] Detected parallel wall pairs

Call generate_pipeline_figure(coord, dtm_path, output_dir) from run_report.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from road_model.process_dtm import (
    getGTAndBand, get_window,
    median_smooth_windowed, local_relief_model, sky_view_factor_xr,
    multiscale_ridge_enhancement,
)
from road_model.ridge_detection import detect_parallel_walls_ridge_pairing, remove_invalid_pairs
from .report_utils import render_hillshaded_dtm, save_figure, style_axes, pair_to_geo
from . import config as cfg


_STEP_LABELS = [
    "① Raw DTM",
    "② Median Smooth",
    "③ Local Relief Model",
    "④ LRM (detection input)",
    "⑤ Sky-View Factor",
    "⑥ Local Relief Model",
    "⑦ LRM Threshold",
    "⑧ Morph Close",
    "⑨ Skeleton",
    "⑩ Line Detection",
]


def _img(ax: plt.Axes, data: np.ndarray, label: str,
         cmap: str = "gray", extent: list = None) -> None:
    kw = dict(origin="upper", cmap=cmap, aspect="auto", interpolation="nearest")
    if extent:
        kw["extent"] = extent
    ax.imshow(data, **kw)
    style_axes(ax, label)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_pipeline_figure(coord: dict, dtm_path: str, output_dir: Path) -> None:
    """
    Build and save the 2×5 pipeline-steps figure for one coordinate.

    Parameters
    ----------
    coord      : dict with keys 'name', 'lat', 'lon', 'size_m'
    dtm_path   : path to merged DTM GeoTIFF
    output_dir : parent output directory (a subdirectory named coord['name'] is created)
    """
    name = coord["name"]
    lat, lon = coord["lat"], coord["lon"]
    size_m = coord.get("size_m", 200)

    print(f"  [pipeline] loading DTM window …")
    gt, band, nodata, _ds = getGTAndBand(dtm_path)
    raw, extent = get_window(lat, lon, size_m, size_m, gt, band, nodata)
    if raw is None or raw.size == 0:
        print(f"  [pipeline] no data — skipped.")
        return

    ext = [extent[0], extent[1], extent[2], extent[3]]

    # ── Pre-processing ────────────────────────────────────────────────────────
    print(f"  [pipeline] pre-processing …")
    smoothed = median_smooth_windowed(raw, nodata=nodata, radius=cfg.MEDIAN_SMOOTH_RADIUS)
    lrm = local_relief_model(smoothed, radius=cfg.LRM_RADIUS, nodata=nodata)
    ridge = multiscale_ridge_enhancement(lrm, scales=cfg.RIDGE_SCALES, nodata=nodata)
    svf = sky_view_factor_xr(smoothed, radius=15, directions=16, nodata=nodata)

    # ── Detection ─────────────────────────────────────────────────────────────
    print(f"  [pipeline] detection steps …")
    nd_mask = raw == nodata
    valid_ridge = np.where(nd_mask, 0.0, ridge.astype(float))

    pairs, (binary, closed, skeleton, labeled, _) = detect_parallel_walls_ridge_pairing(
        ridge_map=valid_ridge,
        target_angle=cfg.TARGET_ANGLE,
        extent=extent,
        spacing_meters=cfg.SPACING_METERS,
        angle_tolerance=cfg.ANGLE_TOLERANCE,
        spacing_tolerance=cfg.SPACING_TOLERANCE,
        ridge_threshold=cfg.RIDGE_THRESHOLD,
        min_ridge_length_meters=cfg.MIN_RIDGE_LENGTH_METERS,
        overlap_ratio=cfg.OVERLAP_RATIO,
        #orientation_source=lrm,
        return_debug=True,
    )
    pairs = remove_invalid_pairs(pairs, angle_tolerance=cfg.ANGLE_TOLERANCE,
                                  target_angle=cfg.TARGET_ANGLE)

    # ── Build figure ──────────────────────────────────────────────────────────
    print(f"  [pipeline] composing figure …")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"LiDAR Processing Pipeline  ·  {name.replace('_', ' ')}  "
        f"({lat:.4f}°N, {lon:.4f}°E)",
        fontsize=13, color="white", y=0.99,
    )

    hs_raw = render_hillshaded_dtm(raw, nodata=nodata, azimuth=315)
    hs_smooth = render_hillshaded_dtm(smoothed, nodata=nodata, azimuth=315)

    lrm_disp = np.where(lrm == nodata, np.nan, lrm)
    svf_disp = svf.astype(float)
    # NaN (nodata) stays NaN — imshow renders it as the axes background

    # Row 1
    axes[0, 0].imshow(hs_raw, origin="upper", extent=ext, aspect="auto")
    style_axes(axes[0, 0], _STEP_LABELS[0])
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    axes[0, 1].imshow(hs_smooth, origin="upper", extent=ext, aspect="auto")
    style_axes(axes[0, 1], _STEP_LABELS[1])
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    im_lrm = axes[0, 2].imshow(lrm_disp, origin="upper", cmap="RdBu_r",
                                 extent=ext, aspect="auto")
    fig.colorbar(im_lrm, ax=axes[0, 2], fraction=0.046, pad=0.03).ax.yaxis.set_tick_params(colors="gray", labelsize=6)
    style_axes(axes[0, 2], _STEP_LABELS[2])
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

    ridge_disp = np.where(ridge == nodata, np.nan, ridge)
    im_ridge = axes[0, 3].imshow(ridge_disp, origin="upper", cmap="hot",
                                   extent=ext, aspect="auto")
    fig.colorbar(im_ridge, ax=axes[0, 3], fraction=0.046, pad=0.03).ax.yaxis.set_tick_params(colors="gray", labelsize=6)
    style_axes(axes[0, 3], "④ Frangi Ridge Response")
    axes[0, 3].set_xticks([]); axes[0, 3].set_yticks([])

    im_svf = axes[0, 4].imshow(svf_disp, origin="upper", cmap="gray",
                                 extent=ext, aspect="auto")
    fig.colorbar(im_svf, ax=axes[0, 4], fraction=0.046, pad=0.03).ax.yaxis.set_tick_params(colors="gray", labelsize=6)
    style_axes(axes[0, 4], _STEP_LABELS[4])
    axes[0, 4].set_xticks([]); axes[0, 4].set_yticks([])

    # Row 2
    labeled_disp = labeled.astype(float)
    labeled_disp[labeled == 0] = np.nan
    im_lab = axes[1, 0].imshow(labeled_disp, origin="upper", cmap="tab20",
                                extent=ext, aspect="auto", interpolation="nearest")
    fig.colorbar(im_lab, ax=axes[1, 0], fraction=0.046, pad=0.03).ax.yaxis.set_tick_params(colors="gray", labelsize=6)
    n_components = labeled.max()
    style_axes(axes[1, 0], f"⑥ Labeled Components  ({n_components})")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    _img(axes[1, 1], binary.astype(np.uint8), "⑦ Ridge Threshold", cmap="gray", extent=ext)
    _img(axes[1, 2], closed.astype(np.uint8), _STEP_LABELS[7], cmap="gray", extent=ext)
    _img(axes[1, 3], skeleton.astype(np.uint8), _STEP_LABELS[8], cmap="gray", extent=ext)

    # Panel ⑩ — pairs on hillshaded DTM
    axes[1, 4].imshow(hs_raw, origin="upper", extent=ext, aspect="auto")
    COLORS = ["#ff4444", "#4499ff"]
    for pair in pairs[:12]:
        for k, geo_seg in enumerate(pair_to_geo(pair, extent, raw.shape)):
            (gx0, gy0), (gx1, gy1) = geo_seg
            axes[1, 4].plot([gx0, gx1], [gy0, gy1],
                            color=COLORS[k], linewidth=2.0, alpha=0.85)
    style_axes(
        axes[1, 4],
        f"{_STEP_LABELS[9]}  ({len(pairs)} pair{'s' if len(pairs) != 1 else ''})",
    )
    axes[1, 4].set_xticks([]); axes[1, 4].set_yticks([])

    # Uniform dark frame
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#111111")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_dir / name, "pipeline_steps.png", dpi=cfg.DPI)
