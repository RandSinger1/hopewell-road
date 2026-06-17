"""
pipeline_steps.py  —  Step-by-step processing pipeline figure.

Produces a 2-row × 5-column figure showing every stage from raw DTM to
detected parallel wall pairs.

Row 1 — Pre-processing → amplitude gate
  [1] Raw DTM (hillshaded)
  [2] Median-smoothed DTM
  [3] Local Relief Model (LRM)
  [4] Multi-scale Frangi ridge response
  [5] Ridge amplitude threshold
  [6] Hessian orientation

Row 2 — Orientation filter → binary → skeleton → pairs + diagnostic
  [7] Angle deviation map
  [8] Along-wall gap fill
  [9] Skeleton
  [10] Labelled skeleton components
  [11] Detected parallel wall pairs
  [12] Hillshade perpendicular to 32° (visual diagnostic)

Call generate_pipeline_figure(coord, dtm_path, output_dir) from run_report.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from road_model.ridge_detection import hessian_ridge_orientation
from .base_compute import get_terrain, get_detection
from .report_utils import (render_hillshaded_dtm, hillshade, save_figure, style_axes,
                            pair_to_geo, pair_style, confirmed_pairs_label)
from . import config as cfg
import parameters


_STEP_LABELS = [
    "1. Raw DTM",
    "2. Median Smooth",
    "3. Local Relief Model",
    "4. Frangi Ridge Response",
    "5. Ridge Amplitude Threshold",
    "6. Hessian Orientation",
    "7. Angle Deviation",
    "8. Binary Mask",
    "9. Along-Wall Gap Fill",
    "10. Skeleton",
    "11. Labeled Components",
    "12. Detected Pairs",
    "13. Hillshade ⊥ 32°",
]

# Slugs follow flat axis order (left-to-right, top-to-bottom in a 2×7 grid).
# None marks the unused last cell in row 1.
_STEP_SLUGS = [
    "01_raw_dtm",               # [0,0]
    "02_median_smooth",         # [0,1]
    "03_local_relief_model",    # [0,2]
    "04_frangi_ridge_response", # [0,3]
    "05_ridge_response",        # [0,4] — amplitude threshold
    "06_hessian_orientation",   # [0,5]
    "07_angle_deviation",       # [0,6] — moved to complete row 0
    "08_binary_mask",           # [1,0] — NEW
    "09_gap_fill",              # [1,1]
    "10_skeleton",              # [1,2]
    "11_labeled_components",    # [1,3]
    "12_detected_pairs",        # [1,4]
    "13_hillshade_perp32",      # [1,5]
    None,                       # [1,6] — empty cell
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


def generate_pipeline_figure(coord: dict, dtm_path: str, output_dir: Path,
                              use_cache: bool = True) -> None:
    """
    Build and save the 2×5 pipeline-steps figure for one coordinate.

    Parameters
    ----------
    coord      : dict with keys 'name', 'lat', 'lon', 'size_m'
    dtm_path   : path to merged DTM GeoTIFF
    output_dir : parent output directory (a subdirectory named coord['name'] is created)
    use_cache  : reuse cached DTM window / smoothing / LRM / detection if available
    """
    name = coord["name"]
    lat, lon = coord["lat"], coord["lon"]

    print(f"  [pipeline] loading DTM window …")
    terrain = get_terrain(coord, dtm_path, output_dir, use_cache=use_cache)
    if terrain is None:
        print(f"  [pipeline] no data — skipped.")
        return
    raw, extent, nodata = terrain["raw"], terrain["extent"], terrain["nodata"]
    smoothed, lrm = terrain["smoothed"], terrain["lrm"]

    ext = [extent[0], extent[1], extent[2], extent[3]]

    # ── Detection (cached: smoothing/LRM/ridge/pairing are the expensive parts) ─
    print(f"  [pipeline] detection steps …")
    det = get_detection(coord, dtm_path, terrain, output_dir, use_cache=use_cache)
    ridge, valid_ridge = det["ridge"], det["valid_ridge"]
    binary, closed, skeleton, labeled = det["binary"], det["closed"], det["skeleton"], det["labeled"]
    pairs = det["pairs_clean"]
    nd_mask = raw == nodata

    # ── Build figure  (2 rows × 7 columns) ───────────────────────────────────
    print(f"  [pipeline] composing figure …")
    fig, axes = plt.subplots(2, 7, figsize=(30, 9))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"LiDAR Processing Pipeline  ·  {name.replace('_', ' ')}  "
        f"({lat:.4f}°N, {lon:.4f}°E)",
        fontsize=13, color="white", y=0.99,
    )

    hs_raw = render_hillshaded_dtm(raw, nodata=nodata, azimuth=315)
    hs_smooth = render_hillshaded_dtm(smoothed, nodata=nodata, azimuth=315)
    lrm_disp = np.where(lrm == nodata, np.nan, lrm)
    # Intense hillshade perpendicular to 32° (azimuth 122°), low sun angle for maximum contrast
    hs_perp32 = hillshade(smoothed, nodata=nodata, azimuth=122.0, altitude=20.0, z_factor=4.0)

    # Hessian orientation & angle-deviation maps (computed once, used in row 1 & 2)
    orient_rad = hessian_ridge_orientation(valid_ridge, sigma=parameters.HESSIAN_SIGMA)
    orient_deg = np.rad2deg(orient_rad)                        # [0, 180)
    ridge_present = valid_ridge > cfg.RIDGE_THRESHOLD * 0.05
    orient_disp = np.where(ridge_present, orient_deg, np.nan)

    target_rad = np.deg2rad(cfg.TARGET_ANGLE)
    diff_rad = np.abs(orient_rad - target_rad)
    diff_rad = np.minimum(diff_rad, np.pi - diff_rad)
    diff_disp = np.where(ridge_present, np.rad2deg(diff_rad), np.nan)

    _ax_colorbars = {}

    def _cb(ax, im, **kw):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, **kw)
        cb.ax.yaxis.set_tick_params(colors="gray", labelsize=6)
        _ax_colorbars[ax] = cb
        return cb

    # ── Row 1: pre-processing → amplitude gate → orientation ─────────────────
    axes[0, 0].imshow(hs_raw,    origin="upper", extent=ext, aspect="auto")
    style_axes(axes[0, 0], "1. Raw DTM")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    axes[0, 1].imshow(hs_smooth, origin="upper", extent=ext, aspect="auto")
    style_axes(axes[0, 1], "2. Median Smooth")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    _cb(axes[0, 2], axes[0, 2].imshow(lrm_disp, origin="upper", cmap="RdBu_r",
                                       extent=ext, aspect="auto"))
    style_axes(axes[0, 2], "3. Local Relief Model")
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

    ridge_disp = np.where(ridge == nodata, np.nan, ridge)
    _cb(axes[0, 3], axes[0, 3].imshow(ridge_disp, origin="upper", cmap="hot",
                                       extent=ext, aspect="auto"))
    style_axes(axes[0, 3], "4. Frangi Ridge Response")
    axes[0, 3].set_xticks([]); axes[0, 3].set_yticks([])

    ridge_thresh_disp = np.where(nd_mask, np.nan, valid_ridge)
    cb_rt = _cb(axes[0, 4], axes[0, 4].imshow(ridge_thresh_disp, origin="upper", cmap="hot",
                                                extent=ext, aspect="auto"))
    cb_rt.ax.axhline(cfg.RIDGE_THRESHOLD, color="cyan", linewidth=1.5, linestyle="--")
    cb_rt.ax.annotate(f"t={cfg.RIDGE_THRESHOLD}", xy=(1.05, cfg.RIDGE_THRESHOLD),
                      xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[0, 4], f"5. Ridge Amplitude Threshold  (t={cfg.RIDGE_THRESHOLD})")
    axes[0, 4].set_xticks([]); axes[0, 4].set_yticks([])

    cb_ori = _cb(axes[0, 5], axes[0, 5].imshow(orient_disp, origin="upper", cmap="hsv",
                                                 vmin=0, vmax=180, extent=ext, aspect="auto"))
    cb_ori.ax.axhline(cfg.TARGET_ANGLE, color="cyan", linewidth=1.2, linestyle="--")
    cb_ori.ax.annotate(f"{cfg.TARGET_ANGLE:.0f}°", xy=(1.05, cfg.TARGET_ANGLE),
                       xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[0, 5], f"6. Hessian Orientation  (target {cfg.TARGET_ANGLE:.0f}°)")
    axes[0, 5].set_xticks([]); axes[0, 5].set_yticks([])

    cb_diff = _cb(axes[0, 6], axes[0, 6].imshow(diff_disp, origin="upper", cmap="RdYlGn_r",
                                                  vmin=0, vmax=90, extent=ext, aspect="auto"))
    cb_diff.ax.axhline(cfg.ANGLE_TOLERANCE, color="cyan", linewidth=1.2, linestyle="--")
    cb_diff.ax.annotate(f"±{cfg.ANGLE_TOLERANCE:.0f}°", xy=(1.05, cfg.ANGLE_TOLERANCE),
                        xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[0, 6], f"7. Angle Deviation  (tol ±{cfg.ANGLE_TOLERANCE:.0f}°)")
    axes[0, 6].set_xticks([]); axes[0, 6].set_yticks([])

    # ── Row 2: binary mask → gap fill → skeleton → labels → pairs + diag ────────
    _img(axes[1, 0], binary.astype(np.uint8),   "8. Binary Mask",          cmap="gray", extent=ext)
    _img(axes[1, 1], closed.astype(np.uint8),   "9. Along-Wall Gap Fill",  cmap="gray", extent=ext)
    _img(axes[1, 2], skeleton.astype(np.uint8), "10. Skeleton",            cmap="gray", extent=ext)

    labeled_disp = labeled.astype(float)
    labeled_disp[labeled == 0] = np.nan
    _cb(axes[1, 3], axes[1, 3].imshow(labeled_disp, origin="upper", cmap="tab20",
                                       extent=ext, aspect="auto", interpolation="nearest"))
    style_axes(axes[1, 3], f"11. Labeled Components  ({labeled.max()})")
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])

    # Panel 12. — pairs on hillshaded DTM
    axes[1, 4].imshow(hs_raw, origin="upper", extent=ext, aspect="auto")
    true_idx = coord.get("true_positive_pair_index")

    # Pixel → metres conversion at the window centre latitude
    rows_n, cols_n = raw.shape
    centre_lat = (extent[2] + extent[3]) / 2
    m_per_px = ((extent[1] - extent[0]) / cols_n * 111_320.0
                * np.cos(np.radians(centre_lat))
                + (extent[3] - extent[2]) / rows_n * 111_320.0) / 2

    for idx, pair in enumerate(pairs[:12]):
        style = pair_style(idx, true_idx)
        geo_segs = pair_to_geo(pair, extent, raw.shape)
        for k, ((gx0, gy0), (gx1, gy1)) in enumerate(geo_segs):
            axes[1, 4].plot([gx0, gx1], [gy0, gy1],
                            color=style["colors"][k], linewidth=style["linewidth"],
                            alpha=style["alpha"], zorder=style["zorder"])

        # Index + spacing label at the midpoint between the two line midpoints
        (ax0, ay0), (ax1, ay1) = geo_segs[0]
        (bx0, by0), (bx1, by1) = geo_segs[1]
        mid_ax, mid_ay = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        mid_bx, mid_by = (bx0 + bx1) / 2, (by0 + by1) / 2
        label_lon = (mid_ax + mid_bx) / 2
        label_lat = (mid_ay + mid_by) / 2
        spacing_m = pair["spacing"] * m_per_px
        axes[1, 4].annotate(
            f"{style['label_text']}  {spacing_m:.0f} m",
            xy=(label_lon, label_lat),
            fontsize=6.5, color=style["label_color"], fontweight="bold",
            ha="center", va="center", zorder=style["zorder"] + 1,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000",
                      alpha=0.6, edgecolor="none"),
        )

    pairs_title = f"12. Detected Pairs  ({len(pairs)} pair{'s' if len(pairs) != 1 else ''})"
    if true_idx is not None:
        pairs_title += f"  ·  {confirmed_pairs_label(true_idx)} confirmed"
    style_axes(axes[1, 4], pairs_title)
    axes[1, 4].set_xticks([]); axes[1, 4].set_yticks([])

    # Panel 13. — visual diagnostic hillshade next to detected pairs
    axes[1, 5].imshow(hs_perp32, origin="upper", cmap="gray", extent=ext, aspect="auto",
                      vmin=0, vmax=1)
    style_axes(axes[1, 5], "13. Hillshade ⊥ 32°  (az 122°, alt 20°)")
    axes[1, 5].set_xticks([]); axes[1, 5].set_yticks([])

    axes[1, 6].set_visible(False)

    # Uniform dark frame
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#111111")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ── Save each step as an individual image ─────────────────────────────────
    step_dir = output_dir / name / "pipeline_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, slug in zip(axes.flat, _STEP_SLUGS):
        if slug is None:
            continue
        bboxes = [ax.get_tightbbox(renderer)]
        cb = _ax_colorbars.get(ax)
        if cb is not None:
            cb_bbox = cb.ax.get_tightbbox(renderer)
            if cb_bbox is not None:
                bboxes.append(cb_bbox)
        bboxes = [b for b in bboxes if b is not None]
        if bboxes:
            bbox_in = Bbox.union(bboxes).expanded(1.02, 1.04).transformed(
                fig.dpi_scale_trans.inverted()
            )
            fig.savefig(step_dir / f"{slug}.png", dpi=cfg.DPI, bbox_inches=bbox_in)
    print(f"    saved pipeline steps → {step_dir}/")

    save_figure(fig, output_dir / name, "pipeline_steps.png", dpi=cfg.DPI)
