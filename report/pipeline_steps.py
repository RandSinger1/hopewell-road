"""
pipeline_steps.py  —  Step-by-step processing pipeline figure.

Produces a 4-row × 3-column figure showing every stage from raw DTM to
detected parallel wall pairs.

Row 0: [1] Raw DTM       [2] Median smooth   [3] LRM
Row 1: [4] Frangi        [5] Ridge amplitude [6] Hessian orient.
Row 2: [7] Angle dev.    [8] Binary mask     [9] Gap fill
Row 3: [10] Skeleton     [11] Components     [12] Detected pairs

Panel 13 (hillshade ⊥ target bearing) is saved separately to
pipeline_steps/13_hillshade_perp32.png but excluded from the main grid.

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


# Slugs follow flat axis order (left-to-right, top-to-bottom in a 4×3 grid).
# Panel 13 (hillshade_perp32) is saved separately, not in the grid.
_STEP_SLUGS = [
    "01_raw_dtm",               # [0,0]
    "02_median_smooth",         # [0,1]
    "03_local_relief_model",    # [0,2]
    "04_frangi_ridge_response", # [1,0]
    "05_ridge_response",        # [1,1] — amplitude threshold
    "06_hessian_orientation",   # [1,2]
    "07_angle_deviation",       # [2,0]
    "08_binary_mask",           # [2,1]
    "09_gap_fill",              # [2,2]
    "10_skeleton",              # [3,0]
    "11_labeled_components",    # [3,1]
    "12_detected_pairs",        # [3,2]
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
    target_angle, mode = det["target_angle"], det["mode"]
    nd_mask = raw == nodata

    # ── Build figure  (4 rows × 3 columns) ───────────────────────────────────
    print(f"  [pipeline] composing figure …")
    fig, axes = plt.subplots(4, 3, figsize=(20, 22))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"LiDAR Processing Pipeline  ·  {name.replace('_', ' ')}  "
        f"({lat:.4f}°N, {lon:.4f}°E)  ·  target {target_angle:.1f}° ({mode})",
        fontsize=13, color="white", y=0.99,
    )

    hs_raw = render_hillshaded_dtm(raw, nodata=nodata, azimuth=315)
    hs_smooth = render_hillshaded_dtm(smoothed, nodata=nodata, azimuth=315)
    lrm_disp = np.where(lrm == nodata, np.nan, lrm)
    # Intense hillshade perpendicular to this site's target bearing, low sun
    # angle for maximum contrast. The bearing itself is mode-dependent (see
    # base_compute.get_detection / utils.corridor_mode_and_angle) — circleville,
    # chillicothe, and high_banks each imply a different expected wall angle.
    perp_azimuth = (target_angle + 90.0) % 360.0
    hs_perp32 = hillshade(smoothed, nodata=nodata, azimuth=perp_azimuth, altitude=20.0, z_factor=4.0)

    # Hessian orientation & angle-deviation maps (computed once, used in row 1 & 2)
    orient_rad = hessian_ridge_orientation(valid_ridge, sigma=parameters.HESSIAN_SIGMA)
    orient_deg = np.rad2deg(orient_rad)                        # [0, 180)
    ridge_present = valid_ridge > cfg.RIDGE_THRESHOLD * 0.05
    orient_disp = np.where(ridge_present, orient_deg, np.nan)

    target_rad = np.deg2rad(target_angle)
    diff_rad = np.abs(orient_rad - target_rad)
    diff_rad = np.minimum(diff_rad, np.pi - diff_rad)
    diff_disp = np.where(ridge_present, np.rad2deg(diff_rad), np.nan)

    _ax_colorbars = {}

    def _cb(ax, im, **kw):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, **kw)
        cb.ax.yaxis.set_tick_params(colors="gray", labelsize=6)
        _ax_colorbars[ax] = cb
        return cb

    # ── Rows 0-3, cols 0-2: panels in reading order (left→right, top→bottom) ─────
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
    _cb(axes[1, 0], axes[1, 0].imshow(ridge_disp, origin="upper", cmap="hot",
                                       extent=ext, aspect="auto"))
    style_axes(axes[1, 0], "4. Frangi Ridge Response")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    ridge_thresh_disp = np.where(nd_mask, np.nan, valid_ridge)
    cb_rt = _cb(axes[1, 1], axes[1, 1].imshow(ridge_thresh_disp, origin="upper", cmap="hot",
                                                extent=ext, aspect="auto"))
    cb_rt.ax.axhline(cfg.RIDGE_THRESHOLD, color="cyan", linewidth=1.5, linestyle="--")
    cb_rt.ax.annotate(f"t={cfg.RIDGE_THRESHOLD}", xy=(1.05, cfg.RIDGE_THRESHOLD),
                      xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[1, 1], f"5. Ridge Amplitude Threshold  (t={cfg.RIDGE_THRESHOLD})")
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    cb_ori = _cb(axes[1, 2], axes[1, 2].imshow(orient_disp, origin="upper", cmap="hsv",
                                                 vmin=0, vmax=180, extent=ext, aspect="auto"))
    cb_ori.ax.axhline(target_angle, color="cyan", linewidth=1.2, linestyle="--")
    cb_ori.ax.annotate(f"{target_angle:.0f}°", xy=(1.05, target_angle),
                       xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[1, 2], f"6. Hessian Orientation  (target {target_angle:.0f}°, {mode})")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])

    cb_diff = _cb(axes[2, 0], axes[2, 0].imshow(diff_disp, origin="upper", cmap="RdYlGn_r",
                                                  vmin=0, vmax=90, extent=ext, aspect="auto"))
    cb_diff.ax.axhline(cfg.ANGLE_TOLERANCE, color="cyan", linewidth=1.2, linestyle="--")
    cb_diff.ax.annotate(f"±{cfg.ANGLE_TOLERANCE:.0f}°", xy=(1.05, cfg.ANGLE_TOLERANCE),
                        xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[2, 0], f"7. Angle Deviation  (tol ±{cfg.ANGLE_TOLERANCE:.0f}°)")
    axes[2, 0].set_xticks([]); axes[2, 0].set_yticks([])

    _img(axes[2, 1], binary.astype(np.uint8),   "8. Binary Mask",         cmap="gray", extent=ext)
    _img(axes[2, 2], closed.astype(np.uint8),   "9. Along-Wall Gap Fill", cmap="gray", extent=ext)
    _img(axes[3, 0], skeleton.astype(np.uint8), "10. Skeleton",           cmap="gray", extent=ext)

    labeled_disp = labeled.astype(float)
    labeled_disp[labeled == 0] = np.nan
    _cb(axes[3, 1], axes[3, 1].imshow(labeled_disp, origin="upper", cmap="tab20",
                                       extent=ext, aspect="auto", interpolation="nearest"))
    style_axes(axes[3, 1], f"11. Labeled Components  ({labeled.max()})")
    axes[3, 1].set_xticks([]); axes[3, 1].set_yticks([])

    # Panel 12. — pairs on hillshaded DTM
    axes[3, 2].imshow(hs_raw, origin="upper", extent=ext, aspect="auto")
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
            axes[3, 2].plot([gx0, gx1], [gy0, gy1],
                            color=style["colors"][k], linewidth=style["linewidth"],
                            alpha=style["alpha"], zorder=style["zorder"])

        # Index + spacing + azimuth label at midpoint between the two line midpoints
        (ax0, ay0), (ax1, ay1) = geo_segs[0]
        (bx0, by0), (bx1, by1) = geo_segs[1]
        mid_ax, mid_ay = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        mid_bx, mid_by = (bx0 + bx1) / 2, (by0 + by1) / 2
        label_lon = (mid_ax + mid_bx) / 2
        label_lat = (mid_ay + mid_by) / 2
        spacing_m = pair["spacing"] * m_per_px
        cos_lat_seg = np.cos(np.radians((ay0 + ay1) / 2))
        pair_azimuth = np.degrees(np.arctan2(
            (ax1 - ax0) * cos_lat_seg * 111_320.0,
            (ay1 - ay0) * 111_320.0,
        )) % 180
        axes[3, 2].annotate(
            f"{style['label_text']}  {spacing_m:.0f} m  {pair_azimuth:.0f}°",
            xy=(label_lon, label_lat),
            fontsize=6.5, color=style["label_color"], fontweight="bold",
            ha="center", va="center", zorder=style["zorder"] + 1,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000",
                      alpha=0.6, edgecolor="none"),
        )

    pairs_title = f"12. Detected Pairs  ({len(pairs)} pair{'s' if len(pairs) != 1 else ''})"
    if true_idx is not None:
        pairs_title += f"  ·  {confirmed_pairs_label(true_idx)} confirmed"
    style_axes(axes[3, 2], pairs_title)
    axes[3, 2].set_xticks([]); axes[3, 2].set_yticks([])

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

    # ── Panel 13 — saved standalone (excluded from the main grid) ────────────
    fig13, ax13 = plt.subplots(1, 1, figsize=(7, 7), facecolor="#1a1a1a")
    ax13.set_facecolor("#111111")
    ax13.imshow(hs_perp32, origin="upper", cmap="gray", extent=ext, aspect="auto",
                vmin=0, vmax=1)
    style_axes(ax13, f"13. Hillshade ⊥ {target_angle:.0f}°  (az {perp_azimuth:.0f}°, alt 20°)")
    ax13.set_xticks([]); ax13.set_yticks([])
    for spine in ax13.spines.values():
        spine.set_edgecolor("#333333")
    plt.tight_layout()
    fig13.savefig(step_dir / "13_hillshade_perp32.png", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig13)
    print(f"    saved panel 13 → {step_dir}/13_hillshade_perp32.png")
