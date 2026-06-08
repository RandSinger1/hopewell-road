"""
pipeline_steps.py  —  Step-by-step processing pipeline figure.

Produces a 2-row × 5-column figure showing every stage from raw DTM to
detected parallel wall pairs.

Row 1 — Pre-processing
  [1] Raw DTM (hillshaded)
  [2] Median-smoothed DTM
  [3] Local Relief Model (LRM)
  [4] Multi-scale Frangi ridge response
  [5] Hessian Orientation
  [6] Hillshade perpendicular to 32°

Row 2 — Detection
  [7] Angle deviation map
  [8] Ridge threshold
  [9] Labelled skeleton components
  [10] Hough reconstruction
  [11] Skeleton
  [12] Detected parallel wall pairs

Call generate_pipeline_figure(coord, dtm_path, output_dir) from run_report.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from road_model.process_dtm import (
    getGTAndBand, get_window,
    median_smooth_windowed, local_relief_model,
    multiscale_ridge_enhancement,
)
from road_model.ridge_detection import detect_parallel_walls_ridge_pairing, remove_invalid_pairs, hessian_ridge_orientation
from .report_utils import render_hillshaded_dtm, hillshade, save_figure, style_axes, pair_to_geo
from . import config as cfg
import parameters


_STEP_LABELS = [
    "① Raw DTM",
    "② Median Smooth",
    "③ Local Relief Model",
    "④ Frangi Ridge Response",
    "⑤ Hessian Orientation",
    "⑥ Hillshade ⊥ 32°",
    "⑦ Angle Deviation",
    "⑧ Ridge Response",
    "⑨ Labeled Components",
    "⑩ Hough Reconstruction",
    "⑪ Skeleton",
    "⑫ Detected Pairs",
]

_STEP_SLUGS = [
    "01_raw_dtm",
    "02_median_smooth",
    "03_local_relief_model",
    "04_frangi_ridge_response",
    "05_hessian_orientation",
    "06_hillshade_perp32",
    "07_angle_deviation",
    "08_ridge_response",
    "09_labeled_components",
    "10_hough_reconstruction",
    "11_skeleton",
    "12_detected_pairs",
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

    # ── Build figure  (2 rows × 6 columns) ───────────────────────────────────
    print(f"  [pipeline] composing figure …")
    fig, axes = plt.subplots(2, 6, figsize=(26, 9))
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

    def _cb(ax, im, **kw):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, **kw)
        cb.ax.yaxis.set_tick_params(colors="gray", labelsize=6)
        return cb

    # ── Row 1: pre-processing + Hessian orientation ───────────────────────────
    axes[0, 0].imshow(hs_raw,    origin="upper", extent=ext, aspect="auto")
    style_axes(axes[0, 0], "① Raw DTM")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    axes[0, 1].imshow(hs_smooth, origin="upper", extent=ext, aspect="auto")
    style_axes(axes[0, 1], "② Median Smooth")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    _cb(axes[0, 2], axes[0, 2].imshow(lrm_disp, origin="upper", cmap="RdBu_r",
                                       extent=ext, aspect="auto"))
    style_axes(axes[0, 2], "③ Local Relief Model")
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

    ridge_disp = np.where(ridge == nodata, np.nan, ridge)
    _cb(axes[0, 3], axes[0, 3].imshow(ridge_disp, origin="upper", cmap="hot",
                                       extent=ext, aspect="auto"))
    style_axes(axes[0, 3], "④ Frangi Ridge Response")
    axes[0, 3].set_xticks([]); axes[0, 3].set_yticks([])

    cb_ori = _cb(axes[0, 4], axes[0, 4].imshow(orient_disp, origin="upper", cmap="hsv",
                                                 vmin=0, vmax=180, extent=ext, aspect="auto"))
    cb_ori.ax.axhline(cfg.TARGET_ANGLE, color="cyan", linewidth=1.2, linestyle="--")
    cb_ori.ax.annotate(f"{cfg.TARGET_ANGLE:.0f}°", xy=(1.05, cfg.TARGET_ANGLE),
                       xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[0, 4], f"⑤ Hessian Orientation  (target {cfg.TARGET_ANGLE:.0f}°)")
    axes[0, 4].set_xticks([]); axes[0, 4].set_yticks([])

    axes[0, 5].imshow(hs_perp32, origin="upper", cmap="gray", extent=ext, aspect="auto",
                      vmin=0, vmax=1)
    style_axes(axes[0, 5], "⑥ Hillshade ⊥ 32°  (az 122°, alt 20°)")
    axes[0, 5].set_xticks([]); axes[0, 5].set_yticks([])

    # ── Row 2: detection pipeline ─────────────────────────────────────────────
    cb_diff = _cb(axes[1, 0], axes[1, 0].imshow(diff_disp, origin="upper", cmap="RdYlGn_r",
                                                  vmin=0, vmax=90, extent=ext, aspect="auto"))
    cb_diff.ax.axhline(cfg.ANGLE_TOLERANCE, color="cyan", linewidth=1.2, linestyle="--")
    cb_diff.ax.annotate(f"±{cfg.ANGLE_TOLERANCE:.0f}°", xy=(1.05, cfg.ANGLE_TOLERANCE),
                        xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[1, 0], f"⑦ Angle Deviation  (tol ±{cfg.ANGLE_TOLERANCE:.0f}°)")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    ridge_thresh_disp = np.where(nd_mask, np.nan, valid_ridge)
    cb_rt = _cb(axes[1, 1], axes[1, 1].imshow(ridge_thresh_disp, origin="upper", cmap="hot",
                                                extent=ext, aspect="auto"))
    cb_rt.ax.axhline(cfg.RIDGE_THRESHOLD, color="cyan", linewidth=1.5, linestyle="--")
    cb_rt.ax.annotate(f"t={cfg.RIDGE_THRESHOLD}", xy=(1.05, cfg.RIDGE_THRESHOLD),
                      xycoords=("axes fraction", "data"), fontsize=5, color="cyan", va="center")
    style_axes(axes[1, 1], f"⑧ Ridge Response  (t={cfg.RIDGE_THRESHOLD})")
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    labeled_disp = labeled.astype(float)
    labeled_disp[labeled == 0] = np.nan
    _cb(axes[1, 2], axes[1, 2].imshow(labeled_disp, origin="upper", cmap="tab20",
                                       extent=ext, aspect="auto", interpolation="nearest"))
    style_axes(axes[1, 2], f"⑨ Labeled Components  ({labeled.max()})")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])

    _img(axes[1, 3], closed.astype(np.uint8),   "⑩ Hough Reconstruction",  cmap="gray", extent=ext)
    _img(axes[1, 4], skeleton.astype(np.uint8), "⑪ Skeleton",     cmap="gray", extent=ext)

    # Panel ⑫ — pairs on hillshaded DTM
    axes[1, 5].imshow(hs_raw, origin="upper", extent=ext, aspect="auto")
    COLORS = ["#ff4444", "#4499ff"]

    # Pixel → metres conversion at the window centre latitude
    rows_n, cols_n = raw.shape
    centre_lat = (extent[2] + extent[3]) / 2
    m_per_px = ((extent[1] - extent[0]) / cols_n * 111_320.0
                * np.cos(np.radians(centre_lat))
                + (extent[3] - extent[2]) / rows_n * 111_320.0) / 2

    for pair in pairs[:12]:
        geo_segs = pair_to_geo(pair, extent, raw.shape)
        for k, ((gx0, gy0), (gx1, gy1)) in enumerate(geo_segs):
            axes[1, 5].plot([gx0, gx1], [gy0, gy1],
                            color=COLORS[k], linewidth=2.0, alpha=0.85)

        # Spacing label at the midpoint between the two line midpoints
        (ax0, ay0), (ax1, ay1) = geo_segs[0]
        (bx0, by0), (bx1, by1) = geo_segs[1]
        mid_ax, mid_ay = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        mid_bx, mid_by = (bx0 + bx1) / 2, (by0 + by1) / 2
        label_lon = (mid_ax + mid_bx) / 2
        label_lat = (mid_ay + mid_by) / 2
        spacing_m = pair["spacing"] * m_per_px
        axes[1, 5].annotate(
            f"{spacing_m:.0f} m",
            xy=(label_lon, label_lat),
            fontsize=6.5, color="white", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000",
                      alpha=0.6, edgecolor="none"),
        )

    style_axes(axes[1, 5],
               f"⑫ Detected Pairs  ({len(pairs)} pair{'s' if len(pairs) != 1 else ''})")
    axes[1, 5].set_xticks([]); axes[1, 5].set_yticks([])

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
    for i, ax in enumerate(axes.flat):
        bbox_disp = ax.get_tightbbox(renderer)
        if bbox_disp is not None:
            bbox_in = bbox_disp.expanded(1.02, 1.04).transformed(
                fig.dpi_scale_trans.inverted()
            )
            fig.savefig(step_dir / f"{_STEP_SLUGS[i]}.png", dpi=cfg.DPI, bbox_inches=bbox_in)
    print(f"    saved pipeline steps → {step_dir}/")

    save_figure(fig, output_dir / name, "pipeline_steps.png", dpi=cfg.DPI)
