"""
cross_sections.py  —  Perpendicular cross-section elevation profiles.

For each configured coordinate, a set of elevation transects is sampled
perpendicular to the road bearing at regular offsets along the road.
All profiles are overlaid on a single figure, colour-coded by offset, so
the dual-raised-wall signature of a Hopewell earthwork is immediately
visible as two symmetric humps flanking the road centreline.

Two panels per coordinate:
  Left  — raw elevation profiles (absolute elevation, metres)
  Right — de-trended profiles   (linear slope removed → wall relief only)

Call generate_cross_section_figure(coord, dtm_path, output_dir) from run_report.py.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_REPORT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _REPORT_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPORT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from process_dtm import getGTAndBand, get_window, median_smooth_windowed
from report_utils import sample_profile, save_figure, style_axes
import config as cfg


def generate_cross_section_figure(coord: dict, dtm_path: str, output_dir: Path) -> None:
    """
    Build and save the cross-section figure for one coordinate.

    Parameters
    ----------
    coord      : dict with keys 'name', 'lat', 'lon', 'size_m'
    dtm_path   : path to merged DTM GeoTIFF
    output_dir : parent output directory
    """
    name = coord["name"]
    lat, lon = coord["lat"], coord["lon"]
    size_m = coord.get("size_m", 200)
    bearing = cfg.ROAD_BEARING_DEG

    print(f"  [cross_sections] loading DTM window …")
    gt, band, nodata, _ds = getGTAndBand(dtm_path)
    raw, extent = get_window(lat, lon, size_m, size_m, gt, band, nodata)
    if raw is None or raw.size == 0:
        print(f"  [cross_sections] no data — skipped.")
        return

    smoothed = median_smooth_windowed(raw, nodata=nodata, radius=cfg.MEDIAN_SMOOTH_RADIUS)

    # Road direction unit vector (east, north) from bearing (CW from north)
    road_east = np.sin(np.radians(bearing))
    road_north = np.cos(np.radians(bearing))

    offsets = cfg.CROSS_SECTION_OFFSETS_M
    half_w = cfg.CROSS_SECTION_HALF_WIDTH_M
    n_samp = cfg.CROSS_SECTION_N_SAMPLES

    n_off = len(offsets)
    profile_cmap = cm.get_cmap("plasma", n_off)

    print(f"  [cross_sections] sampling {n_off} transects …")
    dists_all = np.linspace(-half_w, half_w, n_samp)  # shared x-axis
    profiles_raw: list = []
    profiles_dt: list = []
    valid_offsets: list = []

    for i, off_m in enumerate(offsets):
        lat_c = lat + off_m * road_north / 111320.0
        lon_c = lon + off_m * road_east / (111320.0 * np.cos(np.radians(lat)))

        dists, elev = sample_profile(
            smoothed, extent, lat_c, lon_c,
            bearing, half_w, n_samp, nodata=nodata,
        )

        valid = ~np.isnan(elev)
        if valid.sum() < 4:
            continue

        # De-trend: remove linear regional slope
        coeffs = np.polyfit(dists[valid], elev[valid], 1)
        detrended = elev - np.polyval(coeffs, dists)

        profiles_raw.append(elev)
        profiles_dt.append(detrended)
        valid_offsets.append((i, off_m))

    if not profiles_raw:
        print(f"  [cross_sections] no valid profiles — skipped.")
        return

    n_valid = len(profiles_raw)
    alpha = max(0.30, 1.0 - 0.04 * n_valid)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_raw, ax_dt) = plt.subplots(
        1, 2, figsize=(16, 6), facecolor="#1a1a1a",
        gridspec_kw={"wspace": 0.12},
    )
    fig.suptitle(
        f"Cross-Section Profiles  ·  {name.replace('_', ' ')}  "
        f"({lat:.4f}°N, {lon:.4f}°E)  ·  road bearing {bearing:.0f}°",
        fontsize=12, color="white",
    )

    for (i_off, off_m), elev, detrended in zip(valid_offsets, profiles_raw, profiles_dt):
        color = profile_cmap(i_off / max(1, n_off - 1))
        lbl = f"{off_m:+d} m"
        ax_raw.plot(dists_all, elev,     color=color, alpha=alpha, linewidth=0.9, label=lbl)
        ax_dt.plot(dists_all, detrended, color=color, alpha=alpha, linewidth=0.9)

    # Mean envelope
    stack_raw = np.vstack(profiles_raw)
    stack_dt = np.vstack(profiles_dt)
    mean_raw = np.nanmean(stack_raw, axis=0)
    mean_dt = np.nanmean(stack_dt, axis=0)
    std_dt = np.nanstd(stack_dt, axis=0)

    ax_raw.plot(dists_all, mean_raw, color="white", linewidth=2.0, zorder=5, label="Mean")
    ax_dt.plot(dists_all, mean_dt, color="white", linewidth=2.0, zorder=5, label="Mean")
    ax_dt.fill_between(
        dists_all,
        mean_dt - std_dt, mean_dt + std_dt,
        alpha=0.15, color="white", label="±1 σ",
    )

    # Reference lines
    for ax in (ax_raw, ax_dt):
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_dt.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)

    # Axis labels, estimated wall positions at ±half spacing
    half_sp = cfg.SPACING_METERS / 2
    for ax in (ax_raw, ax_dt):
        ax.axvline(-half_sp, color="#ff8844", linewidth=0.7, linestyle=":",
                   alpha=0.6, label=f"Expected wall (±{half_sp:.0f} m)")
        ax.axvline(+half_sp, color="#ff8844", linewidth=0.7, linestyle=":", alpha=0.6)

    style_axes(ax_raw, "Raw Elevation",
               xlabel="Distance from road centreline (m)",
               ylabel="Elevation (m)")
    style_axes(ax_dt, "De-trended  (regional slope removed)",
               xlabel="Distance from road centreline (m)",
               ylabel="Relative elevation (m)")

    for ax in (ax_raw, ax_dt):
        ax.grid(True, color="#252525", linewidth=0.5)

    # Legend — show up to 8 offset labels + mean
    handles, labels = ax_raw.get_legend_handles_labels()
    # Keep mean + a subset of offsets
    max_show = 8
    skip = max(1, n_valid // max_show)
    kept = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
            if i % skip == 0 or l == "Mean"]
    if kept:
        hs, ls = zip(*kept)
        ax_raw.legend(hs, ls, fontsize=7, ncol=2, facecolor="#1e1e1e",
                       edgecolor="#444", labelcolor="lightgray", loc="best")

    handles2, labels2 = ax_dt.get_legend_handles_labels()
    ax_dt.legend(handles2[-3:], labels2[-3:], fontsize=7,
                  facecolor="#1e1e1e", edgecolor="#444", labelcolor="lightgray")

    # Shared colour bar for offset
    norm = plt.Normalize(vmin=offsets[0], vmax=offsets[-1])
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_raw, ax_dt], orientation="vertical",
                         pad=0.01, fraction=0.012, shrink=0.85)
    cbar.set_label("Along-road offset (m)", color="lightgray", fontsize=8)
    cbar.ax.yaxis.set_tick_params(colors="gray")

    save_figure(fig, output_dir / name, "cross_sections.png", dpi=cfg.DPI)
