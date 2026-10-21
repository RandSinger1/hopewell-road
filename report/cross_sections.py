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

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .base_compute import get_terrain
from .report_utils import (sample_profile, save_figure, style_axes,
                            confirmed_pair_indices, site_target_angle)
from . import config as cfg


def generate_cross_section_figure(coord: dict, dtm_path: str, output_dir: Path,
                                   use_cache: bool = True) -> None:
    """
    Build and save the cross-section figure for one coordinate.

    Only runs once at least one wall pair has been confirmed via
    "true_positive_pair_index" (see REPORT_COORDS in parameters.py) — until
    then there's no confirmed earthwork to sanity-check against the assumed
    road bearing, so the figure is skipped.

    Parameters
    ----------
    coord      : dict with keys 'name', 'lat', 'lon', 'size_m'
    dtm_path   : path to merged DTM GeoTIFF
    output_dir : parent output directory
    use_cache  : reuse cached DTM window / smoothing / LRM if available
    """
    name = coord["name"]
    if not confirmed_pair_indices(coord.get("true_positive_pair_index")):
        print(f"  [cross_sections] no confirmed true-positive pair — skipped.")
        return
    lat, lon = coord["lat"], coord["lon"]
    mode, bearing = site_target_angle(coord)
    print(f"  [cross_sections] target bearing {bearing:.1f}°  (mode={mode})")

    print(f"  [cross_sections] loading DTM window …")
    terrain = get_terrain(coord, dtm_path, output_dir, use_cache=use_cache)
    if terrain is None:
        print(f"  [cross_sections] no data — skipped.")
        return
    smoothed, extent, nodata, lrm = (
        terrain["smoothed"], terrain["extent"], terrain["nodata"], terrain["lrm"],
    )

    # Road direction unit vector (east, north) from bearing (CW from north)
    road_east = np.sin(np.radians(bearing))
    road_north = np.cos(np.radians(bearing))

    offsets = cfg.CROSS_SECTION_OFFSETS_M
    half_w = cfg.CROSS_SECTION_HALF_WIDTH_M
    n_samp = cfg.CROSS_SECTION_N_SAMPLES

    # Each plotted profile is the mean of all transects within the strip
    # between two consecutive offset positions, giving a more complete view
    # of the detected walls than a single sample line.
    n_strips = max(1, len(offsets) - 1)
    profile_cmap = cm.get_cmap("plasma", n_strips)

    print(f"  [cross_sections] sampling {n_strips} strip means ({len(offsets)} offsets) …")
    dists_all = np.linspace(-half_w, half_w, n_samp)  # shared x-axis
    profiles_raw: list = []
    profiles_dt: list = []
    valid_offsets: list = []

    cos_lat = np.cos(np.radians(lat))
    for i in range(len(offsets) - 1):
        off_start = offsets[i]
        off_end   = offsets[i + 1]
        off_mid   = (off_start + off_end) / 2.0

        # Sub-sample ~1 transect per metre within the strip
        n_sub = max(3, abs(int(off_end - off_start)))
        sub_offs = np.linspace(off_start, off_end, n_sub)

        strip_elev = []
        strip_lrm  = []
        for off in sub_offs:
            lat_c = lat + off * road_north / 111320.0
            lon_c = lon + off * road_east / (111320.0 * cos_lat)
            _, elev = sample_profile(
                smoothed, extent, lat_c, lon_c,
                bearing, half_w, n_samp, nodata=nodata,
            )
            _, elev_lrm = sample_profile(
                lrm, extent, lat_c, lon_c,
                bearing, half_w, n_samp, nodata=nodata,
            )
            strip_elev.append(elev)
            strip_lrm.append(elev_lrm)

        mean_elev = np.nanmean(np.vstack(strip_elev), axis=0)
        mean_lrm  = np.nanmean(np.vstack(strip_lrm),  axis=0)

        valid = ~np.isnan(mean_elev)
        if valid.sum() < 4:
            continue

        profiles_raw.append(mean_elev)
        profiles_dt.append(mean_lrm)
        valid_offsets.append((i, off_mid))

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
        f"Cross-Section Strip Means  ·  {name.replace('_', ' ')}  "
        f"({lat:.4f}°N, {lon:.4f}°E)  ·  road bearing {bearing:.0f}°",
        fontsize=12, color="white",
    )

    for (i_off, off_m), elev, detrended in zip(valid_offsets, profiles_raw, profiles_dt):
        color = profile_cmap(i_off / max(1, n_strips - 1))
        lbl = f"{off_m:+.0f} m"
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
    style_axes(ax_dt, "LRM  (local relief model)",
               xlabel="Distance from road centreline (m)",
               ylabel="LRM elevation (m)")

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
    cbar.set_label("Along-road strip midpoint (m)", color="lightgray", fontsize=8)
    cbar.ax.yaxis.set_tick_params(colors="gray")

    save_figure(fig, output_dir / name, "cross_sections.png", dpi=cfg.DPI)
