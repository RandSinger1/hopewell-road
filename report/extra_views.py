"""
extra_views.py  —  Additional diagnostic visualisations.

Generates five supplementary figures per coordinate:

  1. multi_azimuth_hillshade.png
       2×2 grid of hillshades at 315°, 45°, 135°, 225° — reveals features
       that are only visible under specific illumination directions.

  2. 3d_perspective.png
       Interactive-style 3D surface of the DTM with detected wall pairs
       projected at the surface elevation as coloured lines.

  3. ridge_scale_comparison.png
       Frangi vesselness response at six individual sigma values (1–12 px),
       so you can see which scale best highlights the earthwork walls.

  4. elevation_statistics.png
       Three-panel: elevation histogram, slope/gradient-magnitude map,
       and an aspect-direction rose plot.

  5. road_elevation_profile.png
       Two-panel: elevation along the road centreline (with peak markers)
       and the first-derivative slope, highlighting embankment transitions.

Call generate_extra_views(coord, dtm_path, output_dir) from run_report.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3D projection)
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from road_model.process_dtm import (
    getGTAndBand, get_window,
    median_smooth_windowed, local_relief_model, multiscale_ridge_enhancement,
)
from road_model.ridge_detection import detect_parallel_walls_ridge_pairing, remove_invalid_pairs
from .report_utils import hillshade, render_hillshaded_dtm, save_figure, style_axes, pair_to_geo, sample_profile
from . import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Multi-azimuth hillshade
# ─────────────────────────────────────────────────────────────────────────────

def _multi_azimuth_hillshade(raw: np.ndarray, extent: list, nodata: float,
                               name: str, output_dir: Path) -> None:
    azimuths = [315, 45, 135, 225]
    labels = ["315° NW (standard)", "45° NE", "135° SE", "225° SW"]
    ext = [extent[0], extent[1], extent[2], extent[3]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#1a1a1a")
    fig.suptitle(
        f"Multi-Azimuth Hillshade  ·  {name.replace('_', ' ')}",
        fontsize=13, color="white",
    )

    for ax, az, lbl in zip(axes.flat, azimuths, labels):
        hs = hillshade(raw, nodata=nodata, azimuth=az, altitude=45, z_factor=3.0)
        ax.imshow(hs, origin="upper", cmap="gray", extent=ext,
                  aspect="auto", vmin=0, vmax=1)
        style_axes(ax, lbl)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_dir / name, "multi_azimuth_hillshade.png", dpi=cfg.DPI)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  3D perspective
# ─────────────────────────────────────────────────────────────────────────────

def _3d_perspective(raw: np.ndarray, extent: list, nodata: float,
                     name: str, output_dir: Path, pairs: list) -> None:
    rows, cols = raw.shape
    # Downsample to ≤200×200 for render performance
    step = max(1, max(rows, cols) // 200)
    z = raw[::step, ::step].astype(float)
    z[z == nodata] = np.nan

    x_lin = np.linspace(extent[0], extent[1], z.shape[1])
    y_lin = np.linspace(extent[2], extent[3], z.shape[0])
    X, Y = np.meshgrid(x_lin, y_lin)

    valid_z = z[~np.isnan(z)]
    vmin = np.percentile(valid_z, 2) if valid_z.size else 0.0
    vmax = np.percentile(valid_z, 98) if valid_z.size else 1.0

    fig = plt.figure(figsize=(14, 9), facecolor="#1a1a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#111111")

    surf = ax.plot_surface(
        X, Y, z,
        rstride=1, cstride=1,
        cmap="terrain",
        vmin=vmin, vmax=vmax,
        linewidth=0,
        antialiased=True,
        alpha=0.88,
    )

    # Project detected wall pairs just above the surface
    z_top = vmax + (vmax - vmin) * 0.02
    COLORS_3D = ["#ff4444", "#4499ff"]
    for pair in pairs[:10]:
        for k, geo_seg in enumerate(pair_to_geo(pair, extent, raw.shape)):
            (gx0, gy0), (gx1, gy1) = geo_seg
            ax.plot(
                [gx0, gx1], [gy0, gy1], [z_top, z_top],
                color=COLORS_3D[k], linewidth=2.5, alpha=0.9, zorder=10,
            )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=14, pad=0.08)
    cbar.set_label("Elevation (m)", color="lightgray", fontsize=9)
    cbar.ax.yaxis.set_tick_params(colors="gray", labelsize=7)

    ax.set_xlabel("Longitude", color="lightgray", fontsize=8, labelpad=6)
    ax.set_ylabel("Latitude",  color="lightgray", fontsize=8, labelpad=6)
    ax.set_zlabel("Elev (m)",  color="lightgray", fontsize=8, labelpad=6)
    ax.tick_params(colors="gray", labelsize=6)
    ax.set_title(
        f"3D Terrain  ·  {name.replace('_', ' ')}  "
        f"({'%d' % len(pairs)} wall pair{'s' if len(pairs) != 1 else ''} overlaid)",
        color="white", fontsize=11, pad=10,
    )
    ax.view_init(elev=35, azim=225)

    plt.tight_layout()
    save_figure(fig, output_dir / name, "3d_perspective.png", dpi=cfg.DPI)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ridge-scale comparison
# ─────────────────────────────────────────────────────────────────────────────

def _ridge_scale_comparison(smoothed: np.ndarray, extent: list, nodata: float,
                              name: str, output_dir: Path) -> None:
    SCALES = [1, 2, 4, 6, 8, 12]
    ext = [extent[0], extent[1], extent[2], extent[3]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#1a1a1a")
    fig.suptitle(
        f"Frangi Ridge Response by Scale  ·  {name.replace('_', ' ')}",
        fontsize=13, color="white",
    )

    for ax, s in zip(axes.flat, SCALES):
        ridge = multiscale_ridge_enhancement(smoothed, scales=[s], nodata=nodata)
        disp = np.where(ridge == nodata, np.nan, ridge)
        im = ax.imshow(disp, origin="upper", cmap="hot", extent=ext, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.yaxis.set_tick_params(
            colors="gray", labelsize=6)
        style_axes(ax, f"σ = {s} px  (~{s} m feature width)")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_dir / name, "ridge_scale_comparison.png", dpi=cfg.DPI)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Elevation statistics
# ─────────────────────────────────────────────────────────────────────────────

def _elevation_statistics(raw: np.ndarray, nodata: float,
                            name: str, output_dir: Path) -> None:
    valid = raw[raw != nodata].astype(float)
    if valid.size == 0:
        return

    raw_f = raw.astype(float)
    raw_f[raw == nodata] = np.nan
    gy, gx = np.gradient(raw_f)   # NaN propagates to cells adjacent to nodata
    grad_mag = np.hypot(gx, gy)
    aspect_rad = np.arctan2(-gx, gy)  # NaN where gradient is NaN

    fig = plt.figure(figsize=(18, 6), facecolor="#1a1a1a")
    fig.suptitle(
        f"Elevation Statistics  ·  {name.replace('_', ' ')}",
        fontsize=13, color="white",
    )
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # Panel 1 — Histogram
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(valid, bins=80, color="#4488cc", alpha=0.85, edgecolor="none")
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    ax1.axvline(p2,  color="#ff8844", linewidth=1.2, linestyle="--", label=f"2nd pct  {p2:.1f} m")
    ax1.axvline(p98, color="#ffcc44", linewidth=1.2, linestyle="--", label=f"98th pct {p98:.1f} m")
    style_axes(ax1, "Elevation Histogram",
               xlabel="Elevation (m)", ylabel="Pixel count")
    ax1.grid(True, color="#252525", linewidth=0.4)
    ax1.legend(fontsize=8, facecolor="#1e1e1e", edgecolor="#444", labelcolor="lightgray")
    ax1.tick_params(colors="gray")
    ax1.xaxis.label.set_color("lightgray")
    ax1.yaxis.label.set_color("lightgray")

    # Panel 2 — Gradient magnitude map
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(grad_mag, origin="upper", cmap="magma", aspect="auto")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
        colors="gray", labelsize=7)
    style_axes(ax2, "Slope (gradient magnitude, m/m)")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3 — Aspect rose (polar histogram)
    ax3 = fig.add_subplot(gs[2], projection="polar", facecolor="#111111")
    valid_asp = aspect_rad[np.isfinite(aspect_rad)].ravel()
    bins = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(valid_asp, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    ax3.bar(centers, counts, width=width, alpha=0.78, color="#4488cc", edgecolor="none")
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)
    ax3.set_title("Terrain Aspect Rose", color="white", fontsize=9, pad=14)
    ax3.tick_params(colors="gray", labelsize=7)

    save_figure(fig, output_dir / name, "elevation_statistics.png", dpi=cfg.DPI)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Along-road elevation profile
# ─────────────────────────────────────────────────────────────────────────────

def _road_elevation_profile(smoothed: np.ndarray, extent: list, nodata: float,
                              lat: float, lon: float,
                              name: str, output_dir: Path,
                              n_samples: int = 400) -> None:
    bearing = cfg.ROAD_BEARING_DEG
    road_east  = np.sin(np.radians(bearing))
    road_north = np.cos(np.radians(bearing))

    rows, cols = smoothed.shape
    # Estimate road-segment length from the smaller DTM dimension
    px_size_m = (extent[1] - extent[0]) / cols * 111320.0 * np.cos(np.radians(lat))
    road_len_m = min(rows, cols) * px_size_m * 0.80

    dists_m = np.linspace(-road_len_m / 2, road_len_m / 2, n_samples)
    lats_road = lat + dists_m * road_north / 111320.0
    lons_road = lon + dists_m * road_east  / (111320.0 * np.cos(np.radians(lat)))

    col_f = (lons_road - extent[0]) / (extent[1] - extent[0]) * (cols - 1)
    row_f = (extent[3] - lats_road) / (extent[3] - extent[2]) * (rows - 1)
    col_f = np.clip(col_f, 0, cols - 1)
    row_f = np.clip(row_f, 0, rows - 1)

    nd_frac = map_coordinates((smoothed == nodata).astype(float),
                              [row_f, col_f], order=1, mode="nearest")
    elev = map_coordinates(smoothed.astype(float), [row_f, col_f],
                           order=1, mode="nearest")
    elev = np.where(nd_frac > 0, np.nan, elev)  # mask nodata + bilinear-contaminated samples

    slope = np.gradient(elev, dists_m)  # NaN propagates naturally

    fig, (ax_e, ax_s) = plt.subplots(
        2, 1, figsize=(16, 8), facecolor="#1a1a1a",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    fig.suptitle(
        f"Along-Road Elevation Profile  ·  {name.replace('_', ' ')}  "
        f"·  bearing {bearing:.0f}°",
        fontsize=12, color="white",
    )

    ax_e.plot(dists_m, elev, color="#66aaff", linewidth=1.2, alpha=0.95)
    base = np.nanmin(elev) if not np.all(np.isnan(elev)) else 0
    ax_e.fill_between(dists_m, elev, base, alpha=0.18, color="#4488ff")

    # Detected peaks (embankments / walls)
    valid = ~np.isnan(elev)
    if valid.sum() > 20:
        pks, props = find_peaks(elev, prominence=0.2, distance=15)
        troughs, _ = find_peaks(-elev, prominence=0.2, distance=15)
        if pks.size:
            ax_e.scatter(dists_m[pks], elev[pks], color="#ff8844",
                         s=35, zorder=5, label=f"Peaks ({pks.size})", marker="^")
        if troughs.size:
            ax_e.scatter(dists_m[troughs], elev[troughs], color="#44ccff",
                         s=35, zorder=5, label=f"Troughs ({troughs.size})", marker="v")

    style_axes(ax_e, "Elevation along road centreline",
               ylabel="Elevation (m)")
    ax_e.grid(True, color="#252525", linewidth=0.5)
    ax_e.tick_params(colors="gray")
    ax_e.yaxis.label.set_color("lightgray")
    ax_e.legend(fontsize=8, facecolor="#1e1e1e", edgecolor="#444", labelcolor="lightgray")
    ax_e.set_xticklabels([])  # shared x-axis

    ax_s.plot(dists_m, slope, color="#ff8844", linewidth=0.9, alpha=0.85)
    ax_s.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax_s.fill_between(dists_m, slope, 0, where=(slope > 0), alpha=0.2, color="#ff8844")
    ax_s.fill_between(dists_m, slope, 0, where=(slope < 0), alpha=0.2, color="#4499ff")
    style_axes(ax_s, "",
               xlabel="Distance from centre (m)",
               ylabel="Slope (m/m)")
    ax_s.grid(True, color="#252525", linewidth=0.5)
    ax_s.tick_params(colors="gray")
    ax_s.xaxis.label.set_color("lightgray")
    ax_s.yaxis.label.set_color("lightgray")

    save_figure(fig, output_dir / name, "road_elevation_profile.png", dpi=cfg.DPI)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Per-pair perpendicular cross-sections
# ─────────────────────────────────────────────────────────────────────────────

def _detected_pair_profiles(raw: np.ndarray, smoothed: np.ndarray, lrm: np.ndarray,
                             extent: list, nodata: float,
                             pairs: list, name: str, output_dir: Path,
                             n_cross: int = 10) -> None:
    """
    Four-panel figure for the longest detected wall pair.

    Top row  — hillshaded DTM (unchanged) and LRM (slope-normalised) map views,
               with detected wall lines and n_cross colour-coded transect lines
               showing exactly where each profile is sampled.
    Bottom row — raw elevation and de-trended profiles for all n_cross transects,
                 colour-coded to match the map above.
    """
    if not pairs:
        return

    rows, cols = smoothed.shape
    xmin, xmax, ymin, ymax = extent
    center_lat = (ymin + ymax) / 2.0
    px_m = (xmax - xmin) / cols * 111320.0 * np.cos(np.radians(center_lat))

    def _pair_min_length(p):
        l1, l2 = p['line1'], p['line2']
        return min(
            np.hypot(l1[1][0] - l1[0][0], l1[1][1] - l1[0][1]),
            np.hypot(l2[1][0] - l2[0][0], l2[1][1] - l2[0][1]),
        )

    pair = max(pairs, key=_pair_min_length)
    spacing_m = pair['spacing'] * px_m

    geo_segs = pair_to_geo(pair, extent, raw.shape)
    (gx0, gy0), (gx1, gy1) = geo_segs[0]
    (gx2, gy2), (gx3, gy3) = geo_segs[1]

    lat_c = ((gy0 + gy1) + (gy2 + gy3)) / 4.0
    lon_c = ((gx0 + gx1) + (gx2 + gx3)) / 4.0
    cos_lat = np.cos(np.radians(lat_c))

    d_east  = (gx1 - gx0) * cos_lat * 111320.0
    d_north = (gy1 - gy0) * 111320.0
    if abs(d_east) < 1e-9 and abs(d_north) < 1e-9:
        return

    wall_bearing = np.degrees(np.arctan2(d_east, d_north)) % 180.0
    wb_rad = np.radians(wall_bearing)
    sin_b, cos_b = np.sin(wb_rad), np.cos(wb_rad)
    # Perpendicular to wall (bearing + 90°): used for cross-section sampling direction
    perp_east  =  cos_b
    perp_north = -sin_b

    def _along_wall(glon, glat):
        """Signed distance (m) from centroid along the wall axis."""
        de = (glon - lon_c) * cos_lat * 111320.0
        dn = (glat - lat_c) * 111320.0
        return de * sin_b + dn * cos_b

    def _perp_proj(glon, glat):
        """Signed distance (m) from centroid along the cross-section axis."""
        de = (glon - lon_c) * cos_lat * 111320.0
        dn = (glat - lat_c) * 111320.0
        return de * cos_b - dn * sin_b

    # Evenly-space n_cross sample centres inside the pair's along-wall extent
    ts_wall = [_along_wall(gx0, gy0), _along_wall(gx1, gy1),
               _along_wall(gx2, gy2), _along_wall(gx3, gy3)]
    t_min, t_max = min(ts_wall), max(ts_wall)
    edge = (t_max - t_min) * 0.05
    ts = np.linspace(t_min + edge, t_max - edge, n_cross)

    cs_lats = lat_c + ts * cos_b / 111320.0
    cs_lons = lon_c + ts * sin_b / (cos_lat * 111320.0)

    half_w = cfg.SPACING_METERS * 1.5
    n_samp = 200

    pos1 = _perp_proj((gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0)
    pos2 = _perp_proj((gx2 + gx3) / 2.0, (gy2 + gy3) / 2.0)

    # ── Sample profiles ────────────────────────────────────────────────────────
    profiles_raw: list = []
    profiles_dt: list  = []
    valid_ts: list     = []
    valid_lats: list   = []
    valid_lons: list   = []

    for i, (lat_s, lon_s) in enumerate(zip(cs_lats, cs_lons)):
        dists, elev = sample_profile(
            smoothed, extent, lat_s, lon_s,
            wall_bearing, half_w, n_samp, nodata=nodata,
        )
        _, elev_lrm = sample_profile(
            lrm, extent, lat_s, lon_s,
            wall_bearing, half_w, n_samp, nodata=nodata,
        )
        v = ~np.isnan(elev)
        if v.sum() < 10:
            continue
        profiles_raw.append(elev)
        profiles_dt.append(elev_lrm)
        valid_ts.append(ts[i])
        valid_lats.append(lat_s)
        valid_lons.append(lon_s)

    if not profiles_raw:
        return

    n_valid  = len(profiles_raw)
    cmap_cs  = cm.get_cmap("plasma", n_valid)
    dists_x  = np.linspace(-half_w, half_w, n_samp)
    alpha_ln = max(0.45, 1.0 - 0.04 * n_valid)

    # ── Map imagery ────────────────────────────────────────────────────────────
    hs_img  = render_hillshaded_dtm(raw, nodata=nodata,
                                     azimuth=315, altitude=30, z_factor=4.0)
    lrm_f   = np.where(lrm == nodata, np.nan, lrm.astype(float))
    lrm_abs = np.abs(lrm_f[np.isfinite(lrm_f)])
    lrm_vmax = float(np.nanpercentile(lrm_abs, 98)) if lrm_abs.size else 1.0
    ext_map = (xmin, xmax, ymin, ymax)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor="#1a1a1a")
    fig.suptitle(
        f"Detected-Pair Cross Sections  ·  {name.replace('_', ' ')}  "
        f"·  {n_valid} transects  ·  {spacing_m:.0f} m wall spacing  "
        f"·  bearing {wall_bearing:.0f}°",
        fontsize=12, color="white", y=0.985,
    )
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.12,
                          top=0.955, bottom=0.07, left=0.05, right=0.92)

    ax_hs  = fig.add_subplot(gs[0, 0])
    ax_lrm = fig.add_subplot(gs[0, 1])
    ax_raw = fig.add_subplot(gs[1, 0])
    ax_dt  = fig.add_subplot(gs[1, 1])

    # ── Map panels ─────────────────────────────────────────────────────────────
    ax_hs.imshow(hs_img, origin="upper", extent=ext_map, aspect="auto")
    im_lrm = ax_lrm.imshow(lrm_f, origin="upper", cmap="RdBu_r",
                             extent=ext_map, aspect="auto",
                             vmin=-lrm_vmax, vmax=lrm_vmax)
    cbar_lrm = fig.colorbar(im_lrm, ax=ax_lrm, fraction=0.046, pad=0.02, shrink=0.9)
    cbar_lrm.set_label("LRM (m)", color="lightgray", fontsize=8)
    cbar_lrm.ax.yaxis.set_tick_params(colors="gray", labelsize=7)

    WALL_COLORS = ["#ff4444", "#4499ff"]
    for ax in (ax_hs, ax_lrm):
        # Detected wall pair
        for k, ((lx0, ly0), (lx1, ly1)) in enumerate(geo_segs):
            ax.plot([lx0, lx1], [ly0, ly1], color=WALL_COLORS[k],
                    linewidth=2.2, alpha=0.9,
                    label=(f"Wall {k+1}" if ax is ax_hs else "_nolabel_"))
        # Cross-section transect lines, colour-coded by along-wall position
        for j, (lat_s, lon_s) in enumerate(zip(valid_lats, valid_lons)):
            c_j   = cmap_cs(j / max(1, n_valid - 1))
            lat_e = [lat_s + d * perp_north / 111320.0             for d in (-half_w, half_w)]
            lon_e = [lon_s + d * perp_east  / (cos_lat * 111320.0) for d in (-half_w, half_w)]
            ax.plot(lon_e, lat_e, color=c_j, linewidth=0.9, alpha=0.80)
        ax.plot(lon_c, lat_c, "w+", markersize=9, markeredgewidth=1.5, alpha=0.85)

    style_axes(ax_hs,  "DTM Hillshade  (unchanged)")
    style_axes(ax_lrm, "Local Relief Model  (slope-normalised)")
    for ax in (ax_hs, ax_lrm):
        ax.set_xticks([]); ax.set_yticks([])
    ax_hs.legend(fontsize=7, facecolor="#1e1e1e", edgecolor="#444",
                 labelcolor="lightgray", loc="upper left")

    # ── Profile panels ─────────────────────────────────────────────────────────
    stack_raw = np.vstack(profiles_raw)
    stack_dt  = np.vstack(profiles_dt)
    mean_raw  = np.nanmean(stack_raw, axis=0)
    mean_dt   = np.nanmean(stack_dt,  axis=0)
    std_dt    = np.nanstd(stack_dt,   axis=0)

    for j, (elev, detrended) in enumerate(zip(profiles_raw, profiles_dt)):
        c_j = cmap_cs(j / max(1, n_valid - 1))
        ax_raw.plot(dists_x, elev,      color=c_j, linewidth=0.9, alpha=alpha_ln,
                    label=f"{valid_ts[j]:+.0f} m")
        ax_dt.plot( dists_x, detrended, color=c_j, linewidth=0.9, alpha=alpha_ln)

    ax_raw.plot(dists_x, mean_raw, color="white", linewidth=2.0, zorder=5, label="Mean")
    ax_dt.plot( dists_x, mean_dt,  color="white", linewidth=2.0, zorder=5, label="Mean")
    ax_dt.fill_between(dists_x, mean_dt - std_dt, mean_dt + std_dt,
                       alpha=0.15, color="white", label="±1 σ")

    for ax in (ax_raw, ax_dt):
        ax.axvline(pos1, color="#ff4444", linewidth=1.2, linestyle=":", alpha=0.85,
                   label=f"Wall 1  ({pos1:+.0f} m)")
        ax.axvline(pos2, color="#4499ff", linewidth=1.2, linestyle=":", alpha=0.85,
                   label=f"Wall 2  ({pos2:+.0f} m)")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.65)
        ax.grid(True, color="#252525", linewidth=0.5)
    ax_dt.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.65)

    style_axes(ax_raw, "Raw Elevation",
               xlabel="Distance from pair centroid (m)", ylabel="Elevation (m)")
    style_axes(ax_dt,  "LRM  (local relief model)",
               xlabel="Distance from pair centroid (m)", ylabel="LRM elevation (m)")

    # Legend: raw panel shows offset labels + priority lines
    handles_r, labels_r = ax_raw.get_legend_handles_labels()
    priority = {f"Wall 1  ({pos1:+.0f} m)", f"Wall 2  ({pos2:+.0f} m)", "Mean"}
    skip = max(1, n_valid // 6)
    kept = [(h, l) for i, (h, l) in enumerate(zip(handles_r, labels_r))
            if l in priority or i % skip == 0]
    if kept:
        hs_r, ls_r = zip(*kept)
        ax_raw.legend(hs_r, ls_r, fontsize=7, ncol=2, facecolor="#1e1e1e",
                      edgecolor="#444", labelcolor="lightgray", loc="best")

    handles_d, labels_d = ax_dt.get_legend_handles_labels()
    ax_dt.legend(handles_d[-5:], labels_d[-5:], fontsize=7,
                 facecolor="#1e1e1e", edgecolor="#444", labelcolor="lightgray")

    # Shared along-wall-position colourbar for the profile panels
    n_ts = len(valid_ts)
    norm_pos = Normalize(vmin=valid_ts[0],
                         vmax=valid_ts[-1] if n_ts > 1 else valid_ts[0] + 1)
    sm_pos = plt.cm.ScalarMappable(cmap="plasma", norm=norm_pos)
    sm_pos.set_array([])
    cbar_pos = fig.colorbar(sm_pos, ax=[ax_raw, ax_dt],
                             orientation="vertical", pad=0.01, fraction=0.012, shrink=0.85)
    cbar_pos.set_label("Along-wall position (m from centroid)", color="lightgray", fontsize=8)
    cbar_pos.ax.yaxis.set_tick_params(colors="gray")

    save_figure(fig, output_dir / name, "detected_pair_profiles.png", dpi=cfg.DPI)


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def generate_extra_views(coord: dict, dtm_path: str, output_dir: Path) -> None:
    """
    Generate all supplementary figures for one coordinate.

    Parameters
    ----------
    coord      : dict with keys 'name', 'lat', 'lon', 'size_m'
    dtm_path   : path to merged DTM GeoTIFF
    output_dir : parent output directory
    """
    name = coord["name"]
    lat, lon = coord["lat"], coord["lon"]
    size_m = coord.get("size_m", 200)

    print(f"  [extra_views] loading DTM window …")
    gt, band, nodata, _ds = getGTAndBand(dtm_path)
    raw, extent = get_window(lat, lon, size_m, size_m, gt, band, nodata)
    if raw is None or raw.size == 0:
        print(f"  [extra_views] no data — skipped.")
        return

    smoothed = median_smooth_windowed(raw, nodata=nodata, radius=cfg.MEDIAN_SMOOTH_RADIUS)
    lrm = local_relief_model(smoothed, nodata=nodata)

    # Run detection once so the 3D view can overlay pairs
    ridge_for_pairs = multiscale_ridge_enhancement(lrm, scales=cfg.RIDGE_SCALES, nodata=nodata)
    valid_ridge = np.where(raw == nodata, 0.0, ridge_for_pairs.astype(float))
    pairs = detect_parallel_walls_ridge_pairing(
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
        return_debug=False
    )
    # pairs = remove_invalid_pairs(pairs, angle_tolerance=cfg.ANGLE_TOLERANCE,
    #                              target_angle=cfg.TARGET_ANGLE)

    print(f"  [extra_views] multi-azimuth hillshade …")
    _multi_azimuth_hillshade(raw, extent, nodata, name, output_dir)

    print(f"  [extra_views] 3D perspective …")
    _3d_perspective(raw, extent, nodata, name, output_dir, pairs)

    print(f"  [extra_views] ridge-scale comparison …")
    _ridge_scale_comparison(smoothed, extent, nodata, name, output_dir)

    print(f"  [extra_views] elevation statistics …")
    _elevation_statistics(raw, nodata, name, output_dir)

    print(f"  [extra_views] along-road profile …")
    _road_elevation_profile(smoothed, extent, nodata, lat, lon, name, output_dir)

    print(f"  [extra_views] detected-pair cross sections …")
    _detected_pair_profiles(raw, smoothed, lrm, extent, nodata, pairs, name, output_dir)
