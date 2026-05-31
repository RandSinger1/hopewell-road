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

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3D projection)
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

_REPORT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _REPORT_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPORT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from process_dtm import (
    getGTAndBand, get_window,
    median_smooth_windowed, local_relief_model, multiscale_ridge_enhancement,
)
from ridge_detection import detect_parallel_walls_ridge_pairing, remove_invalid_pairs
from report_utils import hillshade, render_hillshaded_dtm, save_figure, style_axes, pair_to_geo, sample_profile
import config as cfg


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

def _detected_pair_profiles(raw: np.ndarray, smoothed: np.ndarray,
                             extent: list, nodata: float,
                             pairs: list, name: str, output_dir: Path) -> None:
    """
    For each detected wall pair, sample an elevation transect perpendicular
    to the walls through the pair's centroid.  All profiles are overlaid on
    one figure so the dual-wall W-shape can be compared across all detections.

    X-axis is distance from the pair centroid (m); walls should appear as
    twin peaks at ±spacing/2.  A de-trended panel removes the regional slope.
    """
    if not pairs:
        return

    rows, cols = smoothed.shape
    xmin, xmax, ymin, ymax = extent
    center_lat = (ymin + ymax) / 2.0
    px_m = (xmax - xmin) / cols * 111320.0 * np.cos(np.radians(center_lat))

    # Fixed sampling extent so all profiles share the same x-axis
    half_w    = cfg.SPACING_METERS * 1.5
    n_samp    = 200
    max_pairs = min(len(pairs), 15)
    cmap_p    = cm.get_cmap("plasma", max_pairs)
    alpha     = max(0.35, 0.85 - 0.04 * max_pairs)

    fig, (ax_raw, ax_dt) = plt.subplots(
        1, 2, figsize=(16, 6), facecolor="#1a1a1a",
        gridspec_kw={"wspace": 0.12},
    )
    fig.suptitle(
        f"Detected-Pair Cross Sections  ·  {name.replace('_', ' ')}  "
        f"·  {max_pairs} pair{'s' if max_pairs != 1 else ''} shown",
        fontsize=12, color="white",
    )

    # All sample calls use the same n_samp and half_w, so dists arrays are
    # identical — we can stack de-trended profiles directly.
    profiles_dt: list = []

    for i, pair in enumerate(pairs[:max_pairs]):
        geo_segs = pair_to_geo(pair, extent, raw.shape)
        (gx0, gy0), (gx1, gy1) = geo_segs[0]   # line1 geo endpoints (lon, lat)
        (gx2, gy2), (gx3, gy3) = geo_segs[1]   # line2 geo endpoints

        lat_c = ((gy0 + gy1) + (gy2 + gy3)) / 4.0
        lon_c = ((gx0 + gx1) + (gx2 + gx3)) / 4.0

        # Wall bearing (CW from N) from line1 geo direction
        d_east  = (gx1 - gx0) * np.cos(np.radians(lat_c)) * 111320.0
        d_north = (gy1 - gy0) * 111320.0
        if abs(d_east) < 1e-9 and abs(d_north) < 1e-9:
            continue
        wall_bearing = np.degrees(np.arctan2(d_east, d_north)) % 180.0

        # sample_profile samples perpendicular to wall_bearing
        dists, elev = sample_profile(
            smoothed, extent, lat_c, lon_c,
            wall_bearing, half_w, n_samp, nodata=nodata,
        )

        valid = ~np.isnan(elev)
        if valid.sum() < 10:
            continue

        coeffs    = np.polyfit(dists[valid], elev[valid], 1)
        detrended = elev - np.polyval(coeffs, dists)

        color     = cmap_p(i / max(1, max_pairs - 1))
        spacing_m = pair['spacing'] * px_m
        lbl       = f"Pair {i + 1}  ({spacing_m:.0f} m)"

        ax_raw.plot(dists, elev,      color=color, linewidth=0.9, alpha=alpha, label=lbl)
        ax_dt.plot(dists, detrended, color=color, linewidth=0.9, alpha=alpha)

        # Project each line's midpoint onto the cross-section axis to get actual wall positions
        wall_bearing_rad = np.radians(wall_bearing)
        cos_b, sin_b = np.cos(wall_bearing_rad), np.sin(wall_bearing_rad)
        cos_lat = np.cos(np.radians(lat_c))

        def _proj(lon_pt, lat_pt):
            d_e = (lon_pt - lon_c) * cos_lat * 111320.0
            d_n = (lat_pt - lat_c) * 111320.0
            return d_e * cos_b - d_n * sin_b

        pos1 = _proj((gx0 + gx1) / 2, (gy0 + gy1) / 2)
        pos2 = _proj((gx2 + gx3) / 2, (gy2 + gy3) / 2)

        for ax in (ax_raw, ax_dt):
            ax.axvline(pos1, color=color, linewidth=0.9, linestyle=":", alpha=0.60)
            ax.axvline(pos2, color=color, linewidth=0.9, linestyle=":", alpha=0.60)

        profiles_dt.append(detrended)

    if not profiles_dt:
        plt.close(fig)
        return

    # Mean ± σ across all de-trended profiles
    if len(profiles_dt) >= 2:
        stack   = np.vstack(profiles_dt)
        mean_dt = np.nanmean(stack, axis=0)
        std_dt  = np.nanstd(stack, axis=0)
        ax_dt.plot(dists, mean_dt, color="white", linewidth=2.0, zorder=5, label="Mean")
        ax_dt.fill_between(dists, mean_dt - std_dt, mean_dt + std_dt,
                           alpha=0.15, color="white", label="±1 σ")

    for ax in (ax_raw, ax_dt):
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)

    style_axes(ax_raw, "Raw Elevation",
               xlabel="Distance from pair centroid (m)", ylabel="Elevation (m)")
    style_axes(ax_dt,  "De-trended  (linear slope removed)",
               xlabel="Distance from pair centroid (m)", ylabel="Relative elevation (m)")
    for ax in (ax_raw, ax_dt):
        ax.grid(True, color="#252525", linewidth=0.5)

    handles, labels = ax_raw.get_legend_handles_labels()
    skip = max(1, len(handles) // 8)
    kept = [(h, l) for k, (h, l) in enumerate(zip(handles, labels)) if k % skip == 0]
    if kept:
        hs, ls = zip(*kept)
        ax_raw.legend(hs, ls, fontsize=7, ncol=2, facecolor="#1e1e1e",
                      edgecolor="#444", labelcolor="lightgray", loc="best")
    h2, l2 = ax_dt.get_legend_handles_labels()
    if h2:
        ax_dt.legend(h2[-3:], l2[-3:], fontsize=7,
                     facecolor="#1e1e1e", edgecolor="#444", labelcolor="lightgray")

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
    pairs = remove_invalid_pairs(pairs, angle_tolerance=cfg.ANGLE_TOLERANCE,
                                  target_angle=cfg.TARGET_ANGLE)

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
    _detected_pair_profiles(raw, smoothed, extent, nodata, pairs, name, output_dir)
