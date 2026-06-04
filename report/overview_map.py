"""
overview_map.py — Study-corridor overview + per-site close-up figures.

Outputs
-------
overview_map.png
    Full corridor map: cities, Newark–Chillicothe route line, every
    REPORT_COORDS site labelled with its detected bearing.

overview_<name>.png  (one per coord)
    Close-up: hillshaded DTM (left) + LRM (right), detected pair lines
    drawn in geo-coordinates, bearing indicator, name / angle.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from road_model.process_dtm import (
    getGTAndBand, get_window,
    median_smooth_windowed, local_relief_model, multiscale_ridge_enhancement,
)
from road_model.ridge_detection import detect_parallel_walls_ridge_pairing, remove_invalid_pairs
from .report_utils import save_figure
from . import config as cfg
import parameters


_CITIES = {
    "Newark":      (40.0, -82.4),
    "Circleville": (39.6,    -82.9),
    "Chillicothe": (39.3,  -82.9),
}

# Length of the corridor-scale bearing indicator (metres each side of dot)
_HALF_LINE_M = 5_000


# ── detection ─────────────────────────────────────────────────────────────────

def _run_detection(coord: dict, dtm_path: str) -> dict:
    """
    Full detection pipeline for one coord window.
    Returns a dict extending `coord` with:
      bearing, detected, raw, lrm, extent, nodata, pairs
    """
    lat, lon   = coord["lat"], coord["lon"]
    size_m     = coord.get("size_m", 200)
    fallback   = cfg.ROAD_BEARING_DEG % 180
    base       = {**coord, "bearing": fallback, "detected": False,
                  "raw": None, "lrm": None, "extent": None,
                  "nodata": parameters.NODATA, "pairs": []}
    try:
        gt, band, nodata, _ds = getGTAndBand(dtm_path)
        raw, extent = get_window(lat, lon, size_m, size_m, gt, band, nodata)
        if raw is None or raw.size == 0:
            return base

        smoothed    = median_smooth_windowed(raw, nodata=nodata, radius=cfg.MEDIAN_SMOOTH_RADIUS)
        lrm         = local_relief_model(smoothed, radius=cfg.LRM_RADIUS, nodata=nodata)
        ridge       = multiscale_ridge_enhancement(lrm, scales=cfg.RIDGE_SCALES, nodata=nodata)
        valid_ridge = np.where(raw == nodata, 0.0, ridge.astype(float))

        result = detect_parallel_walls_ridge_pairing(
            ridge_map=valid_ridge,
            target_angle=cfg.TARGET_ANGLE,
            extent=extent,
            spacing_meters=cfg.SPACING_METERS,
            angle_tolerance=cfg.ANGLE_TOLERANCE,
            spacing_tolerance=cfg.SPACING_TOLERANCE,
            ridge_threshold=cfg.RIDGE_THRESHOLD,
            min_ridge_length_meters=cfg.MIN_RIDGE_LENGTH_METERS,
            overlap_ratio=cfg.OVERLAP_RATIO,
            return_debug=True,
        )
        pairs = result[0] if isinstance(result, tuple) else result
        pairs = remove_invalid_pairs(pairs, angle_tolerance=cfg.ANGLE_TOLERANCE,
                                     target_angle=cfg.TARGET_ANGLE)

        bearing  = fallback
        detected = False
        if pairs:
            detected = True
            (x0, y0), (x1, y1) = pairs[0]["line1"]
            # Image y increases downward (south); flip dy to get north-positive
            bearing = np.degrees(np.arctan2(x1 - x0, -(y1 - y0))) % 180

        return {**coord, "bearing": bearing, "detected": detected,
                "raw": raw, "lrm": lrm, "extent": extent, "nodata": nodata, "pairs": pairs}

    except Exception as exc:
        print(f"    [overview] {coord['name']} detection error: {exc}")
        return base


# ── helpers ───────────────────────────────────────────────────────────────────

def _line_endpoints(lat: float, lon: float, bearing_deg: float, half_m: float):
    """(lon0, lat0), (lon1, lat1) for a line centred at (lat, lon)."""
    b    = np.radians(bearing_deg)
    dlat = half_m * np.cos(b) / 111_320.0
    dlon = half_m * np.sin(b) / (111_320.0 * np.cos(np.radians(lat)))
    return (lon - dlon, lat - dlat), (lon + dlon, lat + dlat)


# ── per-site map ──────────────────────────────────────────────────────────────

def _save_site_figure(s: dict, output_dir: Path) -> None:
    lat, lon   = s["lat"], s["lon"]
    bearing    = s["bearing"]
    detected   = s["detected"]
    name       = s["name"]

    # View extent: ±0.06° around the site so the bearing line fits with room
    pad_lat, pad_lon = 0.06, 0.07
    x0, x1 = lon - pad_lon, lon + pad_lon
    y0, y1 = lat - pad_lat, lat + pad_lat

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a1a")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.grid(True, color="#1a2030", linewidth=0.6)
    ax.tick_params(colors="gray", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.set_xlabel("Longitude", fontsize=8, color="lightgray")
    ax.set_ylabel("Latitude",  fontsize=8, color="lightgray")

    status = f"detected  {bearing:.1f}°" if detected else f"fallback  {bearing:.1f}°"
    ax.set_title(
        f"{name.replace('_', ' ')}  ·  {lat:.4f}°N  {lon:.4f}°E  ·  {status}",
        fontsize=10, color="white", pad=8,
    )

    # bearing line scaled to fit within the view
    color = "#f0c040" if detected else "#ff7060"
    (lx0, ly0), (lx1, ly1) = _line_endpoints(lat, lon, bearing, _HALF_LINE_M * 0.6)
    ax.plot([lx0, lx1], [ly0, ly1], color=color, linewidth=2.5,
            zorder=6, solid_capstyle="round")
    ax.scatter(lon, lat, s=60, color=color, zorder=7)
    ax.annotate(
        f"{name.replace('_', ' ')}\n{bearing:.1f}°",
        xy=(lon, lat), xytext=(8, 5), textcoords="offset points",
        fontsize=8, color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111",
                  edgecolor=color, alpha=0.88, linewidth=0.8),
    )

    # any cities that fall within the view
    for city, (clat, clon) in _CITIES.items():
        if x0 <= clon <= x1 and y0 <= clat <= y1:
            ax.scatter(clon, clat, s=60, color="#5b9bd5", zorder=5, marker="D")
            ax.annotate(city, xy=(clon, clat), xytext=(6, 5),
                        textcoords="offset points",
                        fontsize=8, color="#7eb8e8", fontweight="bold")

    save_figure(fig, output_dir, f"overview_{name}.png", dpi=cfg.DPI)


# ── main entry point ──────────────────────────────────────────────────────────

def generate_overview_map(coords: list[dict], dtm_path: str, output_dir: Path) -> None:
    print("  [overview] running detection at each site …")

    # Run detection once per coord; reuse results for both figures
    sites: list[dict] = []
    for coord in coords:
        s = _run_detection(coord, dtm_path)
        tag = "detected" if s["detected"] else f"fallback {cfg.ROAD_BEARING_DEG:.0f}°"
        print(f"    {s['name']}: {s['bearing']:.1f}°  ({tag})")
        sites.append(s)

    # ── per-site close-up figures ──────────────────────────────────────────────
    for s in sites:
        _save_site_figure(s, output_dir)

    # ── corridor overview map ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#1a1a1a")
    ax.set_facecolor("#0d1117")

    all_lats = [v[0] for v in _CITIES.values()] + [c["lat"] for c in coords]
    all_lons = [v[1] for v in _CITIES.values()] + [c["lon"] for c in coords]
    pad = 0.10
    ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
    ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)

    ax.grid(True, color="#1a2030", linewidth=0.6)
    ax.tick_params(colors="gray", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.set_xlabel("Longitude", fontsize=8, color="lightgray")
    ax.set_ylabel("Latitude",  fontsize=8, color="lightgray")
    ax.set_title(
        "Hopewell Earthwork Corridor  ·  Detected wall bearings",
        fontsize=12, color="white", pad=10,
    )

    # scan-corridor polygon (faint)
    poly = getattr(parameters, "SCAN_POLYGON", [])
    if poly:
        lats_p = [p[0] for p in poly] + [poly[0][0]]
        lons_p = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(lons_p, lats_p, color="#2a3a4a", linewidth=1.2, linestyle="--", zorder=2)

    # Newark → Chillicothe route line
    n_lat, n_lon = _CITIES["Newark"]
    c_lat, c_lon = _CITIES["Chillicothe"]
    ax.plot([n_lon, c_lon], [n_lat, c_lat],
            color="#a0c8e0", linewidth=1.5, linestyle="--",
            alpha=0.6, zorder=3, label="Newark – Chillicothe")

    # cities
    for city, (clat, clon) in _CITIES.items():
        ax.scatter(clon, clat, s=70, color="#5b9bd5", zorder=5, marker="D")
        ax.annotate(city, xy=(clon, clat), xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=9, color="#7eb8e8", fontweight="bold")

    # site dots + bearing lines
    for s in sites:
        lat, lon    = s["lat"], s["lon"]
        bearing     = s["bearing"]
        detected    = s["detected"]
        name        = s["name"].replace("_", " ")
        color       = "#f0c040" if detected else "#ff7060"

        (lx0, ly0), (lx1, ly1) = _line_endpoints(lat, lon, bearing, _HALF_LINE_M)
        ax.plot([lx0, lx1], [ly0, ly1], color=color, linewidth=2.5,
                zorder=6, solid_capstyle="round")
        ax.scatter(lon, lat, s=50, color=color, zorder=7)
        ax.annotate(
            f"{name}\n{bearing:.1f}°",
            xy=(lon, lat), xytext=(8, 4), textcoords="offset points",
            fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111",
                      edgecolor=color, alpha=0.88, linewidth=0.8),
        )

    # legend
    legend_handles = [
        mlines.Line2D([], [], color="#f0c040", linewidth=2.5, label="Detected bearing"),
        mlines.Line2D([], [], color="#ff7060", linewidth=2.5,
                      label=f"Fallback ({cfg.ROAD_BEARING_DEG:.0f}°)"),
        mlines.Line2D([], [], color="#5b9bd5", linewidth=0,
                      marker="D", markersize=7, label="City"),
        mlines.Line2D([], [], color="#a0c8e0", linewidth=1.5,
                      linestyle="--", label="Newark – Chillicothe"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              facecolor="#1e1e1e", edgecolor="#444444",
              labelcolor="lightgray", framealpha=0.9)

    save_figure(fig, output_dir, "overview_map.png", dpi=cfg.DPI)
