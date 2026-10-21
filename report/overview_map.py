"""
overview_map.py — Study-corridor overview + per-site figures.

Outputs
-------
overview_map.png
    Full corridor map: cities, Newark–Chillicothe and Newark–High Banks
    azimuth lines, every REPORT_COORDS site labelled with its detected bearing.

overview_<name>.png  (one per coord)
    Same full corridor map, with the named site highlighted and all
    others dimmed.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .base_compute import get_terrain, get_detection
from .report_utils import save_figure, site_target_angle, pair_to_geo
from . import config as cfg
import parameters
from utils import make_parabola


_CITIES = {
    "Newark":      parameters.NEWARK_COORD,
    "Circleville": parameters.CIRCLEVILLE_COORD,
    "Chillicothe": parameters.CHILLICOTHE_COORD,
    "High Banks":  parameters.HIGH_BANKS_COORD,
}

_HALF_LINE_M = 5_000


# ── detection ─────────────────────────────────────────────────────────────────

def _run_detection(coord: dict, dtm_path: str, output_dir: Path,
                    use_cache: bool = True) -> dict:
    mode, fallback = site_target_angle(coord)
    fallback = fallback % 180
    base     = {**coord, "bearing": fallback, "mode": mode, "detected": False,
                "raw": None, "lrm": None, "extent": None,
                "nodata": parameters.NODATA, "pairs": []}
    try:
        terrain = get_terrain(coord, dtm_path, output_dir, use_cache=use_cache)
        if terrain is None:
            return base
        raw, extent, nodata, lrm = (
            terrain["raw"], terrain["extent"], terrain["nodata"], terrain["lrm"],
        )

        det = get_detection(coord, dtm_path, terrain, output_dir, use_cache=use_cache)
        pairs = det["pairs_clean"]
        mode  = det.get("mode", mode)

        bearing  = fallback
        detected = False
        if pairs:
            detected = True
            # Convert to geo coords then apply cos(lat) so _line_endpoints
            # (which works in physical metres) draws the line at the correct angle.
            geo_segs = pair_to_geo(pairs[0], extent, raw.shape)
            (gx0, gy0), (gx1, gy1) = geo_segs[0]
            cos_lat = np.cos(np.radians((gy0 + gy1) / 2))
            bearing = np.degrees(np.arctan2(
                (gx1 - gx0) * cos_lat * 111_320.0,
                (gy1 - gy0) * 111_320.0,
            )) % 180

        return {**coord, "bearing": bearing, "mode": mode, "detected": detected,
                "raw": raw, "lrm": lrm, "extent": extent, "nodata": nodata, "pairs": pairs}

    except Exception as exc:
        print(f"    [overview] {coord['name']} detection error: {exc}")
        return base


# ── helpers ───────────────────────────────────────────────────────────────────

def _line_endpoints(lat: float, lon: float, bearing_deg: float, half_m: float):
    b    = np.radians(bearing_deg)
    dlat = half_m * np.cos(b) / 111_320.0
    dlon = half_m * np.sin(b) / (111_320.0 * np.cos(np.radians(lat)))
    return (lon - dlon, lat - dlat), (lon + dlon, lat + dlat)


def _corridor_extent(coords: list[dict], pad: float = 0.10):
    all_lats = [v[0] for v in _CITIES.values()] + [c["lat"] for c in coords]
    all_lons = [v[1] for v in _CITIES.values()] + [c["lon"] for c in coords]
    return (min(all_lons) - pad, max(all_lons) + pad,
            min(all_lats) - pad, max(all_lats) + pad)


def _newark_chillicothe_bearing() -> float:
    n_lat, n_lon = _CITIES["Newark"]
    c_lat, c_lon = _CITIES["Chillicothe"]
    cos_lat = np.cos(np.radians((n_lat + c_lat) / 2))
    return np.degrees(np.arctan2((c_lon - n_lon) * cos_lat, c_lat - n_lat)) % 360


def _newark_high_banks_bearing() -> float:
    n_lat, n_lon = _CITIES["Newark"]
    h_lat, h_lon = _CITIES["High Banks"]
    cos_lat = np.cos(np.radians((n_lat + h_lat) / 2))
    return np.degrees(np.arctan2((h_lon - n_lon) * cos_lat, h_lat - n_lat)) % 360


# ── shared map drawing ────────────────────────────────────────────────────────

def _draw_circleville_parabola(ax: plt.Axes) -> None:
    """Draw the parabolic Circleville corridor curve."""
    curve_fn, _ = make_parabola(
        parameters.PARABOLA_A, parameters.PARABOLA_B,
        parameters.PARABOLA_C, parameters.PARABOLA_X_OFFSET,
    )
    c_lon = _CITIES["Circleville"][1]
    n_lon = _CITIES["Newark"][1]
    lons = np.linspace(c_lon, n_lon, 300)
    lats = np.array([curve_fn(lon) for lon in lons])
    ax.plot(lons, lats, color="#90d090", linewidth=1.5, linestyle="--",
            alpha=0.8, zorder=3)
    mid = len(lons) // 2
    ax.annotate("circleville", xy=(lons[mid], lats[mid]),
                xytext=(-50, 6), textcoords="offset points",
                fontsize=7, color="#90d090", alpha=0.85)


def _draw_corridor_map(ax: plt.Axes, sites: list[dict],
                       focus_name: str | None = None,
                       path_name: str | None = None) -> None:
    """
    Draw the full corridor map onto ax.

    focus_name : if given, that site is highlighted; all others are dimmed.
                 If None, all sites are drawn at equal weight (overview mode).
    path_name  : corridor path being displayed; draws the parabola when
                 "circleville".
    """
    coords = [{k: s[k] for k in ("lat", "lon", "name")} for s in sites]
    xmin, xmax, ymin, ymax = _corridor_extent(coords)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color="#1a2030", linewidth=0.6)
    ax.tick_params(colors="gray", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.set_xlabel("Longitude", fontsize=8, color="lightgray")
    ax.set_ylabel("Latitude",  fontsize=8, color="lightgray")

    # Newark → Chillicothe azimuth line
    n_lat, n_lon = _CITIES["Newark"]
    c_lat, c_lon = _CITIES["Chillicothe"]
    az = _newark_chillicothe_bearing()
    ax.plot([n_lon, c_lon], [n_lat, c_lat],
            color="#a0c8e0", linewidth=1.5, linestyle="--",
            alpha=0.7, zorder=3)
    # azimuth label near midpoint
    mid_lon = (n_lon + c_lon) / 2
    mid_lat = (n_lat + c_lat) / 2
    ax.annotate(f"az {az:.1f}°", xy=(mid_lon, mid_lat),
                xytext=(-30, 6), textcoords="offset points",
                fontsize=7, color="#a0c8e0", alpha=0.85)

    # Newark → High Banks azimuth line
    h_lat, h_lon = _CITIES["High Banks"]
    az_hb = _newark_high_banks_bearing()
    ax.plot([n_lon, h_lon], [n_lat, h_lat],
            color="#cc99ff", linewidth=1.5, linestyle="--",
            alpha=0.7, zorder=3)
    mid_lon_hb = (n_lon + h_lon) / 2
    mid_lat_hb = (n_lat + h_lat) / 2
    ax.annotate(f"az {az_hb:.1f}°", xy=(mid_lon_hb, mid_lat_hb),
                xytext=(-30, 6), textcoords="offset points",
                fontsize=7, color="#cc99ff", alpha=0.85)

    # Circleville parabola (always shown — completes the three-leg corridor picture)
    _draw_circleville_parabola(ax)

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
        name        = s["name"]
        is_focus    = (focus_name is None or name == focus_name)

        if is_focus:
            color     = "#f0c040" if detected else "#ff7060"
            lw        = 2.5
            dot_size  = 80
            font_size = 8.5
            alpha     = 1.0
            zorder    = 8
        else:
            color     = "#888888"
            lw        = 1.2
            dot_size  = 25
            font_size = 7.0
            alpha     = 0.45
            zorder    = 5

        (lx0, ly0), (lx1, ly1) = _line_endpoints(lat, lon, bearing, _HALF_LINE_M)
        ax.plot([lx0, lx1], [ly0, ly1], color=color, linewidth=lw,
                alpha=alpha, zorder=zorder, solid_capstyle="round")
        ax.scatter(lon, lat, s=dot_size, color=color, alpha=alpha, zorder=zorder + 1)

        if is_focus:
            ax.annotate(
                f"{name.replace('_', ' ')}\n{bearing:.1f}°",
                xy=(lon, lat), xytext=(9, 5), textcoords="offset points",
                fontsize=font_size, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                          edgecolor=color, alpha=0.92, linewidth=1.0),
                zorder=zorder + 2,
            )
        else:
            ax.annotate(
                name.replace("_", " "),
                xy=(lon, lat), xytext=(6, 4), textcoords="offset points",
                fontsize=font_size, color=color, alpha=alpha,
                zorder=zorder + 2,
            )


# ── per-site figure ───────────────────────────────────────────────────────────

def _legend_handles(path_name: str | None) -> list:
    handles = [
        mlines.Line2D([], [], color="#f0c040", linewidth=2.5, label="Detected bearing"),
        mlines.Line2D([], [], color="#ff7060", linewidth=2.5,
                      label="Fallback (corridor mode bearing)"),
        mlines.Line2D([], [], color="#888888", linewidth=1.2, alpha=0.5,
                      label="Other sites"),
        mlines.Line2D([], [], color="#5b9bd5", linewidth=0,
                      marker="D", markersize=7, label="City"),
        mlines.Line2D([], [], color="#a0c8e0", linewidth=1.5,
                      linestyle="--", label="Newark – Chillicothe"),
        mlines.Line2D([], [], color="#cc99ff", linewidth=1.5,
                      linestyle="--", label="Newark – High Banks"),
    ]
    handles.append(mlines.Line2D([], [], color="#90d090", linewidth=1.5,
                                 linestyle="--", label="Circleville parabola"))
    return handles


def _save_site_figure(focus: dict, all_sites: list[dict], output_dir: Path,
                      path_name: str | None = None) -> None:
    name    = focus["name"]
    bearing = focus["bearing"]
    mode    = focus["mode"]
    detected = focus["detected"]
    status  = (f"detected {bearing:.1f}°" if detected
               else f"fallback {bearing:.1f}° ({mode})")

    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#1a1a1a")
    ax.set_facecolor("#0d1117")
    ax.set_title(
        f"Hopewell Corridor  ·  {name.replace('_', ' ')}  ·  {status}",
        fontsize=12, color="white", pad=10,
    )

    _draw_corridor_map(ax, all_sites, focus_name=name, path_name=path_name)

    ax.legend(handles=_legend_handles(path_name), fontsize=8, loc="lower right",
              facecolor="#1e1e1e", edgecolor="#444444",
              labelcolor="lightgray", framealpha=0.9)

    save_figure(fig, output_dir, f"overview_{name}.png", dpi=cfg.DPI)


# ── main entry point ──────────────────────────────────────────────────────────

def generate_overview_map(coords: list[dict], dtm_path: str, output_dir: Path,
                           use_cache: bool = True,
                           path_name: str | None = None) -> None:
    print("  [overview] running detection at each site …")

    sites: list[dict] = []
    for coord in coords:
        s = _run_detection(coord, dtm_path, output_dir, use_cache=use_cache)
        tag = "detected" if s["detected"] else f"fallback ({s['mode']})"
        print(f"    {s['name']}: {s['bearing']:.1f}°  ({tag})")
        sites.append(s)

    # per-site figures (full map, focused site highlighted)
    for s in sites:
        _save_site_figure(s, sites, output_dir, path_name=path_name)

    # corridor overview (all sites equal weight)
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#1a1a1a")
    ax.set_facecolor("#0d1117")
    path_label = f"  ·  {path_name}" if path_name else ""
    ax.set_title(
        f"Hopewell Earthwork Corridor{path_label}  ·  Detected wall bearings",
        fontsize=12, color="white", pad=10,
    )
    _draw_corridor_map(ax, sites, focus_name=None, path_name=path_name)

    overview_handles = [h for h in _legend_handles(path_name)
                        if h.get_label() != "Other sites"]
    ax.legend(handles=overview_handles, fontsize=8, loc="lower right",
              facecolor="#1e1e1e", edgecolor="#444444",
              labelcolor="lightgray", framealpha=0.9)

    save_figure(fig, output_dir, "overview_map.png", dpi=cfg.DPI)
