"""
report_utils.py  —  Shared helpers for all report figure generators.

Provides: path setup, hillshading, colourising, DTM profile sampling,
and figure-save / axes-styling utilities.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from utils import corridor_mode_and_angle, corridor_angle_for_mode


# ─── Target angle resolution ──────────────────────────────────────────────────

def site_target_angle(coord: dict) -> tuple:
    """
    Resolve the expected wall bearing for one REPORT_COORDS site.

    Three ways to resolve it, checked in order:
      1. "target_angle" in the REPORT_COORDS entry — a fixed angle override,
         reported as mode "manual".
      2. "mode" in the REPORT_COORDS entry — one of "circleville",
         "chillicothe", "high_banks". Forces that segment's locally expected
         bearing at the site's longitude (utils.corridor_angle_for_mode),
         instead of guessing which leg the site is on.
      3. Neither set — auto-detect whichever named corridor segment passes
         closest to the site (utils.corridor_mode_and_angle).

    Returns (mode, angle_deg).
    """
    override = coord.get("target_angle")
    if override is not None:
        return "manual", override
    mode = coord.get("mode")
    if mode is not None:
        return mode, corridor_angle_for_mode(mode, coord["lon"])
    return corridor_mode_and_angle(coord["lat"], coord["lon"])


# ─── Figure helpers ───────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, output_dir: Path, filename: str, dpi: int = 150) -> None:
    """Save and close a figure; create output_dir if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {path}")


def style_axes(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title_color: str = "white",
) -> None:
    """Apply consistent dark-theme styling to an Axes."""
    ax.set_title(title, fontsize=9, pad=4, color=title_color)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color="lightgray")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color="lightgray")
    ax.tick_params(colors="gray", labelsize=7)
    ax.set_facecolor("#111111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


# ─── Terrain rendering ────────────────────────────────────────────────────────

def hillshade(
    elev: np.ndarray,
    nodata: float = -9999,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 2.0,
) -> np.ndarray:
    """
    Classic hillshade.  Returns a [0, 1] float array.
    Nodata cells are set to 0.5 (neutral grey).
    """
    mask = elev == nodata
    fill = float(np.nanmedian(np.where(mask, np.nan, elev.astype(float))))
    e = np.where(mask, fill, elev).astype(float)

    gy, gx = np.gradient(e)
    slope = np.arctan(z_factor * np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)

    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)

    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0.0, 1.0)
    hs[mask] = 0.5
    return hs


def apply_terrain_colormap(
    elev: np.ndarray,
    nodata: float = -9999,
    cmap_name: str = "terrain",
    pct_clip: tuple = (2, 98),
) -> np.ndarray:
    """Normalise elevation and apply a matplotlib colourmap → RGBA (H, W, 4)."""
    mask = elev == nodata
    e = elev.astype(float)
    valid = e[~mask]
    if valid.size == 0:
        return np.zeros((*elev.shape, 4))
    lo = np.percentile(valid, pct_clip[0])
    hi = np.percentile(valid, pct_clip[1])
    norm = np.clip((e - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    norm[mask] = 0.0
    rgba = plt.get_cmap(cmap_name)(norm)
    rgba[mask, 3] = 0.0
    return rgba


def render_hillshaded_dtm(
    elev: np.ndarray,
    nodata: float = -9999,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 2.0,
    cmap_name: str = "terrain",
    hs_strength: float = 0.65,
) -> np.ndarray:
    """Terrain-coloured + hillshaded RGBA image ready for imshow."""
    hs = hillshade(elev, nodata=nodata, azimuth=azimuth,
                   altitude=altitude, z_factor=z_factor)
    rgba = apply_terrain_colormap(elev, nodata=nodata, cmap_name=cmap_name)
    rgba[..., :3] *= (1.0 - hs_strength) + hs_strength * hs[..., None]
    rgba[..., :3] = np.clip(rgba[..., :3], 0.0, 1.0)
    return rgba


# ─── Cross-section sampling ───────────────────────────────────────────────────

def sample_profile(
    elev: np.ndarray,
    extent: list,
    lat0: float,
    lon0: float,
    road_bearing_deg: float,
    half_width_m: float,
    n_samples: int,
    nodata: float = -9999,
) -> tuple:
    """
    Sample a perpendicular elevation transect through (lat0, lon0).

    The transect runs perpendicular to road_bearing_deg (CW from north),
    extending ±half_width_m from the road centreline.

    Returns
    -------
    dists_m : (n_samples,) array — distance from centreline in metres
    elevs   : (n_samples,) array — elevation; NaN where nodata or out-of-bounds
    """
    perp_bearing_rad = np.radians(road_bearing_deg + 90.0)
    perp_east = np.sin(perp_bearing_rad)
    perp_north = np.cos(perp_bearing_rad)

    dists_m = np.linspace(-half_width_m, half_width_m, n_samples)

    lats = lat0 + dists_m * perp_north / 111320.0
    lons = lon0 + dists_m * perp_east / (111320.0 * np.cos(np.radians(lat0)))

    rows, cols = elev.shape
    xmin, xmax, ymin, ymax = extent

    col_f = (lons - xmin) / (xmax - xmin) * (cols - 1)
    row_f = (ymax - lats) / (ymax - ymin) * (rows - 1)

    in_bounds = (col_f >= 0) & (col_f <= cols - 1) & (row_f >= 0) & (row_f <= rows - 1)
    col_f = np.clip(col_f, 0, cols - 1)
    row_f = np.clip(row_f, 0, rows - 1)

    nd_frac = map_coordinates(
        (elev == nodata).astype(np.float64), [row_f, col_f], order=1, mode="nearest"
    )
    sampled = map_coordinates(
        elev.astype(float), [row_f, col_f], order=1, mode="nearest"
    )
    sampled = np.where(~in_bounds | (nd_frac > 0), np.nan, sampled)
    return dists_m, sampled


# ─── Pair-line pixel → geo coordinate helper ─────────────────────────────────

def pair_to_geo(pair: dict, extent: list, shape: tuple) -> list:
    """
    Convert a ridge_pairing pair dict (pixel coords) to geographic line
    endpoints: [ ((lon0, lat0), (lon1, lat1)), ((lon0, lat0), (lon1, lat1)) ]
    """
    rows, cols = shape
    dx = (extent[1] - extent[0]) / cols
    dy = (extent[3] - extent[2]) / rows

    geo = []
    for key in ("line1", "line2"):
        (x0, y0), (x1, y1) = pair[key]
        gx0 = extent[0] + (x0 + 0.5) * dx
        gy0 = extent[2] + (rows - y0 - 0.5) * dy
        gx1 = extent[0] + (x1 + 0.5) * dx
        gy1 = extent[2] + (rows - y1 - 0.5) * dy
        geo.append(((gx0, gy0), (gx1, gy1)))
    return geo


# ─── True-positive / false-positive pair styling ──────────────────────────────

TP_COLORS = ("#ff4444", "#4499ff")   # line1 / line2 colour when confirmed (or unmarked)
FP_COLOR = "#666666"                 # both lines, dimmed grey, when marked a false positive


def confirmed_pair_indices(true_positive_idx) -> set:
    """
    Normalize a REPORT_COORDS "true_positive_pair_index" value into a set of
    confirmed indices. Accepts None, a single int, or a list/tuple/set of ints.
    """
    if true_positive_idx is None:
        return set()
    if isinstance(true_positive_idx, (list, tuple, set)):
        return set(true_positive_idx)
    return {true_positive_idx}


def confirmed_pairs_label(true_positive_idx) -> str:
    """Human-readable '#0, #2' label for one or more confirmed pair indices."""
    return ", ".join(f"#{i}" for i in sorted(confirmed_pair_indices(true_positive_idx)))


def pair_style(idx: int, true_positive_idx) -> dict:
    """
    Drawing style for detected pair `idx`, given the coordinate's configured
    true_positive_pair_index (None if not yet confirmed; an int or a
    list/tuple/set of ints if one or more pairs have been confirmed).

    Unmarked (true_positive_idx is None): every pair drawn at full strength,
    labelled with its index, so the index can be read off the figure and
    written into REPORT_COORDS as "true_positive_pair_index".

    Marked: only the confirmed indices are drawn at full strength; every
    other pair is greyed out (kept visible, not removed, for context).
    """
    confirmed = confirmed_pair_indices(true_positive_idx)
    is_unmarked = not confirmed
    is_confirmed = is_unmarked or idx in confirmed
    return {
        "colors": list(TP_COLORS) if is_confirmed else [FP_COLOR, FP_COLOR],
        "linewidth": 2.2 if is_confirmed else 1.0,
        "alpha": 0.9 if is_confirmed else 0.35,
        "zorder": 5 if is_confirmed else 1,
        "label_color": "white" if is_confirmed else "#999999",
        "label_text": (f"#{idx}" if is_unmarked else f"#{idx} ✓") if is_confirmed else f"#{idx}",
    }
