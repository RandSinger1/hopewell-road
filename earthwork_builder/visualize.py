"""
visualize.py — Direction-independent terrain views of a site for labelling.

A single-light hillshade hides any wall that runs parallel to the light.
These views make earthworks readable at every orientation:

  combo  — (default) mdh darkened by slope and tinted by local relief (VAT-style);
  mdh    — multi-directional hillshade, contrast-stretched (unstretched it is
           flat grey: averaging lights evens out the shading): lights at 225/270/315/360°, each pixel
           weighted towards the light that hits its slope most obliquely
           (Mark 1992, as in `gdaldem hillshade -multidirectional`);
  slope  — slope magnitude, steep = dark; no light direction at all;
  lrm    — local relief (elevation minus EW_VIZ_LRM_RADIUS_M mean surface),
           diverging blue (hollow) → white → red (bump);

Each view is computed once per site from ground.tif and cached next to it as
an RGBA GeoTIFF in EW_WORK_CRS (viz_<mode>_<VIZ_VER>_<ground_token>.tif, so a
rebuilt ground.tif always gets fresh views); ``render_tile`` warps it into
Web-Mercator XYZ tiles, so outlines drawn over it stay aligned.
"""

from __future__ import annotations

import hashlib
import io
import math
import threading
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from osgeo import gdal, osr
from PIL import Image
from pyproj import CRS

import parameters
from earthwork_processor.features import local_relief
from lidar.ground_grid import GroundGrid, fill_gaps

gdal.UseExceptions()

VIZ_VER = "v2"          # bump when the rendering changes, to invalidate cached views
MODES = {"combo": "Combined", "mdh": "Hillshade, all directions", "slope": "Slope",
         "lrm": "Local relief"}             # first = default view
TILE = 256
_MERC = 20037508.342789244
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


# ─── Terrain views (float arrays in [0, 1]) ──────────────────────────────────

def _gradient(z: np.ndarray, pixel_m: float, z_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """∂z/∂x (east) and ∂z/∂y (north), exaggerated by z_factor."""
    gy, gx = np.gradient(z.astype(np.float64) * z_factor, pixel_m)
    return gx, -gy                                   # rows run south


def multidirectional_hillshade(z, pixel_m, z_factor=parameters.EW_VIZ_Z_FACTOR,
                               altitude=parameters.EW_VIZ_ALTITUDE) -> np.ndarray:
    dzdx, dzdy = _gradient(z, pixel_m, z_factor)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    # Downslope direction as a compass azimuth (clockwise from north)
    aspect = np.arctan2(-dzdx, -dzdy)
    zen = math.radians(90.0 - altitude)
    shade = np.zeros_like(slope)
    weight = np.zeros_like(slope)
    for az_deg in (225.0, 270.0, 315.0, 360.0):
        az = math.radians(az_deg)
        # Light from azimuth az: cos(az - aspect) > 0 when the slope faces the light
        hs = math.cos(zen) * np.cos(slope) + math.sin(zen) * np.sin(slope) * np.cos(az - aspect)
        w = np.sin(aspect - az) ** 2                 # favour lights that graze the slope
        shade += w * np.clip(hs, 0, 1)
        weight += w
    flat = weight < 1e-9                             # perfectly flat: every weight vanishes
    out = np.where(flat, math.cos(zen), shade / np.where(flat, 1.0, weight))
    return np.clip(out, 0.0, 1.0)


def _stretch(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def slope_view(z, pixel_m, z_factor=parameters.EW_VIZ_Z_FACTOR, pct=parameters.EW_VIZ_CLIP_PCT) -> np.ndarray:
    dzdx, dzdy = _gradient(z, pixel_m, z_factor)
    s = np.hypot(dzdx, dzdy)
    hi = max(np.percentile(s, pct), 1e-6)
    return 1.0 - np.clip(s / hi, 0.0, 1.0)            # steep = dark


def relief_view(z, pixel_m, radius_m=parameters.EW_VIZ_LRM_RADIUS_M,
                pct=parameters.EW_VIZ_CLIP_PCT) -> np.ndarray:
    """Local relief scaled to [0, 1] with 0.5 = level, symmetric about zero."""
    lrm = local_relief(z, max(1, int(round(radius_m / pixel_m))))
    lrm = np.nan_to_num(lrm)
    hi = max(np.percentile(np.abs(lrm), pct), 1e-3)
    return 0.5 + 0.5 * np.clip(lrm / hi, -1.0, 1.0)


def render_rgb(z: np.ndarray, pixel_m: float, mode: str) -> np.ndarray:
    """(H, W, 3) float RGB in [0, 1] for one view of an elevation grid."""
    if mode == "mdh":
        return np.repeat(_stretch(multidirectional_hillshade(z, pixel_m))[..., None], 3, axis=2)
    if mode == "slope":
        return np.repeat(slope_view(z, pixel_m)[..., None], 3, axis=2)
    if mode == "lrm":
        return colormaps["RdBu_r"](relief_view(z, pixel_m))[..., :3]
    if mode == "combo":
        shade = 0.25 + 0.75 * _stretch(multidirectional_hillshade(z, pixel_m))
        dark = slope_view(z, pixel_m)
        base = shade * (0.45 + 0.55 * dark)           # slope darkens edges from any side
        tint = colormaps["RdBu_r"](relief_view(z, pixel_m))[..., :3]
        rgb = base[..., None] * (0.6 + 0.4 * tint / np.maximum(tint.mean(axis=2, keepdims=True), 1e-6))
        return np.clip(rgb, 0.0, 1.0)
    raise ValueError(f"unknown view {mode!r}; choose from {sorted(MODES)}")


# ─── Cached RGBA GeoTIFF per site and view ───────────────────────────────────

def ground_token(ground_path: Path) -> str:
    """Identifies one version of ground.tif (size + mtime_ns): any rebuild, even one
    that leaves an older mtime (copies, sync clients), gets a new token."""
    st = Path(ground_path).stat()
    return hashlib.sha1(f"{st.st_size}|{st.st_mtime_ns}|{VIZ_VER}".encode()).hexdigest()[:10]


def view_path(ground_path: Path, mode: str) -> Path:
    return Path(ground_path).parent / f"viz_{mode}_{VIZ_VER}_{ground_token(ground_path)}.tif"


def ensure_view(ground_path: Path, mode: str) -> Path:
    """The site's cached view of exactly this ground.tif, built on first use
    (views of earlier ground.tif versions are deleted then)."""
    out = view_path(ground_path, mode)
    key = str(out)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:                                        # tile requests arrive in parallel
        if out.exists():
            return out
        for old in out.parent.glob(f"viz_{mode}_*.tif"):
            try:
                old.unlink()
            except OSError:
                pass                                  # still open by a tile render; next build retries
        grid = GroundGrid.from_geotiff(ground_path)
        valid = np.isfinite(grid.z)
        z = fill_gaps(grid.z, 1e9) if not valid.all() else grid.z
        rgb = render_rgb(z, grid.pixel_m, mode)
        rgba = np.dstack([rgb, valid.astype(float)])
        data = np.round(rgba * 255).astype(np.uint8)

        tmp = out.with_suffix(".tmp.tif")
        ds = gdal.GetDriverByName("GTiff").Create(str(tmp), data.shape[1], data.shape[0], 4, gdal.GDT_Byte,
                                                  options=["COMPRESS=DEFLATE", "TILED=YES", "PHOTOMETRIC=RGB",
                                                           "ALPHA=YES"])
        ds.SetGeoTransform(grid.geotransform())
        srs = osr.SpatialReference()
        srs.ImportFromWkt(CRS.from_user_input(grid.crs).to_wkt())
        ds.SetProjection(srs.ExportToWkt())
        for i in range(4):
            ds.GetRasterBand(i + 1).WriteArray(data[..., i])
        ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        ds.FlushCache()
        ds = None
        tmp.replace(out)
        return out


# ─── XYZ tiles ────────────────────────────────────────────────────────────────

def _tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    size = 2 * _MERC / (2 ** z)
    west = -_MERC + x * size
    north = _MERC - y * size
    return west, north - size, west + size, north


def empty_tile() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def render_tile(view_tif: Path, z: int, x: int, y: int) -> bytes:
    """A 256 px Web-Mercator PNG tile of a cached view (transparent outside the site)."""
    ds = gdal.Warp("", str(view_tif), format="MEM", dstSRS="EPSG:3857",
                   outputBounds=_tile_bounds_3857(z, x, y), width=TILE, height=TILE,
                   resampleAlg=gdal.GRA_Bilinear if z <= 19 else gdal.GRA_Cubic,
                   dstAlpha=False)
    arr = np.dstack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(4)]).astype(np.uint8)
    ds = None
    if not arr[..., 3].any():
        return empty_tile()
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG", optimize=False)
    return buf.getvalue()
