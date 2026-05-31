"""Render a GeoTIFF DTM as hillshaded RGBA XYZ map tiles."""

import hashlib
import io
import logging
import math
import os
import pickle
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
import params

_log = logging.getLogger(__name__)

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    _HAS_GDAL = True
except ImportError:
    _HAS_GDAL = False

try:
    import rasterio
    from rasterio.enums import Resampling as _Resampling
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


TILE_SIZE = params.TILE_SIZE_PX
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".dtm_cache")


# Bump whenever rendering parameters (colors, hillshade, etc.) change,
# so cached tiles are automatically invalidated.
RENDER_VER = "v4"

# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _source_hash(path: str) -> str:
    stat = os.stat(path)
    raw = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime}|{RENDER_VER}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tile coordinate math
# ---------------------------------------------------------------------------

def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in EPSG:4326 for an XYZ tile."""
    n = 2 ** z
    west  = x / n * 360.0 - 180.0
    east  = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return west, south, east, north


# ---------------------------------------------------------------------------
# GDAL overview pyramids
# ---------------------------------------------------------------------------

def build_overviews(path: str) -> None:
    """Build external GDAL overview pyramids (.ovr sidecar) for fast low-zoom reads.

    Opening the file read-only causes GDAL to write overviews to a sidecar
    <filename>.ovr rather than modifying the original file.  Subsequent
    gdal.Warp calls automatically select the best overview level, cutting
    render time by 5–50× for tiles at zoom 13–15.

    Safe to call repeatedly — returns immediately if overviews already exist.
    """
    if not _HAS_GDAL:
        return
    ds = gdal.Open(path)
    if ds is None:
        return
    if ds.GetRasterBand(1).GetOverviewCount() > 0:
        _log.info("Overviews already present: %s", os.path.basename(path))
        ds = None
        return
    ds = None

    _log.info("Building overview pyramids for %s …", os.path.basename(path))
    ds = gdal.Open(path)   # read-only → external .ovr sidecar
    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64, 128, 256])
    ds = None
    _log.info("Overviews complete: %s.ovr", os.path.basename(path))


def _lon_to_tile_x(lon: float, n: int) -> int:
    return int((lon + 180.0) / 360.0 * n)


def _lat_to_tile_y(lat: float, n: int) -> int:
    lat_r = math.radians(lat)
    return int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)


def _tile_center_lat(ty: int, n: int) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (ty + 0.5) / n))))


def _tile_center_lon(tx: int, n: int) -> float:
    return (tx + 0.5) / n * 360.0 - 180.0


def _seg_dist_m(px: float, py: float,
                ax: float, ay: float, bx: float, by: float) -> float:
    """Distance in metres from point (px,py) to segment (ax,ay)→(bx,by)."""
    dx = (bx - ax) * _LON_M_APPROX
    dy = (by - ay) * _LAT_M
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return math.sqrt(((px - ax) * _LON_M_APPROX) ** 2 + ((py - ay) * _LAT_M) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * _LON_M_APPROX * dx + (py - ay) * _LAT_M * dy) / len_sq))
    return math.sqrt(((px - ax) * _LON_M_APPROX - t * dx) ** 2 +
                     ((py - ay) * _LAT_M          - t * dy) ** 2)


_LAT_M        = params.METERS_PER_DEGREE_LAT
_LON_M_APPROX = params.METERS_PER_DEGREE_LON_AT_40N


def iter_corridor_tiles(z: int,
                        corridors: list[tuple],
                        dtm_bounds: dict):
    """
    Yield (x, y) tile indices at zoom z whose centres fall within the
    specified buffer of any supplied corridor segment.

    corridors — list of (lon1, lat1, lon2, lat2, buffer_km)
    dtm_bounds — {west, south, east, north}; tiles outside are skipped.
    """
    n = 1 << z
    seen: set = set()

    for lon1, lat1, lon2, lat2, buf_km in corridors:
        buf_m   = buf_km * 1000.0
        buf_lat = buf_m / _LAT_M
        buf_lon = buf_m / _LON_M_APPROX

        # Axis-aligned search box clamped to DTM extent
        lo_lon = max(min(lon1, lon2) - buf_lon, dtm_bounds['west'])
        hi_lon = min(max(lon1, lon2) + buf_lon, dtm_bounds['east'])
        lo_lat = max(min(lat1, lat2) - buf_lat, dtm_bounds['south'])
        hi_lat = min(max(lat1, lat2) + buf_lat, dtm_bounds['north'])
        if lo_lon >= hi_lon or lo_lat >= hi_lat:
            continue

        tx_lo = max(0, _lon_to_tile_x(lo_lon, n))
        tx_hi = min(n - 1, _lon_to_tile_x(hi_lon, n))
        ty_lo = max(0, _lat_to_tile_y(hi_lat, n))   # lat↑ → y↓
        ty_hi = min(n - 1, _lat_to_tile_y(lo_lat, n))

        for tx in range(tx_lo, tx_hi + 1):
            for ty in range(ty_lo, ty_hi + 1):
                if (tx, ty) in seen:
                    continue
                cx = _tile_center_lon(tx, n)
                cy = _tile_center_lat(ty, n)
                if _seg_dist_m(cx, cy, lon1, lat1, lon2, lat2) <= buf_m:
                    seen.add((tx, ty))
                    yield tx, ty


# ---------------------------------------------------------------------------
# DTM metadata (WGS84 bounds + global elevation range)
# ---------------------------------------------------------------------------

def read_dtm_info(path: str) -> tuple[dict, tuple[float, float]]:
    """
    Return (bounds_wgs84, (elev_lo, elev_hi)).
    Cached to disk; subsequent calls for the same file version are instant.
    """
    src_hash = _source_hash(path)
    cache_file = os.path.join(_CACHE_DIR, src_hash, "info.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    bounds = _read_bounds_wgs84(path)
    lo, hi = _read_elev_range(path)

    result = bounds, (lo, hi)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def _read_bounds_wgs84(path: str) -> dict:
    if _HAS_GDAL:
        return _bounds_wgs84_gdal(path)
    if _HAS_RASTERIO:
        return _bounds_wgs84_rasterio(path)
    raise RuntimeError("Neither GDAL nor rasterio is installed.")


def _bounds_wgs84_gdal(path: str) -> dict:
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open: {path}")
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    src_srs = ds.GetSpatialRef()

    corners = [
        (gt[0],             gt[3]),
        (gt[0] + gt[1] * w, gt[3]),
        (gt[0],             gt[3] + gt[5] * h),
        (gt[0] + gt[1] * w, gt[3] + gt[5] * h),
    ]

    if src_srs and src_srs.GetAuthorityCode(None) != '4326':
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src_srs, dst_srs)
        pts = [ct.TransformPoint(x, y)[:2] for x, y in corners]
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
    else:
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]

    ds = None
    return {'west': min(lons), 'south': min(lats), 'east': max(lons), 'north': max(lats)}


def _bounds_wgs84_rasterio(path: str) -> dict:
    from rasterio.warp import transform_bounds
    from rasterio.crs import CRS
    with rasterio.open(path) as src:
        b = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
    return {'west': b[0], 'south': b[1], 'east': b[2], 'north': b[3]}


def _read_elev_range(path: str) -> tuple[float, float]:
    """Compute 2nd–98th percentile elevation using a downsampled read."""
    elev, _, nodata = _read(path, max_dim=2000)
    if nodata is not None:
        mask = (elev == nodata) | ~np.isfinite(elev)
    else:
        mask = ~np.isfinite(elev)
    valid = elev[~mask]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [2, 98])
    return float(lo), float(hi)


def read_region_elev_range(path: str, west: float, south: float,
                            east: float, north: float) -> tuple[float, float]:
    """
    Compute 2nd–98th percentile elevation for a specific geographic region.
    Uses a 256×256 sample for speed.  Falls back to the global range if the
    region contains fewer than 10 valid pixels (e.g. mostly outside the DTM).
    """
    if _HAS_GDAL:
        opts = gdal.WarpOptions(
            format='MEM',
            dstSRS='EPSG:4326',
            outputBounds=[west, south, east, north],
            width=256,
            height=256,
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=-9999.0,
        )
        mem_ds = gdal.Warp('', path, options=opts)
        if mem_ds is None:
            return _read_elev_range(path)
        data = mem_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        mem_ds = None
        mask = (data == -9999.0) | ~np.isfinite(data)
    elif _HAS_RASTERIO:
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        dst_transform = from_bounds(west, south, east, north, 256, 256)
        dst = np.full((256, 256), np.nan, dtype=np.float64)
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=CRS.from_epsg(4326),
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
        data, mask = dst, ~np.isfinite(dst)
    else:
        raise RuntimeError("Neither GDAL nor rasterio is installed.")

    valid = data[~mask]
    if valid.size < 10:
        return _read_elev_range(path)
    lo, hi = np.percentile(valid, [2, 98])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------

def render_tile(path: str, z: int, x: int, y: int,
                elev_range: tuple[float, float],
                dtm_bounds: dict) -> bytes:
    """
    Render a TILE_SIZE×TILE_SIZE XYZ tile as PNG bytes.
    Returns a transparent tile when the DTM does not cover the requested tile.
    Tiles are cached to disk after the first render.
    """
    west, south, east, north = tile_bounds(z, x, y)

    dw = dtm_bounds['west'];  de = dtm_bounds['east']
    ds_ = dtm_bounds['south']; dn = dtm_bounds['north']
    if east <= dw or west >= de or north <= ds_ or south >= dn:
        return _empty_tile()

    src_hash = _source_hash(path)
    lo_q, hi_q = int(round(elev_range[0])), int(round(elev_range[1]))
    tile_path = os.path.join(_CACHE_DIR, src_hash, "tiles", str(z), str(x),
                             f"{y}_{lo_q}_{hi_q}.png")
    if os.path.exists(tile_path):
        with open(tile_path, "rb") as f:
            return f.read()

    if _HAS_GDAL:
        data, nodata = _warp_tile_gdal(path, west, south, east, north)
    elif _HAS_RASTERIO:
        data, nodata = _warp_tile_rasterio(path, west, south, east, north)
    else:
        raise RuntimeError("Neither GDAL nor rasterio is installed.")

    # Metres per pixel for this tile (lon direction, approximate at 40 °N)
    cell_m = (east - west) * _LON_M_APPROX / TILE_SIZE
    png_bytes = _array_to_png(data, nodata, elev_range, cell_m=cell_m)

    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
    with open(tile_path, "wb") as f:
        f.write(png_bytes)

    return png_bytes


def _warp_tile_gdal(path: str, west: float, south: float,
                    east: float, north: float) -> tuple[np.ndarray, float]:
    opts = gdal.WarpOptions(
        format='MEM',
        dstSRS='EPSG:4326',
        outputBounds=[west, south, east, north],
        width=TILE_SIZE,
        height=TILE_SIZE,
        resampleAlg=gdal.GRA_Bilinear,
        dstNodata=-9999.0,
    )
    mem_ds = gdal.Warp('', path, options=opts)
    if mem_ds is None:
        return np.full((TILE_SIZE, TILE_SIZE), np.nan), None
    data = mem_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    mem_ds = None
    return data, -9999.0


def _warp_tile_rasterio(path: str, west: float, south: float,
                        east: float, north: float) -> tuple[np.ndarray, float | None]:
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    dst_transform = from_bounds(west, south, east, north, TILE_SIZE, TILE_SIZE)
    dst_crs = CRS.from_epsg(4326)
    dst = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float64)

    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        nodata = src.nodata
    return dst, nodata


# ---------------------------------------------------------------------------
# Pixel-level rendering helpers
# ---------------------------------------------------------------------------

def _empty_tile() -> bytes:
    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _array_to_png(elev: np.ndarray, nodata_val,
                  elev_range: tuple[float, float],
                  cell_m: float = 10.0) -> bytes:
    if nodata_val is not None:
        mask = (elev == nodata_val) | ~np.isfinite(elev)
    else:
        mask = ~np.isfinite(elev)

    if np.all(mask):
        return _empty_tile()

    elev_filled = _fill_nodata(elev, mask)
    hs  = _hillshade(elev_filled, mask, cell_m=cell_m)

    # Pure grayscale hillshade — floor at 8 so shadows stay readable
    gray  = np.clip(8 + hs * 247, 0, 255).astype(np.uint8)
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    rgba  = np.dstack([gray, gray, gray, alpha])

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _fill_nodata(elev: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = elev.copy()
    if np.any(~mask):
        out[mask] = np.nanmedian(elev[~mask])
    return out


def _hillshade(elev: np.ndarray, mask: np.ndarray,
               azimuth: float = params.HILLSHADE_AZIMUTH,
               altitude: float = params.HILLSHADE_ALTITUDE,
               cell_m: float = 10.0,
               z_factor: float = params.HILLSHADE_Z_FACTOR) -> np.ndarray:
    # Standard GIS hillshade: slope = arctan(z_factor * |∇elev / cell_m|)
    # z_factor > 1 exaggerates vertical relief; cell_m converts pixels → metres.
    dy, dx = np.gradient(elev / cell_m)
    az_rad  = np.radians(360.0 - azimuth + 90.0)
    alt_rad = np.radians(altitude)
    slope   = np.arctan(z_factor * np.sqrt(dx**2 + dy**2))
    aspect  = np.arctan2(-dy, dx)
    hs = (np.sin(alt_rad) * np.cos(slope)
          + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    hs = np.clip(hs, 0.0, 1.0)
    hs[mask] = 0.5
    return hs


def _terrain_rgb(elev: np.ndarray, mask: np.ndarray,
                 lo: float = None, hi: float = None) -> np.ndarray:
    valid = elev[~mask]
    if valid.size == 0:
        return np.zeros((*elev.shape, 3), dtype=np.float32)

    if lo is None or hi is None:
        lo, hi = np.percentile(valid, [2, 98])

    span = hi - lo if hi > lo else 1.0
    t = np.clip((elev - lo) / span, 0.0, 1.0)

    # 8-stop vivid ramp: deep blue → teal → vivid green → lime → gold → orange → red → snow
    stops_t = np.array([0.00, 0.10, 0.22, 0.38, 0.54, 0.70, 0.87, 1.0])
    stops_r = np.array([   8,  18,   55, 150,  245, 255,  205, 255], dtype=np.float32)
    stops_g = np.array([  28, 108,  170, 215,  210, 130,   40, 255], dtype=np.float32)
    stops_b = np.array([ 118, 150,   70,  25,   15,   5,   15, 255], dtype=np.float32)

    flat = t.ravel()
    r = np.interp(flat, stops_t, stops_r).reshape(elev.shape)
    g = np.interp(flat, stops_t, stops_g).reshape(elev.shape)
    b = np.interp(flat, stops_t, stops_b).reshape(elev.shape)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Low-res read helper (used only for elevation range stats)
# ---------------------------------------------------------------------------

MAX_DIM = params.DTM_MAX_READ_DIM


def _read(path: str, max_dim: int):
    if _HAS_GDAL:
        return _read_gdal(path, max_dim)
    if _HAS_RASTERIO:
        return _read_rasterio(path, max_dim)
    raise RuntimeError("Neither GDAL nor rasterio is installed.")


def _read_gdal(path: str, max_dim: int):
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open: {path}")
    band = ds.GetRasterBand(1)
    h, w = ds.RasterYSize, ds.RasterXSize
    scale = min(1.0, max_dim / max(h, w, 1))
    out_h = max(1, int(h * scale))
    out_w = max(1, int(w * scale))
    data = band.ReadAsArray(0, 0, w, h, out_w, out_h).astype(np.float64)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    bounds = {
        "west":  gt[0],
        "north": gt[3],
        "east":  gt[0] + gt[1] * w,
        "south": gt[3] + gt[5] * h,
    }
    ds = None
    return data, bounds, nodata


def _read_rasterio(path: str, max_dim: int):
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_dim / max(h, w, 1))
        out_h = max(1, int(h * scale))
        out_w = max(1, int(w * scale))
        data = src.read(1, out_shape=(out_h, out_w),
                        resampling=_Resampling.bilinear).astype(np.float64)
        nodata = src.nodata
        b = src.bounds
        bounds = {"west": b.left, "south": b.bottom, "east": b.right, "north": b.top}
    return data, bounds, nodata
