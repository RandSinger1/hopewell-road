"""
dem_source.py — Ground grids from the USGS 3DEP 1 m DEM instead of point clouds.

    grid = fetch_dem_grid((xmin, ymin, xmax, ymax))     # work-CRS bounds, metres

Returns the same GroundGrid that GridAccumulator builds from class-2 points,
on the same pixel grid, so everything downstream is source-agnostic.  The
3DEP ImageServer resamples to EW_WORK_CRS server-side; requests larger than
the service limit are split into pieces.

Use ``ground_grid(bounds, source, paths)`` to build a grid from whichever
source a FeatureSpec names.
"""

from __future__ import annotations

import time
import uuid

import numpy as np
from osgeo import gdal
from pyproj import CRS

import parameters
from lidar.ground_grid import GridAccumulator, GroundGrid, fill_gaps

gdal.UseExceptions()

_MAX_PX = 4000            # service allows 8000; smaller pieces fail/retry cheaper


def _export(bounds: tuple[float, float, float, float], w: int, h: int, retries: int = 5) -> np.ndarray:
    import requests

    epsg = CRS.from_user_input(parameters.EW_WORK_CRS).to_epsg()
    params = {"bbox": ",".join(f"{v:.3f}" for v in bounds), "bboxSR": epsg, "imageSR": epsg,
              "size": f"{w},{h}", "format": "tiff", "pixelType": "F32",
              "noData": parameters.NODATA, "noDataInterpretation": "esriNoDataMatchAny",
              "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    for attempt in range(retries):
        try:
            r = requests.get(f"{parameters.EW_DEM_SERVICE}/exportImage", params=params, timeout=300)
            r.raise_for_status()
            if not r.content.startswith((b"II*", b"MM\x00*")):
                raise ValueError(f"DEM service returned no TIFF: {r.content[:200]!r}")
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * 2 ** attempt)
    path = f"/vsimem/dem_{uuid.uuid4().hex}.tif"
    gdal.FileFromMemBuffer(path, r.content)
    try:
        ds = gdal.Open(path)
        z = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        ds = None
    finally:
        gdal.Unlink(path)
    z[(z <= parameters.NODATA + 1) | (z < -1000) | ~np.isfinite(z)] = np.nan
    return z


def fetch_dem_grid(bounds: tuple[float, float, float, float],
                   pixel_m: float = parameters.EW_PIXEL_M) -> GroundGrid:
    xmin, ymin, xmax, ymax = bounds
    w = int(np.ceil((xmax - xmin) / pixel_m))
    h = int(np.ceil((ymax - ymin) / pixel_m))
    z = np.full((h, w), np.nan, np.float32)
    for r0 in range(0, h, _MAX_PX):
        for c0 in range(0, w, _MAX_PX):
            ph, pw = min(_MAX_PX, h - r0), min(_MAX_PX, w - c0)
            x0, y1 = xmin + c0 * pixel_m, ymax - r0 * pixel_m
            z[r0:r0 + ph, c0:c0 + pw] = _export((x0, y1 - ph * pixel_m, x0 + pw * pixel_m, y1), pw, ph)
    return GroundGrid(fill_gaps(z, parameters.EW_MAX_FILL_M / pixel_m), xmin, ymax, pixel_m)


def ground_grid(bounds: tuple[float, float, float, float], source: str, paths=None,
                pixel_m: float = parameters.EW_PIXEL_M) -> GroundGrid:
    """Ground grid over work-CRS ``bounds`` from ``source``: "dem", or "points"
    (vendor ground class of the LAZ files in ``paths``, read one at a time)."""
    if source == "dem":
        return fetch_dem_grid(bounds, pixel_m)
    if source != "points":
        raise ValueError(f"Unknown ground source {source!r} (expected 'points' or 'dem')")
    from lidar import point_io

    acc = GridAccumulator(bounds, pixel_m)
    for p in paths or []:
        acc.add(point_io.read_tile(p, classes=parameters.EW_GROUND_CLASSES, work_bounds=bounds).xyz)
    return acc.finish()
