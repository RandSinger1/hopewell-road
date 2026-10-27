"""
ground_grid.py — Bin ground points into a regular metric elevation grid.

The earthwork tools work on this grid rather than on merged_dtm.tif: it is
built straight from the vendor ground class, lives in ``EW_WORK_CRS`` with
square ``EW_PIXEL_M`` pixels (merged_dtm.tif is EPSG:4326 with non-square
0.30 m cells), and can be accumulated tile by tile so a statewide scan never
holds more than one tile's points in memory.

Row 0 is the north edge; ``x0, y0`` are the work-CRS coordinates of the
grid's top-left corner.  Elevations are metres, NaN = no data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from pyproj import CRS
from scipy.ndimage import distance_transform_edt

import parameters

gdal.UseExceptions()


@dataclass
class GroundGrid:
    z: np.ndarray                     # (H, W) float32 metres, NaN = no data
    x0: float                         # work-CRS x of the left edge
    y0: float                         # work-CRS y of the top edge
    pixel_m: float = parameters.EW_PIXEL_M
    crs: str = parameters.EW_WORK_CRS

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in the work CRS."""
        h, w = self.z.shape
        return self.x0, self.y0 - h * self.pixel_m, self.x0 + w * self.pixel_m, self.y0

    def to_pixel(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Work-CRS coordinates → fractional (col, row); pixel centres are at +0.5."""
        return (np.asarray(x) - self.x0) / self.pixel_m, (self.y0 - np.asarray(y)) / self.pixel_m

    def to_world(self, col, row) -> tuple[np.ndarray, np.ndarray]:
        return self.x0 + np.asarray(col) * self.pixel_m, self.y0 - np.asarray(row) * self.pixel_m

    def crop(self, col0: int, row0: int, w: int, h: int) -> "GroundGrid":
        x, y = self.to_world(col0, row0)
        return GroundGrid(self.z[row0:row0 + h, col0:col0 + w], float(x), float(y),
                          self.pixel_m, self.crs)

    # ── GeoTIFF IO ────────────────────────────────────────────────────────────

    def geotransform(self) -> tuple:
        return (self.x0, self.pixel_m, 0.0, self.y0, 0.0, -self.pixel_m)

    def to_geotiff(self, path: Path | str, bands: np.ndarray | None = None,
                   band_names: list[str] | None = None) -> Path:
        """Write the elevation grid — or ``bands`` (C, H, W) on the same grid."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.z[None] if bands is None else bands
        nodata = parameters.NODATA
        ds = gdal.GetDriverByName("GTiff").Create(
            str(path), data.shape[2], data.shape[1], data.shape[0], gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES"])
        ds.SetGeoTransform(self.geotransform())
        srs = osr.SpatialReference()
        srs.ImportFromWkt(CRS.from_user_input(self.crs).to_wkt())
        ds.SetProjection(srs.ExportToWkt())
        for i, band in enumerate(data):
            b = ds.GetRasterBand(i + 1)
            b.WriteArray(np.where(np.isfinite(band), band, nodata).astype(np.float32))
            b.SetNoDataValue(nodata)
            if band_names:
                b.SetDescription(band_names[i])
        ds.FlushCache()
        ds = None
        return path

    @classmethod
    def from_geotiff(cls, path: Path | str) -> "GroundGrid":
        ds = gdal.Open(str(path))
        gt = ds.GetGeoTransform()
        if abs(gt[1] + gt[5]) > 1e-9 or gt[2] or gt[4]:
            raise ValueError(f"{path}: expected a north-up grid with square pixels")
        band = ds.GetRasterBand(1)
        z = band.ReadAsArray().astype(np.float32)
        nd = band.GetNoDataValue()
        if nd is not None:
            z[z == nd] = np.nan
        crs = CRS.from_wkt(ds.GetProjection())
        epsg = crs.to_epsg()
        return cls(z, gt[0], gt[3], gt[1], f"EPSG:{epsg}" if epsg else crs.to_wkt())


class GridAccumulator:
    """Mean-bin points into a grid incrementally:

        acc = GridAccumulator(bounds)
        for tile in tiles: acc.add(read_tile(tile, classes=(2,)).xyz)
        grid = acc.finish()
    """

    def __init__(self, bounds: tuple[float, float, float, float],
                 pixel_m: float = parameters.EW_PIXEL_M, crs: str = parameters.EW_WORK_CRS):
        xmin, ymin, xmax, ymax = bounds
        self.x0, self.y0, self.pixel_m, self.crs = xmin, ymax, pixel_m, crs
        self.w = int(np.ceil((xmax - xmin) / pixel_m))
        self.h = int(np.ceil((ymax - ymin) / pixel_m))
        self._sum = np.zeros(self.h * self.w, np.float64)
        self._cnt = np.zeros(self.h * self.w, np.int32)

    def add(self, xyz: np.ndarray) -> None:
        col = np.floor((xyz[:, 0] - self.x0) / self.pixel_m).astype(np.int64)
        row = np.floor((self.y0 - xyz[:, 1]) / self.pixel_m).astype(np.int64)
        ok = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        idx = row[ok] * self.w + col[ok]
        self._sum += np.bincount(idx, weights=xyz[ok, 2], minlength=self._sum.size)
        self._cnt += np.bincount(idx, minlength=self._cnt.size).astype(np.int32)

    @property
    def count(self) -> np.ndarray:
        return self._cnt.reshape(self.h, self.w)

    def finish(self, max_fill_m: float = parameters.EW_MAX_FILL_M) -> GroundGrid:
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (self._sum / self._cnt).reshape(self.h, self.w).astype(np.float32)
        return GroundGrid(fill_gaps(z, max_fill_m / self.pixel_m), self.x0, self.y0,
                          self.pixel_m, self.crs)


def fill_gaps(z: np.ndarray, max_px: float) -> np.ndarray:
    """Fill NaN cells from their nearest valid cell when it is ≤ ``max_px`` away."""
    hole = ~np.isfinite(z)
    if not hole.any() or hole.all():
        return z
    dist, (ri, ci) = distance_transform_edt(hole, return_indices=True)
    out = z[ri, ci]
    out[dist > max_px] = np.nan
    return out
