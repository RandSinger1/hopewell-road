"""
augment.py — Cut (optionally rotated / scaled / exaggerated) chips from a GroundGrid.

Augmentation is applied to the *elevation* grid and to the label outlines
with one shared affine map, and features are recomputed on the result.  So a
rotated chip's dx/dy channels are the true gradient of the rotated surface
(no hand-rotated vector fields), and masks are rasterised from exactly
transformed polygons (no nearest-neighbour mask resampling).

Chip pixel (col, row) maps to source-grid pixel coordinates by

    src = centre + scale · R(θ) · ((col, row) − c),   c = (N − 1) / 2

so scale > 1 shows more ground per chip pixel (objects look smaller).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.draw import polygon as draw_polygon

from earthwork_processor.features import FeatureSpec, compute_features

if TYPE_CHECKING:     # type hints only: training (dataset.py → rasterize) must import without GDAL (Colab)
    from lidar.ground_grid import GroundGrid


@dataclass(frozen=True)
class ChipTransform:
    center_col: float          # chip centre in source-grid *index* coordinates
    center_row: float          # (pixel centres at integers, as map_coordinates)
    angle_deg: float = 0.0     # counter-clockwise, as displayed (north up)
    scale: float = 1.0
    z_exaggeration: float = 1.0
    noise_m: float = 0.0

    def _matrix(self) -> np.ndarray:
        # Content turns CCW on screen when each chip pixel samples the source
        # at R(−θ)·offset; with rows growing downward that is [[c, −s], [s, c]].
        t = np.radians(self.angle_deg)
        c, s = np.cos(t), np.sin(t)
        return self.scale * np.array([[c, -s], [s, c]])

    def chip_to_src(self, cols: np.ndarray, rows: np.ndarray, size: int):
        h = (size - 1) / 2.0
        a = self._matrix()
        dc, dr = cols - h, rows - h
        return (self.center_col + a[0, 0] * dc + a[0, 1] * dr,
                self.center_row + a[1, 0] * dc + a[1, 1] * dr)

    def src_to_chip(self, cols: np.ndarray, rows: np.ndarray, size: int):
        h = (size - 1) / 2.0
        inv = np.linalg.inv(self._matrix())
        dc, dr = cols - self.center_col, rows - self.center_row
        return h + inv[0, 0] * dc + inv[0, 1] * dr, h + inv[1, 0] * dc + inv[1, 1] * dr


def sample_elevation(grid: GroundGrid, tf: ChipTransform, size: int,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """(size, size) elevation chip under ``tf`` (NaN outside data)."""
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float64)
    sc, sr = tf.chip_to_src(cols, rows, size)
    valid = np.isfinite(grid.z)
    zf = np.where(valid, grid.z, 0.0)
    z = map_coordinates(zf, [sr, sc], order=1, mode="constant", cval=0.0)
    ok = map_coordinates(valid.astype(np.float32), [sr, sc], order=1, mode="constant", cval=0.0) > 0.999
    z = np.where(ok, z, np.nan)
    if tf.z_exaggeration != 1.0 and ok.any():
        ref = np.nanmedian(z)
        z = ref + (z - ref) * tf.z_exaggeration
    if tf.noise_m > 0 and rng is not None:
        z = z + rng.normal(0.0, tf.noise_m, z.shape)
    return z.astype(np.float32)


def make_chip(grid: GroundGrid, tf: ChipTransform, spec: FeatureSpec, size: int,
              polygons_px: list[np.ndarray], rng: np.random.Generator | None = None):
    """Features and label outlines for one chip.

    polygons_px: outlines as (K, 2) arrays of source-grid (col, row) in the
    edge convention of GroundGrid.to_pixel (pixel i spans [i, i+1)).
    Returns (features (C, size, size), valid (size, size), outlines in chip
    (col, row), same edge convention — unclipped, in input order)."""
    pad = spec.pad_px
    big = size + 2 * pad
    # Same centre, larger canvas: the pad gives features their full neighbourhood
    z = sample_elevation(grid, tf, big, rng)
    feats, valid = compute_features(z, spec)
    feats, valid = feats[:, pad:pad + size, pad:pad + size], valid[pad:pad + size, pad:pad + size]
    out = []
    for poly in polygons_px:
        # edge → index coords, map, back to edge coords on the unpadded chip
        c, r = tf.src_to_chip(poly[:, 0] - 0.5, poly[:, 1] - 0.5, big)
        out.append(np.column_stack([c - pad + 0.5, r - pad + 0.5]).astype(np.float32))
    return feats, valid, out


def rasterize(poly_px: np.ndarray, size: int) -> np.ndarray:
    """Boolean (size, size) mask of a polygon given in pixel (col, row) coords.
    Pixel (c, r) covers [c, c+1) × [r, r+1) and is inside when its centre is
    (cv2.fillPoly would also fill every pixel the outline touches)."""
    mask = np.zeros((size, size), bool)
    rr, cc = draw_polygon(poly_px[:, 1] - 0.5, poly_px[:, 0] - 0.5, shape=mask.shape)
    mask[rr, cc] = True
    return mask


def rasterize_rings(rings: list[np.ndarray], size: int) -> np.ndarray:
    """rasterize() of an outline with holes: rings[0] is the exterior, the rest
    are interiors (e.g. an enclosure's inside) cut out of the mask."""
    mask = rasterize(rings[0], size)
    for hole in rings[1:]:
        mask &= ~rasterize(hole, size)
    return mask
