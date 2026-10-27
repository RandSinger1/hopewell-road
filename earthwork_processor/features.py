"""
features.py — Per-pixel surface features from a GroundGrid via local PCA.

For every grid cell, the ground cells within ±r pixels are treated as 3-D
points (x, y, z) and their 3×3 covariance is eigen-decomposed:

  * the eigenvector of the smallest eigenvalue is the surface normal n;
    the gradient follows as dx = −n_x/n_z, dy = −n_y/n_z (rise per metre
    east / north), magnitude = |∇z|;
  * λ_min / (λ_0+λ_1+λ_2) is the PCA "surface variation" — ~0 on a plane,
    larger on ridges, ditches and breaks of slope (channel 'curvature').

This is exactly the per-point PCA normal, evaluated on the gridded ground
surface instead of on raw points: the covariance moments are box-filter sums,
so the cost is O(pixels·r) instead of a k-NN query per point, which is what
makes a statewide scan feasible — and both training and the scan use this
one function, so their inputs are guaranteed to match.

Coordinate conventions: columns increase east (+x), rows increase south
(−y).  dx and dy are *vector* channels: rotating or flipping a feature stack
must also rotate / negate them (``rot90`` / ``flip`` below).  Augmentation in
build_dataset.py sidesteps this by transforming the elevation grid and
recomputing features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import correlate1d

import parameters

# How each channel transforms under rotation / reflection
CHANNEL_KIND = {"dx": "vx", "dy": "vy", "magnitude": "scalar",
                "lrm": "scalar", "curvature": "scalar"}

_EIGH_CHUNK = 1_000_000


@dataclass(frozen=True)
class FeatureSpec:
    channels: tuple = tuple(parameters.EW_CHANNELS)
    pixel_m: float = parameters.EW_PIXEL_M
    normal_radius_m: float = parameters.EW_NORMAL_RADIUS_M
    lrm_radius_m: float = parameters.EW_LRM_RADIUS_M
    ground_source: str = parameters.EW_GROUND_SOURCE   # how z was made: "points" | "dem"

    def __post_init__(self):
        unknown = set(self.channels) - set(CHANNEL_KIND)
        if unknown:
            raise ValueError(f"Unknown feature channel(s) {sorted(unknown)}; "
                             f"available: {sorted(CHANNEL_KIND)}")
        if self.ground_source not in ("points", "dem"):
            raise ValueError(f"ground_source must be 'points' or 'dem', not {self.ground_source!r}")

    @property
    def normal_radius_px(self) -> int:
        return max(1, int(round(self.normal_radius_m / self.pixel_m)))

    @property
    def lrm_radius_px(self) -> int:
        return max(1, int(round(self.lrm_radius_m / self.pixel_m)))

    @property
    def pad_px(self) -> int:
        """Context (pixels) a chip needs on each side for its features to be exact."""
        r = self.normal_radius_px
        if "lrm" in self.channels:
            r = max(r, self.lrm_radius_px)
        return r + 1

    def to_dict(self) -> dict:
        return {**asdict(self), "channels": list(self.channels)}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        # Specs written before ground_source existed were all point-derived
        return cls(**{"ground_source": "points", **d, "channels": tuple(d["channels"])})


def _sep(a: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Separable correlation: kx along columns (x), ky along rows."""
    return correlate1d(correlate1d(a, kx, axis=1, mode="constant"), ky, axis=0, mode="constant")


def _smallest_eigvec(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending, (N, 3)) and the unit eigenvector of the smallest
    one ((N, 3)) for a stack of symmetric 3×3 matrices.

    Closed form (Smith 1961 trigonometric solution + cross product of the rows
    of A − λ_min·I); ~50× faster than batched np.linalg.eigh.  Exact here
    because λ_min (vertical scatter, ~cm²) is always far below λ_mid
    (horizontal scatter, ~r²/3 m²), so the eigenvector is well separated."""
    a00, a11, a22 = a[:, 0, 0], a[:, 1, 1], a[:, 2, 2]
    a01, a02, a12 = a[:, 0, 1], a[:, 0, 2], a[:, 1, 2]
    q = (a00 + a11 + a22) / 3.0
    p1 = a01 ** 2 + a02 ** 2 + a12 ** 2
    p = np.sqrt(((a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1) / 6.0)
    ps = np.where(p > 0, p, 1.0)
    b00, b11, b22 = (a00 - q) / ps, (a11 - q) / ps, (a22 - q) / ps
    b01, b02, b12 = a01 / ps, a02 / ps, a12 / ps
    det = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
           + b02 * (b01 * b12 - b11 * b02))
    phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
    hi = q + 2.0 * p * np.cos(phi)
    lo = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    lam = np.stack([lo, 3.0 * q - hi - lo, hi], axis=1)

    m = a - lo[:, None, None] * np.eye(3)
    cands = np.stack([np.cross(m[:, 0], m[:, 1]), np.cross(m[:, 0], m[:, 2]),
                      np.cross(m[:, 1], m[:, 2])], axis=1)          # (N, 3, 3)
    norms = np.linalg.norm(cands, axis=2)
    best = norms.argmax(axis=1)
    rows = np.arange(len(a))
    vec = cands[rows, best] / np.maximum(norms[rows, best], 1e-300)[:, None]
    return lam, vec


def eigen_surface(z: np.ndarray, pixel_m: float, radius_px: int) -> dict[str, np.ndarray]:
    """Local-PCA surface descriptors of an elevation grid (NaN = no data).

    Returns dx, dy, magnitude, curvature (float32, NaN where undefined) and
    'valid' (cell has data and ≥ half its neighbourhood has data)."""
    h, w = z.shape
    valid = np.isfinite(z)
    m = valid.astype(np.float64)
    ref = np.nanmedian(z) if valid.any() else 0.0
    zc = np.where(valid, z - ref, 0.0)          # centred: keeps z² moments well-conditioned

    o = np.arange(-radius_px, radius_px + 1) * pixel_m
    one = np.ones_like(o)
    u, v = o, -o                                 # neighbour offsets in x (east) and y (north)

    s0 = _sep(m, one, one)
    ok = valid & (s0 >= 0.5 * len(o) ** 2)
    out = {k: np.full((h, w), np.nan, np.float32) for k in ("dx", "dy", "magnitude", "curvature")}
    out["valid"] = ok
    if not ok.any():
        return out

    # Raw moments Σx, Σy, Σz, Σxx, … over each neighbourhood (flattened)
    mom = {k: a.ravel() for k, a in {
        "x": _sep(m, u, one), "y": _sep(m, one, v), "z": _sep(zc, one, one),
        "xx": _sep(m, u * u, one), "yy": _sep(m, one, v * v), "zz": _sep(zc * zc, one, one),
        "xy": _sep(m, u, v), "xz": _sep(zc, u, one), "yz": _sep(zc, one, v)}.items()}
    s0 = s0.ravel()

    idx = np.flatnonzero(ok)
    for start in range(0, idx.size, _EIGH_CHUNK):
        sel = idx[start:start + _EIGH_CHUNK]
        inv = 1.0 / s0[sel]
        e = {k: mom[k][sel] * inv for k in mom}
        cov = np.empty((sel.size, 3, 3))
        cov[:, 0, 0] = e["xx"] - e["x"] ** 2
        cov[:, 1, 1] = e["yy"] - e["y"] ** 2
        cov[:, 2, 2] = e["zz"] - e["z"] ** 2
        cov[:, 0, 1] = cov[:, 1, 0] = e["xy"] - e["x"] * e["y"]
        cov[:, 0, 2] = cov[:, 2, 0] = e["xz"] - e["x"] * e["z"]
        cov[:, 1, 2] = cov[:, 2, 1] = e["yz"] - e["y"] * e["z"]
        lam, n = _smallest_eigvec(cov)
        n = n * np.sign(n[:, 2:3] + 1e-12)       # orient normals upward
        nz = np.maximum(n[:, 2], 1e-6)
        dx, dy = -n[:, 0] / nz, -n[:, 1] / nz
        lam = np.clip(lam, 0.0, None)
        r, c = np.divmod(sel, w)
        out["dx"][r, c] = dx
        out["dy"][r, c] = dy
        out["magnitude"][r, c] = np.hypot(dx, dy)
        out["curvature"][r, c] = lam[:, 0] / np.maximum(lam.sum(axis=1), 1e-12)
    return out


def local_relief(z: np.ndarray, radius_px: int) -> np.ndarray:
    """z minus its nodata-aware box mean (separable, so O(r) per pixel —
    road_model's local_relief_model uses a dense 2-D kernel, too slow statewide)."""
    valid = np.isfinite(z)
    one = np.ones(2 * radius_px + 1)
    zc = np.where(valid, z, 0.0)
    s0 = _sep(valid.astype(np.float64), one, one)
    with np.errstate(invalid="ignore", divide="ignore"):
        lrm = z - _sep(zc, one, one) / s0
    return lrm.astype(np.float32)


def compute_features(z: np.ndarray, spec: FeatureSpec = FeatureSpec()) -> tuple[np.ndarray, np.ndarray]:
    """(C, H, W) float32 feature stack in ``spec.channels`` order, plus the
    (H, W) validity mask.  Invalid cells are 0 in every channel."""
    surf = eigen_surface(z, spec.pixel_m, spec.normal_radius_px)
    if "lrm" in spec.channels:
        surf["lrm"] = local_relief(z, spec.lrm_radius_px)
    valid = surf["valid"]
    feats = np.stack([surf[c] for c in spec.channels]).astype(np.float32)
    feats[:, ~valid] = 0.0
    return np.nan_to_num(feats, copy=False), valid


# ─── Symmetry transforms for feature stacks ──────────────────────────────────

def _vector_index(channels) -> tuple[int | None, int | None]:
    kinds = [CHANNEL_KIND[c] for c in channels]
    ix = kinds.index("vx") if "vx" in kinds else None
    iy = kinds.index("vy") if "vy" in kinds else None
    return ix, iy


def rot90(feats: np.ndarray, k: int, channels) -> np.ndarray:
    """Rotate a (C, H, W) stack by k×90° counter-clockwise (as displayed, north
    up), rotating the (dx, dy) vector with it."""
    k %= 4
    out = np.rot90(feats, k, axes=(1, 2)).copy()
    ix, iy = _vector_index(channels)
    if k and (ix is None) != (iy is None):
        raise ValueError("rot90 needs both dx and dy channels (or neither)")
    for _ in range(k if ix is not None else 0):
        gx, gy = out[ix].copy(), out[iy].copy()
        out[ix], out[iy] = -gy, gx               # CCW: (x, y) → (−y, x)
    return out


def flip(feats: np.ndarray, axis: str, channels) -> np.ndarray:
    """Mirror a (C, H, W) stack: axis='lr' (east↔west) or 'ud' (north↔south)."""
    ix, iy = _vector_index(channels)
    if axis == "lr":
        out = feats[:, :, ::-1].copy()
        if ix is not None:
            out[ix] *= -1
    elif axis == "ud":
        out = feats[:, ::-1, :].copy()
        if iy is not None:
            out[iy] *= -1
    else:
        raise ValueError(f"axis must be 'lr' or 'ud', not {axis!r}")
    return out
