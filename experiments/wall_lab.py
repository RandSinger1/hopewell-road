"""
wall_lab.py — Run the report's terrain → ridge → pair-detection chain on a
named site with any parameter overridden, without touching parameters.py.

    from experiments.wall_lab import Params, run
    res = run("Van Voorhis Wall", Params(ridge_threshold=0.05))

Mirrors report/base_compute.py (get_terrain + get_detection) but keeps
results in memory and caches each stage separately, so a sweep over a
detection-only parameter (e.g. spacing_tolerance) doesn't redo the median
smoothing / LRM / Frangi work.

Some detector settings are read from the ``parameters`` module inside
ridge_detection.py rather than passed as arguments (HESSIAN_SIGMA,
BINARY_MIN_OBJECT_SIZE, WALL_GAP_WINDOW_RATIO, WALL_FILL_THRESHOLD,
PAIRING_ANGLE_TOLERANCE_DEG); run() patches them for the duration of the call.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache

import numpy as np

import parameters
from road_model.process_dtm import (
    getGTAndBand, get_window,
    median_smooth_windowed, local_relief_model, multiscale_ridge_enhancement,
)
from road_model.ridge_detection import detect_parallel_walls_ridge_pairing, remove_invalid_pairs
from report.report_utils import site_target_angle


# ─── Sites ────────────────────────────────────────────────────────────────────

SITES: dict[str, dict] = {c["name"]: c for c in parameters.REPORT_COORDS}


def site(name_or_coord) -> dict:
    """A REPORT_COORDS entry by name (or pass a coord dict straight through)."""
    if isinstance(name_or_coord, dict):
        return name_or_coord
    try:
        return SITES[name_or_coord]
    except KeyError:
        raise KeyError(f"{name_or_coord!r} not in REPORT_COORDS; have {list(SITES)}") from None


# ─── Parameters ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Params:
    """Every knob in the detection chain, defaulting to parameters.py."""
    # window
    size_m: float | None = None                 # half-window; None = the site's size_m
    # terrain (§4)
    median_radius: int = parameters.MEDIAN_SMOOTH_RADIUS
    lrm_radius: int = parameters.LRM_RADIUS
    lrm_method: str = parameters.LRM_METHOD
    ridge_scales: tuple = tuple(parameters.RIDGE_SCALES)
    # detection (§5)
    target_angle: float | None = None           # None = site_target_angle(coord)
    angle_tolerance: float = parameters.ANGLE_TOLERANCE
    ridge_threshold: float = parameters.RIDGE_THRESHOLD
    spacing_meters: float = parameters.SPACING_METERS
    spacing_tolerance: float = parameters.SPACING_TOLERANCE
    min_ridge_length_meters: float = parameters.MIN_RIDGE_LENGTH_METERS
    overlap_ratio: float = parameters.OVERLAP_RATIO
    gap_fill: bool = True
    # module-level settings read inside ridge_detection.py
    hessian_sigma: float = parameters.HESSIAN_SIGMA
    binary_min_object_size: int = parameters.BINARY_MIN_OBJECT_SIZE
    wall_gap_window_ratio: float = parameters.WALL_GAP_WINDOW_RATIO
    wall_fill_threshold: float = parameters.WALL_FILL_THRESHOLD
    pairing_angle_tolerance_deg: float = parameters.PAIRING_ANGLE_TOLERANCE_DEG

    def __post_init__(self):
        # lists aren't hashable; accept them for convenience
        object.__setattr__(self, "ridge_scales", tuple(self.ridge_scales))

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def diff(self, other: "Params | None" = None) -> dict:
        """Fields that differ from `other` (default: parameters.py defaults)."""
        other = other or Params()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(other, f.name)}


_PATCHED = {
    "hessian_sigma": "HESSIAN_SIGMA",
    "binary_min_object_size": "BINARY_MIN_OBJECT_SIZE",
    "wall_gap_window_ratio": "WALL_GAP_WINDOW_RATIO",
    "wall_fill_threshold": "WALL_FILL_THRESHOLD",
    "pairing_angle_tolerance_deg": "PAIRING_ANGLE_TOLERANCE_DEG",
}


@contextlib.contextmanager
def _patched_parameters(p: Params):
    saved = {attr: getattr(parameters, attr) for attr in _PATCHED.values()}
    try:
        for field_name, attr in _PATCHED.items():
            setattr(parameters, attr, getattr(p, field_name))
        yield
    finally:
        for attr, v in saved.items():
            setattr(parameters, attr, v)


# ─── Cached stages ────────────────────────────────────────────────────────────

DTM_PATH: str = parameters.REPORT_DTM_PATH


@lru_cache(maxsize=4)
def _dtm(path: str):
    gt, band, nodata, ds = getGTAndBand(path)
    return gt, band, nodata, ds          # keep ds alive or band dangles


@lru_cache(maxsize=16)
def _raw(path, lat, lon, size_m):
    gt, band, nodata, _ = _dtm(path)
    raw, extent = get_window(lat, lon, size_m, size_m, gt, band, nodata)
    if raw is None or raw.size == 0:
        raise ValueError(f"No DTM data at ({lat}, {lon}) in {path}")
    return raw, tuple(extent), nodata


@lru_cache(maxsize=32)
def _terrain(path, lat, lon, size_m, median_radius, lrm_radius, lrm_method):
    raw, extent, nodata = _raw(path, lat, lon, size_m)
    smoothed = median_smooth_windowed(raw, nodata=nodata, radius=median_radius)
    lrm = local_relief_model(smoothed, radius=lrm_radius, nodata=nodata, method=lrm_method)
    return smoothed, lrm


@lru_cache(maxsize=32)
def _ridge(path, lat, lon, size_m, median_radius, lrm_radius, lrm_method, scales):
    raw, _, nodata = _raw(path, lat, lon, size_m)
    _, lrm = _terrain(path, lat, lon, size_m, median_radius, lrm_radius, lrm_method)
    ridge = multiscale_ridge_enhancement(lrm, scales=list(scales), nodata=nodata)
    return np.where(raw == nodata, 0.0, ridge.astype(float))


def clear_cache():
    for f in (_dtm, _raw, _terrain, _ridge):
        f.cache_clear()


# ─── Run ──────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    coord: dict
    params: Params
    target_angle: float
    mode: str
    raw: np.ndarray
    extent: tuple                 # [xmin, xmax, ymin, ymax] lon/lat degrees
    nodata: float
    smoothed: np.ndarray
    lrm: np.ndarray
    ridge: np.ndarray
    pairs: list                   # after remove_invalid_pairs
    pairs_raw: list               # detector output before remove_invalid_pairs
    binary: np.ndarray
    closed: np.ndarray
    skeleton: np.ndarray
    orientation_mask: np.ndarray
    m_per_px: float = field(init=False)

    def __post_init__(self):
        lat = 0.5 * (self.extent[2] + self.extent[3])
        self.m_per_px = ((self.extent[1] - self.extent[0]) * 111_320.0 * np.cos(np.radians(lat))
                         / self.raw.shape[1])

    def pair_table(self) -> list[dict]:
        """Per-pair summary in metres."""
        rows = []
        for i, p in enumerate(self.pairs):
            l1, l2 = p["line1"], p["line2"]
            len1 = np.hypot(l1[1][0] - l1[0][0], l1[1][1] - l1[0][1]) * self.m_per_px
            len2 = np.hypot(l2[1][0] - l2[0][0], l2[1][1] - l2[0][1]) * self.m_per_px
            rows.append({"pair": i, "spacing_m": round(float(p["spacing"] * self.m_per_px), 1),
                         "len1_m": round(float(len1), 1), "len2_m": round(float(len2), 1),
                         "angle_diff": round(float(p["angle_diff"]), 2),
                         "score": round(float(p["score"]), 3)})
        return rows


def run(name_or_coord, params: Params | None = None, dtm_path: str | None = None) -> Result:
    """Full chain for one site with `params` (default: parameters.py values)."""
    p = params or Params()
    coord = site(name_or_coord)
    path = dtm_path or DTM_PATH
    lat, lon = coord["lat"], coord["lon"]
    size_m = p.size_m or coord.get("size_m", 200)

    if p.target_angle is None:
        mode, target_angle = site_target_angle(coord)
    else:
        mode, target_angle = "manual", p.target_angle

    raw, extent, nodata = _raw(path, lat, lon, size_m)
    smoothed, lrm = _terrain(path, lat, lon, size_m, p.median_radius, p.lrm_radius, p.lrm_method)
    ridge = _ridge(path, lat, lon, size_m, p.median_radius, p.lrm_radius, p.lrm_method,
                   p.ridge_scales)

    with _patched_parameters(p):
        pairs_raw, debug = detect_parallel_walls_ridge_pairing(
            ridge_map=ridge, target_angle=target_angle, extent=list(extent),
            spacing_meters=p.spacing_meters, angle_tolerance=p.angle_tolerance,
            spacing_tolerance=p.spacing_tolerance, ridge_threshold=p.ridge_threshold,
            min_ridge_length_meters=p.min_ridge_length_meters,
            overlap_ratio=p.overlap_ratio, return_debug=True, gap_fill=p.gap_fill,
        )
    binary, closed, skeleton, _labeled, orientation_mask = debug
    # remove_invalid_pairs mutates 'spacing'; give it copies so pairs_raw stays raw
    pairs = remove_invalid_pairs([dict(q) for q in pairs_raw],
                                 angle_tolerance=p.angle_tolerance, target_angle=target_angle)

    return Result(coord=coord, params=p, target_angle=target_angle, mode=mode,
                  raw=raw, extent=extent, nodata=nodata, smoothed=smoothed, lrm=lrm,
                  ridge=ridge, pairs=pairs, pairs_raw=pairs_raw, binary=binary,
                  closed=closed, skeleton=skeleton, orientation_mask=orientation_mask)


def sweep(name_or_coord, param: str, values, base: Params | None = None) -> list[Result]:
    """run() once per value of one Params field."""
    base = base or Params()
    return [run(name_or_coord, base.with_(**{param: v})) for v in values]


def grid(name_or_coord, base: Params | None = None, **axes) -> list[Result]:
    """run() over the cartesian product of several fields, e.g.
    grid(site, ridge_threshold=[.05, .1], spacing_tolerance=[.2, .3])."""
    import itertools
    base = base or Params()
    keys = list(axes)
    return [run(name_or_coord, base.with_(**dict(zip(keys, combo))))
            for combo in itertools.product(*(axes[k] for k in keys))]
