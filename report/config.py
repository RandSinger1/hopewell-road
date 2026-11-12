"""
config.py — Backward-compatible shim; all values live in parameters.py.

Report scripts do ``import config as cfg`` and access names like
``cfg.DTM_PATH``.  To change any value, edit parameters.py in the repo root.
"""
from parameters import (  # noqa: F401 (re-exported as cfg.*)
    REPORT_DTM_PATH          as DTM_PATH,
    REPORT_DPI               as DPI,
    REPORT_COORDS,
    ROAD_BEARING_DEG,
    CROSS_SECTION_OFFSETS_M,
    CROSS_SECTION_HALF_WIDTH_M,
    CROSS_SECTION_N_SAMPLES,
    MEDIAN_SMOOTH_RADIUS,
    LRM_RADIUS,
    LRM_METHOD,
    RIDGE_SCALES,
    RIDGE_THRESHOLD,
    SPACING_METERS,
    ANGLE_TOLERANCE,
    MIN_RIDGE_LENGTH_METERS,
    SPACING_TOLERANCE,
    OVERLAP_RATIO,
)
