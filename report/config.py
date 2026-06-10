"""
config.py — Backward-compatible shim; all values now live in params.py.

Report scripts do ``import config as cfg`` and access names like
``cfg.DTM_PATH``.  To change any value, edit params.py in the repo root.
"""
from parameters import (
    REPORT_DTM_PATH          as DTM_PATH,
    REPORT_OUTPUT_DIR        as OUTPUT_DIR,
    REPORT_DPI               as DPI,
    REPORT_COORDS,
    ROAD_BEARING_DEG,
    CROSS_SECTION_OFFSETS_M,
    CROSS_SECTION_HALF_WIDTH_M,
    CROSS_SECTION_N_SAMPLES,
    NODATA,
    MEDIAN_SMOOTH_RADIUS,
    LRM_RADIUS,
    LRM_METHOD,
    RIDGE_SCALES,
    SVF_RADIUS,
    SVF_DIRECTIONS,
    RIDGE_THRESHOLD,
    MORPH_DISK_RADIUS,
    TARGET_ANGLE,
    SPACING_METERS,
    ANGLE_TOLERANCE,
    PAIRING_ANGLE_TOLERANCE_DEG,
    MIN_RIDGE_LENGTH_METERS,
    SPACING_TOLERANCE,
    OVERLAP_RATIO,
)
