"""
params.py — Single source of truth for every tunable parameter in KhthonLiDAR.

All processing scripts, the Flask app, and the report generator import from
this file.  To change any algorithm setting, edit it here and here only.

Sections
--------
1.  Geography & coordinate-system constants
2.  Nodata sentinel
3.  Point-cloud ingestion  (pdal_pipeline.py)
4.  DTM pre-processing     (process_dtm.py)
5.  Ridge / skeleton detection  (ridge_detection.py)
6.  Template-matching detector  (road_modeler.py)
7.  Area scan               (range_searcher.py)
8.  Map viewer / tile server    (app/server.py, app/dtm_renderer.py)
9.  Report generation       (report/)
"""

from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# §1  GEOGRAPHY & COORDINATE-SYSTEM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# WGS-84 approximations used for fast flat-earth conversions throughout the
# codebase.  Accurate to within ~0.3 % for the study corridor.
METERS_PER_DEGREE_LAT: float = 111_320.0
"""Metres per degree of latitude (WGS-84 mean value)."""

METERS_PER_DEGREE_LON_AT_40N: float = 85_000.0
"""Metres per degree of longitude at ~40 °N  (≈ cos(40°) × 111 320)."""

# Approximate centre of the Hopewell Road study corridor.
# Used as the reference latitude in unit-conversion helpers.
STUDY_CENTER_LAT: float = 40.0
STUDY_CENTER_LON: float = -82.4


# ═══════════════════════════════════════════════════════════════════════════════
# §2  NODATA SENTINEL
# ═══════════════════════════════════════════════════════════════════════════════

NODATA: float = -9999.0
"""Sentinel value written to raster cells with no valid elevation data."""


# ═══════════════════════════════════════════════════════════════════════════════
# §3  POINT-CLOUD INGESTION  (pdal_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Statistical outlier removal (filters.outlier) ────────────────────────────

OUTLIER_MEAN_K: int = 8
"""Number of nearest neighbours used to compute the local mean distance."""

OUTLIER_MULTIPLIER: float = 2.0
"""Points whose distance from the local mean exceeds this many standard
deviations are flagged as noise."""

# ── SMRF ground classification (filters.smrf) ────────────────────────────────
# Simple Morphological Filter — Pingel et al. (2013).
# Increase SMRF_WINDOW / decrease SMRF_SLOPE for flatter, more forested terrain.

SMRF_CELL: float = 1.0
"""Initial raster cell size (m) for the progressive morphological surface."""

SMRF_WINDOW: float = 24.0
"""Maximum window radius (m) for the progressive filter.  Larger values
classify points on top of wide, low vegetation as ground."""

SMRF_SLOPE: float = 0.2
"""Maximum allowed slope (m/m) between adjacent ground cells."""

SMRF_THRESHOLD: float = 0.5
"""Maximum elevation above the model surface (m) for a point to be ground."""

SMRF_SCALAR: float = 1.25
"""Window-size growth factor between progressive filter iterations."""

SMRF_CUT: float = 0.0
"""Distance below the model surface; points within this band are always ground."""

SMRF_RETURNS: str = "last,only"
"""LAS return types passed to the SMRF filter ('last,only' keeps last returns
and single returns, which are most likely to reach the ground)."""

# ── DTM rasterisation (writers.gdal) ─────────────────────────────────────────

DTM_RESOLUTION: float = 1.0
"""Output grid cell size in metres.  Must match the point density; 1 m is
appropriate for the ~8 pts/m² density of OGRIP Ohio LiDAR."""

DTM_INTERP_METHOD: str = "idw"
"""PDAL/GDAL interpolation algorithm for DTM gridding.
'idw' (inverse-distance weighting) is robust for sparse returns."""

# ── Source CRS (used when DTM tiles have no embedded projection) ──────────────

SOURCE_CRS: str = "EPSG:6551"
"""Coordinate reference system of the source LiDAR tiles, used as a fallback
when the CRS cannot be read from the file metadata.  EPSG:6551 """
# ── Warp / mosaic settings (gdal.Warp) ───────────────────────────────────────

WARP_RESAMPLE: str = "bilinear"
"""GDAL resampling algorithm used when reprojecting tiles to EPSG:4326."""

WARP_NODATA: float = NODATA
"""No-data value written into the warped / merged output raster."""

WARP_CREATION_OPTS: list = [
    "COMPRESS=LZW",     # lossless compression — cuts file size ~50 %
    "TILED=YES",        # tiled layout for fast partial reads
    "BLOCKXSIZE=256",
    "BLOCKYSIZE=256",
    "BIGTIFF=YES",      # allow files > 4 GB
]
"""GDAL creation options for the merged GeoTIFF."""


# ═══════════════════════════════════════════════════════════════════════════════
# §4  DTM PRE-PROCESSING  (process_dtm.py)
# ═══════════════════════════════════════════════════════════════════════════════

MEDIAN_SMOOTH_RADIUS: int = 5
"""Kernel half-size (cells) for the nodata-aware median filter.
Full kernel = (2 × radius + 1)².  Removes sub-metre noise while
preserving sharp earthwork edges."""

LRM_RADIUS: int = 40
"""Neighbourhood half-size (cells) for the Local Relief Model surface estimate.
The LRM = elevation − local_surface highlights micro-topography."""

LRM_METHOD: str = "median"
"""Surface estimator used to compute the regional baseline in the LRM.
'mean'   — box-filter mean (fast, slight blurring of sharp edges).
'median' — median filter  (slower but preserves sharp edges better,
           consistent with the upstream median-smoothing step)."""

RIDGE_SCALES: list = [1, 2, 3, 6]
"""Frangi vesselness filter scales (σ, in pixels ≈ metres at 1 m/px).
The maximum response across all scales is used.  Smaller σ → narrow ridges;
larger σ → broad earthen banks."""

SCAN_RIDGE_SCALES: list = [4]
"""Ridge scales used during the fast area scan (range_searcher.py).
Single scale is ~4× faster than the full RIDGE_SCALES set."""

SVF_RADIUS: int = 15
"""Maximum horizon-search radius (cells) for sky-view factor computation."""

SVF_DIRECTIONS: int = 16
"""Number of azimuth directions sampled per cell for sky-view factor."""


# ═══════════════════════════════════════════════════════════════════════════════
# §5  RIDGE / SKELETON DETECTION  (ridge_detection.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Road / earthwork geometry ─────────────────────────────────────────────────

ROAD_BEARING_DEG: float = 32.0
"""Bearing of the Hopewell Road corridor (degrees clockwise from north).
Used to define the expected wall orientation and to orient cross-section
transects in the report."""

TARGET_ANGLE: float = 32.0
"""Expected wall orientation in image space (degrees counter-clockwise from
the +x axis).  Calibrated against the Van Voorhis ground-truth site."""

SPACING_METERS: float = 38.0
"""Expected centre-to-centre distance between the two parallel earthen walls
(metres).  Hopewell Road earthworks are typically 38–42 m apart."""

# ── Hessian orientation filter ────────────────────────────────────────────────

HESSIAN_SIGMA: float = 2.0
"""Gaussian smoothing σ (pixels) applied before computing the Hessian matrix
for ridge-orientation estimation.  Larger values → less sensitive to noise
but blur thin ridges."""
# TODO: Should I raise this?

ANGLE_TOLERANCE: float = 15.0
"""Pixels whose Hessian orientation deviates more than ±ANGLE_TOLERANCE
degrees from TARGET_ANGLE are excluded from the binary ridge mask."""

# ── Binary thresholding & morphology ─────────────────────────────────────────

RIDGE_THRESHOLD: float = .1
"""Minimum LRM elevation (metres above local mean surface) to include a pixel
in the binary ridge mask.  Lower → more candidates, more false positives."""

MORPH_DISK_RADIUS: float = 10
"""Structuring-element radius (pixels) for binary closing before skeletonisation.
Closes small gaps in fragmented ridge segments."""

BINARY_MIN_OBJECT_SIZE: int = 200
"""Connected components with fewer than this many pixels are removed before
skeletonisation (remove_small_objects).  Filters isolated speckle."""

WALL_GAP_WINDOW_RATIO: float = 2
"""Along-wall integration window half-length as a fraction of MIN_RIDGE_LENGTH_METERS.
window_px = WALL_GAP_WINDOW_RATIO × MIN_RIDGE_LENGTH_METERS × 2 + 1.
Larger values bridge wider orthogonal cuts; raise to 2–4 for frequent cuts."""

WALL_FILL_THRESHOLD: float = 0.1
"""Minimum along-wall pixel density (0–1) required to fill a gap.
0.3 = a gap pixel is recovered if ≥30 % of its along-wall window contains
wall evidence.  Lower → more aggressive gap filling; raise to tighten."""

# ── Line fitting & parallel-pair matching ─────────────────────────────────────

MIN_RIDGE_LENGTH_METERS: float = 15.0
"""Skeleton components shorter than this (metres) are discarded before
straight-line fitting.  Suppresses short, wandering fragments."""

PAIRING_ANGLE_TOLERANCE_DEG: float = 5.0
"""Maximum angular difference (degrees) between two fitted lines to be
considered a parallel pair.  Very tight — ensures truly parallel walls."""

SPACING_TOLERANCE: float = 0.3
"""Relative tolerance on the wall-to-wall spacing, e.g. 0.20 = ±20 %.
A detected pair whose spacing deviates more than this from SPACING_METERS
is rejected."""

OVERLAP_RATIO: float = 0.3
# TODO Maybe i could try making this 0.
"""Minimum fractional overlap required along the shared axis of two lines
before they qualify as a parallel pair.  Prevents pairing short fragments
against long lines with little spatial coincidence."""


# ═══════════════════════════════════════════════════════════════════════════════
# §6  TEMPLATE-MATCHING DETECTOR  (road_modeler.py)
# ═══════════════════════════════════════════════════════════════════════════════
# This detector uses matched-filter NCC rather than skeletonisation.
# Parameters are intentionally distinct from §5 because the two detectors
# operate on different representations (raw LRM vs ridge probability).

TMPL_SPACING_M: float = 38.0
"""Wall centre-to-centre spacing used when building the matched-filter kernel."""

TMPL_WALL_WIDTH_M: float = 2.0
"""Gaussian σ (metres) for each individual wall ridge in the kernel."""

TMPL_ANGLE_RANGE: tuple = (110, 160)
"""Search range (min, max) of wall orientations in degrees CCW from +x."""

TMPL_ANGLE_STEP: int = 1
"""Angular search increment (degrees)."""

TMPL_KERNEL_LENGTH_M: float = 90.0
"""Length of the matched-filter kernel along the wall direction (metres)."""

TMPL_THRESHOLD: float = 0.3
"""Minimum normalised cross-correlation score to accept a detection peak."""

TMPL_MIN_DISTANCE_M: float = 1.0
"""Minimum spatial separation (metres) between two accepted detection peaks."""

TMPL_METHOD: str = "rotation"
"""'rotation' — fast separable filter (recommended for large images).
'cv2_match' — compact OpenCV kernels, slightly more accurate."""


# ═══════════════════════════════════════════════════════════════════════════════
# §7  AREA SCAN  (range_searcher.py)
# ═══════════════════════════════════════════════════════════════════════════════

SCAN_DTM_PATH: str = "data/loaded_tiles/Whole_Road/merged_dtm.tif"
"""Path to the merged DTM used for the full-corridor scan (relative to repo root)."""

SCAN_TILE_SIZE_M: int = 600
"""Edge length of each square search tile (metres)."""

SCAN_OVERLAP_M: int = 100
"""Overlap between adjacent tiles (metres).  Prevents earthworks straddling
a tile boundary from being missed entirely."""

SCAN_MAX_WORKERS: int = 10
"""Maximum parallel worker processes for the tile scan."""

SCAN_POLYGON: list = [
    # (lat, lon) vertices that bound the search area.
    (39.5,  -82.9),   # Circleville
    (40.0,  -82.4),    # Newark
    (39.3,  -82.9),    # Chillicothe
]
"""Polygon vertices (lat, lon) bounding the Hopewell Road study corridor.
Tiles whose centres fall outside this polygon are skipped."""

# ── Named anchor coordinates ──────────────────────────────────────────────────

CIRCLEVILLE_COORD: tuple = (39.5,  -82.9)
"""(lat, lon) of Circleville, Ohio — southern anchor of the corridor."""

NEWARK_COORD: tuple = (40.0, -82.4)
"""(lat, lon) of Newark, Ohio — northern anchor of the corridor."""

CHILLICOTHE_COORD: tuple = (39.3,-82.9)
"""(lat, lon) of Chillicothe, Ohio — south-west anchor of the corridor."""

HIGH_BANKS_COORD: tuple = (39.2, -82.9)

# ── Corridor polynomial (circleville mode) ────────────────────────────────────

CORRIDOR_HALF_WIDTH_M: float = 5000.0
"""Corridor half-width (metres) on each side of the centreline.
Used for tile generation in range_searcher and for the map overlay polygon."""

PARABOLA_A: float = 1.391264
"""Coefficient a in the corridor parabola  y = a*(x-x0)^2 + b*(x-x0) + c."""

PARABOLA_B: float = 0.199026
"""Coefficient b in the corridor parabola."""

PARABOLA_C: float = 39.5
"""Coefficient c (latitude at x = x0) in the corridor parabola."""

PARABOLA_X_OFFSET: float = -82.9
"""Longitude offset x0 for the parabola  (= CIRCLEVILLE_COORD[1])."""


# ═══════════════════════════════════════════════════════════════════════════════
# §8  MAP VIEWER / TILE SERVER  (app/server.py, app/dtm_renderer.py)
# ═══════════════════════════════════════════════════════════════════════════════

SERVER_PORT: int = 5000
"""TCP port the Flask / Waitress server listens on."""

SERVER_THREADS: int = 8
"""Number of Waitress worker threads."""

# ── OSM road-detection (app/server.py) ───────────────────────────────────────

OSM_ROAD_THRESHOLD_M: float = 20.0
"""A detected line whose midpoint is within this distance of an OSM road
segment is flagged as a likely road artefact, not an earthwork."""

OSM_CACHE_TTL_S: int = 86_400
"""Lifetime of the local Overpass API response cache (seconds; default 24 h).
Delete data/.osm_roads_cache.json to force a fresh fetch."""

OSM_QUERY_TIMEOUT_S: int = 65
"""HTTP request timeout for Overpass API queries (seconds)."""

OSM_BBOX_MARGIN_DEG: float = 0.02
"""Degrees of padding added around the bounding box of all loaded lines
before querying the Overpass API.  Prevents edge-clip artefacts."""

# ── DTM tile renderer (app/dtm_renderer.py) ──────────────────────────────────

TILE_SIZE_PX: int = 256
"""XYZ tile size in pixels (both axes).  Standard Web Mercator tile size."""

HILLSHADE_AZIMUTH: float = 315.0
"""Light-source azimuth for DTM tile hillshading (degrees clockwise from north).
315° = north-west, the cartographic convention."""

HILLSHADE_ALTITUDE: float = 30.0
"""Light-source elevation above the horizon (degrees).  Lower values (< 45°)
increase shadow contrast and reveal subtle micro-topography."""

HILLSHADE_Z_FACTOR: float = 4.0
"""Vertical exaggeration applied to the DTM before hillshading in the tile
renderer.  Higher values make earthworks more visible at small scales."""

DTM_MAX_READ_DIM: int = 3000
"""Maximum pixels per axis when reading the DTM at reduced resolution for
global elevation-range statistics.  Caps memory use on very large mosaics."""

PREWARM_ZOOM_SCHEDULE: list = [
    # (zoom_level, corridor_buffer_km) — finer zooms with shrinking buffers
    (13, 4.0),
    (14, 3.0),
    (15, 2.0),
    (16, 1.5),
]
"""Priority pre-warm schedule: list of (zoom, buffer_km) pairs processed in
order from coarsest to finest, so the most-visible tiles are cached first."""


# ═══════════════════════════════════════════════════════════════════════════════
# §9  REPORT GENERATION  (report/)
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_DTM_PATH: str = "data/loaded_tiles/Whole_Road/merged_dtm.tif"
"""Path to the merged DTM for the report (relative to repo root).
Can differ from SCAN_DTM_PATH if a higher-resolution mosaic is available."""

REPORT_OUTPUT_DIR: Path = Path("report/output")
"""Root directory for all generated report figures (relative to repo root)."""

REPORT_DPI: int = 150
"""Resolution for saved report figures (dots per inch)."""

REPORT_COORDS: list = [
    # Each entry: name (used in filenames), lat/lon (WGS-84), size_m (half-window).
    # 'note' is displayed in the run log only; it does not affect processing.
    # Optional 'true_positive_pair_index': index (or list of indices) of the
    # detected wall pair(s) that are the real earthwork, when more than one
    # parallel-wall pair is detected. Workflow: run the report once, read the
    # pair indices off the "12. Detected Pairs" panel in pipeline_steps.png
    # (every pair is drawn and labelled #0, #1, ... on that first pass), set
    # this key to that index (e.g. 0) or a list of indices (e.g. [0, 2]) if
    # more than one pair is a confirmed earthwork, then run again — the
    # confirmed pair(s) are highlighted and all others greyed out.
    #
    # Target-angle resolution (report.report_utils.site_target_angle): by
    # default the expected wall bearing is auto-detected from whichever of
    # the three corridor segments passes closest to the site. Two ways to
    # override that per-entry:
    #   'mode': 'circleville' | 'chillicothe' | 'high_banks' — force this
    #       segment's locally expected bearing at the site's longitude
    #       (use when a site is near where legs diverge and auto-detect
    #       might pick the wrong one).
    #   'target_angle': <degrees> — force a fixed angle outright (reported
    #       as mode 'manual'). Takes priority over 'mode' if both are set.
    {
        "name": "Van Voorhis Wall",
        "lat":   40.0, "lon": -82.4, "size_m": 200,
        "note": "Ground truth — confirmed earthwork, 41 m width",
        "true_positive_pair_index": [0, 8]
    },
    {
        "name": "Beaver Run",
        "lat":   39.9, "lon": -82.5, "size_m": 200,
        "note": "Proposed route, 37 m width",
        "true_positive_pair_index": 0
    },
    {
        "name": "Route 40",
        "lat":   39.9, "lon": -82.5, "size_m": 200,
        "note": "Proposed route, 33 m width",
        "true_positive_pair_index": 1
    },
    {
        "name": "Blacklick Eastern",
        "lat": 39.9, "lon": -82.5, "size_m": 200,
        "note": "Proposed route"    
    },
    {
        "name": "Route 159",
        "lat": 39.4, "lon": -82.9, "size_m": 200,
        "note": "Not visible, but worth checking out"
    },
    {
        "name": "Canal Rd",
        "lat":   39.8, "lon": -82.5, "size_m": 200,
        "note": "Proposed route",
        "true_positive_pair_index": 0
    },
    {
        "name": "Mount Zion Farm",
        "lat": 39.7, "lon": -82.7, "size_m": 200,
        "note": "eyeballed this one"
    },
    {
        "name": "Another one near mount zion",
        "lat": 39.7, "lon": -82.7, "size_m": 200,
        "note": "eyeballed this one too"
    },
    {
        "name": "Wilson Road Resivoir",
        "lat": 39.7, "lon": -82.7, "size_m": 200,
        "note": "Very weird"
    },
    {
        "name": "HB Remnant",
        "lat":39.5, "lon": -82.7, "size_m": 200,
        "note": "Very correct looking, but faint, high bank work",
        "mode": "high_banks",
    },
    {
        "name": "Rainbow Drive",
        "lat": 39.7, "lon": -82.6, "size_m": 200,
        "note": "probably trees",
        "mode": "high_banks",
    }
    # {
    #     "name": "Examine Later 3",
    #     "lat": 39.7, "lon": -82.7, "size_m": 200,
    #     "note": "Curvy road looking thing?
    # }
    # {
    #     "name": "Examine Later",
    #     "lat": 39.6, "lon": -82.7, "size_m": 200,
    #     "note": "Too much tree cover.  Second pass"
    # },
    # {
    #     "name": "Examine Later 2",
    #     "lat": 39.5, "lon": -82.8, "size_m": 200,
    #     "note": "Too much tree cover.  Second pass"
    # },
    
    # {
    #     "name": "Route 33",
    #     "lat":   39.7, "lon": -82.6, "size_m": 200,
    #     "note": "Proposed route",
    # },
    # {
    #     "name": "Woodlot",
    #     "lat":   39.8, "lon": -82.5, "size_m": 200,
    #     "note": "Probably irrigation feature",
    # },
    # {
    #     "name": "Route 33",
    #     "lat":   39.9, "lon": -82.8, "size_m": 200,
    #     "note": "Very confident, possibly new discovery",
    # },
    # {
    #     "name": "single line near circleville",
    #     "lat": 39.6, "lon": -82.8, "size_m": 200,
    #     "note": "single line near circleville"
    # }
    # {
    #     "name": "Shapes in woodlot?",
    #     "lat": 39.6, "lon": -82.7, "size_m": 200,
    #     "note": "Shapes in woodlot? "
    # }
    # {
    #     "name": "Farm border, trees? Probably a coincidence",
    #     "lat": 39.6, "lon": -82.7, "size_m": 200,
    #     "note": "Farm border, trees? Probably a coincidence",
    # },
    # {
    #     "name": "Weird shape",
    #     "lat": 39.7, "lon": -82.7, "size_m": 200,
    #     "note": "Weird shape",
    # }
    
]
"""Coordinates to include in the report.  Add entries here to process
additional sites without touching any other file."""

# ── Cross-section parameters ──────────────────────────────────────────────────

CROSS_SECTION_OFFSETS_M: list = list(range(-50, 51, 10))
"""Along-road offsets (metres) at which perpendicular transects are sampled.
Positive = forward along ROAD_BEARING_DEG, negative = behind."""

CROSS_SECTION_HALF_WIDTH_M: float = 80.0
"""Each transect extends ±CROSS_SECTION_HALF_WIDTH_M metres either side of
the road centreline."""

CROSS_SECTION_N_SAMPLES: int = 160
"""Number of elevation samples per transect."""
