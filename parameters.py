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
10. Earthwork dataset, model & statewide scan
                            (earthwork_builder/, earthwork_processor/, earthwork_inference/)
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

# ── Tile-list filtering (collect_images.py → lidar/tile_filter.py) ───────────

DROP_SUPERSEDED_TILES: bool = True
"""Drop tiles from an older collection when newer collections already cover
them (e.g. legacy OH_SOUTH_2006/2007 under OH_Statewide 2020/2021).  Disable
(or pass --keep-superseded) for deliberately multi-year areas like Hart_Island."""

SUPERSEDED_COVERAGE_THRESHOLD: float = 0.95
"""Fraction of a tile's AOI-clipped footprint that newer collections must
cover for the tile to be dropped.  Partially-covered older tiles are kept to
fill gaps; merge_dtms warps them first so newer data wins in overlaps."""

# ── Statistical outlier removal (filters.outlier) ────────────────────────────

OUTLIER_MEAN_K: int = 8
"""Number of nearest neighbours used to compute the local mean distance."""

OUTLIER_MULTIPLIER: float = 2.0
"""Points whose distance from the local mean exceeds this many standard
deviations are flagged as noise."""

# ── SMRF ground classification (filters.smrf) ────────────────────────────────
# Simple Morphological Filter — Pingel et al. (2013).
# Increase SMRF_WINDOW / decrease SMRF_SLOPE for flatter, more forested terrain.
#
# UNITS: PDAL applies these in the point cloud's CRS units.  The Ohio tiles are
# NAD83(2011) / Ohio South (ftUS) + NAVD88 height (ftUS), so every distance in
# this block and DTM_RESOLUTION below is in US SURVEY FEET (1 ft = 0.3048 m),
# not metres.  PDAL's own documented defaults (window 18, threshold 0.5) assume
# metres.

SMRF_CELL: float = 1.0
"""Initial raster cell size (ftUS) for the progressive morphological surface.
1.0 ft = 0.30 m."""

SMRF_WINDOW: float = 24.0
"""Maximum window radius (ftUS) for the progressive filter.  24 ft = 7.3 m.
Larger values remove wider above-ground objects (buildings, tree clumps)."""

SMRF_SLOPE: float = 0.2
"""Maximum allowed slope between adjacent ground cells (rise/run, unitless —
the same in feet or metres)."""

SMRF_THRESHOLD: float = 0.5
"""Maximum height above the provisional surface (ftUS) for a point to be
ground.  0.5 ft = 0.15 m."""

SMRF_SCALAR: float = 1.25
"""Elevation scaling factor applied to the threshold on steep cells (unitless)."""

SMRF_CUT: float = 0.0
"""Optional cut-net height (ftUS) for a pre-opening of the surface; 0 disables it."""

SMRF_RETURNS: str = "last,only"
"""LAS return types passed to the SMRF filter ('last,only' keeps last returns
and single returns, which are most likely to reach the ground)."""

# ── DTM rasterisation (writers.gdal) ─────────────────────────────────────────

DTM_RESOLUTION: float = 1.0
"""Output grid cell size in the tiles' CRS units, i.e. US survey FEET:
1.0 → 0.30 m pixels, which is the pixel size of merged_dtm.tif.  The
OH_Statewide 2020/2021 data (served via USGS; same data OGRIP distributes) is
dense enough for this: ~12–22 pts/m² measured at the Van Voorhis / Route 40
sites.

Per-tile DTMs keep elevations in feet; gdal.Warp to EPSG:4326 in merge_dtms
converts heights, so merged_dtm.tif elevations (and everything derived from
it — LRM, cross-sections) are in METRES."""

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

# UNITS: everything here and in §5 marked "pixels"/"cells" is in DTM pixels,
# and merged_dtm.tif pixels are 0.30 m (DTM_RESOLUTION = 1 ftUS), NOT 1 m.
# Multiply by 0.30 for the size on the ground.  If DTM_RESOLUTION changes,
# every pixel-based value below changes its ground size with it.

MEDIAN_SMOOTH_RADIUS: int = 5
"""Kernel half-size (pixels) for the nodata-aware median filter.
Full kernel = (2 × radius + 1)² = 11 × 11 px ≈ 3.4 m square on the ground.
Removes sub-metre noise while preserving sharp earthwork edges."""

LRM_RADIUS: int = 40
"""Neighbourhood half-size (pixels) for the Local Relief Model surface estimate.
40 px ≈ 12 m on the ground (81 px ≈ 25 m full window) — narrower than the
38–42 m wall spacing.  The LRM = elevation − local_surface highlights
micro-topography; values are in metres."""

LRM_METHOD: str = "median"
"""Surface estimator used to compute the regional baseline in the LRM.
'mean'   — box-filter mean (fast, slight blurring of sharp edges).
'median' — median filter  (slower but preserves sharp edges better,
           consistent with the upstream median-smoothing step)."""

RIDGE_SCALES: list = [1, 2, 3, 6]
"""Frangi vesselness filter scales (Gaussian σ in pixels; × 0.30 for metres,
so [1, 2, 3, 6] ≈ 0.3, 0.6, 0.9, 1.8 m).  The maximum response across all
scales is used.  Smaller σ → narrow ridges; larger σ → broad earthen banks."""

SCAN_RIDGE_SCALES: list = [4]
"""Ridge scales (σ, pixels; 4 px ≈ 1.2 m) used during the fast area scan
(range_searcher.py).  Single scale is ~4× faster than the full RIDGE_SCALES set."""

SVF_RADIUS: int = 15
"""Maximum horizon-search radius (pixels; 15 px ≈ 4.6 m) for sky-view factor."""

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
"""Gaussian smoothing σ (pixels; 2 px ≈ 0.6 m) applied before computing the
Hessian matrix for ridge-orientation estimation.  Larger values → less
sensitive to noise but blur thin ridges."""
# TODO: Should I raise this?

ANGLE_TOLERANCE: float = 15.0
"""Pixels whose Hessian orientation deviates more than ±ANGLE_TOLERANCE
degrees from TARGET_ANGLE are excluded from the binary ridge mask."""

# ── Binary thresholding & morphology ─────────────────────────────────────────

RIDGE_THRESHOLD: float = .1
"""Minimum Frangi ridge response (unitless, 0–1 — the vesselness map is
normalised per window, not an elevation) for a pixel to enter the binary
ridge mask.  Lower → more candidates, more false positives."""

MORPH_DISK_RADIUS: float = 10
"""Structuring-element radius (pixels; ≈ 3 m) for binary closing.  Currently
unused by ridge_detection.py (along-wall gap filling replaced it); only
re-exported via report/config.py."""

BINARY_MIN_OBJECT_SIZE: int = 200
"""Connected components with fewer than this many pixels are removed before
skeletonisation (remove_small_objects).  200 px ≈ 18 m² on the ground at
0.30 m pixels.  Filters isolated speckle."""

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
        "mode": "all",
        "true_positive_pair_index": 0
    },
    {
        "name": "Beaver Run",
        "lat":   39.9, "lon": -82.5, "size_m": 200,
        "note": "Proposed route, 37 m width",
        "mode": "all",
        "true_positive_pair_index": 0
    },
    {
        "name": "Route 40",
        "lat":   39.9, "lon": -82.5, "size_m": 200,
        "note": "Proposed route, 33 m width",
        "mode": "all",
        "true_positive_pair_index": 1
    },
    # {
    #     "name": "Blacklick Eastern",
    #     "lat": 39.9, "lon": -82.5, "size_m": 200,
    #     "note": "Proposed route",
    #     "mode": "chillicothe, high_banks"
    # },
    # {
    #     "name": "Route 159",
    #     "lat": 39.4, "lon": -82.9, "size_m": 200,
    #     "note": "Not visible, but worth checking out",
    #     "mode": "chillicothe"
    # },
    # {
    #     "name": "Canal Rd",
    #     "lat":   39.8, "lon": -82.5, "size_m": 200,
    #     "note": "Proposed route",
    #     "mode": "all"
    #     #"true_positive_pair_index": 0
    # },
    # {
    #     "name": "Mount Zion Farm",
    #     "lat": 39.7, "lon": -82.7, "size_m": 200,
    #     "note": "eyeballed this one",
    #     "mode": "circleville",
    # },
    # {
    #     "name": "Mount Zion Farm 2",
    #     "lat": 39.7, "lon": -82.7, "size_m": 200,
    #     "note": "eyeballed this one too"
    # },
    {
        "name": "Wilson Road Resivoir",
        "lat": 39.7, "lon": -82.7, "size_m": 200,
        "note": "Very weird",
        "mode": "circleville",
    },
    {
        "name": "HB Remnant",
        "lat":39.5, "lon": -82.7, "size_m": 200,
        "note": "Very correct looking, but faint, high bank work",
        "mode": "high_banks",
        "true_positive_pair_index": [0,1]

    },
    # {
    #     "name": "Rainbow Drive",
    #     "lat": 39.7, "lon": -82.6, "size_m": 200,
    #     "note": "probably trees",
    #     "mode": "high_banks",
    # }
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


# ═══════════════════════════════════════════════════════════════════════════════
# §10  EARTHWORK DATASET, MODEL & STATEWIDE SCAN
#      (earthwork_builder/, earthwork_processor/, earthwork_inference/)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Storage & source data ─────────────────────────────────────────────────────

EW_ROOT: Path = Path("data/earthworks")
"""Root for everything the earthwork tools write: sites/, dataset/, models/,
scan/, tile_cache/ (relative to repo root, gitignored with the rest of data/)."""

EW_SITES_FILE: Path = EW_ROOT / "sites.json"
"""List of known-earthwork coordinates the builder app loads and prepares."""

EW_TILE_CACHE: Path = EW_ROOT / "tile_cache"
"""Downloaded raw LAZ tiles shared by the builder and the scan."""

EW_COLLECTION_PREFIX: str = "OH_Statewide"
"""Only tiles from collections whose name starts with this are used (the
OSIP III 2020/2021 QL2 flights, whose vendor ground class we rely on)."""

EW_FALLBACK_COLLECTIONS: bool = True
"""Site windows that no EW_COLLECTION_PREFIX tile touches (a few spots OSIP III
skipped, e.g. Johnson's Island) use the newest non-legacy collection there
instead (point_io.fallback_collection; recorded as 'collections' in site.json).
The statewide scan's tile index is OH_Statewide only, so it never scans them."""

EW_GROUND_CLASSES: tuple = (2,)
"""LAS classes treated as bare earth.  The vendor (OGRIP) class 2 is used
as-is — no SMRF re-classification."""

EW_WORK_CRS: str = "EPSG:6346"
"""Metric CRS every earthwork grid lives in: NAD83(2011) / UTM 17N.  One CRS
for the whole state (tiles arrive in Ohio North *and* Ohio South ftUS), with
square metre pixels so rotation augmentation is geometrically exact.  West of
84°W Ohio is technically UTM 16; the 17N scale error there is ~0.1 %."""

EW_FALLBACK_SRS: str = "EPSG:6551+6360"
"""CRS assumed for a tile whose header carries none (Ohio South ftUS + NAVD88 ftUS)."""

EW_GROUND_SOURCE: str = "points"
"""Where the ground elevation grid comes from — for training sites AND the scan:
  "points" — bin the vendor class-2 points of the LAZ tiles (what the labels
             were drawn on; the scan must download every tile: tens of TB);
  "dem"    — USGS 3DEP 1 m bare-earth DEM (made from the same flights) via
             EW_DEM_SERVICE: one request per scan block, point clouds fetched
             only for candidates — orders of magnitude less download.
The two are co-registered (~4 cm median offset at Van Voorhis) but the DEM is
smoother (feature correlation ~0.74), so a model only works on the source it
was trained on.  The choice is stored in the feature spec, hence in the
dataset manifest and checkpoint; the scan always uses the checkpoint's source.
Changing it needs `prepare --force`, a new dataset and retraining."""

EW_DEM_SERVICE: str = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
"""ArcGIS ImageServer for EW_GROUND_SOURCE = "dem" (exportImage, ≤ 8000 px per side)."""

# ── Sites (earthwork_builder) ─────────────────────────────────────────────────

EW_SITE_HALF_M: float = 500.0
"""Default half-size (m) of the square window cropped around each site.  Must
comfortably exceed half the chip diagonal (EW_CHIP_PX·EW_PIXEL_M·0.71) so
rotated chips still land on data, and include empty ground around the work."""

EW_EARTHWORK_TYPES: list = ["enclosure", "mound", "wall", "road", "other"]
"""Builder 'type' choices for real earthworks.  Descriptive only — they all
train as 'earthwork' unless listed in EW_CLASSES."""

EW_NEGATIVE_TYPES: list = ["levee", "rail grade", "road embankment", "field terrace",
                           "spoil / quarry", "not an earthwork"]
"""Builder 'type' choices for look-alikes.  These outlines are never training
targets; build_dataset centres extra chips on them (hard negatives) so the
model learns what to ignore."""

EW_UNVERIFIED_TYPE: str = "unverified"
"""Type given to scan candidates imported into the builder.  A site cannot be
marked reviewed while any outline still has this type."""

EW_CANDIDATE_HALF_M: float = 300.0
"""Window half-size (m) for scan candidates imported as builder sites."""

EW_BUILDER_PORT: int = 5001
"""Port for the earthwork_builder Flask app (the map viewer uses SERVER_PORT)."""

EW_VIZ_Z_FACTOR: float = 3.0
"""Vertical exaggeration for the builder's hillshade / slope views (earthworks
are often < 1 m high on near-flat ground)."""

EW_VIZ_ALTITUDE: float = 35.0
"""Sun altitude (degrees) for the builder's multi-directional hillshade."""

EW_VIZ_LRM_RADIUS_M: float = 10.0
"""Half-size (m) of the mean surface subtracted for the builder's local-relief view."""

EW_VIZ_CLIP_PCT: float = 98.0
"""Percentile used to stretch slope and local relief to full contrast per site."""

# ── Ground grid + features (earthwork_processor) ─────────────────────────────

EW_PIXEL_M: float = 1.0
"""Ground-grid / model pixel size (metres).  Hopewell walls are 5–15 m wide
and enclosures up to ~400 m across, so 1 m keeps them all inside one chip;
the 0.30 m DTM used by road_model/ would make chips 11× larger for no gain."""

EW_MAX_FILL_M: float = 5.0
"""Grid cells with no ground point are filled from the nearest ground cell
when one is within this distance (m); larger holes (water, buildings) stay NaN."""

EW_NORMAL_RADIUS_M: float = 3.0
"""Half-size (m) of the neighbourhood whose ground points are PCA'd for the
surface normal (eigenvector of the smallest eigenvalue).  3 m → a 7×7 window
at 1 m pixels: wide enough to suppress point noise, narrow enough to keep a
low wall's flanks."""

EW_LRM_RADIUS_M: float = 15.0
"""Half-size (m) of the mean surface subtracted for the optional 'lrm' channel."""

EW_CHANNELS: tuple = ("dx", "dy", "magnitude", "lrm", "curvature")
"""Per-pixel model inputs, in order.  Available (earthwork_processor/features.py):
  dx, dy      — surface gradient ∂z/∂x (east), ∂z/∂y (north) from the PCA normal
  magnitude   — |∇z|
  lrm         — elevation minus EW_LRM_RADIUS_M mean surface (m)
  curvature   — PCA surface variation λ_min / Σλ (roughness / break of slope)
lrm and curvature are included because slope alone is dominated by hillsides
(slope std is only ~0.015 at Van Voorhis) and magnitude is derivable from
dx, dy; the model's first conv adapts to any channel count.
Changing this requires rebuilding the dataset and retraining."""

# ── Training chips (earthwork_processor/build_dataset.py) ────────────────────

EW_CHIP_PX: int = 512
"""Chip edge length in pixels (512 px × 1 m = 512 m)."""

EW_CHIPS_PER_LABEL: int = 12
"""Augmented chips generated around each labelled polygon."""

EW_CHIPS_PER_NEGATIVE: int = 6
"""Augmented chips centred on each look-alike (EW_NEGATIVE_TYPES) outline."""

EW_BACKGROUND_CHIPS_PER_SITE: int = 8
"""Extra chips per site centred at random (most contain no earthwork) — the
empty-ground examples a detector needs to learn what is *not* an earthwork."""

EW_LOOKALIKE_TRAINING: str = "ignore"
"""How look-alike outlines (EW_NEGATIVE_TYPES) are used in training:
  "hard_negative" — EW_CHIPS_PER_NEGATIVE extra chips centred on each, so the
                    model is actively taught to ignore them (fewer false positives);
  "background"    — no extra chips; a look-alike that happens to fall inside
                    another chip is ordinary unlabelled ground;
  "ignore"        — no extra chips, and every chip blanks the look-alike's
                    pixels (valid = False → channel mean) so the model gets no
                    signal about it either way.  Most permissive: look-alikes
                    near known works stay possible detections.
Changing this needs a dataset rebuild + retraining."""

EW_BACKGROUND_SITE_CHIPS: int = 16
"""Random chips per *background* site ("background": true in sites.json: a
window with no known earthwork, picked to show terrain the earthwork windows
lack).  Used instead of EW_BACKGROUND_CHIPS_PER_SITE for those sites."""

EW_BACKGROUND_MIN_SEP_M: float = 3000.0
"""add_background --random: minimum distance (m) from a random background
site to any existing site, so it never overlaps a known earthwork window."""

EW_BACKGROUND_MIN_COVER: float = 0.95
"""add_background --random: fraction of the window that OH_Statewide tiles must
cover (rejects points outside Ohio, in Lake Erie, or in coverage gaps)."""

EW_AUG_ROTATION: bool = True
"""Rotate chips by a uniform random angle (the DTM is rotated, then features
are recomputed, so the dx/dy vector channels stay consistent)."""

EW_AUG_SCALE: tuple = (0.9, 1.1)
"""Horizontal scale range.  Kept narrow on purpose: real earthwork size is a
discriminating cue, and scaling distorts it."""

EW_AUG_Z_EXAGGERATION: tuple = (0.7, 1.5)
"""Vertical scale range — mimics plough-flattened vs. well-preserved works."""

EW_AUG_NOISE_M: float = 0.02
"""σ (m) of Gaussian elevation noise added to augmented chips."""

EW_MIN_INSTANCE_PX: int = 40
"""Drop a label from a chip if fewer than this many of its pixels are visible."""

EW_VAL_FRACTION: float = 0.2
"""Fraction of *sites* (never chips — chips of one site are near-duplicates)
held out for validation.  Override per site with "split" in sites.json."""

EW_SEED: int = 0
"""Seed for chip sampling, splits and training."""

# ── Model & training (earthwork_inference/) ──────────────────────────────────

EW_CLASSES: tuple = ("earthwork",)
"""Detector classes (background is implicit).  Every labelled polygon maps to
'earthwork' unless its 'type' is itself listed here."""

EW_PRETRAINED: bool = True
"""Start from COCO-pretrained Mask R-CNN weights (ResNet-50-FPN v2)."""

EW_EPOCHS: int = 40
EW_BATCH_SIZE: int = 4
EW_LEARNING_RATE: float = 1e-4
EW_WEIGHT_DECAY: float = 1e-4

# ── Statewide scan (earthwork_inference/scan.py) ─────────────────────────────

EW_OHIO_BBOX: tuple = (-84.83, 38.40, -80.51, 41.98)
"""(min_lon, min_lat, max_lon, max_lat) of Ohio, used to grid index queries
and scan blocks.  Blocks with no OH_Statewide tiles are skipped."""

EW_INDEX_CELL_DEG: float = 0.1
"""Cell size (degrees) for gridding ScienceBase index queries; tiles are
~380 m squares, so a 0.1° cell returns ~600 OH tiles (cap is 5000)."""

EW_SCAN_BLOCK_M: float = 2048.0
"""Core size (m) of each scan block.  Tiles are only ~380 m, smaller than many
earthworks, so the scan grids ground from many tiles into one block."""

EW_SCAN_HALO_M: float = 256.0
"""Extra context (m) read around each block core.  A detection is kept only by
the block whose core contains its centre, so works up to ~2× this across are
seen whole and never reported twice."""

EW_SCAN_STRIDE_PX: int = 384
"""Sliding-window stride inside a block (chip overlap = EW_CHIP_PX − stride)."""

EW_SCORE_THRESHOLD: float = 0.5
"""Minimum detector score for a candidate to be saved."""

EW_NMS_IOU: float = 0.3
"""Box IoU above which overlapping detections from adjacent chips are merged."""

EW_MIN_CANDIDATE_M2: float = 5.0
"""Candidates smaller than this (m²) are discarded."""

EW_CANDIDATE_MARGIN_M: float = 3.0
"""Margin (m) around a candidate's outline for the saved point-cloud crop."""

EW_TILE_CACHE_GB: float = 40.0
"""Disk cap for EW_TILE_CACHE during a scan; least-recently-used tiles are
evicted (raw Ohio LAZ is tens of TB, it cannot all be kept)."""

EW_DOWNLOAD_WORKERS: int = 4
"""Concurrent tile downloads (the scan prefetches the next block's tiles).
rockyweb.usgs.gov answers bursts with HTTP 429; 4 matches generate_tiles.py."""
