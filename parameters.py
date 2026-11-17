"""
parameters.py — Single source of truth for every tunable parameter in the
hopewell-road repo (Great Hopewell Road detection from LiDAR).

All processing scripts, the Flask app, and the report generator import from
this file.  To change any algorithm setting, edit it here and here only.

Sections
--------
1.  Geography & coordinate-system constants
2.  Nodata sentinel
3.  Point-cloud ingestion  (pdal_pipeline.py)
4.  DTM pre-processing     (process_dtm.py)
5.  Ridge / skeleton detection  (ridge_detection.py)
6.  Area scan               (range_searcher.py)
7.  Map viewer / tile server    (app/server.py, app/dtm_renderer.py)
8.  Report generation       (report/)

The statewide earthwork detector (formerly §10 here) lives in the sibling
ohio-earthwork-detector repo, with its own parameters.py.
"""



# ═══════════════════════════════════════════════════════════════════════════════
# §1  GEOGRAPHY & COORDINATE-SYSTEM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# WGS-84 approximations used for fast flat-earth conversions throughout the
# codebase.  Accurate to within ~0.3 % for the study corridor.
METERS_PER_DEGREE_LAT: float = 111_320.0
"""Metres per degree of latitude (WGS-84 mean value)."""

METERS_PER_DEGREE_LON_AT_40N: float = 85_000.0
"""Metres per degree of longitude at ~40 °N  (≈ cos(40°) × 111 320)."""


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
dense enough for this: 12.5 / 21.8 pts/m² (all returns), ~8 ground pts/m², at
the Route 40 / Van Voorhis sites; across the 98 earthwork sites the ground
(class 2) density has a median of ~8 pts/m² (middle half 6–14).

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

# ── Point access (lidar/point_io.py — used by experiments/lidar_lab.py) ───────

POINT_WORK_CRS: str = "EPSG:6346"
"""Metric CRS in which point_io compares a LAZ file's header extent with its listing
(NAD83(2011) / UTM 17N; tiles arrive in Ohio North *or* South ftUS)."""

POINT_FALLBACK_SRS: str = "EPSG:6551+6360"
"""CRS assumed for a LAZ tile whose header carries none (Ohio South ftUS + NAVD88 ftUS)."""

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
# §6  AREA SCAN  (range_searcher.py)
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

# Survey/candidate site coordinates (REPORT_COORDS) live in private_coords.py next
# to this file.  It is gitignored and must never be committed: exact site locations
# stay off git and GitHub.  Without it, private_coords_example.py (tracked) supplies
# an empty REPORT_COORDS.  The corridor anchors below are well-known public
# earthworks and may stay exact in git.

def _load_private_coords():
    import importlib.util
    from pathlib import Path as _Path
    here = _Path(__file__).resolve().parent
    path = here / "private_coords.py"
    if not path.exists():
        path = here / "private_coords_example.py"
    spec = importlib.util.spec_from_file_location("_private_coords", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path.name


_PRIVATE, PRIVATE_COORDS_FILE = _load_private_coords()
"""Which file REPORT_COORDS came from ("private_coords.py" = the real list)."""


# ── Named anchor coordinates ──────────────────────────────────────────────────

CIRCLEVILLE_COORD: tuple = (39.599460,  -82.948154)
"""(lat, lon) southern anchor of the corridor, at Circleville."""

NEWARK_COORD: tuple = (40.052548, -82.444544)
"""(lat, lon) northern anchor of the corridor, at Newark."""

CHILLICOTHE_COORD: tuple = (39.3864135,-82.9824134)
"""(lat, lon) south-west anchor of the corridor, at Chillicothe."""

HIGH_BANKS_COORD: tuple = (39.28886498, -82.9220224)
"""(lat, lon) southern anchor of the high_banks leg."""

# ── Corridor polynomial (circleville mode) ────────────────────────────────────

CORRIDOR_HALF_WIDTH_M: float = 5000.0
"""Corridor half-width (metres) on each side of the centreline.
Used for tile generation in range_searcher and for the map overlay polygon."""

PARABOLA_A: float = 1.391264
"""Coefficient a in the corridor parabola  y = a*(x-x0)^2 + b*(x-x0) + c."""

PARABOLA_B: float = 0.199026
"""Coefficient b in the corridor parabola."""

PARABOLA_C: float = 39.599460
"""Coefficient c (latitude at x = x0) in the corridor parabola."""

PARABOLA_X_OFFSET: float = -82.9459666
"""Longitude offset x0 for the parabola  (= CIRCLEVILLE_COORD[1])."""


# ═══════════════════════════════════════════════════════════════════════════════
# §7  MAP VIEWER / TILE SERVER  (app/server.py, app/dtm_renderer.py)
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
# §8  REPORT GENERATION  (report/)
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_DTM_PATH: str = "data/loaded_tiles/Whole_Road/merged_dtm.tif"
"""Path to the merged DTM for the report (relative to repo root).
Can differ from SCAN_DTM_PATH if a higher-resolution mosaic is available."""


REPORT_DPI: int = 150
"""Resolution for saved report figures (dots per inch)."""

REPORT_MAP_SHOW_LATLON: bool = False
"""Draw latitude/longitude tick labels on the corridor overview maps
(report/overview_map.py).  Off by default: the maps place candidate sites to
~0.01°, and exact site locations are not published (see private_coords.py)."""

REPORT_COORDS: list = _PRIVATE.REPORT_COORDS
"""Coordinates to include in the report: edit REPORT_COORDS in private_coords.py
(gitignored; entry format documented there) to process
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
