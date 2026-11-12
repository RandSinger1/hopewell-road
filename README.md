# hopewell-road

LiDAR-based archaeological survey tool for detecting the remains of the Great
Hopewell Road along the
[Hopewell Road](https://en.wikipedia.org/wiki/Hopewell_tradition) corridor in Ohio.
The Hopewell culture (100 BCE – 500 CE) built an extensive network of ceremonial
earthen roads connecting major sites from Newark to Chillicothe; most of this
infrastructure is invisible at ground level but survives as subtle parallel earthen
banks detectable in high-resolution terrain data.

This project processes publicly available USGS LiDAR point-cloud data into Digital
Terrain Models (DTMs) and applies ridge / matched-filter detection algorithms to
locate parallel wall features consistent with Hopewell road geometry (~38 m
centre-to-centre spacing, bearing ~32° from north).

> The learned, statewide detector for earthworks of any shape (mounds,
> enclosures, effigies) used to live here too; it is now the sibling
> [`earthworks`](../earthworks) repo.  Both were split from `khthonlidar` and
> share its git history.  The earthworks repo reads this repo's downloaded
> tiles (`data/saved_file_lists/`, `data/loaded_tiles/`) instead of
> downloading them again, when the two are checked out side by side.

---

## Project Structure

```text
hopewell-road/
├── parameters.py            Single source of truth for all tunable settings
├── utils.py                 Unit-conversion helpers and corridor geometry
├── requirements.txt         Core pipeline dependencies
│
├── lidar/                   Step-by-step pipeline scripts
│   ├── collect_images.py    Query USGS ScienceBase for LAZ tile URLs
│   ├── generate_tiles.py    Download LAZ tiles in parallel
│   ├── pdal_pipeline.py     Denoise → ground classify (SMRF) → DTM → mosaic (merges sub-areas)
│   ├── point_io.py          Tile lookup + download for experiments/lidar_lab.py
│   ├── quick_view.py        Low-resolution DTM preview image
│   └── run_pipeline.py      One-shot runner for the full pipeline
│
├── road_model/              Feature-detection algorithms
│   ├── process_dtm.py       Shared raster helpers (median smoothing, LRM, Frangi ridge)
│   ├── ridge_detection.py   Hessian ridge + parallel-wall pairing
│   ├── road_modeler.py      Template matched-filter (NCC) detector
│   └── range_searcher.py    Tile-parallel corridor scan
│
├── app/                     Interactive map viewer (Flask + Leaflet)
│   ├── server.py
│   ├── dtm_renderer.py
│   ├── templates/index.html
│   ├── static/
│   └── requirements.txt
│
├── report/                  Automated report generator (figures + .docx per corridor leg)
├── experiments/             Parameter sweeps: wall_lab (detection), lidar_lab (SMRF/DTM),
│                            sentinel_lab (Sentinel-2 history of the sites)
│
└── data/
    ├── polygons.json         GeoJSON polygon definitions for survey areas
    └── saved_file_lists/     Cached tile URL lists (per area/sub-area)
```

> **Run all scripts from the project root** (`hopewell-road/`).  Relative data paths
> (`data/`, `data/loaded_tiles/`, etc.) are resolved from the working directory.

---

## Installation

### Automated setup (recommended)

**Windows:**

```powershell
.\setup_env.ps1
```

**Linux / macOS:**

```bash
chmod +x setup_env.sh && ./setup_env.sh
```

Both scripts run all steps below in order.  Pass optional arguments to override
the environment name or Python version:

- PowerShell: `.\setup_env.ps1 -EnvName myenv -PythonVersion 3.11`
- Bash: `./setup_env.sh myenv 3.11`

---

### Manual setup

### 1. Clone with submodules

AOI2List and laszip-python are tracked as git submodules.  Always clone with:

```bash
git clone --recurse-submodules https://github.com/<you>/hopewell-road.git
```

Or if you already have the repo:

```bash
git submodule update --init
```

### 2. Conda environment

GDAL and PDAL require native system libraries and must be installed via conda
**before** the Python packages:

```bash
conda create -n hopewell-road python=3.11
conda activate hopewell-road
conda install -c conda-forge gdal python-pdal
pip install -r requirements.txt
```

### 3. AOI2List

AOI2List has no pip-installable package; it is imported directly from the submodule.
Install its own dependency:

```bash
pip install -r AOI2List/requirements.txt
```

### 4. laszip extension

**Windows (pre-built wheel — fastest):**

```bash
pip install laszip-python/laszip-0.3.0-cp312-cp312-win_amd64.whl
```

**macOS / Linux (build from the submodule):**

```bash
pip install ./laszip-python
```

(Requires CMake ≥ 3.15 and a C++14 compiler.)

### 5. Web viewer dependencies

```bash
pip install -r app/requirements.txt
```

---

## Quick start — full pipeline

`lidar/run_pipeline.py` runs all steps in sequence and accepts `--skip-*` flags for
any step you have already completed:

```bash
# Download, process, and preview a single area
python -m lidar.run_pipeline Hopewell_Road

# Two areas merged into one DTM
python -m lidar.run_pipeline Hopewell_Road Circleville_Road

# All flight years separately (Hart Island multi-year dataset)
python -m lidar.run_pipeline Hart_Island --separate-year

# Shorthand for both road segments
python -m lidar.run_pipeline full_road

# Resume after tiles are already downloaded
python -m lidar.run_pipeline Hopewell_Road --skip-collect --skip-download
```

---

## Step-by-step usage

### 1. Collect tile metadata

```bash
python -m lidar.collect_images Hopewell_Road
```

Saves tile URL lists to `data/saved_file_lists/<area>_<n>.json`.

### 2. Download tiles

```bash
python -m lidar.generate_tiles Hopewell_Road_0
```

Downloads LAZ tiles to `data/loaded_tiles/<area>/raw_tiles/`.

### 3. Process point clouds → DTM

```bash
python -m lidar.pdal_pipeline Hopewell_Road --area-splits 15
python -m lidar.pdal_pipeline full_road --area-splits 15     # Hopewell_Road + Circleville_Road
```

Per sub-area: denoise → SMRF ground → per-tile DTMs, then warps and mosaics every
sub-area into one `merged_dtm.tif` (`full_road` → `data/loaded_tiles/Whole_Road/`).

### 4. Scan the corridor for wall pairs

```bash
python -m road_model.range_searcher --mode chillicothe     # or circleville / high_banks / polygon
```

Tiles the corridor and runs the Hessian ridge + parallel-wall pairing detector
(`road_model/ridge_detection.py`) on each tile; writes PNG overlays and
`line_overlay.json` files to `data/locations/<mode>_<w>_<h>/`.  The template
matched-filter detector (`road_model/road_modeler.py`,
`detect_parallel_walls_template`) is a library alternative with no command of
its own (settings: `parameters.py` §6).

### 5. Report

```bash
python -m report.run_report --path chillicothe       # figures per REPORT_COORDS site
python -m report.generate_report_doc                 # .docx per corridor path
```

### 6. Quick preview

```bash
python -m lidar.quick_view \
    --input data/loaded_tiles/Hopewell_Road/merged_dtm.tif \
    --step 100
```

### 7. Launch the map viewer

```bash
python -m app.server \
    --folder data/locations/600_600 \
    --dtm    data/loaded_tiles/Whole_Road/merged_dtm.tif \
    --port   5000
```

Then open **<http://127.0.0.1:5000>** in a browser.

Both `--folder` and `--dtm` are optional; you can load them from the sidebar UI
after the server starts.

---

## Configuration

All tunable parameters live in [`parameters.py`](parameters.py).  Edit that file
to change algorithm settings, detection thresholds, corridor geometry, or output
paths without touching any processing script.

Key sections:

| Section | Parameters |
| ------- | ---------- |
| §3 Point-cloud ingestion | SMRF ground classification, outlier removal, point_io access |
| §4 DTM pre-processing | LRM radius, Frangi ridge scales |
| §5 Ridge detection | `SPACING_METERS`, `ROAD_BEARING_DEG`, angle tolerance |
| §6 Template matching | NCC kernel geometry, score threshold (road_modeler defaults) |
| §7 Area scan | Scan tile size, overlap, corridor anchors, parabola coefficients |
| §8 Map viewer | Port, hillshade settings, tile renderer |
| §9 Report | Output coords, DPI, cross-section offsets |

---

## Survey Areas

| Name | Description |
| ---- | ----------- |
| `Hopewell_Road` | Primary Hopewell earthwork survey area |
| `Circleville_Road` | Circleville area, split into 10 sub-polygons |
| `Hart_Island` | Hart Island, multiple flight years (2014/2015/2021/2023) |
| `Zorro_Ranch` | Additional survey area |

Area polygon boundaries are defined in [`data/polygons.json`](data/polygons.json).
