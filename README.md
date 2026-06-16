# KhthonLiDAR

LiDAR-based archaeological survey tool for detecting ancient earthworks along the
[Hopewell Road](https://en.wikipedia.org/wiki/Hopewell_tradition) corridor in Ohio.
The Hopewell culture (100 BCE – 500 CE) built an extensive network of ceremonial
earthen roads connecting major sites from Newark to Chillicothe; most of this
infrastructure is invisible at ground level but survives as subtle parallel earthen
banks detectable in high-resolution terrain data.

This project processes publicly available USGS LiDAR point-cloud data into Digital
Terrain Models (DTMs) and applies ridge / matched-filter detection algorithms to
locate parallel wall features consistent with Hopewell road geometry (~38 m
centre-to-centre spacing, bearing ~32° from north).

---

## Project Structure

```text
khthonlidar/
├── parameters.py            Single source of truth for all tunable settings
├── utils.py                 Unit-conversion helpers and corridor geometry
├── requirements.txt         Core pipeline dependencies
│
├── lidar/                   Step-by-step pipeline scripts
│   ├── collect_images.py    Query USGS ScienceBase for LAZ tile URLs
│   ├── generate_tiles.py    Download LAZ tiles in parallel
│   ├── pdal_pipeline.py     Denoise → ground classify (SMRF) → DTM → mosaic
│   ├── merge_multiple_areas.py  Merge per-area sub-mosaics
│   ├── quick_view.py        Low-resolution DTM preview image
│   └── run_pipeline.py      One-shot runner for the full pipeline
│
├── road_model/              Feature-detection algorithms
│   ├── process_dtm.py       Shared raster helpers (LRM, SVF, Frangi ridge)
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
├── report/                  Automated report generator
│
└── data/
    ├── polygons.json         GeoJSON polygon definitions for survey areas
    └── saved_file_lists/     Cached tile URL lists (per area/sub-area)
```

> **Run all scripts from the project root** (`khthonlidar/`).  Relative data paths
> (`data/`, `data/loaded_tiles/`, etc.) are resolved from the working directory.

---

## Installation

### 1. Clone with submodules

AOI2List and laszip-python are tracked as git submodules.  Always clone with:

```bash
git clone --recurse-submodules https://github.com/<you>/khthonlidar.git
```

Or if you already have the repo:

```bash
git submodule update --init
```

### 2. Conda environment

GDAL and PDAL require native system libraries and must be installed via conda
**before** the Python packages:

```bash
conda create -n khthonlidar python=3.11
conda activate khthonlidar
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

`run_pipeline.py` runs all steps in sequence and accepts `--skip-*` flags for
any step you have already completed:

```bash
# Download, process, and preview a single area
python run_pipeline.py Hopewell_Road

# Two areas merged into one DTM
python run_pipeline.py Hopewell_Road Circleville_Road

# All flight years separately (Hart Island multi-year dataset)
python run_pipeline.py Hart_Island --separate-year

# Shorthand for both road segments
python run_pipeline.py full_road

# Resume after tiles are already downloaded
python run_pipeline.py Hopewell_Road --skip-collect --skip-download
```

---

## Step-by-step usage

### 1. Collect tile metadata

```bash
python lidar/collect_images.py Hopewell_Road
```

Saves tile URL lists to `data/saved_file_lists/<area>_<n>.json`.

### 2. Download tiles

```bash
python lidar/generate_tiles.py Hopewell_Road_0
```

Downloads LAZ tiles to `data/loaded_tiles/<area>/raw_tiles/`.

### 3. Process point clouds → DTM

```bash
python lidar/pdal_pipeline.py
```

Edit the `areas` list at the bottom of the script.  Outputs `merged_dtm.tif`
per area under `data/loaded_tiles/<area>/`.

### 4. Merge sub-areas (optional)

```bash
python lidar/merge_multiple_areas.py
```

Merges `Circleville_Road_0` … `Circleville_Road_9` into
`data/loaded_tiles/Circleville_Road/merged_dtm.tif`.

### 5. Detect features

```bash
# Hessian ridge + parallel-wall pairing
python road_model/ridge_detection.py

# Template matched-filter (NCC) detector
python road_model/road_modeler.py
```

Both write PNG overlays and JSON coordinate files to `data/locations/`.

### 6. Quick preview

```bash
python lidar/quick_view.py \
    --input data/loaded_tiles/Hopewell_Road/merged_dtm.tif \
    --step 100
```

### 7. Launch the map viewer

```bash
python app/server.py \
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
| §3 Point-cloud ingestion | SMRF ground classification, outlier removal |
| §4 DTM pre-processing | LRM radius, Frangi ridge scales |
| §5 Ridge detection | `SPACING_METERS`, `ROAD_BEARING_DEG`, angle tolerance |
| §6 Template matching | NCC kernel geometry, score threshold |
| §7 Area scan | Scan tile size, corridor polygon |
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
