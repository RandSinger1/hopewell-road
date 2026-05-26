# HopewellRoadLiDAR

LiDAR-based archaeological survey tool for detecting ancient earthworks (roads, walls) in Ohio. Processes publicly available USGS LiDAR point-cloud data into Digital Terrain Models (DTMs) and applies ridge/matched-filter detection algorithms to locate parallel wall features.

## Pipeline Overview

```
1. collect_images.py       Query USGS ScienceBase API for LiDAR tile URLs
        ↓
2. generate_tiles.py       Download LAZ tiles and cache locally
        ↓
3. pdal_pipeline.py        Denoise → ground classify (SMRF) → generate per-tile DTMs → warp mosaic to WGS84
        ↓
4. merge_multiple_areas.py Merge per-area sub-mosaics into a single DTM (optional)
        ↓
5. ridge_detection.py      Hessian ridge enhancement + parallel-wall pairing
   road_modeler.py         Template matched-filter detection (alternative)
        ↓
6. quick_view.py           Low-resolution preview of any DTM GeoTIFF
```

`process_dtm.py` provides shared raster helpers (windowed reads, LRM, SVF, Frangi ridge enhancement) used by detection scripts.

`utils.py` provides unit-conversion utilities (feet/meters/degrees/pixels) and kernel creation.

`polygons.json` stores GeoJSON polygon definitions for survey areas (Hopewell Road, Circleville Road, Hart Island, etc.).

---

## Installation

GDAL and PDAL require native system libraries and are best installed via conda before the Python packages:

```bash
conda create -n hopewellroadlidar python=3.11
conda activate hopewellroadlidar
conda install -c conda-forge gdal python-pdal
pip install -r requirements.txt
```

The local `laszip` C++ extension is pre-built as a wheel in `laszip-python/`:

```bash
pip install laszip-python/laszip-0.3.0-cp312-cp312-win_amd64.whl
```

---

## Usage

### 1. Collect tile metadata

```bash
python collect_images.py          # default area: Hart_Island
```

Saves tile URL lists to `saved_file_lists/<area>.json`.

### 2. Download tiles

```bash
python generate_tiles.py <area>   # e.g. Hopewell_Road_0
```

Downloads LAZ tiles to `loaded_tiles/<area>/raw_tiles/`.

### 3. Process point clouds → DTM

```bash
python pdal_pipeline.py
```

Edit the `areas` list at the bottom of the script. Outputs `merged_dtm.tif` per area under `loaded_tiles/<area>/`.

### 4. Merge sub-areas (optional)

```bash
python merge_multiple_areas.py
```

Merges `Circleville_Road_0` … `Circleville_Road_9` into `Circleville_Road/merged_dtm.tif`.

### 5. Detect features

```bash
python ridge_detection.py         # Hessian ridge + parallel-wall pairing
python road_modeler.py            # Template matched-filter detection
```

Both scripts write PNG overlays and JSON coordinate files to `locations/`.

### 6. Quick preview

```bash
python quick_view.py --input loaded_tiles/Hopewell_Road/merged_dtm.tif --step 100
```

---

## Survey Areas

| Name | Description |
|---|---|
| `Hopewell_Road` | Primary Hopewell earthwork survey area |
| `Circleville_Road` | Circleville area, split into 10 sub-polygons |
| `Hart_Island` | Hart Island, multiple flight years (2014/2015/2021/2023) |

Area polygon boundaries are defined in `polygons.json`.
