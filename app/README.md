# KhthonLiDAR Viewer

Interactive map viewer for LiDAR range-searcher results. Displays detected earthwork
wall pairs on top of a hillshaded DTM, with reference lines connecting the major
Hopewell ceremonial sites in Ohio.

## Quick start

```bash
# From the project root (khthonlidar/)
pip install -r app/requirements.txt   # GDAL is expected from the conda environment

python app/server.py \
    --folder data/locations/600_600 \
    --dtm    data/loaded_tiles/Whole_Road/merged_dtm.tif \
    --port   5000
```

Then open **<http://127.0.0.1:5000>** in a browser.

Both `--folder` and `--dtm` are optional — you can load them from the sidebar UI
after the server starts.

## Command-line options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--folder` | *(none)* | Relative or absolute path to a locations subfolder, e.g. `data/locations/600_600` |
| `--dtm` | *(none)* | Path to a merged DTM GeoTIFF, e.g. `data/loaded_tiles/Whole_Road/merged_dtm.tif` |
| `--port` | `5000` | Local port to serve on |

## Folder layout

```text
app/
  server.py          Flask backend (API routes + line/DTM data loading)
  dtm_renderer.py    Converts a GeoTIFF to a hillshaded RGBA PNG via GDAL
  templates/
    index.html       Single-page UI (sidebar + Leaflet map)
  static/
    map.js           Leaflet map logic, zoom-dependent line loading, reference lines
    style.css        Dark theme
  requirements.txt   Python dependencies
```

## What it reads

**Detected lines** — all `line_overlay.json` files found recursively under the chosen
`--folder`.  Each file is a JSON array of wall-pair objects:

```json
[{
  "line_0_xs": -82.4, "line_0_ys": 40.0,
  "line_0_xe": -82.4, "line_0_ye": 40.0,
  "line_0_length": 0.00110, "line_0_angle": 0.96,
  "line_1_xs": "...", "...": "..."
}]
```

Both `line_0` and `line_1` of each pair are drawn as individual segments.

**DTM** — any GeoTIFF in WGS-84 (EPSG:4326), typically a `merged_dtm.tif` produced
by `lidar/pdal_pipeline.py`.  It is rendered once at startup
(hillshade + terrain colour ramp) and served as a Leaflet image overlay.

## Map controls

| Action | Effect |
| ------ | ------ |
| Scroll / `+` / `-` | Zoom in/out |
| Click-drag | Pan |
| **`L` key** | Toggle detected lines on/off |
| **`R` key** | Toggle reference lines on/off |
| **`D` key** | Toggle DTM overlay on/off |
| Lines button (map) | Toggle detected lines (same as `L`) |

## Zoom-dependent rendering

Line density is reduced at small scales so the map stays readable:

| Zoom | Minimum line length shown |
| ---- | ------------------------- |
| < 11 | > 0.005 ° (~550 m) |
| 11 | > 0.002 ° (~220 m) |
| 12 | > 0.001 ° (~110 m) |
| 13 | > 0.0005 ° (~55 m) |
| ≥ 14 | all lines |

## Reference lines

Two thin dashed lines are always drawn over the map:

- **Blue** — straight line from the Newark Earthworks to Chillicothe / Mound City
- **Yellow** — quadratic Bézier arc from the Newark Earthworks to Circleville,
  curving northward to suggest the ceremonial alignment arc

Coordinates are taken from the project's `metadata.json` survey anchors.
