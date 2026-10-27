"""
tile_index.py — Resumable SQLite index of every OH_Statewide LAZ tile.

    python -m earthwork_inference.tile_index                  # all of Ohio
    python -m earthwork_inference.tile_index --bbox -82.6 39.9 -82.4 40.1

ScienceBase caps a query at a few thousand items and OSIP III tiles are only
~380 m squares (~800k statewide), so Ohio is queried in EW_INDEX_CELL_DEG
cells.  Finished cells are recorded, so an interrupted build resumes where it
stopped.  Tiles already in data/saved_file_lists/ are imported first, with no
network needed.

Index: EW_ROOT/scan/ohio_tiles.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

import parameters
from lidar import point_io
from lidar.tile_filter import collection_of

INDEX_PATH = parameters.EW_ROOT / "scan" / "ohio_tiles.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    tile_id TEXT PRIMARY KEY, url TEXT NOT NULL, collection TEXT,
    min_lon REAL, min_lat REAL, max_lon REAL, max_lat REAL, flight_date TEXT);
CREATE INDEX IF NOT EXISTS tiles_lon ON tiles (min_lon, max_lon);
CREATE INDEX IF NOT EXISTS tiles_lat ON tiles (min_lat, max_lat);
CREATE TABLE IF NOT EXISTS cells (
    i INTEGER, j INTEGER, n INTEGER, PRIMARY KEY (i, j));
"""


def open_index(path: Path = INDEX_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(_SCHEMA)
    return db


def _insert(db: sqlite3.Connection, tiles: list[dict]) -> int:
    rows = [(t["tile_id"], t["url"], collection_of(t), t["min_lon"], t["min_lat"],
             t["max_lon"], t["max_lat"], t.get("flight_date", "")) for t in tiles
            if t.get("min_lon") is not None]
    before = db.total_changes
    db.executemany("INSERT OR IGNORE INTO tiles VALUES (?,?,?,?,?,?,?,?)", rows)
    return db.total_changes - before


def import_saved_lists(db: sqlite3.Connection) -> int:
    n = 0
    for lst in sorted(point_io.SAVED_LISTS.glob("*.json")):
        if not lst.stem.endswith(("_old", ".unfiltered")):
            n += _insert(db, point_io.statewide_only(json.loads(lst.read_text())))
    db.commit()
    return n


def _query_cell(bbox: tuple[float, float, float, float], retries: int = 6) -> list[dict]:
    from AOI2List.src.aoi2list import (bbox_to_wkt_polygon, build_spatial_query_filter,
                                       extract_laz_tiles_with_metadata, fetch_sciencebase_items)
    filt = build_spatial_query_filter(bbox_to_wkt_polygon(*bbox))
    for attempt in range(retries):
        try:
            return extract_laz_tiles_with_metadata(fetch_sciencebase_items(filt, max_items=5000))
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  query failed ({e}); retry in {wait}s")
            time.sleep(wait)
    return []


def build_index(db: sqlite3.Connection, bbox=parameters.EW_OHIO_BBOX,
                cell_deg: float = parameters.EW_INDEX_CELL_DEG) -> None:
    # Cells sit on a global lon/lat grid, so runs over different --bbox values
    # agree on which cells are already done.
    w, s, e, n = bbox
    cols = range(int(np.floor(w / cell_deg)), int(np.ceil(e / cell_deg)))
    rows = range(int(np.floor(s / cell_deg)), int(np.ceil(n / cell_deg)))
    done = {(r["i"], r["j"]) for r in db.execute("SELECT i, j FROM cells")}
    todo = [(i, j) for j in rows for i in cols if (i, j) not in done]
    print(f"Index: {len(cols) * len(rows)} cells, {len(todo)} to query")
    for k, (i, j) in enumerate(todo, 1):
        cell = (i * cell_deg, j * cell_deg, (i + 1) * cell_deg, (j + 1) * cell_deg)
        tiles = point_io.statewide_only(_query_cell(cell))
        added = _insert(db, tiles)
        db.execute("INSERT OR REPLACE INTO cells VALUES (?,?,?)", (i, j, len(tiles)))
        db.commit()
        if added or k % 50 == 0:
            total = db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
            print(f"  [{k}/{len(todo)}] cell {i},{j}: +{added} tiles (total {total:,})")


def tiles_in_bbox(db: sqlite3.Connection, bbox: tuple[float, float, float, float]) -> list[dict]:
    w, s, e, n = bbox
    rows = db.execute("SELECT * FROM tiles WHERE min_lon <= ? AND max_lon >= ? "
                      "AND min_lat <= ? AND max_lat >= ?", (e, w, n, s))
    return [dict(r, local=None) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    default=parameters.EW_OHIO_BBOX, help="Lon/lat box to index (default: Ohio)")
    ap.add_argument("--db", type=Path, default=INDEX_PATH)
    args = ap.parse_args()
    db = open_index(args.db)
    print(f"Imported {import_saved_lists(db):,} tile(s) from saved lists")
    build_index(db, tuple(args.bbox))
    print(f"Done: {db.execute('SELECT COUNT(*) FROM tiles').fetchone()[0]:,} tiles in {args.db}")


if __name__ == "__main__":
    main()
