"""
scan.py — Scan every OH_Statewide tile for earthworks, saving candidates as it goes.

    python -m earthwork_inference.tile_index                       # once, builds the tile index
    python -m earthwork_inference.scan --run ohio_v1               # resumable; Ctrl-C is safe
    python -m earthwork_inference.scan --run test --bbox -82.50 40.00 -82.40 40.08 --max-blocks 4
    python -m earthwork_inference.scan --run ohio_v1 --export      # merge candidates into one file

Tiles (~380 m) are smaller than many earthworks, so the state is cut into
EW_SCAN_BLOCK_M blocks in EW_WORK_CRS, each with an EW_SCAN_HALO_M halo, and
only blocks the tile index covers (i.e. inside Ohio) are scanned.  A candidate
is kept only by the block whose core contains its centroid: each is reported
once, and seen whole if it is up to ~2×halo across.  Blocks run in serpentine
order while the next block's ground grid is built in the background.

The ground grid comes from the *checkpoint's* ground source (FeatureSpec):
  "points" — every overlapping tile is downloaded and its class 2 binned;
  "dem"    — one 3DEP 1 m DEM request per block; LAZ tiles are downloaded
             only around candidates, for their saved point clouds.
The tile cache is capped at EW_TILE_CACHE_GB (least-recently-used evicted).

Run state (EW_ROOT/scan/<run>/):
    run.json                      checkpoint + thresholds the run was started with
    scan.sqlite                   per-block status + every candidate
    candidates/<id>/outline.shp   candidate outline (WGS84) with score / label / area
    candidates/<id>/points.laz    every point (all classes, native CRS) around it
    candidates.shp / .geojson     written by --export

Review candidates in the builder with `python -m earthwork_builder.import_candidates --run <run>`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.ops import transform as shp_transform

import parameters
from earthwork_builder.labels import write_vector
from earthwork_inference.tile_index import INDEX_PATH, open_index, tiles_in_bbox
from lidar import point_io
from lidar.dem_source import ground_grid
from lidar.ground_grid import GroundGrid

SCAN_ROOT = parameters.EW_ROOT / "scan"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
    bi INTEGER, bj INTEGER, status TEXT, tiles INTEGER, candidates INTEGER,
    seconds REAL, error TEXT, updated REAL, PRIMARY KEY (bi, bj));
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, bi INTEGER, bj INTEGER, score REAL, label TEXT,
    area_m2 REAL, lon REAL, lat REAL, wkt TEXT, dir TEXT);
"""


# ─── Blocks ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Block:
    bi: int
    bj: int
    core: tuple[float, float, float, float]      # work CRS xmin, ymin, xmax, ymax

    @property
    def padded(self) -> tuple[float, float, float, float]:
        h = parameters.EW_SCAN_HALO_M
        x0, y0, x1, y1 = self.core
        return x0 - h, y0 - h, x1 + h, y1 + h


def lonlat_bbox(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bounds
    lon, lat = point_io.work_to_lonlat(np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]))
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def state_blocks(bbox=parameters.EW_OHIO_BBOX) -> list[Block]:
    """Serpentine-ordered blocks covering a lon/lat box (block grid anchored at
    the work-CRS origin, so any sub-box reuses the statewide block ids)."""
    w, s, e, n = bbox
    lons, lats = np.linspace(w, e, 9), np.linspace(s, n, 9)
    gx, gy = np.meshgrid(lons, lats)
    x, y = point_io.lonlat_to_work(gx.ravel(), gy.ravel())
    size = parameters.EW_SCAN_BLOCK_M
    i0, i1 = int(np.floor(x.min() / size)), int(np.ceil(x.max() / size))
    j0, j1 = int(np.floor(y.min() / size)), int(np.ceil(y.max() / size))
    blocks = []
    for row, bj in enumerate(range(j1 - 1, j0 - 1, -1)):            # north → south
        cols = range(i0, i1) if row % 2 == 0 else range(i1 - 1, i0 - 1, -1)
        blocks += [Block(bi, bj, (bi * size, bj * size, (bi + 1) * size, (bj + 1) * size)) for bi in cols]
    return blocks


# ─── Tile cache ───────────────────────────────────────────────────────────────

class TileCache:
    """Download-on-demand LAZ cache with an LRU size cap."""

    def __init__(self, root: Path = parameters.EW_TILE_CACHE, cap_gb: float = parameters.EW_TILE_CACHE_GB,
                 workers: int = parameters.EW_DOWNLOAD_WORKERS):
        self.root, self.cap = Path(root), cap_gb * 1e9
        self.pool = ThreadPoolExecutor(workers)

    def path_for(self, tile: dict) -> Path:
        return Path(tile["local"]) if tile.get("local") else point_io.cache_path(tile, self.root)

    def ensure(self, tiles: list[dict]) -> list[Path]:
        paths = list(self.pool.map(lambda t: point_io.fetch_tile(t, self.root), tiles))
        now = time.time()
        for p in paths:
            if p.parent == self.root:
                os.utime(p, (now, now))           # mark recently used
        return paths

    def evict(self, protect: set[Path]) -> None:
        files = sorted(self.root.glob("*.laz"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.cap:
                break
            if p in protect:
                continue
            total -= p.stat().st_size
            p.unlink(missing_ok=True)


# ─── Scan ─────────────────────────────────────────────────────────────────────

def _open_run(run_dir: Path, config: dict) -> sqlite3.Connection:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "run.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old.get("checkpoint") != config["checkpoint"]:
            raise SystemExit(f"Run {run_dir.name} was started with {old.get('checkpoint')}; "
                             f"use a new --run name for a different model")
    else:
        cfg_path.write_text(json.dumps(config, indent=2))
    db = sqlite3.connect(run_dir / "scan.sqlite")
    db.row_factory = sqlite3.Row
    db.executescript(_SCHEMA)
    return db


def _set_block(db, b: Block, status: str, **kw) -> None:
    db.execute("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)",
               (b.bi, b.bj, status, kw.get("tiles"), kw.get("candidates"),
                kw.get("seconds"), kw.get("error"), time.time()))
    db.commit()


def _clear_block(db, b: Block, run_dir: Path) -> None:
    db.execute("DELETE FROM candidates WHERE bi = ? AND bj = ?", (b.bi, b.bj))
    for d in (run_dir / "candidates").glob(f"{b.bi}_{b.bj}_*"):
        shutil.rmtree(d, ignore_errors=True)


def _to_lonlat(geom):
    return shp_transform(lambda x, y, z=None: point_io.work_to_lonlat(x, y), geom)


def load_block_grid(b: Block, tiles: list[dict], source: str, cache: TileCache) -> GroundGrid:
    """Ground grid over the padded block (runs in the prefetch thread)."""
    paths = cache.ensure(tiles) if source == "points" else None
    return ground_grid(b.padded, source, paths)


def save_candidate(run_dir: Path, cid: str, cand, tiles: list[dict], cache: TileCache) -> dict:
    """Write outline.shp + points.laz for one candidate; returns its DB row.
    Only the tiles under the candidate are needed (downloaded if not cached)."""
    d = run_dir / "candidates" / cid
    outline_ll = _to_lonlat(cand.outline)
    props = {"id": cid, "score": round(cand.score, 4), "label": cand.label,
             "area_m2": round(cand.area_m2, 1), "status": "new"}
    write_vector(d / "outline.shp", [(outline_ll, props)])

    m = parameters.EW_CANDIDATE_MARGIN_M
    x0, y0, x1, y1 = cand.outline.bounds
    crop = (x0 - m, y0 - m, x1 + m, y1 + m)
    w, s, e, n = lonlat_bbox(crop)
    near = [t for t in tiles
            if t["min_lon"] <= e and t["max_lon"] >= w and t["min_lat"] <= n and t["max_lat"] >= s]
    cloud = point_io.read_tiles(cache.ensure(near), work_bounds=crop)
    if len(cloud):
        point_io.write_laz(cloud, d / "points.laz")
    c = outline_ll.centroid
    return {"id": cid, "score": cand.score, "label": cand.label, "area_m2": cand.area_m2,
            "lon": c.x, "lat": c.y, "wkt": outline_ll.wkt, "dir": str(d)}


def scan_block(b: Block, grid: GroundGrid, tiles: list[dict], detector, run_dir: Path,
               cache: TileCache) -> list[dict]:
    if not np.isfinite(grid.z).any():
        return []
    rows = []
    x0, y0, x1, y1 = b.core
    for cand in detector.detect(grid):
        c = cand.outline.centroid
        if not (x0 <= c.x < x1 and y0 <= c.y < y1):
            continue                      # its centroid belongs to a neighbouring block
        row = save_candidate(run_dir, f"{b.bi}_{b.bj}_{len(rows)}", cand, tiles, cache)
        rows.append({**row, "bi": b.bi, "bj": b.bj})
    return rows


def run_scan(run: str, checkpoint: Path, bbox, max_blocks: int | None, retry_failed: bool,
             device: str | None) -> None:
    from earthwork_inference.detector import Detector   # torch import only when scanning

    index = open_index(INDEX_PATH)
    if not index.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]:
        raise SystemExit("Tile index is empty - run `python -m earthwork_inference.tile_index` first")
    detector = Detector(checkpoint, device=device)
    source = detector.spec.ground_source          # the source the model was trained on
    run_dir = SCAN_ROOT / run
    config = {"checkpoint": str(checkpoint), "ground_source": source,
              "score_threshold": parameters.EW_SCORE_THRESHOLD,
              "block_m": parameters.EW_SCAN_BLOCK_M, "halo_m": parameters.EW_SCAN_HALO_M,
              "stride_px": parameters.EW_SCAN_STRIDE_PX}
    db = _open_run(run_dir, config)
    skip = {"done", "empty"} | (set() if retry_failed else {"failed"})
    finished = {(r["bi"], r["bj"]) for r in db.execute("SELECT bi, bj, status FROM blocks")
                if r["status"] in skip}
    todo = [b for b in state_blocks(bbox) if (b.bi, b.bj) not in finished]
    if max_blocks:
        todo = todo[:max_blocks]
    print(f"Run {run}: {len(todo)} block(s) to scan, ground={source}, device={detector.device}")

    cache = TileCache()
    prefetch = ThreadPoolExecutor(1)
    local = point_io.local_tile_paths()           # reuse tiles already under data/loaded_tiles

    def block_tiles(b: Block) -> list[dict]:
        tiles = tiles_in_bbox(index, lonlat_bbox(b.padded))
        for t in tiles:
            t["local"] = local.get(t["tile_id"])
        return tiles

    pending: dict[Block, tuple[list[dict], Future]] = {}

    def queue(b: Block) -> None:
        tiles = block_tiles(b)
        fut = prefetch.submit(load_block_grid, b, tiles, source, cache) if tiles else None
        pending[b] = (tiles, fut)

    try:
        for k, b in enumerate(todo):
            if b not in pending:
                queue(b)
            tiles, fut = pending.pop(b)
            if k + 1 < len(todo):
                queue(todo[k + 1])                    # build the next block's grid meanwhile
            if not tiles:
                _set_block(db, b, "empty", tiles=0)   # outside Ohio / no OH_Statewide coverage
                continue
            t0 = time.time()
            try:
                grid = fut.result()
                _clear_block(db, b, run_dir)          # leftovers of an earlier failed attempt
                rows = scan_block(b, grid, tiles, detector, run_dir, cache)
                db.executemany("INSERT OR REPLACE INTO candidates VALUES "
                               "(:id,:bi,:bj,:score,:label,:area_m2,:lon,:lat,:wkt,:dir)", rows)
                _set_block(db, b, "done", tiles=len(tiles), candidates=len(rows),
                           seconds=round(time.time() - t0, 1))
                print(f"[{k + 1}/{len(todo)}] block {b.bi},{b.bj}: {len(tiles)} tiles, "
                      f"{len(rows)} candidate(s), {time.time() - t0:.0f}s")
            except Exception as e:
                _set_block(db, b, "failed", tiles=len(tiles), error=f"{type(e).__name__}: {e}")
                print(f"[{k + 1}/{len(todo)}] block {b.bi},{b.bj} FAILED: {e}")
            # Never evict the next block's tiles, finished downloading or not
            cache.evict({cache.path_for(t) for nxt, _ in pending.values() for t in nxt})
    except KeyboardInterrupt:
        print("Interrupted - finished blocks are saved; re-run the same command to resume.")
    finally:
        prefetch.shutdown(wait=False, cancel_futures=True)


# ─── Export ───────────────────────────────────────────────────────────────────

def export(run: str, min_score: float = 0.0) -> None:
    run_dir = SCAN_ROOT / run
    db = sqlite3.connect(run_dir / "scan.sqlite")
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM candidates WHERE score >= ? ORDER BY score DESC", (min_score,)).fetchall()
    feats = [(shapely_wkt.loads(r["wkt"]),
              {"id": r["id"], "score": round(r["score"], 4), "label": r["label"],
               "area_m2": round(r["area_m2"], 1), "status": "new"}) for r in rows]
    for ext in (".shp", ".geojson"):
        write_vector(run_dir / f"candidates{ext}", feats)
    stats = dict(db.execute("SELECT status, COUNT(*) FROM blocks GROUP BY status").fetchall())
    print(f"{len(rows)} candidate(s) -> {run_dir / 'candidates.shp'}   blocks: {stats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Run name (folder under data/earthworks/scan/)")
    ap.add_argument("--checkpoint", type=Path, default=parameters.EW_ROOT / "models" / "default" / "best.pt")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    default=parameters.EW_OHIO_BBOX, help="Lon/lat box to scan (default: Ohio)")
    ap.add_argument("--max-blocks", type=int, help="Stop after this many blocks (for testing)")
    ap.add_argument("--retry-failed", action="store_true", help="Re-scan blocks that failed before")
    ap.add_argument("--device", help="torch device (default: cuda if available)")
    ap.add_argument("--export", action="store_true", help="Write candidates.shp/.geojson and exit")
    ap.add_argument("--min-score", type=float, default=0.0, help="--export: score cut-off")
    args = ap.parse_args()
    if args.export:
        export(args.run, args.min_score)
    else:
        run_scan(args.run, args.checkpoint, tuple(args.bbox), args.max_blocks, args.retry_failed, args.device)


if __name__ == "__main__":
    main()
