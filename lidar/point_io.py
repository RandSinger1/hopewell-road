"""
point_io.py — Find and download USGS LAZ tiles (used by experiments/lidar_lab.py).

Tiles come from the saved lists (``saved_list_tiles``: data/saved_file_lists +
already-downloaded raw files under data/loaded_tiles) and are fetched with
``fetch_tile``, which downloads atomically (.part → rename), backs off on
HTTP 429 and checks every cached or local file's header extent against its
listing (``extent_matches``: ScienceBase lists some tile ids twice, a full tile
and a ~95 m sliver with the same filename).  Extents are compared in
``POINT_WORK_CRS``; LAZ headers are read with PDAL.

Trimmed copy of the earthworks repo's lidar/point_io.py, which also reads and
crops points into its metric work CRS: port lookup/download fixes to both.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import numpy as np
import pdal
from pyproj import CRS, Transformer

import parameters
from lidar.tile_filter import tile_file_stem

SAVED_LISTS = Path("data/saved_file_lists")
LOADED_TILES = Path("data/loaded_tiles")


# ─── Tile lookup ──────────────────────────────────────────────────────────────

def _intersects(t: dict, bbox: tuple[float, float, float, float]) -> bool:
    w, s, e, n = bbox
    return t["min_lon"] <= e and t["max_lon"] >= w and t["min_lat"] <= n and t["max_lat"] >= s


def lonlat_bbox(lat: float, lon: float, half_m: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a square of half-size ``half_m`` around a point."""
    dlat = half_m / parameters.METERS_PER_DEGREE_LAT
    dlon = half_m / (parameters.METERS_PER_DEGREE_LAT * np.cos(np.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def saved_list_tiles(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Every tile in data/saved_file_lists/*.json whose bbox intersects ``bbox``,
    deduplicated by tile_id, each with 'local' = an already-downloaded raw file
    (USGS-stem or legacy tile_<i> name) or None."""
    found: dict[str, dict] = {}
    for lst in sorted(SAVED_LISTS.glob("*.json")):
        if lst.stem.endswith(("_old", ".unfiltered")):
            continue
        raw_dir = LOADED_TILES / lst.stem / "raw_tiles"
        for i, t in enumerate(json.loads(lst.read_text())):
            if not _intersects(t, bbox):
                continue
            entry = found.setdefault(t["tile_id"], dict(t, local=None))
            if entry["local"] is None:
                for p in (raw_dir / f"{tile_file_stem(t['url'])}.laz", raw_dir / f"tile_{i}.laz"):
                    if p.exists():
                        entry["local"] = str(p)
                        break
    return list(found.values())








def work_box(t: dict):
    """A listing's footprint (its lon/lat bbox) as a shapely box in POINT_WORK_CRS."""
    import shapely
    xs, ys = lonlat_to_work(np.array([t["min_lon"], t["max_lon"], t["max_lon"], t["min_lon"]]),
                            np.array([t["min_lat"], t["min_lat"], t["max_lat"], t["max_lat"]]))
    return shapely.box(min(xs), min(ys), max(xs), max(ys))


def _block(t: dict) -> str:
    """Delivery folder of a listing, e.g. 'OH_StatewideP3_6_B21' (…/<block>/LAZ/<file>)."""
    return PurePosixPath(urlparse(t["url"]).path).parts[-3]












# ─── Download ─────────────────────────────────────────────────────────────────

def fetch_tile(tile: dict, cache_dir: Path, attempts: int = 4) -> Path:
    """Local path of a tile, downloading it into ``cache_dir`` if needed.

    Writes to a .part file and renames, so an interrupted download is never
    mistaken for a complete tile on the next run.  On top of the session's
    short retries, waits 30 s → 4 min between attempts: rockyweb answers
    bursts with HTTP 429, which a multi-day scan has to ride out."""
    if tile.get("local") and extent_matches(tile["local"], tile):
        return Path(tile["local"])
    dest = cache_path(tile, cache_dir)
    if dest.exists():
        if extent_matches(dest, tile):
            return dest
        # The other file of a duplicated tile id (cached before variants got their
        # own names), or a corrupt download: replace it
        print(f"  {dest.name}: cached file does not match its listing; downloading it again")
        dest.unlink()
    import requests
    from lidar.generate_tiles import _make_session

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".laz.part")
    for attempt in range(attempts):
        try:
            with _make_session().get(tile["url"], timeout=300, stream=True) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                # A connection dropped mid-body raises nothing: check the length
                expected = r.headers.get("Content-Length")
                if expected and not r.headers.get("Content-Encoding") and part.stat().st_size != int(expected):
                    raise requests.RequestException(
                        f"incomplete download of {dest.name}: {part.stat().st_size} of {expected} bytes")
            break
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(30 * 2 ** attempt)
    os.replace(part, dest)
    if not extent_matches(dest, tile):
        raise RuntimeError(f"{dest.name}: downloaded file's extent does not match its listing")
    return dest


def cache_path(tile: dict, cache_dir: Path) -> Path:
    """Cache file of a listing: the USGS file stem, plus the delivery block for
    ``variant`` listings (two different files sharing one tile id)."""
    stem = tile_file_stem(tile["url"])
    return Path(cache_dir) / (f"{stem}__{_block(tile)}.laz" if tile.get("variant") else f"{stem}.laz")


def file_work_bounds(path: Path | str) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of a LAZ file's header, in POINT_WORK_CRS (header-only read)."""
    p = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(path), "count": 0,
                                   "default_srs": parameters.POINT_FALLBACK_SRS}]))
    p.execute()
    md = p.metadata["metadata"]["readers.las"]
    crs = parse_crs(_srs_wkt(p.metadata))
    xs, ys = to_work(np.array([md["minx"], md["maxx"], md["maxx"], md["minx"]]),
                     np.array([md["miny"], md["miny"], md["maxy"], md["maxy"]]), crs)
    return min(xs), min(ys), max(xs), max(ys)


_extent_ok: dict[tuple, bool] = {}


def extent_matches(path: Path | str, tile: dict, tol_m: float = 15.0) -> bool:
    """Does the file's header extent match the listing's footprint?  Catches the
    wrong file of a duplicated tile id (a 95 m sliver vs the full tile) and
    unreadable files.  Memoised per (path, mtime, size)."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = (str(path), st.st_mtime_ns, st.st_size, tile["url"])
    if key not in _extent_ok:
        try:
            got = file_work_bounds(path)
            want = work_box(tile).bounds
            _extent_ok[key] = all(abs(a - b) <= tol_m for a, b in zip(got, want))
        except RuntimeError:
            _extent_ok[key] = False
    return _extent_ok[key]


# ─── CRS handling ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TileCRS:
    wkt: str              # full (usually compound) CRS of the tile
    horizontal: CRS       # its horizontal part
    z_to_m: float         # vertical unit → metres


@lru_cache(maxsize=16)
def parse_crs(wkt: str) -> TileCRS:
    crs = CRS.from_user_input(wkt or parameters.POINT_FALLBACK_SRS)
    subs = crs.sub_crs_list or [crs]
    horiz = subs[0]
    vert = subs[1] if len(subs) > 1 else None
    if vert is not None:
        z_to_m = vert.axis_info[0].unit_conversion_factor
    else:
        # No vertical CRS declared: assume heights share the horizontal unit
        z_to_m = horiz.axis_info[0].unit_conversion_factor
    return TileCRS(wkt=crs.to_wkt(), horizontal=horiz, z_to_m=z_to_m)


@lru_cache(maxsize=16)
def _transformer(src_wkt: str, dst: str) -> Transformer:
    return Transformer.from_crs(CRS.from_wkt(src_wkt), CRS.from_user_input(dst), always_xy=True)


def to_work(x: np.ndarray, y: np.ndarray, crs: TileCRS,
            work_crs: str = parameters.POINT_WORK_CRS) -> tuple[np.ndarray, np.ndarray]:
    return _transformer(crs.horizontal.to_wkt(), work_crs).transform(x, y)


def lonlat_to_work(lon, lat, work_crs: str = parameters.POINT_WORK_CRS):
    return _transformer(CRS.from_user_input("EPSG:4326").to_wkt(), work_crs).transform(lon, lat)


def _srs_wkt(metadata: dict) -> str:
    srs = metadata.get("metadata", {}).get("readers.las", {}).get("srs", {})
    return srs.get("compoundwkt") or srs.get("wkt") or ""
