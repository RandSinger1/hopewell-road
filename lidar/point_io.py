"""
point_io.py — Find, download and read USGS LAZ tiles into one metric CRS.

Shared by experiments/lidar_lab.py (site crops), earthwork_builder/ (site
point clouds) and earthwork_inference/scan.py (statewide blocks).

All LAZ reading and writing goes through PDAL (laspy has no LAZ backend in
pdal_env).  Every reader returns two views of the same points:

  * ``points`` — the PDAL structured array, untouched, in the tile's own CRS
    and units (Ohio North/South ftUS).  This is what gets written back out,
    so saved crops keep every attribute at full fidelity.
  * ``xyz``    — (N, 3) float64 in ``parameters.EW_WORK_CRS`` (UTM 17N) with
    Z in metres.  This is what ground grids and features are built from.

The XY transform is done with pyproj on the horizontal CRS only and Z is
scaled by the vertical unit, so no geoid / ellipsoid conversion ever touches
heights.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pdal
from pyproj import CRS, Transformer

import parameters
from lidar.tile_filter import collection_of, tile_file_stem

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


def query_tiles(bbox: tuple[float, float, float, float], retries: int = 5) -> list[dict]:
    """Ask ScienceBase (via the AOI2List submodule) for tiles intersecting ``bbox``."""
    from AOI2List.src.aoi2list import bbox_to_wkt_polygon, query_polygon

    wkt = bbox_to_wkt_polygon(*bbox)
    for attempt in range(retries):
        try:
            tiles = query_polygon(wkt)[0]
            return [dict(t, local=None) for t in tiles if _intersects(t, bbox)]
        except Exception as e:  # network / ScienceBase hiccups
            if attempt == retries - 1:
                raise
            print(f"  ScienceBase query failed ({e}); retrying ...")
    return []


def statewide_only(tiles: list[dict], prefix: str = parameters.EW_COLLECTION_PREFIX) -> list[dict]:
    """Keep tiles from the OSIP III statewide collections."""
    return [t for t in tiles if collection_of(t).startswith(prefix)]


def find_tiles(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Statewide tiles covering ``bbox``.  ScienceBase is authoritative (the saved
    lists only cover the corridor AOIs, so they can cover a site partially);
    tiles already downloaded for those lists are reused via 'local'.  Falls back
    to the saved lists alone when ScienceBase is unreachable."""
    saved = {t["tile_id"]: t for t in statewide_only(saved_list_tiles(bbox))}
    try:
        tiles = statewide_only(query_tiles(bbox))
    except Exception as e:
        print(f"  ScienceBase unavailable ({e}); using saved tile lists only")
        return list(saved.values())
    for t in tiles:
        if t["tile_id"] in saved:
            t["local"] = saved[t["tile_id"]]["local"]
    return tiles


def local_tile_paths() -> dict[str, str]:
    """tile_id → already-downloaded raw file, for every tile in the saved lists
    (resolves the legacy tile_<i>.laz names as well)."""
    world = (-180.0, -90.0, 180.0, 90.0)
    return {t["tile_id"]: t["local"] for t in saved_list_tiles(world) if t["local"]}


# ─── Download ─────────────────────────────────────────────────────────────────

def fetch_tile(tile: dict, cache_dir: Path = parameters.EW_TILE_CACHE, attempts: int = 4) -> Path:
    """Local path of a tile, downloading it into ``cache_dir`` if needed.

    Writes to a .part file and renames, so an interrupted download is never
    mistaken for a complete tile on the next run.  On top of the session's
    short retries, waits 30 s → 4 min between attempts: rockyweb answers
    bursts with HTTP 429, which a multi-day scan has to ride out."""
    if tile.get("local"):
        return Path(tile["local"])
    dest = Path(cache_dir) / f"{tile_file_stem(tile['url'])}.laz"
    if dest.exists():
        return dest
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
            break
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(30 * 2 ** attempt)
    os.replace(part, dest)
    return dest


# ─── CRS handling ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TileCRS:
    wkt: str              # full (usually compound) CRS of the tile
    horizontal: CRS       # its horizontal part
    z_to_m: float         # vertical unit → metres


@lru_cache(maxsize=16)
def parse_crs(wkt: str) -> TileCRS:
    crs = CRS.from_user_input(wkt or parameters.EW_FALLBACK_SRS)
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
            work_crs: str = parameters.EW_WORK_CRS) -> tuple[np.ndarray, np.ndarray]:
    return _transformer(crs.horizontal.to_wkt(), work_crs).transform(x, y)


def work_to_lonlat(x, y, work_crs: str = parameters.EW_WORK_CRS):
    return _transformer(CRS.from_user_input(work_crs).to_wkt(), "EPSG:4326").transform(x, y)


def lonlat_to_work(lon, lat, work_crs: str = parameters.EW_WORK_CRS):
    return _transformer(CRS.from_user_input("EPSG:4326").to_wkt(), work_crs).transform(lon, lat)


# ─── Reading ──────────────────────────────────────────────────────────────────

@dataclass
class PointCloud:
    points: np.ndarray    # PDAL structured array, native CRS / units
    crs: TileCRS          # native CRS of ``points``
    xyz: np.ndarray       # (N, 3) float64: work-CRS metres, Z in metres

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, mask: np.ndarray) -> "PointCloud":
        return PointCloud(self.points[mask], self.crs, self.xyz[mask])


def _srs_wkt(metadata: dict) -> str:
    srs = metadata.get("metadata", {}).get("readers.las", {}).get("srs", {})
    return srs.get("compoundwkt") or srs.get("wkt") or ""


def read_tile(path: Path | str, classes: tuple | None = None,
              work_bounds: tuple[float, float, float, float] | None = None) -> PointCloud:
    """Read one LAZ tile.  ``classes`` keeps only those LAS classes;
    ``work_bounds`` = (xmin, ymin, xmax, ymax) in the work CRS crops to a box."""
    stages: list[dict] = [{"type": "readers.las", "filename": str(path),
                           "default_srs": parameters.EW_FALLBACK_SRS}]
    if classes:
        stages.append({"type": "filters.range",
                       "limits": ",".join(f"Classification[{c}:{c}]" for c in classes)})
    p = pdal.Pipeline(json.dumps(stages))
    p.execute()
    pts = p.arrays[0]
    crs = parse_crs(_srs_wkt(p.metadata))
    wx, wy = to_work(pts["X"], pts["Y"], crs)
    xyz = np.column_stack([wx, wy, pts["Z"] * crs.z_to_m])
    pc = PointCloud(pts, crs, xyz)
    if work_bounds is not None:
        x0, y0, x1, y1 = work_bounds
        pc = pc.subset((wx >= x0) & (wx <= x1) & (wy >= y0) & (wy <= y1))
    return pc


def read_tiles(paths: list, classes: tuple | None = None,
               work_bounds: tuple[float, float, float, float] | None = None) -> PointCloud:
    """Read several tiles into one cloud.  Tiles in a different native CRS than
    the first (Ohio North vs South) have their native X/Y/Z converted to it, and
    only the dimensions every tile shares are kept, so ``points`` stays writable."""
    clouds = [read_tile(p, classes, work_bounds) for p in paths]
    clouds = [c for c in clouds if len(c)] or clouds[:1]
    if not clouds:
        raise ValueError("No tiles to read")
    ref = clouds[0].crs
    common = [n for n in clouds[0].points.dtype.names
              if all(n in c.points.dtype.names for c in clouds)]
    parts = []
    for c in clouds:
        pts = c.points[common].copy()
        if c.crs.wkt != ref.wkt:
            nx, ny = _transformer(c.crs.horizontal.to_wkt(), ref.horizontal.to_wkt()).transform(
                pts["X"], pts["Y"])
            pts["X"], pts["Y"] = nx, ny
            pts["Z"] = pts["Z"] * (c.crs.z_to_m / ref.z_to_m)
        parts.append(pts)
    # Repack so the result is a plain (non-view) structured array
    points = np.concatenate([np.asarray(p, dtype=parts[0].dtype) for p in parts])
    return PointCloud(points, ref, np.concatenate([c.xyz for c in clouds]))


def write_laz(cloud: PointCloud, path: Path | str) -> Path:
    """Write ``cloud.points`` (native CRS, all attributes) to a LAZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = {"type": "writers.las", "filename": str(path), "compression": "laszip",
         "a_srs": cloud.crs.wkt, "minor_version": 4, "extra_dims": "all",
         "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}
    pdal.Pipeline(json.dumps([w]), arrays=[cloud.points]).execute()
    return path
