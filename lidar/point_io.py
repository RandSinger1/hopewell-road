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
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

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


def _collection_year(name: str) -> int:
    years = [int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d\d(?!\d)", name)]
    return max(years, default=0)


def work_box(t: dict):
    """A listing's footprint (its lon/lat bbox) as a shapely box in EW_WORK_CRS."""
    import shapely
    xs, ys = lonlat_to_work(np.array([t["min_lon"], t["max_lon"], t["max_lon"], t["min_lon"]]),
                            np.array([t["min_lat"], t["min_lat"], t["max_lat"], t["max_lat"]]))
    return shapely.box(min(xs), min(ys), max(xs), max(ys))


def _block(t: dict) -> str:
    """Delivery folder of a listing, e.g. 'OH_StatewideP3_6_B21' (…/<block>/LAZ/<file>)."""
    return PurePosixPath(urlparse(t["url"]).path).parts[-3]


def unique_tiles(tiles: list[dict], tol_m: float = 15.0) -> list[dict]:
    """Drop listings that add no points.  ScienceBase lists some OSIP III tile ids
    twice, under neighbouring delivery blocks (e.g. OH_StatewideP3_6_B21 / _7_B21):
    one is the full tile, the other a ~95 m sliver along the block edge that lies
    inside it.  These are different files with the same name.  A listing whose
    footprint lies inside another listing of the same id is dropped (exact
    duplicates included); listings that remain sharing an id get ``variant`` so
    they are cached under different names (``cache_path``)."""
    by_url = {t["url"]: t for t in tiles}
    groups: dict[str, list[dict]] = {}
    for t in by_url.values():
        groups.setdefault(t["tile_id"], []).append(t)
    out = []
    for ts in groups.values():
        boxes = [work_box(t) for t in ts]
        # biggest first, so of two equal footprints the first is kept
        order = sorted(range(len(ts)), key=lambda i: -boxes[i].area)
        kept: list[int] = []
        for i in order:
            if not any(boxes[k].buffer(tol_m).contains(boxes[i]) for k in kept):
                kept.append(i)
        for i in kept:
            out.append(dict(ts[i], variant=len(kept) > 1))
    return out


def fill_tiles(window, covered, candidates: list[dict], min_frac: float = 0.002) -> list[dict]:
    """Tiles that fill the part of ``window`` (shapely, work CRS) outside ``covered``,
    from non-legacy ``candidates``: newest collection first, each tile tagged with
    ``clip`` = the still-uncovered area it alone supplies, so points never overlap.
    For spots OSIP III skipped (Johnson's Island, much of Delaware County, ...)."""
    gap = window.difference(covered)
    out = []
    modern = [t for t in candidates if "/legacy/" not in t["url"]
              and not collection_of(t).startswith(parameters.EW_COLLECTION_PREFIX)]
    for coll in sorted({collection_of(t) for t in modern}, key=lambda c: (-_collection_year(c), c)):
        if gap.area < min_frac * window.area:
            break
        ts = [t for t in unique_tiles([t for t in modern if collection_of(t) == coll])
              if work_box(t).intersects(gap)]
        if not ts:
            continue
        clip = gap
        out += [dict(t, clip=clip) for t in ts]
        gap = gap.difference(shapely_union([work_box(t) for t in ts]))
    return out


def shapely_union(geoms):
    import shapely
    return shapely.union_all(geoms) if geoms else shapely.Polygon()


def find_tiles(bbox: tuple[float, float, float, float],
               fallback: bool = parameters.EW_FALLBACK_COLLECTIONS,
               window=None) -> list[dict]:
    """Statewide tiles covering ``bbox``.  ScienceBase is authoritative (the saved
    lists only cover the corridor AOIs, so they can cover a site partially);
    tiles already downloaded for those lists are reused via 'local'.  Falls back
    to the saved lists alone when ScienceBase is unreachable.  With ``fallback``,
    the part of ``window`` (shapely, work CRS; default: ``bbox``) no statewide tile
    covers is filled by ``fill_tiles`` (those tiles carry a ``clip`` geometry)."""
    saved = {t["tile_id"]: t for t in statewide_only(saved_list_tiles(bbox))}
    try:
        everything = query_tiles(bbox)
    except Exception as e:
        print(f"  ScienceBase unavailable ({e}); using saved tile lists only")
        return list(saved.values())
    tiles = unique_tiles(statewide_only(everything))
    for t in tiles:
        if t["tile_id"] in saved and not t["variant"]:
            t["local"] = saved[t["tile_id"]]["local"]
    if fallback:
        if window is None:
            window = work_box({"min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3]})
        tiles += fill_tiles(window, shapely_union([work_box(t) for t in tiles]), everything)
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


def cache_path(tile: dict, cache_dir: Path = parameters.EW_TILE_CACHE) -> Path:
    """Cache file of a listing: the USGS file stem, plus the delivery block for
    ``variant`` listings (two different files sharing one tile id)."""
    stem = tile_file_stem(tile["url"])
    return Path(cache_dir) / (f"{stem}__{_block(tile)}.laz" if tile.get("variant") else f"{stem}.laz")


def file_work_bounds(path: Path | str) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of a LAZ file's header, in EW_WORK_CRS (header-only read)."""
    p = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(path), "count": 0,
                                   "default_srs": parameters.EW_FALLBACK_SRS}]))
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


def _native_to(points: np.ndarray, crs: TileCRS, ref: TileCRS) -> np.ndarray:
    """``points`` (native X/Y/Z in ``crs``) converted to ``ref``'s native CRS / units."""
    if crs.wkt == ref.wkt:
        return points
    pts = points.copy()
    nx, ny = _transformer(crs.horizontal.to_wkt(), ref.horizontal.to_wkt()).transform(pts["X"], pts["Y"])
    pts["X"], pts["Y"] = nx, ny
    pts["Z"] = pts["Z"] * (crs.z_to_m / ref.z_to_m)
    return pts


def read_tiles(paths: list, classes: tuple | None = None,
               work_bounds: tuple[float, float, float, float] | None = None) -> PointCloud:
    """Read several tiles into one cloud.  Tiles in a different native CRS than
    the first (Ohio North vs South) have their native X/Y/Z converted to it, and
    only the dimensions every tile shares are kept, so ``points`` stays writable.
    Holds everything in memory (~75 B/point) — for big windows use crop_tiles_to_laz."""
    clouds = [read_tile(p, classes, work_bounds) for p in paths]
    clouds = [c for c in clouds if len(c)] or clouds[:1]
    if not clouds:
        raise ValueError("No tiles to read")
    ref = clouds[0].crs
    common = [n for n in clouds[0].points.dtype.names
              if all(n in c.points.dtype.names for c in clouds)]
    parts = [_native_to(c.points[common].copy(), c.crs, ref) for c in clouds]
    # Repack so the result is a plain (non-view) structured array
    points = np.concatenate([np.asarray(p, dtype=parts[0].dtype) for p in parts])
    return PointCloud(points, ref, np.concatenate([c.xyz for c in clouds]))


def crop_tiles_to_laz(paths: list, work_bounds: tuple[float, float, float, float], out: Path | str,
                      on_cloud=None, refetch=None, clips=None) -> tuple[int, TileCRS | None]:
    """Crop tiles to ``work_bounds`` into one LAZ without holding the window in memory.

    Each tile is read, cropped and written as a chunk file (in the first non-empty
    tile's native CRS, as read_tiles does), then PDAL stream-merges the chunks.
    ``on_cloud(cloud)`` sees every cropped tile (e.g. to bin ground points).
    ``refetch(path) -> path`` replaces a tile PDAL cannot read (a corrupt or
    truncated download); it is tried once per tile.  ``clips[i]`` (shapely, work
    CRS, or None) keeps only tile i's points inside it (fill tiles: see fill_tiles).
    Returns (point count, native CRS of ``out``); writes nothing if no point falls inside."""
    out = Path(out)
    chunk_dir = out.with_name(out.stem + ".chunks")
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True)
    ref, chunks, n = None, [], 0
    try:
        for i, p in enumerate(paths):
            try:
                cloud = read_tile(p, work_bounds=work_bounds)
            except RuntimeError as e:                 # PDAL: unreadable LAZ
                if refetch is None:
                    raise RuntimeError(f"{Path(p).name}: {e}") from e
                print(f"  {Path(p).name} is unreadable ({e}); downloading it again")
                cloud = read_tile(refetch(p), work_bounds=work_bounds)
            clip = clips[i] if clips else None
            if clip is not None and len(cloud):
                import shapely
                shapely.prepare(clip)
                cloud = cloud.subset(shapely.contains_xy(clip, cloud.xyz[:, 0], cloud.xyz[:, 1]))
            if not len(cloud):
                continue
            if on_cloud is not None:
                on_cloud(cloud)
            ref = ref or cloud.crs
            pts = _native_to(cloud.points, cloud.crs, ref)
            chunks.append(write_laz(PointCloud(pts, ref, cloud.xyz), chunk_dir / f"{len(chunks)}.laz"))
            n += len(cloud)
            del cloud, pts
        if chunks:
            tmp = out.with_suffix(".tmp.laz")
            stages = [{"type": "readers.las", "filename": str(c)} for c in chunks]
            stages.append({"type": "writers.las", "filename": str(tmp), "compression": "laszip",
                           "a_srs": ref.wkt, "minor_version": 4, "extra_dims": "all", "forward": "all"})
            merge = pdal.Pipeline(json.dumps(stages))
            merge.execute_streaming(chunk_size=1_000_000)
            del merge                                 # its readers hold the chunk files open
            os.replace(tmp, out)
    finally:
        import gc
        gc.collect()
        for _ in range(5):                            # OneDrive briefly holds the emptied folder
            shutil.rmtree(chunk_dir, ignore_errors=True)
            if not chunk_dir.exists():
                break
            time.sleep(1)
    return n, ref


def iter_xyz(path: Path | str, classes: tuple | None = None, chunk_size: int = 2_000_000):
    """Work-CRS (N, 3) xyz of a LAZ file, chunk by chunk (constant memory)."""
    head = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(path), "count": 0,
                                      "default_srs": parameters.EW_FALLBACK_SRS}]))
    head.execute()
    crs = parse_crs(_srs_wkt(head.metadata))
    stages: list[dict] = [{"type": "readers.las", "filename": str(path),
                           "default_srs": parameters.EW_FALLBACK_SRS}]
    if classes:
        stages.append({"type": "filters.range",
                       "limits": ",".join(f"Classification[{c}:{c}]" for c in classes)})
    for pts in pdal.Pipeline(json.dumps(stages)).iterator(chunk_size=chunk_size):
        if len(pts):
            wx, wy = to_work(pts["X"], pts["Y"], crs)
            yield np.column_stack([wx, wy, pts["Z"] * crs.z_to_m])


def write_laz(cloud: PointCloud, path: Path | str) -> Path:
    """Write ``cloud.points`` (native CRS, all attributes) to a LAZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = {"type": "writers.las", "filename": str(path), "compression": "laszip",
         "a_srs": cloud.crs.wkt, "minor_version": 4, "extra_dims": "all",
         "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}
    pdal.Pipeline(json.dumps([w]), arrays=[cloud.points]).execute()
    return path
