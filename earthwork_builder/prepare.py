"""
prepare.py — Download and crop OH_Statewide LiDAR for every site in sites.json.

    python -m earthwork_builder.prepare                 # all unprepared sites
    python -m earthwork_builder.prepare --only Newark_Octagon --force

Per site (window = square of half-size ``half_m`` in EW_WORK_CRS):
  1. find the OH_Statewide tiles covering it (saved lists, else ScienceBase) —
     or, where OSIP III has none, the newest non-legacy collection
     (EW_FALLBACK_COLLECTIONS);
  2. download them into EW_TILE_CACHE (shared with the scan);
  3. write ``points.laz`` — every point in the window, native CRS, all classes;
  4. build ``ground.tif`` from EW_GROUND_SOURCE: the vendor ground class
     (EW_GROUND_CLASSES) of those points, or the 3DEP 1 m DEM;
  5. write ``site.json`` with the window, tiles, point counts and ground source.

A site whose lat/lon/half_m in sites.json differ from those in its site.json
(moved or resized) is stale (``Site.stale``) and rebuilt from scratch.
A prepared site whose ground.tif came from another source than the current
EW_GROUND_SOURCE gets only its ground.tif rebuilt (no re-download).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import parameters
from earthwork_builder.labels import Site, load_sites
from lidar import point_io
from lidar.dem_source import ground_grid
from lidar.ground_grid import GridAccumulator, GroundGrid
from lidar.tile_filter import collection_of


def site_window(site: Site) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in the work CRS, snapped to the pixel grid."""
    cx, cy = point_io.lonlat_to_work(site.lon, site.lat)
    p = parameters.EW_PIXEL_M
    x0 = np.floor((cx - site.half_m) / p) * p
    y0 = np.floor((cy - site.half_m) / p) * p
    n = int(np.ceil(2 * site.half_m / p))
    return float(x0), float(y0), float(x0 + n * p), float(y0 + n * p)


def window_lonlat_bbox(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    lon, lat = point_io.work_to_lonlat(np.array([xmin, xmax, xmax, xmin]),
                                       np.array([ymin, ymin, ymax, ymax]))
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def _write_ground(site: Site, bounds, source: str) -> GroundGrid:
    """ground.tif from ``source`` ("points" reads the site's own points.laz)."""
    grid = ground_grid(bounds, source, paths=[site.points_path])
    grid.to_geotiff(site.ground_path)
    return grid


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
    except ImportError:
        return True                                   # can't tell: assume the lock is live
    return psutil.pid_exists(pid)


@contextmanager
def _site_lock(site: Site):
    """One prepare per site across processes (the builder server and the CLI may
    both pick up a site sites.json moved).  A lock left by a dead process is taken over."""
    site.dir.mkdir(parents=True, exist_ok=True)
    lock = site.dir / ".prepare.lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                raise RuntimeError(f"{site.name}: already being prepared by process {pid}")
            lock.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"{site.name}: could not take {lock}")
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def prepare_site(site: Site, force: bool = False, log=print) -> dict:
    """Build points.laz / ground.tif / site.json for one site; returns site.json."""
    if site.prepared and not force and _ground_source_ok(site):
        return json.loads((site.dir / "site.json").read_text())
    with _site_lock(site):
        return _prepare_site(site, force, log)


def _ground_source_ok(site: Site) -> bool:
    meta = json.loads((site.dir / "site.json").read_text())
    return meta.get("ground_source", "points") == parameters.EW_GROUND_SOURCE


def _prepare_site(site: Site, force: bool, log) -> dict:
    source = parameters.EW_GROUND_SOURCE
    if site.prepared and not force:
        meta = json.loads((site.dir / "site.json").read_text())
        if meta.get("ground_source", "points") == source:
            return meta
        log(f"[{site.name}] rebuilding ground.tif from {source!r} (was {meta.get('ground_source', 'points')!r})")
        grid = _write_ground(site, tuple(meta["work_bounds"]), source)
        meta.update(ground_source=source,
                    grid_nodata_pct=round(100 * float(np.isnan(grid.z).mean()), 2))
        (site.dir / "site.json").write_text(json.dumps(meta, indent=2))
        return meta
    if site.stale:
        log(f"[{site.name}] coords or half_m changed since it was prepared: rebuilding")
    # Unprepared until the rebuild completes: a run that dies half way must not
    # leave new points.laz next to an old ground.tif that still looks prepared
    (site.dir / "site.json").unlink(missing_ok=True)
    bounds = site_window(site)
    import shapely
    window = shapely.box(*bounds)
    tiles = point_io.find_tiles(window_lonlat_bbox(bounds), window=window)
    if not tiles:
        raise RuntimeError(f"{site.name}: no {parameters.EW_COLLECTION_PREFIX} (or newer non-legacy) "
                           f"tiles cover this site")
    collections = sorted({collection_of(t) for t in tiles})
    fill = sorted({collection_of(t) for t in tiles if "clip" in t})
    if fill:
        main = window.intersection(point_io.shapely_union(
            [point_io.work_box(t) for t in tiles if "clip" not in t]))
        log(f"[{site.name}] {parameters.EW_COLLECTION_PREFIX} covers {main.area / window.area:.0%} of the "
            f"window; filling the rest from {', '.join(fill)}")
    gap = window.difference(point_io.shapely_union([point_io.work_box(t) for t in tiles]))
    if gap.area > 0.002 * window.area:
        log(f"[{site.name}] warning: {gap.area / window.area:.0%} of the window has no LiDAR listing at all")
    log(f"[{site.name}] {len(tiles)} tile(s): downloading ...")
    with ThreadPoolExecutor(parameters.EW_DOWNLOAD_WORKERS) as pool:
        paths = list(pool.map(point_io.fetch_tile, tiles))

    # One tile at a time: a 3 km window is ~600M points, far more than fits in RAM
    log(f"[{site.name}] cropping {len(paths)} tile(s) ...")
    acc = GridAccumulator(bounds, parameters.EW_PIXEL_M)
    n_ground = 0

    def on_cloud(cloud):
        nonlocal n_ground
        ground = np.isin(cloud.points["Classification"], parameters.EW_GROUND_CLASSES)
        n_ground += int(ground.sum())
        if source == "points":
            acc.add(cloud.xyz[ground])

    tile_of = {str(p): t for p, t in zip(paths, tiles)}

    def refetch(path):
        """Corrupt tile: drop the cached copy (never a saved-list original) and download again."""
        log(f"[{site.name}] {Path(path).name} is corrupt: downloading it again")
        cached = Path(parameters.EW_TILE_CACHE) / Path(path).name
        if Path(path).resolve() == cached.resolve():
            cached.unlink(missing_ok=True)
        return point_io.fetch_tile(dict(tile_of[str(path)], local=None))

    n_points, crs = point_io.crop_tiles_to_laz(paths, bounds, site.points_path, on_cloud, refetch,
                                               clips=[t.get("clip") for t in tiles])
    if not n_points:
        raise RuntimeError(f"{site.name}: tiles contain no points inside the window")

    log(f"[{site.name}] ground grid from {source!r} ...")
    grid = acc.finish() if source == "points" else ground_grid(bounds, source)
    grid.to_geotiff(site.ground_path)

    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    meta = {
        "name": site.name, "lat": site.lat, "lon": site.lon, "half_m": site.half_m,
        "work_crs": parameters.EW_WORK_CRS, "work_bounds": bounds,
        "lonlat_bbox": window_lonlat_bbox(bounds), "pixel_m": grid.pixel_m,
        "ground_source": source,
        "native_crs": crs.wkt,
        "collections": collections,
        "tiles": [{"tile_id": t["tile_id"], "url": t["url"], **({"fill": True} if "clip" in t else {})}
                  for t in tiles],
        "points": n_points, "ground_points": n_ground,
        "ground_per_m2": round(n_ground / area, 2),
        "grid_nodata_pct": round(100 * float(np.isnan(grid.z).mean()), 2),
    }
    (site.dir / "site.json").write_text(json.dumps(meta, indent=2))
    log(f"[{site.name}] done: {meta['points']:,} points, {meta['ground_per_m2']} ground/m2, "
        f"{meta['grid_nodata_pct']}% grid nodata")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="SLUG", help="Prepare only these sites (slug or name)")
    ap.add_argument("--force", action="store_true", help="Rebuild sites that are already prepared")
    args = ap.parse_args()

    sites = load_sites()
    if args.only:
        want = set(args.only)
        sites = [s for s in sites if s.slug in want or s.name in want]
    failed = []
    for site in sites:
        try:
            prepare_site(site, force=args.force)
        except Exception as e:
            print(f"[{site.name}] FAILED: {e}")
            failed.append(site.name)
    if failed:
        raise SystemExit(f"{len(failed)} site(s) failed: {failed}")


if __name__ == "__main__":
    main()
