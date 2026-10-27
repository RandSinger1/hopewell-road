"""
prepare.py — Download and crop OH_Statewide LiDAR for every site in sites.json.

    python -m earthwork_builder.prepare                 # all unprepared sites
    python -m earthwork_builder.prepare --only Newark_Octagon --force

Per site (window = square of half-size ``half_m`` in EW_WORK_CRS):
  1. find the OH_Statewide tiles covering it (saved lists, else ScienceBase);
  2. download them into EW_TILE_CACHE (shared with the scan);
  3. write ``points.laz`` — every point in the window, native CRS, all classes;
  4. build ``ground.tif`` from EW_GROUND_SOURCE: the vendor ground class
     (EW_GROUND_CLASSES) of those points, or the 3DEP 1 m DEM;
  5. write ``site.json`` with the window, tiles, point counts and ground source.

A prepared site whose ground.tif came from another source than the current
EW_GROUND_SOURCE gets only its ground.tif rebuilt (no re-download).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import parameters
from earthwork_builder.labels import Site, load_sites
from lidar import point_io
from lidar.dem_source import ground_grid
from lidar.ground_grid import GroundGrid


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


def prepare_site(site: Site, force: bool = False, log=print) -> dict:
    """Build points.laz / ground.tif / site.json for one site; returns site.json."""
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
    bounds = site_window(site)
    tiles = point_io.find_tiles(window_lonlat_bbox(bounds))
    if not tiles:
        raise RuntimeError(f"{site.name}: no {parameters.EW_COLLECTION_PREFIX} tiles cover this site")
    log(f"[{site.name}] {len(tiles)} tile(s): downloading ...")
    with ThreadPoolExecutor(parameters.EW_DOWNLOAD_WORKERS) as pool:
        paths = list(pool.map(point_io.fetch_tile, tiles))

    log(f"[{site.name}] reading + cropping ...")
    cloud = point_io.read_tiles(paths, work_bounds=bounds)
    if not len(cloud):
        raise RuntimeError(f"{site.name}: tiles contain no points inside the window")
    point_io.write_laz(cloud, site.points_path)

    ground = np.isin(cloud.points["Classification"], parameters.EW_GROUND_CLASSES)
    log(f"[{site.name}] ground grid from {source!r} ...")
    grid = _write_ground(site, bounds, source)

    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    meta = {
        "name": site.name, "lat": site.lat, "lon": site.lon, "half_m": site.half_m,
        "work_crs": parameters.EW_WORK_CRS, "work_bounds": bounds,
        "lonlat_bbox": window_lonlat_bbox(bounds), "pixel_m": grid.pixel_m,
        "ground_source": source,
        "native_crs": cloud.crs.wkt,
        "tiles": [{"tile_id": t["tile_id"], "url": t["url"]} for t in tiles],
        "points": int(len(cloud)), "ground_points": int(ground.sum()),
        "ground_per_m2": round(float(ground.sum()) / area, 2),
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
