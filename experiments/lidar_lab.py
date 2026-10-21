"""
lidar_lab.py — Re-run the point-cloud → DTM stage (lidar/pdal_pipeline.py)
on a crop around one site with any setting overridden, then hand the DTM to
wall_lab so the same wall detection runs on it.

    from experiments.lidar_lab import LidarParams, run_lidar
    lr = run_lidar("Van Voorhis Wall", LidarParams(smrf_window=60))
    lr.wall            # a wall_lab.Result on the re-generated DTM
    lr.stats           # point counts, ground fraction, density, DTM gaps

Stages mirror make_pipeline / make_dtm_pipeline: filters.outlier →
filters.smrf → Classification[2:2] → writers.gdal (in the tiles' CRS) →
gdal.Warp to EPSG:4326 (bilinear, like merge_dtms).

UNITS: the Ohio tiles are NAD83(2011) / Ohio South (ftUS) + NAVD88 height
(ftUS), and PDAL applies every distance in the CRS's units.  So all SMRF
distances (cell, window, threshold, cut) and dtm_resolution are in **US survey
feet**, as are the values in parameters.py.  Use ``ft(m)`` to convert metres.
The DTM handed to wall_lab is in **metres** vertically: gdal.Warp from the
compound ftUS CRS to EPSG:4326 converts heights, exactly as merge_dtms does
for merged_dtm.tif (per-tile DTMs are feet; the merged mosaic is metres).

Caching: the cropped point cloud is saved under experiments/.cache/points/,
each DTM under experiments/.cache/dtm/ (named by a hash of every input), and
ground classification is memoised in memory, so repeating a run is cheap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, replace, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pdal
from osgeo import gdal
from pyproj import Transformer

import parameters
from lidar.tile_filter import tile_file_stem, tile_year
from experiments import wall_lab

gdal.UseExceptions()

FT_PER_M = 1 / 0.3048006096          # US survey foot
CRS = parameters.SOURCE_CRS          # EPSG:6551, Ohio South ftUS (horizontal)
CRS_3D = f"{CRS}+6360"                # + NAVD88 height (ftUS), as the tiles declare
CACHE = Path("experiments/.cache")


def ft(metres: float) -> float:
    return metres * FT_PER_M


# ─── Parameters ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LidarParams:
    """Point-cloud → DTM settings, defaulting to parameters.py §3 (units: ftUS)."""
    # which flights to use where collections overlap
    collection: str = "newest"            # "newest" | "oldest" | "all"
    # ground classification
    classifier: str = "smrf"              # "smrf" | "vendor" (keep the tiles' own class 2)
    max_scan_angle: float | None = None   # drop points with |ScanAngleRank| above this (deg)
    outlier: bool = True
    outlier_mean_k: int = parameters.OUTLIER_MEAN_K
    outlier_multiplier: float = parameters.OUTLIER_MULTIPLIER
    smrf_ignore_noise: bool = False       # production doesn't pass ignore=Classification[7:7]
    smrf_cell: float = parameters.SMRF_CELL
    smrf_window: float = parameters.SMRF_WINDOW
    smrf_slope: float = parameters.SMRF_SLOPE
    smrf_threshold: float = parameters.SMRF_THRESHOLD
    smrf_scalar: float = parameters.SMRF_SCALAR
    smrf_cut: float = parameters.SMRF_CUT
    smrf_returns: str = parameters.SMRF_RETURNS
    # rasterisation (writers.gdal)
    dtm_resolution: float = parameters.DTM_RESOLUTION
    interp: str = parameters.DTM_INTERP_METHOD   # idw | mean | min | max | count | stdev
    gdal_radius: float | None = None             # None = PDAL default (resolution·√2)
    gdal_window_size: int = 0                    # >0 fills holes from neighbouring cells

    def with_(self, **kw) -> "LidarParams":
        return replace(self, **kw)

    def diff(self, other: "LidarParams | None" = None) -> dict:
        other = other or LidarParams()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(other, f.name)}

    def _ground_key(self) -> tuple:
        names = ["classifier", "max_scan_angle", "outlier", "outlier_mean_k", "outlier_multiplier",
                 "smrf_ignore_noise", "smrf_cell", "smrf_window", "smrf_slope",
                 "smrf_threshold", "smrf_scalar", "smrf_cut", "smrf_returns"]
        return tuple(getattr(self, n) for n in names)


# ─── Tiles + crop ─────────────────────────────────────────────────────────────

def _bbox_ft(coord: dict, half_m: float) -> tuple[float, float, float, float]:
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x, y = tr.transform(coord["lon"], coord["lat"])
    h = ft(half_m)
    return x - h, x + h, y - h, y + h


def site_tiles(coord: dict, half_m: float) -> list[dict]:
    """Every saved-list tile whose bbox intersects the site window, deduplicated,
    each with 'local' = an existing raw file (stem or legacy tile_<i> name) or None."""
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * np.cos(np.radians(coord["lat"])))
    found: dict[str, dict] = {}
    for lst in sorted(Path("data/saved_file_lists").glob("*.json")):
        if lst.stem.endswith(("_old", ".unfiltered")):
            continue
        raw_dir = Path("data/loaded_tiles") / lst.stem / "raw_tiles"
        for i, t in enumerate(json.loads(lst.read_text())):
            if not (t["min_lat"] <= coord["lat"] + dlat and t["max_lat"] >= coord["lat"] - dlat and
                    t["min_lon"] <= coord["lon"] + dlon and t["max_lon"] >= coord["lon"] - dlon):
                continue
            entry = found.setdefault(t["tile_id"], dict(t, local=None))
            if entry["local"] is None:
                for p in (raw_dir / f"{tile_file_stem(t['url'])}.laz", raw_dir / f"tile_{i}.laz"):
                    if p.exists():
                        entry["local"] = str(p)
                        break
    return list(found.values())


def _fetch(tile: dict) -> str:
    if tile["local"]:
        return tile["local"]
    import requests
    dest = CACHE / "laz" / f"{tile_file_stem(tile['url'])}.laz"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {tile['tile_id']} …")
        r = requests.get(tile["url"], timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return str(dest)


def _select(tiles: list[dict], collection: str) -> list[dict]:
    years = sorted({tile_year(t) or 0 for t in tiles})
    if collection == "all":
        return sorted(tiles, key=lambda t: tile_year(t) or 0)
    want = years[-1] if collection == "newest" else years[0]
    return [t for t in tiles if (tile_year(t) or 0) == want]


@lru_cache(maxsize=8)
def _points(name: str, half_m: float, collection: str) -> np.ndarray:
    coord = wall_lab.site(name)
    key = f"{name.replace(' ', '_')}_{half_m:g}m_{collection}"
    cache = CACHE / "points" / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    tiles = _select(site_tiles(coord, half_m), collection)
    if not tiles:
        raise ValueError(f"No tiles cover {name}")
    x0, x1, y0, y1 = _bbox_ft(coord, half_m)
    stages = [{"type": "readers.las", "filename": _fetch(t), "default_srs": CRS} for t in tiles]
    stages += [{"type": "filters.merge"},
               {"type": "filters.crop", "bounds": f"([{x0},{x1}],[{y0},{y1}])"}]
    print(f"  reading {len(tiles)} tile(s) for {name}: {[t['tile_id'] for t in tiles]}")
    p = pdal.Pipeline(json.dumps(stages))
    p.execute()
    pts = p.arrays[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, pts)
    return pts


# ─── Ground classification + rasterisation ───────────────────────────────────

@lru_cache(maxsize=24)
def _ground(name: str, half_m: float, collection: str, gkey: tuple) -> np.ndarray:
    lp = dict(zip(["classifier", "max_scan_angle", "outlier", "outlier_mean_k", "outlier_multiplier",
                   "smrf_ignore_noise", "smrf_cell", "smrf_window", "smrf_slope",
                   "smrf_threshold", "smrf_scalar", "smrf_cut", "smrf_returns"], gkey))
    pts = _points(name, half_m, collection)
    if lp["max_scan_angle"] is not None:
        pts = pts[np.abs(pts["ScanAngleRank"]) <= lp["max_scan_angle"]]
    if lp["classifier"] == "vendor":
        return pts[pts["Classification"] == 2]
    stages = []
    if lp["outlier"]:
        stages.append({"type": "filters.outlier", "method": "statistical",
                       "mean_k": lp["outlier_mean_k"], "multiplier": lp["outlier_multiplier"]})
    smrf = {"type": "filters.smrf", "cell": lp["smrf_cell"], "window": lp["smrf_window"],
            "slope": lp["smrf_slope"], "threshold": lp["smrf_threshold"],
            "scalar": lp["smrf_scalar"], "cut": lp["smrf_cut"], "returns": lp["smrf_returns"]}
    if lp["smrf_ignore_noise"]:
        smrf["ignore"] = "Classification[7:7]"
    stages += [smrf, {"type": "filters.range", "limits": "Classification[2:2]"}]
    p = pdal.Pipeline(json.dumps(stages), arrays=[pts])
    p.execute()
    return p.arrays[0]


def _dtm_path(name, half_m, lp: LidarParams) -> Path:
    h = hashlib.sha1(json.dumps([name, half_m, asdict(lp)], sort_keys=True).encode()).hexdigest()[:12]
    return CACHE / "dtm" / f"{name.replace(' ', '_')}_{h}.tif"


def _rasterize(name: str, half_m: float, lp: LidarParams, ground: np.ndarray) -> Path:
    out = _dtm_path(name, half_m, lp)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = _bbox_ft(wall_lab.site(name), half_m)
    native = out.with_suffix(".native.tif")
    w = {"type": "writers.gdal", "filename": str(native), "resolution": lp.dtm_resolution,
         "output_type": lp.interp, "gdaldriver": "GTiff", "nodata": parameters.NODATA,
         "bounds": f"([{x0},{x1}],[{y0},{y1}])", "override_srs": CRS_3D,
         "window_size": lp.gdal_window_size}
    if lp.gdal_radius is not None:
        w["radius"] = lp.gdal_radius
    pdal.Pipeline(json.dumps([w]), arrays=[ground]).execute()
    gdal.Warp(str(out), str(native), dstSRS="EPSG:4326", resampleAlg=parameters.WARP_RESAMPLE,
              srcNodata=parameters.NODATA, dstNodata=parameters.NODATA)
    native.unlink(missing_ok=True)
    return out


# ─── Run ──────────────────────────────────────────────────────────────────────

@dataclass
class LidarResult:
    coord: dict
    lidar: LidarParams
    dtm_path: str
    stats: dict
    wall: wall_lab.Result
    ground_xy: np.ndarray = field(repr=False)    # (N, 2) ground points, feet, for density maps
    bbox_ft: tuple = field(repr=False)


def run_lidar(name_or_coord, lidar: LidarParams | None = None,
              wall: wall_lab.Params | None = None, size_m: float | None = None,
              margin_m: float = 40) -> LidarResult:
    """Crop → classify → rasterise → detect walls.  `size_m` is the wall window
    half-size (default: site's size_m); the crop adds `margin_m` so LRM / SMRF
    edge effects stay outside the window."""
    lp = lidar or LidarParams()
    wp_ = wall or wall_lab.Params()
    coord = wall_lab.site(name_or_coord)
    name = coord["name"]
    size_m = size_m or wp_.size_m or coord.get("size_m", 200)
    half_m = size_m + margin_m

    pts = _points(name, half_m, lp.collection)
    ground = _ground(name, half_m, lp.collection, lp._ground_key())
    dtm = _rasterize(name, half_m, lp, ground)
    res = wall_lab.run(coord, wp_.with_(size_m=size_m), dtm_path=str(dtm))

    area_m2 = (2 * half_m) ** 2
    nd = res.raw == res.nodata
    stats = {
        "points": int(len(pts)),
        "ground": int(len(ground)),
        "ground_pct": round(100 * len(ground) / max(len(pts), 1), 1),
        "pts_per_m2": round(len(pts) / area_m2, 2),
        "ground_per_m2": round(len(ground) / area_m2, 2),
        "vendor_ground_pct": round(100 * float(np.mean(pts["Classification"] == 2)), 1),
        "dtm_nodata_pct": round(100 * float(nd.mean()), 2),
        "pairs": len(res.pairs),
    }
    return LidarResult(coord=coord, lidar=lp, dtm_path=str(dtm), stats=stats, wall=res,
                       ground_xy=np.column_stack([ground["X"], ground["Y"]]),
                       bbox_ft=_bbox_ft(coord, half_m))


def sweep_lidar(name_or_coord, param: str, values, base: LidarParams | None = None,
                wall: wall_lab.Params | None = None, size_m: float | None = None) -> list[LidarResult]:
    base = base or LidarParams()
    return [run_lidar(name_or_coord, base.with_(**{param: v}), wall=wall, size_m=size_m)
            for v in values]


def clear_cache(disk: bool = False):
    _points.cache_clear()
    _ground.cache_clear()
    if disk:
        import shutil
        shutil.rmtree(CACHE, ignore_errors=True)
