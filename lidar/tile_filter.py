"""
tile_filter.py — Drop LiDAR tiles that a newer collection already covers.

USGS ScienceBase returns every collection that intersects an AOI, so the
Ohio corridor lists mix the legacy OSIP I flights (OH_SOUTH_2006/2007,
~1 pt/m²) with the OSIP III statewide flights (OH_Statewide_Phase2_2020 /
Phase3_2021, QL2).  Mosaicking both lets the sparse legacy surface overwrite
the newer one wherever it happens to be warped last.

A tile is "superseded" when at least ``threshold`` of its footprint (clipped
to the AOI, since newer tiles are only listed inside the AOI) is covered by
tiles from strictly newer collections.  Coverage is estimated on a regular
grid of sample points inside the tile's lon/lat bbox.

Tiles are also named by their USGS file stem (``tile_file_stem``) so a
filtered list doesn't shift the on-disk names of the tiles that remain.
"""

import re
from pathlib import PurePosixPath
from urllib.parse import urlparse

import numpy as np
from matplotlib.path import Path as MplPath

_YEAR_RE = re.compile(r"_((?:19|20)\d{2})(?=_|$)")


def tile_file_stem(url: str) -> str:
    """File stem of a tile URL, e.g. 'USGS_LPC_OH_SOUTH_2006_001055'."""
    return PurePosixPath(urlparse(url).path).stem


def collection_of(tile: dict) -> str:
    """Collection (project) name from a tile's URL path, e.g. 'OH_SOUTH_2006'."""
    parts = PurePosixPath(urlparse(tile["url"]).path).parts
    if "Projects" in parts:
        rest = [p for p in parts[parts.index("Projects") + 1:] if p != "legacy"]
        if rest:
            return rest[0]
    return tile_file_stem(tile["url"])


def collection_year(name: str) -> int | None:
    """First 4-digit year embedded in a collection or tile name, or None."""
    m = _YEAR_RE.search(name)
    return int(m.group(1)) if m else None


def tile_year(tile: dict) -> int | None:
    return collection_year(collection_of(tile))


def _sample_points(t: dict, grid: int) -> np.ndarray:
    xs = np.linspace(t["min_lon"], t["max_lon"], grid + 2)[1:-1]
    ys = np.linspace(t["min_lat"], t["max_lat"], grid + 2)[1:-1]
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def drop_superseded(tiles: list[dict], aoi: list | None = None,
                    threshold: float = 0.95, grid: int = 16
                    ) -> tuple[list[dict], list[dict]]:
    """Split ``tiles`` into (kept, dropped).

    aoi       : optional polygon as a list of (lon, lat) vertices; coverage is
                measured only over the part of each tile inside it.
    threshold : fraction of the (AOI-clipped) footprint that newer
                collections must cover for a tile to be dropped.
    grid      : sample points per bbox side used to estimate coverage.
    """
    years = [tile_year(t) for t in tiles]
    if len({y for y in years if y is not None}) < 2:
        return list(tiles), []

    boxes = np.array([[t["min_lon"], t["min_lat"], t["max_lon"], t["max_lat"]]
                      for t in tiles])
    aoi_path = MplPath(np.asarray(aoi, dtype=float)) if aoi else None

    kept, dropped = [], []
    for t, y, box in zip(tiles, years, boxes):
        if y is None:
            kept.append(t)
            continue
        newer = np.array([yy is not None and yy > y for yy in years])
        # Only newer boxes that actually intersect this tile
        newer &= ((boxes[:, 0] < box[2]) & (boxes[:, 2] > box[0]) &
                  (boxes[:, 1] < box[3]) & (boxes[:, 3] > box[1]))
        if not newer.any():
            kept.append(t)
            continue

        pts = _sample_points(t, grid)
        if aoi_path is not None:
            inside = aoi_path.contains_points(pts)
            if not inside.any():
                # Tile only clips the AOI edge; a newer tile intersects it, so
                # it contributes nothing the newer data doesn't.
                dropped.append(t)
                continue
            pts = pts[inside]

        nb = boxes[newer]
        covered = ((pts[:, None, 0] >= nb[None, :, 0]) & (pts[:, None, 0] <= nb[None, :, 2]) &
                   (pts[:, None, 1] >= nb[None, :, 1]) & (pts[:, None, 1] <= nb[None, :, 3])
                   ).any(axis=1)
        (dropped if covered.mean() >= threshold else kept).append(t)
    return kept, dropped


def summarize(tiles: list[dict]) -> str:
    """'OH_SOUTH_2006: 52, OH_Statewide_Phase2_2020: 636'-style count string."""
    counts: dict[str, int] = {}
    for t in tiles:
        c = collection_of(t)
        counts[c] = counts.get(c, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"


def merge_order_key(path) -> tuple[int, str]:
    """Sort key putting older collections first so newer DTM tiles win overlaps
    (gdal.Warp lets later sources overwrite earlier ones)."""
    name = PurePosixPath(str(path).replace("\\", "/")).name
    return (collection_year(name.rsplit(".", 1)[0]) or 0, name)
