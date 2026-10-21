"""
prune_superseded.py - One-off migration for tiles downloaded before
lidar/tile_filter.py existed.

Per sub-area:
  1. Spot-check that tile_<i> files really are entry <i> of the saved list
     (DTM tile centre must fall inside the listed lon/lat bbox).  Aborts the
     sub-area on any mismatch — renaming on a wrong mapping would mislabel data.
  2. Rename tile_<i>.laz / tile_<i>_ground.laz / tile_<i>_dtm.tif to the USGS
     file stem (the naming generate_tiles.py now uses).
  3. Drop superseded tiles (drop_superseded): delete their raw/ground/DTM
     files, back up the original list to <sub_area>.unfiltered.json, and
     rewrite <sub_area>.json without them.

Dry run by default; pass --apply to change anything.  Afterwards re-run the
merge (e.g. python -m lidar.run_pipeline full_road --skip-collect
--skip-download --skip-pdal) to rebuild merged_dtm.tif.

Usage:
  python -m lidar.prune_superseded full_road
  python -m lidar.prune_superseded Hopewell_Road_5 Circleville_Road_8 --apply
"""

import argparse
import json
import re
from pathlib import Path

from osgeo import gdal, osr

import parameters
from lidar.collect_images import get_polygons
from lidar.tile_filter import drop_superseded, summarize, tile_file_stem, tile_year

gdal.UseExceptions()

LISTS = Path("data/saved_file_lists")
TILES = Path("data/loaded_tiles")
# (sub-directory, filename suffix) for each per-tile artefact
ARTEFACTS = [("raw_tiles", ".laz"), ("ground_tiles", "_ground.laz"), ("dtm_tiles", "_dtm.tif")]


def _expand(names: list[str]) -> list[str]:
    out = []
    for n in names:
        bases = ["Hopewell_Road", "Circleville_Road"] if n == "full_road" else [n]
        for b in bases:
            numbered = sorted((p.stem for p in LISTS.glob(f"{b}_*.json")
                               if re.fullmatch(rf"{re.escape(b)}_\d+", p.stem)),
                              key=lambda s: int(s.rsplit("_", 1)[1]))
            out += numbered or [b]
    return out


def _aoi_for(sub_area: str):
    m = re.fullmatch(r"(.+)_(\d+)", sub_area)
    with open("data/polygons.json") as f:
        polygons = json.load(f)
    if m and m.group(1) in polygons:
        base = m.group(1)
        n = sum(1 for p in LISTS.glob(f"{base}_*.json")
                if re.fullmatch(rf"{re.escape(base)}_\d+", p.stem))
        return get_polygons(base, n_splits=n).get(sub_area)
    return polygons.get(sub_area)


def _centre_lonlat(tif: Path) -> tuple[float, float]:
    ds = gdal.Open(str(tif))
    gt = ds.GetGeoTransform()
    cx = gt[0] + gt[1] * ds.RasterXSize / 2
    cy = gt[3] + gt[5] * ds.RasterYSize / 2
    src = osr.SpatialReference()
    if ds.GetProjection():
        src.ImportFromWkt(ds.GetProjection())
    else:
        src.SetFromUserInput(str(parameters.SOURCE_CRS))
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lon, lat, _ = osr.CoordinateTransformation(src, dst).TransformPoint(cx, cy)
    return lon, lat


def _verify_mapping(sub_area: str, tiles: list[dict], n_checks: int) -> bool:
    """Spot-check tile_<i>_dtm.tif against list entry i at spread-out indices.

    Checks newest-collection tiles only (small files; OneDrive may have to
    download each one checked)."""
    dtm_dir = TILES / sub_area / "dtm_tiles"
    newest = max((tile_year(t) or 0) for t in tiles)
    candidates = [i for i, t in enumerate(tiles)
                  if (tile_year(t) or 0) == newest and (dtm_dir / f"tile_{i}_dtm.tif").exists()]
    if not candidates:
        return True
    picks = sorted({candidates[round(k * (len(candidates) - 1) / max(n_checks - 1, 1))]
                    for k in range(n_checks)})
    tol = 0.002  # deg; list bboxes are lon/lat envelopes of projected tiles
    for i in picks:
        t = tiles[i]
        lon, lat = _centre_lonlat(dtm_dir / f"tile_{i}_dtm.tif")
        if not (t["min_lon"] - tol <= lon <= t["max_lon"] + tol and
                t["min_lat"] - tol <= lat <= t["max_lat"] + tol):
            print(f"  MISMATCH tile_{i}_dtm.tif centre ({lon:.5f}, {lat:.5f}) "
                  f"is outside {t['tile_id']}'s bbox")
            return False
    print(f"  mapping verified on {len(picks)} tile(s): {picks}")
    return True


def prune(sub_area: str, apply: bool, n_checks: int, threshold: float) -> None:
    print(f"\n[{sub_area}]")
    list_path = LISTS / f"{sub_area}.json"
    if not list_path.exists():
        print("  no saved list, skipping")
        return
    tiles = json.loads(list_path.read_text())
    area_dir = TILES / sub_area

    # 1–2. index names → stem names
    to_rename = []
    for i, t in enumerate(tiles):
        stem = tile_file_stem(t["url"])
        for sub, suffix in ARTEFACTS:
            old = area_dir / sub / f"tile_{i}{suffix}"
            if old.exists():
                to_rename.append((old, area_dir / sub / f"{stem}{suffix}"))
    stray = [p for sub, _ in ARTEFACTS for p in (area_dir / sub).glob("tile_*")
             if re.fullmatch(r"tile_(\d+)(_ground|_dtm)?", p.stem)
             and int(re.match(r"tile_(\d+)", p.stem).group(1)) >= len(tiles)]
    if stray:
        print(f"  {len(stray)} tile_<i> file(s) have an index beyond the list "
              f"({len(tiles)} entries) — list doesn't match the download; skipping")
        return
    if to_rename:
        if not _verify_mapping(sub_area, tiles, n_checks):
            print("  skipping: index-to-tile mapping is wrong, nothing changed")
            return
        print(f"  rename {len(to_rename)} file(s) tile_<i> -> <tile_id>")
        if apply:
            for old, new in to_rename:
                if new.exists():
                    print(f"    {new.name} already exists, leaving {old.name}")
                else:
                    old.rename(new)

    # 3. drop superseded
    kept, dropped = drop_superseded(tiles, aoi=_aoi_for(sub_area), threshold=threshold)
    if not dropped:
        print("  no superseded tiles")
        return
    print(f"  drop {len(dropped)} superseded tile(s): {summarize(dropped)}")
    print(f"  keep {len(kept)}: {summarize(kept)}")
    n_files = 0
    for t in dropped:
        stem = tile_file_stem(t["url"])
        idx = tiles.index(t)
        for sub, suffix in ARTEFACTS:
            # After a dry run the files still carry index names
            for p in (area_dir / sub / f"{stem}{suffix}", area_dir / sub / f"tile_{idx}{suffix}"):
                if p.exists():
                    n_files += 1
                    if apply:
                        p.unlink()
    print(f"  delete {n_files} file(s) of dropped tiles")
    if apply:
        backup = LISTS / f"{sub_area}.unfiltered.json"
        if not backup.exists():
            backup.write_text(list_path.read_text())
        list_path.write_text(json.dumps(kept))
        print(f"  list rewritten ({len(tiles)} -> {len(kept)}); original in {backup.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("areas", nargs="+",
                    help="Sub-areas or base areas (expanded to their numbered lists); 'full_road' allowed")
    ap.add_argument("--apply", action="store_true", help="Actually rename/delete/rewrite (default: dry run)")
    ap.add_argument("--checks", type=int, default=3, help="DTM tiles to spot-check per sub-area (default 3)")
    ap.add_argument("--threshold", type=float, default=parameters.SUPERSEDED_COVERAGE_THRESHOLD)
    args = ap.parse_args()
    if not args.apply:
        print("DRY RUN - pass --apply to change files.")
    for sa in _expand(args.areas):
        prune(sa, args.apply, args.checks, args.threshold)


if __name__ == "__main__":
    main()
