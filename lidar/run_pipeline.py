"""
run_pipeline.py — Full LiDAR pipeline in one shot.

Steps:
  1. collect_images  — fetch tile metadata from the USGS/OGRIP API
  2. generate_tiles  — download raw LAZ tiles in parallel
  3. pdal_pipeline   — denoise, ground-classify, rasterise; merge all DTMs
  4. quick_view      — save a preview image of the merged DTM

Usage:
  python run_pipeline.py Hopewell_Road
  python run_pipeline.py Hopewell_Road Circleville_Road        # merged DTM
  python run_pipeline.py Hart_Island --separate-year           # one DTM per year
  python run_pipeline.py Hopewell_Road --area-splits 5
  python run_pipeline.py full_road                             # shorthand for both roads
"""

import argparse
import json
import re
from multiprocessing import freeze_support
from pathlib import Path

from lidar.collect_images import main as _collect_images
from lidar.generate_tiles import generate_tiles
from lidar.pdal_pipeline import process_tiles, merge_dtms
from lidar.quick_view import quick_view_dtm
import parameters


# ── helpers ──────────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"_(\d{4})$")


def _resolve_area(area: str) -> list[str]:
    """Expand shorthand aliases to constituent area names."""
    if area == "full_road":
        return ["Hopewell_Road", "Circleville_Road"]
    return [area]


def _discover_sub_areas(area: str) -> list[str]:
    """Discover sub-areas from previously saved JSON file lists."""
    found = sorted(
        p for p in Path("data/saved_file_lists").glob(f"{area}_*.json")
        if re.fullmatch(rf"{re.escape(area)}_\d+", p.stem)
    )
    if found:
        return [p.stem for p in found]
    return [area]


def _load_tiles_for(sub_area: str) -> list[str]:
    path = Path("data/saved_file_lists") / f"{sub_area}.json"
    if not path.exists():
        print(f"  [warn] no file list found for {sub_area}, skipping")
        return []
    with open(path) as f:
        return [t["url"] for t in json.load(f)]


def _merged_output_dir(areas: list[str]) -> Path:
    if len(areas) == 1:
        return Path("data/loaded_tiles") / areas[0]
    if set(areas) == {"Hopewell_Road", "Circleville_Road"}:
        return Path("data/loaded_tiles/Whole_Road")
    return Path("data/loaded_tiles") / "_".join(areas)


def _year_of(sub_area: str) -> str | None:
    m = _YEAR_RE.search(sub_area)
    return m.group(1) if m else None


# ── pipeline steps ────────────────────────────────────────────────────────────

def step_collect(area: str, n_splits: int) -> list[str]:
    print(f"\n=== Step 1 — collect images [{area}] ===")
    print(f"  Fetching metadata for {area} …")
    return _collect_images(area, n_splits=n_splits)


def step_generate(sub_areas: list[str], max_workers: int = 4):
    print(f"\n=== Step 2 — generate tiles [{len(sub_areas)} sub-areas] ===")
    for sub_area in sub_areas:
        tiles = _load_tiles_for(sub_area)
        if not tiles:
            continue
        print(f"  Downloading {len(tiles)} tiles → {sub_area} …")
        generate_tiles(tiles, sub_area, max_workers=max_workers)


def step_pdal(sub_areas: list[str], max_workers: int = 5):
    print(f"\n=== Step 3 — PDAL pipeline [{len(sub_areas)} sub-areas] ===")
    for sa in sub_areas:
        print(f"  [{sa}] starting …")
        try:
            process_tiles(sa, max_workers=max_workers)
            print(f"  [{sa}] done")
        except Exception as exc:
            print(f"  [{sa}] ERROR: {exc}")


def step_merge(areas: list[str], sub_areas: list[str], separate_year: bool, workers: int = 1):
    print("\n=== Step 4 — merge DTMs ===")
    if len(sub_areas) <= 1:
        print("  Only one sub-area — skipping merge.")
        return

    out_dir = _merged_output_dir(areas)

    if separate_year:
        by_year: dict[str, list[str]] = {}
        for sa in sub_areas:
            year = _year_of(sa)
            if year:
                by_year.setdefault(year, []).append(sa)

        if by_year:
            for year, year_subs in sorted(by_year.items()):
                out_path = out_dir / f"merged_dtm_{year}.tif"
                print(f"  Merging {len(year_subs)} sub-area(s) for {year} → {out_path}")
                merge_dtms(year_subs, out_path, workers=workers)
            return

    out_path = out_dir / "merged_dtm.tif"
    print(f"  Merging {len(sub_areas)} sub-areas → {out_path}")
    merge_dtms(sub_areas, out_path, workers=workers)


def step_view(areas: list[str], sub_areas: list[str], separate_year: bool, sample_step: int):
    print("\n=== Step 5 — quick view ===")
    out_dir = _merged_output_dir(areas)
    label = out_dir.name  # e.g. "Hart_Island", "Whole_Road"
    Path("data/imgs").mkdir(parents=True, exist_ok=True)

    renders: list[tuple[str, str]] = []  # (dtm_path, out_png)
    if separate_year:
        years = sorted(y for sa in sub_areas if (y := _year_of(sa)))
        for year in years:
            renders.append((
                str(out_dir / f"merged_dtm_{year}.tif"),
                f"data/imgs/quick_view_{label}_{year}.png",
            ))
    elif len(sub_areas) > 1:
        renders.append((
            str(out_dir / "merged_dtm.tif"),
            f"data/imgs/quick_view_{label}.png",
        ))
    else:
        dtm_dir = Path(f"data/loaded_tiles/{sub_areas[0]}/dtm_tiles")
        tifs = sorted(dtm_dir.glob("*.tif")) if dtm_dir.exists() else []
        dtm_path = str(tifs[0]) if tifs else parameters.REPORT_DTM_PATH
        renders.append((dtm_path, f"data/imgs/quick_view_{sub_areas[0]}.png"))

    show_cities = (label == "Whole_Road")
    for dtm_path, out_png in renders:
        print(f"  Rendering {dtm_path} → {out_png}")
        quick_view_dtm(dtm_path, sample_step=sample_step, cmap="terrain", outpath=out_png,
                       show_cities=show_cities)
        print(f"  Saved: {out_png}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full KhthonLiDAR pipeline end-to-end."
    )
    parser.add_argument(
        "area",
        type=str,
        nargs="+",
        help=(
            "Area(s) to process. Pass multiple names to merge them into one DTM. "
            "'full_road' expands to Hopewell_Road + Circleville_Road."
        ),
    )
    parser.add_argument(
        "--separate-year", action="store_true",
        help="Produce a separate merged DTM per year (e.g. Hart_Island 2014, 2015, …)",
    )
    parser.add_argument(
        "--area-splits", type=int, default=15, metavar="N",
        help="Sub-segments to split road-type polygons into (default 10)",
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Skip Step 1 (metadata already downloaded)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Step 2 (tiles already downloaded)",
    )
    parser.add_argument(
        "--skip-pdal", action="store_true",
        help="Skip Step 3 (PDAL processing already done)",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip Step 4 (DTM merge already done)",
    )
    parser.add_argument(
        "--skip-view", action="store_true",
        help="Skip Step 5 (quick view already done)",
    )
    parser.add_argument(
        "--download-workers", type=int, default=4,
        help="Parallel download threads (default 4)",
    )
    parser.add_argument(
        "--pdal-workers", type=int, default=5,
        help="Parallel PDAL processes (default 5)",
    )
    parser.add_argument(
        "--sample-step", type=int, default=1000,
        help="Pixel sampling step for quick view (default 1000)",
    )
    args = parser.parse_args()

    # Expand aliases (full_road → Hopewell_Road Circleville_Road)
    areas: list[str] = []
    for a in args.area:
        areas.extend(_resolve_area(a))

    # Step 1: collect metadata and determine sub-areas
    all_sub_areas: list[str] = []
    if not args.skip_collect:
        for area in areas:
            all_sub_areas.extend(step_collect(area, args.area_splits))
    else:
        for area in areas:
            all_sub_areas.extend(_discover_sub_areas(area))

    if not all_sub_areas:
        print("No sub-areas found — aborting.")
        return

    if not args.skip_download:
        step_generate(all_sub_areas, max_workers=args.download_workers)

    if not args.skip_pdal:
        step_pdal(all_sub_areas, max_workers=args.pdal_workers)

    if not args.skip_merge:
        step_merge(areas, all_sub_areas, args.separate_year, workers=args.pdal_workers)

    if not args.skip_view:
        step_view(areas, all_sub_areas, args.separate_year, args.sample_step)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    freeze_support()
    main()
