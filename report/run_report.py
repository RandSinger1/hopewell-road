"""
run_report.py  —  Report generator entry point.

Usage
-----
    cd khthonlidar
    python -m report.run_report

Output
------
    report/output/<coord_name>/
        pipeline_steps.png          — 2×5 step-by-step pipeline grid
        cross_sections.png          — perpendicular elevation profiles
        multi_azimuth_hillshade.png — 4-direction hillshade comparison
        3d_perspective.png          — 3D terrain surface with wall overlays
        ridge_scale_comparison.png  — Frangi response at six sigma values
        elevation_statistics.png    — histogram, slope map, aspect rose
        road_elevation_profile.png  — along-road elevation + slope panel
        .cache/                     — cached DTM window / smoothing / LRM /
                                       detection results (see report/cache.py)

Edit report/config.py to add coordinates before running.

Caching
-------
Loading the DTM window and running median-smoothing / LRM / ridge
detection is the expensive part of report generation. Those results are
cached per coordinate (report/output/<name>/.cache/) and reused on the
next run as long as the coordinate and the relevant parameters.py
settings haven't changed. Pass --no-cache to force a full recompute.

Marking the true-positive wall pair
------------------------------------
When a coordinate has multiple detected parallel-wall pairs, only one is
usually the real earthwork; the rest are false positives. Workflow:

  1. Run the report once. The "12. Detected Pairs" panel in
     pipeline_steps.png (and the 3D/profile figures in extra_views) draws
     every candidate pair, each labelled with its index (#0, #1, …).
  2. Pick the correct index (or indices, if more than one pair is a real
     earthwork) from the figure and add it to that coordinate's
     REPORT_COORDS entry in parameters.py:
         "true_positive_pair_index": 0
     or, for multiple confirmed pairs:
         "true_positive_pair_index": [0, 2]
  3. Run again (--only that coordinate, cache still warm) — the confirmed
     pair(s) are highlighted at full strength and every other pair is
     greyed out, instead of all pairs looking equally valid. When more than
     one pair is confirmed, detected_pair_profiles.png is generated once per
     confirmed index (e.g. detected_pair_profiles_0.png, _2.png).

    python -m report.run_report --only "Van Voorhis Wall"
"""

import argparse
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path

from . import config as cfg
from .pipeline_steps import generate_pipeline_figure
from .cross_sections import generate_cross_section_figure
from .extra_views import generate_extra_views
from .overview_map import generate_overview_map


def _parse_args():
    p = argparse.ArgumentParser(description="KhthonLiDAR report generator")
    p.add_argument("--no-cache", action="store_true",
                    help="ignore cached terrain/detection results and recompute everything")
    p.add_argument("--only", action="append", default=None, metavar="NAME",
                    help="restrict the run to this coordinate name "
                         "(repeatable, e.g. --only Foo --only Bar)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    use_cache = not args.no_cache

    _repo_root = Path(__file__).resolve().parent.parent
    dtm_path = str(_repo_root / cfg.DTM_PATH)
    output_dir = _repo_root / cfg.OUTPUT_DIR

    if not cfg.REPORT_COORDS:
        print(
            "\nNo coordinates configured.\n"
            "Open report/config.py, add entries to REPORT_COORDS, then re-run.\n"
            "\nExample entry:\n"
            '  {"name": "Hopewell_Seg_A", "lat": 40.0, "lon": -82.4, "size_m": 200}\n'
        )
        return

    coords = cfg.REPORT_COORDS
    if args.only:
        wanted = set(args.only)
        coords = [c for c in coords if c["name"] in wanted]
        missing = wanted - {c["name"] for c in coords}
        if missing:
            print(f"  [WARN] --only name(s) not found in REPORT_COORDS: {sorted(missing)}")
        if not coords:
            print("  No matching coordinates — nothing to do.")
            return

    print(f"\n{'═' * 60}")
    print(f"  khthonlidar  ·  Report Generator")
    print(f"  DTM    : {dtm_path}")
    print(f"  Output : {output_dir}")
    print(f"  Coords : {len(coords)}")
    print(f"  Cache  : {'on' if use_cache else 'off (--no-cache)'}")
    print(f"{'═' * 60}\n")

    print("── Overview map ──")
    try:
        generate_overview_map(coords, dtm_path, output_dir, use_cache=use_cache)
    except Exception as exc:
        import traceback
        print(f"  [ERROR] overview_map: {exc}")
        traceback.print_exc()
    for coord in coords:
        name = coord["name"]
        t0 = time.time()
        print(f"── {name}  (lat={coord['lat']}, lon={coord['lon']}) ──")

        _run("pipeline_steps",  generate_pipeline_figure,
             coord, dtm_path, output_dir, use_cache=use_cache)
        _run("cross_sections",  generate_cross_section_figure,
             coord, dtm_path, output_dir, use_cache=use_cache)
        _run("extra_views",     generate_extra_views,
             coord, dtm_path, output_dir, use_cache=use_cache)

        print(f"  ✓ {name} done in {time.time() - t0:.1f}s  →  {output_dir / name}/\n")

    print(f"{'═' * 60}")
    print(f"  Report complete.  Open {output_dir} to view results.")
    print(f"{'═' * 60}\n")


def _run(label: str, fn, coord: dict, dtm_path: str, output_dir: Path, **kwargs) -> None:
    try:
        fn(coord, dtm_path, output_dir, **kwargs)
    except Exception as exc:
        import traceback
        print(f"  [ERROR] {label}: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
