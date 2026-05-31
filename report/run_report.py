"""
run_report.py  —  Report generator entry point.

Usage
-----
    cd khthonlidar
    python report/run_report.py

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

Edit report/config.py to add coordinates before running.
"""

import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path

_REPORT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _REPORT_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPORT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg
from pipeline_steps import generate_pipeline_figure
from cross_sections import generate_cross_section_figure
from extra_views import generate_extra_views


def main() -> None:
    dtm_path = str(_REPO_ROOT / cfg.DTM_PATH)
    output_dir = _REPO_ROOT / cfg.OUTPUT_DIR

    if not cfg.REPORT_COORDS:
        print(
            "\nNo coordinates configured.\n"
            "Open report/config.py, add entries to REPORT_COORDS, then re-run.\n"
            "\nExample entry:\n"
            '  {"name": "Hopewell_Seg_A", "lat": 40.0, "lon": -82.4, "size_m": 200}\n'
        )
        return

    print(f"\n{'═' * 60}")
    print(f"  khthonlidar  ·  Report Generator")
    print(f"  DTM    : {dtm_path}")
    print(f"  Output : {output_dir}")
    print(f"  Coords : {len(cfg.REPORT_COORDS)}")
    print(f"{'═' * 60}\n")

    for coord in cfg.REPORT_COORDS:
        name = coord["name"]
        t0 = time.time()
        print(f"── {name}  (lat={coord['lat']}, lon={coord['lon']}) ──")

        _run("pipeline_steps",  generate_pipeline_figure,       coord, dtm_path, output_dir)
        _run("cross_sections",  generate_cross_section_figure,  coord, dtm_path, output_dir)
        _run("extra_views",     generate_extra_views,            coord, dtm_path, output_dir)

        print(f"  ✓ {name} done in {time.time() - t0:.1f}s  →  {output_dir / name}/\n")

    print(f"{'═' * 60}")
    print(f"  Report complete.  Open {output_dir} to view results.")
    print(f"{'═' * 60}\n")


def _run(label: str, fn, coord: dict, dtm_path: str, output_dir: Path) -> None:
    try:
        fn(coord, dtm_path, output_dir)
    except Exception as exc:
        import traceback
        print(f"  [ERROR] {label}: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
