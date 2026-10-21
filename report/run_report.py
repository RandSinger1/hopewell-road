"""
run_report.py  —  Report figure generator entry point.

Generates per-site figures into path-specific output directories:

    report/output/circleville/   sites with mode "circleville" or "all"
    report/output/chillicothe/   sites with mode "chillicothe" or "all"
    report/output/high_banks/    sites with mode "high_banks" or "all"

Within each directory every site's target azimuth is forced to that
corridor path's geometry (parabola tangent for circleville, fixed
bearing for chillicothe / high_banks).

Per-site figure generation (pipeline_steps, cross_sections, extra_views)
runs in parallel — one worker process per site.  The overview map runs
last, after all site caches are warm (detection results already on disk).

Usage
-----
    cd khthonlidar
    python -m report.run_report                          # all paths, all sites
    python -m report.run_report --path chillicothe       # one path
    python -m report.run_report --only "Van Voorhis Wall"  # one site, all paths
    python -m report.run_report --workers 4              # explicit parallelism
    python -m report.run_report --no-cache               # force full recompute
"""

import argparse
import io
import multiprocessing as mp
import os
import sys
sys.dont_write_bytecode = True
import time
import traceback
from pathlib import Path

from . import config as cfg
from .pipeline_steps import generate_pipeline_figure
from .cross_sections import generate_cross_section_figure
from .extra_views import generate_extra_views
from .overview_map import generate_overview_map
from .report_utils import site_paths
from .generate_report_doc import generate_report_doc

_ALL_PATHS = ("circleville", "chillicothe", "high_banks")
_REPO_REPORT = Path(__file__).resolve().parent


# ── worker (module-level so it is picklable on Windows) ──────────────────────

def _site_worker(args):
    """
    Generate pipeline_steps, cross_sections, and extra_views for one site.
    Runs in a subprocess; stdout is captured and returned so the parent can
    print each site's log as a block instead of interleaved lines.
    """
    coord, dtm_path, output_dir, use_cache = args
    name = coord["name"]
    t0 = time.time()
    errors = []

    buf = io.StringIO()
    _orig = sys.stdout
    sys.stdout = buf
    try:
        for label, fn in [
            ("pipeline_steps", generate_pipeline_figure),
            ("cross_sections", generate_cross_section_figure),
            ("extra_views",    generate_extra_views),
        ]:
            try:
                fn(coord, dtm_path, output_dir, use_cache=use_cache)
            except Exception:
                errors.append(f"  [ERROR] {name}/{label}:\n{traceback.format_exc()}")
    finally:
        sys.stdout = _orig

    return name, time.time() - t0, errors, buf.getvalue()


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="KhthonLiDAR per-path figure generator")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore cached results and recompute everything")
    p.add_argument("--only", action="append", default=None, metavar="NAME",
                   help="restrict to this site name (repeatable)")
    p.add_argument("--path", action="append", default=None, metavar="PATH",
                   dest="paths", choices=list(_ALL_PATHS),
                   help="generate only this path's output dir (repeatable; "
                        f"default: all three)")
    p.add_argument("--workers", type=int,
                   default=min(os.cpu_count() or 4, 8),
                   metavar="N",
                   help="parallel worker processes per path (default: cpu_count, max 8)")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = _parse_args()
    use_cache  = not args.no_cache
    req_paths  = tuple(args.paths) if args.paths else _ALL_PATHS
    only_names = set(args.only) if args.only else None

    _repo_root = _REPO_REPORT.parent
    dtm_path   = str(_repo_root / cfg.DTM_PATH)

    if not cfg.REPORT_COORDS:
        print("No coordinates configured in REPORT_COORDS — nothing to do.")
        return

    all_coords = cfg.REPORT_COORDS
    all_path_results: dict = {}   # {path_name: [forced_coord, ...]} for report doc

    for path_name in req_paths:
        path_coords = [c for c in all_coords if path_name in site_paths(c)]
        if only_names:
            path_coords = [c for c in path_coords if c["name"] in only_names]
        if not path_coords:
            print(f"  [path={path_name}] no matching sites — skipping.\n")
            continue

        path_output_dir = _REPO_REPORT / "output" / path_name
        n_workers = min(len(path_coords), args.workers)

        print(f"\n{'═' * 60}")
        print(f"  khthonlidar  ·  {path_name}")
        print(f"  DTM     : {dtm_path}")
        print(f"  Output  : {path_output_dir}")
        print(f"  Sites   : {len(path_coords)}")
        print(f"  Workers : {n_workers}")
        print(f"  Cache   : {'on' if use_cache else 'off (--no-cache)'}")
        print(f"{'═' * 60}\n")

        # Force this path's azimuth on every coord
        forced_coords = [{**c, "mode": path_name} for c in path_coords]
        all_path_results[path_name] = forced_coords
        work = [(c, dtm_path, path_output_dir, use_cache) for c in forced_coords]

        # ── per-site figures (parallel) ───────────────────────────────────────
        t_path = time.time()
        if n_workers > 1:
            with mp.Pool(processes=n_workers) as pool:
                for name, elapsed, errors, log in pool.imap_unordered(_site_worker, work):
                    print(f"── {name}  ({elapsed:.1f}s) ──")
                    if log.strip():
                        for line in log.rstrip().splitlines():
                            print(f"    {line}")
                    for err in errors:
                        print(err)
                    print()
        else:
            for a in work:
                name, elapsed, errors, log = _site_worker(a)
                print(f"── {name}  ({elapsed:.1f}s) ──")
                if log.strip():
                    for line in log.rstrip().splitlines():
                        print(f"    {line}")
                for err in errors:
                    print(err)
                print()

        # ── overview map (after site caches are warm) ─────────────────────────
        print("── Overview map ──")
        try:
            generate_overview_map(forced_coords, dtm_path, path_output_dir,
                                  use_cache=use_cache, path_name=path_name)
        except Exception as exc:
            print(f"  [ERROR] overview_map: {exc}")
            traceback.print_exc()

        print(f"\n  ✓ {path_name} complete in {time.time() - t_path:.1f}s\n")

    print(f"{'═' * 60}")
    print(f"  Done.  Outputs in {_REPO_REPORT / 'output'}/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
