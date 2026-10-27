"""
import_candidates.py — Turn scan candidates into builder sites for review.

    python -m earthwork_builder.import_candidates --run ohio_v1 [--min-score 0.7] [--limit 50]
    python -m earthwork_builder.prepare          # download LiDAR for the new sites

Each candidate becomes a site named "cand <id>" (window EW_CANDIDATE_HALF_M)
whose labels.geojson is prefilled with the predicted outline, typed
EW_UNVERIFIED_TYPE.  In the builder, retype it as an earthwork type (true
positive — adjust the outline), as a look-alike type (false positive, becomes
a hard negative), or delete it; outline anything it missed; then mark the
site reviewed and rebuild the dataset.  This is the active-learning loop:
every reviewed batch teaches the next model about its own mistakes.

Candidates already imported (same site name) are skipped, so re-running after
more of the scan has finished only adds the new ones.
"""

from __future__ import annotations

import argparse
import sqlite3

from shapely import wkt as shapely_wkt
from shapely.geometry import mapping

import parameters
from earthwork_builder.labels import Site, load_sites, save_labels, save_sites

SCAN_ROOT = parameters.EW_ROOT / "scan"


def import_candidates(run: str, min_score: float = 0.0, limit: int | None = None) -> list[Site]:
    db_path = SCAN_ROOT / run / "scan.sqlite"
    if not db_path.exists():
        raise SystemExit(f"No scan run at {db_path}")
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM candidates WHERE score >= ? ORDER BY score DESC",
                      (min_score,)).fetchall()
    sites = load_sites()
    existing = {s.slug for s in sites}
    added = []
    for r in rows:
        if limit is not None and len(added) >= limit:
            break
        site = Site(name=f"cand {r['id']}", lat=round(r["lat"], 7), lon=round(r["lon"], 7),
                    half_m=parameters.EW_CANDIDATE_HALF_M,
                    notes=f"scan {run}: score {r['score']:.2f}, {r['area_m2']:.0f} m2")
        if site.slug in existing:
            continue
        outline = {"type": "Feature", "geometry": mapping(shapely_wkt.loads(r["wkt"])),
                   "properties": {"type": parameters.EW_UNVERIFIED_TYPE,
                                  "notes": f"predicted, score {r['score']:.2f}"}}
        save_labels(site, {"type": "FeatureCollection", "features": [outline], "reviewed": False})
        added.append(site)
        existing.add(site.slug)
    save_sites(sites + added)
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Scan run name (folder under data/earthworks/scan/)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--limit", type=int, help="Import at most this many (highest scores first)")
    args = ap.parse_args()
    added = import_candidates(args.run, args.min_score, args.limit)
    print(f"Added {len(added)} candidate site(s) to {parameters.EW_SITES_FILE}; "
          f"run `python -m earthwork_builder.prepare` to download their LiDAR")


if __name__ == "__main__":
    main()
