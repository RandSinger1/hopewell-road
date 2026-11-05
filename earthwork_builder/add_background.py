"""
add_background.py — Add background sites (windows with no known earthwork) to sites.json.

    python -m earthwork_builder.add_background --random 20 [--seed 1] [--bbox W S E N]
    python -m earthwork_builder.add_background --at 39.9 -82.9 "bg Columbus downtown"
    python -m earthwork_builder.prepare          # download their LiDAR (or let the builder do it)

The earthwork windows only show terrain within ~500 m of known works (mostly
river terraces and farmland), so a first model has never seen towns, quarries,
strip mines, dissected hills or interchanges.  Background sites fill that gap:
they are flagged "background": true and build_dataset samples
EW_BACKGROUND_SITE_CHIPS random chips from each.

--random draws points uniformly over the box (default: Ohio), keeps those at
least EW_BACKGROUND_MIN_SEP_M from every existing site and covered (≥
EW_BACKGROUND_MIN_COVER) by OH_Statewide tiles — one ScienceBase query per try.
--at adds one hand-picked window (pick terrain types the model gets wrong).

Like any site, a background site trains only once marked reviewed in the
builder: glance over it, outline any earthwork you do find (it then trains as
a normal positive), and tick reviewed.  --reviewed marks them reviewed up front
instead (an unknown earthwork in a random 1 km window is rare, and would only
cost one wrong negative).
"""

from __future__ import annotations

import argparse

import numpy as np

import parameters
from earthwork_builder.labels import Site, load_sites, save_labels, save_sites
from lidar import point_io


def _distance_m(lat0, lon0, lat1, lon1) -> float:
    dy = (lat1 - lat0) * parameters.METERS_PER_DEGREE_LAT
    dx = (lon1 - lon0) * parameters.METERS_PER_DEGREE_LAT * np.cos(np.radians((lat0 + lat1) / 2))
    return float(np.hypot(dx, dy))


def coverage(lat: float, lon: float, half_m: float) -> float:
    """Fraction of the window that OH_Statewide tiles cover (ScienceBase query)."""
    bbox = point_io.lonlat_bbox(lat, lon, half_m)
    window = point_io.work_box({"min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3]})
    tiles = point_io.statewide_only(point_io.query_tiles(bbox))
    covered = point_io.shapely_union([point_io.work_box(t) for t in tiles])
    return covered.intersection(window).area / window.area


def random_sites(n: int, seed: int, bbox=parameters.EW_OHIO_BBOX, half_m: float = parameters.EW_SITE_HALF_M,
                 existing: list[Site] = (), max_tries: int | None = None) -> list[Site]:
    rng = np.random.default_rng(seed)
    taken = [(s.lat, s.lon) for s in existing]
    w, s, e, nth = bbox
    out, tries = [], 0
    max_tries = max_tries or 20 * n
    while len(out) < n and tries < max_tries:
        tries += 1
        lat, lon = round(float(rng.uniform(s, nth)), 5), round(float(rng.uniform(w, e)), 5)
        if any(_distance_m(lat, lon, a, b) < parameters.EW_BACKGROUND_MIN_SEP_M for a, b in taken):
            continue
        try:
            cov = coverage(lat, lon, half_m)
        except Exception as ex:
            print(f"  {lat:.5f}, {lon:.5f}: query failed ({ex}) - skipped")
            continue
        if cov < parameters.EW_BACKGROUND_MIN_COVER:
            continue
        site = Site(name=f"bg {lat:.5f} {lon:.5f}", lat=lat, lon=lon, half_m=half_m,
                    notes=f"random background (seed {seed})", extra={"background": True})
        print(f"  [{len(out) + 1}/{n}] {site.name}")
        out.append(site)
        taken.append((lat, lon))
    if len(out) < n:
        print(f"  only {len(out)} of {n} found in {tries} tries")
    return out


def add(new: list[Site], reviewed: bool = False) -> list[Site]:
    sites = load_sites()
    slugs = {s.slug for s in sites}
    added = [s for s in new if s.slug not in slugs]
    for s in new:
        if s.slug in slugs:
            print(f"  {s.name}: a site with that name already exists - skipped")
    if reviewed:
        for s in added:
            save_labels(s, {"type": "FeatureCollection", "features": [], "reviewed": True})
    save_sites(sites + added)
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--random", type=int, metavar="N", help="Add N random background sites")
    what.add_argument("--at", nargs=3, metavar=("LAT", "LON", "NAME"), help="Add one background site here")
    ap.add_argument("--seed", type=int, default=parameters.EW_SEED)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), default=parameters.EW_OHIO_BBOX)
    ap.add_argument("--half-m", type=float, default=parameters.EW_SITE_HALF_M)
    ap.add_argument("--notes", default="")
    ap.add_argument("--reviewed", action="store_true",
                    help="Mark the new sites reviewed (no outlines) without looking at them")
    args = ap.parse_args()
    if args.random:
        new = random_sites(args.random, args.seed, tuple(args.bbox), args.half_m, load_sites())
    else:
        lat, lon, name = float(args.at[0]), float(args.at[1]), args.at[2]
        new = [Site(name=name, lat=lat, lon=lon, half_m=args.half_m,
                    notes=args.notes or "background", extra={"background": True})]
    added = add(new, args.reviewed)
    print(f"Added {len(added)} background site(s) to {parameters.EW_SITES_FILE}; "
          f"run `python -m earthwork_builder.prepare` (or the builder) to download their LiDAR")


if __name__ == "__main__":
    main()
