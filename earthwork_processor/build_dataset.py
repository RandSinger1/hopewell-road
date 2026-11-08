"""
build_dataset.py — Turn reviewed builder sites into augmented training chips.

    python -m earthwork_processor.build_dataset [--name default] [--only SLUG ...]
                                                [--classes mound earthwork]

For every site whose labels.geojson is marked reviewed:
  * sites/<slug>/features.tif — the full-window feature stack (one band per
    channel, EW_WORK_CRS) for inspection in QGIS;
  * EW_CHIPS_PER_LABEL chips around each earthwork outline (one un-augmented,
    the rest randomly rotated / scaled / exaggerated / noised, with the outline
    jittered inside the chip);
  * EW_CHIPS_PER_NEGATIVE chips around each look-alike outline
    (EW_NEGATIVE_TYPES: levees, rail grades, ...) — hard negatives: the
    look-alike is in the chip but is NOT a target.  Only with
    EW_LOOKALIKE_TRAINING = "hard_negative"; with "ignore" look-alike pixels
    are also blanked (valid = False) in every chip;
  * EW_BACKGROUND_CHIPS_PER_SITE chips centred at random, which mostly
    contain no earthwork (EW_BACKGROUND_SITE_CHIPS for "background" sites).

Outlines of EW_EXCLUDED_TYPES (e.g. "road") are left out: no chips of their
own, and their pixels are blanked (valid = False) in every chip.

Outline-centred chips stop at EW_MAX_OUTLINE_CHIPS_PER_SITE chips or
EW_MAX_INSTANCES_PER_SITE instances per site (taken round-robin over the
outlines), so a dense mound group cannot dominate: uncapped, Mound City's
23 mounds were 73% of v1's training instances.

Output: EW_ROOT/dataset/<name>/
    chips/<slug>_<k>.npz   feats (C,N,N) float16, valid (N,N), poly_xy (M,2),
                           poly_len (K,), poly_cls (K,)  — outlines in chip px
    manifest.json          feature spec, classes, per-site split, chip list,
                           train-split channel mean/std (used by the model)

Sites are split train/val as a whole (a site's chips are near-duplicates of
each other, so splitting chips would leak).  Set "split" in sites.json to pin one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

import parameters
from earthwork_builder.labels import Site, is_negative, label_polygons, load_sites, read_labels
from earthwork_processor.augment import ChipTransform, make_chip, rasterize_rings
from earthwork_processor.features import FeatureSpec, compute_features
from lidar import point_io
from lidar.ground_grid import GroundGrid

DATASET_ROOT = parameters.EW_ROOT / "dataset"

LOOKALIKE_MODES = ("hard_negative", "background", "ignore")
_IGNORE_PAD_PX = 3                 # "ignore" also blanks this margin around a look-alike
EXCLUDED = -1                      # class_index of an EW_EXCLUDED_TYPES outline


def class_index(label_type: str, classes=None) -> int | None:
    """Target class of an outline type, None for a look-alike (hard negative),
    EXCLUDED for an EW_EXCLUDED_TYPES outline (blanked, never trained on)."""
    if is_negative(label_type):
        return None
    if label_type in parameters.EW_EXCLUDED_TYPES:
        return EXCLUDED
    if label_type == parameters.EW_UNVERIFIED_TYPE:
        raise ValueError(f"'{label_type}' outline in a reviewed site")
    classes = list(classes or parameters.EW_CLASSES)
    if label_type in classes:
        return classes.index(label_type)
    if "earthwork" in classes:
        return classes.index("earthwork")
    raise ValueError(f"Label type {label_type!r} not in EW_CLASSES and no 'earthwork' fallback")


def site_split(site: Site) -> str:
    """"split" from sites.json, else a hash of the site's "group" (sites whose
    windows overlap share a group, so they never straddle train/val), else of its slug."""
    if site.split in ("train", "val"):
        return site.split
    key = site.extra.get("group") or site.slug
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 1000
    return "val" if h < 1000 * parameters.EW_VAL_FRACTION else "train"


def outlines_px(site: Site, grid: GroundGrid, classes=None) -> list[tuple[list[np.ndarray], int | None]]:
    """[(rings, class index / None for a look-alike / EXCLUDED), …]; rings = [exterior, *holes],
    each (K, 2) in grid (col, row).  Holes are interiors drawn inside an outline
    (an enclosure's inside): not part of the instance, so they train as background."""
    out = []
    for poly, props in label_polygons(site):
        rings = []
        for ring in [poly.exterior, *poly.interiors]:
            lon, lat = np.asarray(ring.coords).T
            x, y = point_io.lonlat_to_work(lon, lat)
            c, r = grid.to_pixel(x, y)
            rings.append(np.column_stack([c, r]))
        out.append((rings, class_index(props.get("type", "other"), classes)))
    return out


def _random_transform(rng, col, row, augment: bool) -> ChipTransform:
    if not augment:
        return ChipTransform(col, row)
    return ChipTransform(
        col, row,
        angle_deg=rng.uniform(0, 360) if parameters.EW_AUG_ROTATION else 0.0,
        scale=rng.uniform(*parameters.EW_AUG_SCALE),
        z_exaggeration=rng.uniform(*parameters.EW_AUG_Z_EXAGGERATION),
        noise_m=parameters.EW_AUG_NOISE_M,
    )


def chip_plan(rng, grid: GroundGrid, outlines, size: int,
              n_background: int = parameters.EW_BACKGROUND_CHIPS_PER_SITE,
              lookalikes: str = parameters.EW_LOOKALIKE_TRAINING) -> list[tuple[ChipTransform, bool]]:
    """Centres + augmentations for one site's chips: [(transform, outline-centred?), ...].

    Outline-centred chips come round-robin over the outlines (in random order):
    every outline's un-augmented chip first, then each one's first augmented
    chip, ... — so when build() stops early (EW_MAX_OUTLINE_CHIPS_PER_SITE /
    EW_MAX_INSTANCES_PER_SITE) the chips it kept still cover all outlines.
    Background chips follow."""
    half = size / 2
    n_negative = parameters.EW_CHIPS_PER_NEGATIVE if lookalikes == "hard_negative" else 0
    per_outline = []
    for (poly, *_), cls in outlines:
        centre = poly.mean(axis=0) - 0.5                   # edge → index coords
        radius = np.linalg.norm(poly - poly.mean(axis=0), axis=1).max()
        # Jitter so the outline lands anywhere in the chip while staying (mostly) inside
        max_shift = max(0.0, 0.8 * half - radius)
        n = (n_negative if cls is None else 0 if cls == EXCLUDED else parameters.EW_CHIPS_PER_LABEL)
        tfs = []
        for k in range(n):
            shift = np.zeros(2)
            if k:
                ang, rad = rng.uniform(0, 2 * np.pi), max_shift * np.sqrt(rng.uniform())
                shift = rad * np.array([np.cos(ang), np.sin(ang)])
            tfs.append(_random_transform(rng, centre[0] + shift[0], centre[1] + shift[1], augment=k > 0))
        per_outline.append(tfs)
    order = rng.permutation(len(per_outline)) if per_outline else []
    rounds = max((len(t) for t in per_outline), default=0)
    plan = [(per_outline[i][k], True) for k in range(rounds) for i in order if k < len(per_outline[i])]
    rows, cols = np.nonzero(np.isfinite(grid.z))
    if len(rows):
        for _ in range(n_background):
            i = rng.integers(len(rows))
            plan.append((_random_transform(rng, float(cols[i]), float(rows[i]), augment=True), False))
    return plan


def _stack(rings: list[np.ndarray]) -> np.ndarray:
    return (np.concatenate(rings) if rings else np.zeros((0, 2))).astype(np.float32)


class _Stats:
    """Streaming per-channel mean / std over valid pixels."""

    def __init__(self, c: int):
        self.n = 0
        self.s = np.zeros(c)
        self.ss = np.zeros(c)

    def add(self, feats: np.ndarray, valid: np.ndarray) -> None:
        v = feats[:, valid].astype(np.float64)
        self.n += v.shape[1]
        self.s += v.sum(axis=1)
        self.ss += (v ** 2).sum(axis=1)

    def result(self) -> tuple[list[float], list[float]]:
        mean = self.s / max(self.n, 1)
        std = np.sqrt(np.maximum(self.ss / max(self.n, 1) - mean ** 2, 1e-12))
        return mean.tolist(), std.tolist()


def build(name: str = "default", only: list[str] | None = None, spec: FeatureSpec | None = None,
          lookalikes: str = parameters.EW_LOOKALIKE_TRAINING, classes=None) -> Path:
    """classes: target classes, default EW_CLASSES (types not listed train as "earthwork")."""
    classes = tuple(classes or parameters.EW_CLASSES)
    if lookalikes not in LOOKALIKE_MODES:
        raise ValueError(f"EW_LOOKALIKE_TRAINING must be one of {LOOKALIKE_MODES}, not {lookalikes!r}")
    spec = spec or FeatureSpec()
    size = parameters.EW_CHIP_PX
    out = DATASET_ROOT / name
    chips_dir = out / "chips"
    if chips_dir.exists():
        shutil.rmtree(chips_dir)
    chips_dir.mkdir(parents=True)

    rng = np.random.default_rng(parameters.EW_SEED)
    stats = _Stats(len(spec.channels))
    manifest = {"feature_spec": spec.to_dict(), "chip_px": size,
                "classes": list(classes), "lookalikes": lookalikes,
                "excluded_types": list(parameters.EW_EXCLUDED_TYPES),
                "max_outline_chips_per_site": parameters.EW_MAX_OUTLINE_CHIPS_PER_SITE,
                "max_instances_per_site": parameters.EW_MAX_INSTANCES_PER_SITE,
                "sites": {}, "chips": []}

    for site in load_sites():
        if only and site.slug not in only and site.name not in only:
            continue
        if not site.prepared:
            why = "moved/resized since prepared" if site.stale else "not prepared"
            print(f"[{site.name}] skipped: {why} (run earthwork_builder.prepare)")
            continue
        fc = read_labels(site)
        if fc.get("rejected"):
            print(f"[{site.name}] skipped: rejected ({fc.get('reject_reason') or 'no reason given'})")
            continue
        if not fc.get("reviewed"):
            print(f"[{site.name}] skipped: labels not marked reviewed")
            continue
        source = json.loads((site.dir / "site.json").read_text()).get("ground_source", "points")
        if source != spec.ground_source:
            raise ValueError(f"{site.name}: ground.tif is from {source!r} but the spec wants "
                             f"{spec.ground_source!r} - run `python -m earthwork_builder.prepare` to rebuild it")
        grid = GroundGrid.from_geotiff(site.ground_path)
        if abs(grid.pixel_m - spec.pixel_m) > 1e-6:
            raise ValueError(f"{site.name}: ground.tif is {grid.pixel_m} m/px but the spec wants "
                             f"{spec.pixel_m} - re-run prepare --force")
        feats, _ = compute_features(grid.z, spec)
        grid.to_geotiff(site.dir / "features.tif", bands=feats, band_names=list(spec.channels))

        outlines = outlines_px(site, grid, classes)
        split = site_split(site)
        n = n_outline_chips = n_instances = 0
        all_rings = [ring for rings, _ in outlines for ring in rings]
        n_bg = parameters.EW_BACKGROUND_SITE_CHIPS if site.background else parameters.EW_BACKGROUND_CHIPS_PER_SITE
        for tf, outline_centred in chip_plan(rng, grid, outlines, size, n_bg, lookalikes):
            if outline_centred and (n_outline_chips >= parameters.EW_MAX_OUTLINE_CHIPS_PER_SITE
                                    or n_instances >= parameters.EW_MAX_INSTANCES_PER_SITE):
                continue                      # site has its share - keep dense groups from dominating
            f, valid, flat = make_chip(grid, tf, spec, size, all_rings, rng)
            if valid.mean() < 0.5:
                continue
            keep, k = [], 0                   # regroup the transformed rings per outline
            blank = np.zeros((size, size), bool)
            for rings, cls in outlines:
                chip_rings, k = flat[k:k + len(rings)], k + len(rings)
                if cls is None or cls == EXCLUDED:
                    if cls == EXCLUDED or lookalikes == "ignore":
                        blank |= rasterize_rings(chip_rings, size)
                elif rasterize_rings(chip_rings, size).sum() >= parameters.EW_MIN_INSTANCE_PX:
                    keep.append((chip_rings, cls))
            if blank.any():
                # Never blank an earthwork a look-alike overlaps (e.g. a road through a wall)
                blank = binary_dilation(blank, iterations=_IGNORE_PAD_PX)
                for rings, _ in keep:
                    blank &= ~rasterize_rings(rings, size)
                valid = valid & ~blank
            holes = [(h, j) for j, (rings, _) in enumerate(keep) for h in rings[1:]]
            fname = f"{site.slug}_{n:04d}.npz"
            np.savez_compressed(
                chips_dir / fname, feats=f.astype(np.float16), valid=valid,
                poly_xy=_stack([r[0] for r, _ in keep]),
                poly_len=np.array([len(r[0]) for r, _ in keep], np.int32),
                poly_cls=np.array([c for _, c in keep], np.int32),
                hole_xy=_stack([h for h, _ in holes]),
                hole_len=np.array([len(h) for h, _ in holes], np.int32),
                hole_of=np.array([j for _, j in holes], np.int32))
            manifest["chips"].append({"file": f"chips/{fname}", "site": site.slug,
                                      "split": split, "instances": len(keep)})
            if split == "train":
                stats.add(f, valid)
            n += 1
            n_instances += len(keep)
            n_outline_chips += outline_centred
        n_neg = sum(cls is None for _, cls in outlines)
        n_excl = sum(cls == EXCLUDED for _, cls in outlines)
        n_lab = len(outlines) - n_neg - n_excl
        manifest["sites"][site.slug] = {"split": split, "labels": n_lab, "lookalikes": n_neg,
                                        "excluded": n_excl, "chips": n, "instances": n_instances,
                                        "background": site.background}
        kind = " (background site)" if site.background else ""
        excl = f", {n_excl} excluded" if n_excl else ""
        print(f"[{site.name}] {split}{kind}: {n_lab} earthwork(s), "
              f"{n_neg} look-alike(s){excl} -> {n} chips, {n_instances} instances")

    if not manifest["chips"]:
        raise SystemExit("No chips written - are any sites prepared and marked reviewed?")
    manifest["channel_mean"], manifest["channel_std"] = stats.result()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_val = sum(c["split"] == "val" for c in manifest["chips"])
    print(f"Dataset {out}: {len(manifest['chips'])} chips ({n_val} val)")
    if not n_val:
        print("  warning: no validation sites - set \"split\": \"val\" on a site in sites.json")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="default", help="Dataset folder under data/earthworks/dataset/")
    ap.add_argument("--only", nargs="+", metavar="SLUG", help="Only these sites")
    ap.add_argument("--lookalikes", choices=LOOKALIKE_MODES, default=parameters.EW_LOOKALIKE_TRAINING,
                    help="How look-alike outlines train (default: EW_LOOKALIKE_TRAINING)")
    ap.add_argument("--classes", nargs="+", metavar="CLASS",
                    help="Target classes (default: EW_CLASSES), e.g. --classes mound earthwork: "
                         "types not listed train as 'earthwork'")
    args = ap.parse_args()
    build(args.name, args.only, lookalikes=args.lookalikes, classes=args.classes)


if __name__ == "__main__":
    main()
