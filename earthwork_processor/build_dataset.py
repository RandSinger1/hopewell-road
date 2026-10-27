"""
build_dataset.py — Turn reviewed builder sites into augmented training chips.

    python -m earthwork_processor.build_dataset [--name default] [--only SLUG ...]

For every site whose labels.geojson is marked reviewed:
  * sites/<slug>/features.tif — the full-window feature stack (one band per
    channel, EW_WORK_CRS) for inspection in QGIS;
  * EW_CHIPS_PER_LABEL chips around each earthwork outline (one un-augmented,
    the rest randomly rotated / scaled / exaggerated / noised, with the outline
    jittered inside the chip);
  * EW_CHIPS_PER_NEGATIVE chips around each look-alike outline
    (EW_NEGATIVE_TYPES: levees, rail grades, ...) — hard negatives: the
    look-alike is in the chip but is NOT a target;
  * EW_BACKGROUND_CHIPS_PER_SITE chips centred at random, which mostly
    contain no earthwork.

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

import parameters
from earthwork_builder.labels import Site, is_negative, label_polygons, load_sites, read_labels
from earthwork_processor.augment import ChipTransform, make_chip, rasterize
from earthwork_processor.features import FeatureSpec, compute_features
from lidar import point_io
from lidar.ground_grid import GroundGrid

DATASET_ROOT = parameters.EW_ROOT / "dataset"


def class_index(label_type: str) -> int | None:
    """Target class of an outline type, or None for a look-alike (hard negative)."""
    if is_negative(label_type):
        return None
    if label_type == parameters.EW_UNVERIFIED_TYPE:
        raise ValueError(f"'{label_type}' outline in a reviewed site")
    classes = list(parameters.EW_CLASSES)
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


def outlines_px(site: Site, grid: GroundGrid) -> list[tuple[np.ndarray, int | None]]:
    """[(outline (K, 2) in grid (col, row), class index or None for a look-alike), …]."""
    out = []
    for poly, props in label_polygons(site):
        lon, lat = np.asarray(poly.exterior.coords).T
        x, y = point_io.lonlat_to_work(lon, lat)
        c, r = grid.to_pixel(x, y)
        out.append((np.column_stack([c, r]), class_index(props.get("type", "other"))))
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


def chip_plan(rng, grid: GroundGrid, outlines, size: int) -> list[ChipTransform]:
    """Centres + augmentations for one site's chips."""
    plan = []
    half = size / 2
    for poly, cls in outlines:
        centre = poly.mean(axis=0) - 0.5                    # edge → index coords
        radius = np.linalg.norm(poly - poly.mean(axis=0), axis=1).max()
        # Jitter so the outline lands anywhere in the chip while staying (mostly) inside
        max_shift = max(0.0, 0.8 * half - radius)
        n = parameters.EW_CHIPS_PER_LABEL if cls is not None else parameters.EW_CHIPS_PER_NEGATIVE
        for k in range(n):
            shift = np.zeros(2)
            if k:
                ang, rad = rng.uniform(0, 2 * np.pi), max_shift * np.sqrt(rng.uniform())
                shift = rad * np.array([np.cos(ang), np.sin(ang)])
            plan.append(_random_transform(rng, centre[0] + shift[0], centre[1] + shift[1], augment=k > 0))
    rows, cols = np.nonzero(np.isfinite(grid.z))
    if len(rows):
        for _ in range(parameters.EW_BACKGROUND_CHIPS_PER_SITE):
            i = rng.integers(len(rows))
            plan.append(_random_transform(rng, float(cols[i]), float(rows[i]), augment=True))
    return plan


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


def build(name: str = "default", only: list[str] | None = None, spec: FeatureSpec | None = None) -> Path:
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
                "classes": list(parameters.EW_CLASSES), "sites": {}, "chips": []}

    for site in load_sites():
        if only and site.slug not in only and site.name not in only:
            continue
        if not site.prepared:
            print(f"[{site.name}] skipped: not prepared (run earthwork_builder.prepare)")
            continue
        if not read_labels(site).get("reviewed"):
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

        outlines = outlines_px(site, grid)
        split = site_split(site)
        n = 0
        for tf in chip_plan(rng, grid, outlines, size):
            f, valid, polys = make_chip(grid, tf, spec, size, [p for p, _ in outlines], rng)
            if valid.mean() < 0.5:
                continue
            keep = [(p, cls) for p, (_, cls) in zip(polys, outlines)
                    if cls is not None and rasterize(p, size).sum() >= parameters.EW_MIN_INSTANCE_PX]
            fname = f"{site.slug}_{n:04d}.npz"
            np.savez_compressed(
                chips_dir / fname, feats=f.astype(np.float16), valid=valid,
                poly_xy=(np.concatenate([p for p, _ in keep]) if keep else np.zeros((0, 2))).astype(np.float32),
                poly_len=np.array([len(p) for p, _ in keep], np.int32),
                poly_cls=np.array([c for _, c in keep], np.int32))
            manifest["chips"].append({"file": f"chips/{fname}", "site": site.slug,
                                      "split": split, "instances": len(keep)})
            if split == "train":
                stats.add(f, valid)
            n += 1
        n_neg = sum(cls is None for _, cls in outlines)
        manifest["sites"][site.slug] = {"split": split, "labels": len(outlines) - n_neg,
                                        "lookalikes": n_neg, "chips": n}
        print(f"[{site.name}] {split}: {len(outlines) - n_neg} earthwork(s), {n_neg} look-alike(s) -> {n} chips")

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
    args = ap.parse_args()
    build(args.name, args.only)


if __name__ == "__main__":
    main()
