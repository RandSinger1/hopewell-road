"""
cache.py — Disk cache for expensive per-coordinate raster computations.

Loading a DTM window, median-smoothing it, building the local relief
model, and running ridge/pair detection are by far the most expensive
steps in report generation, and four different figure generators
(pipeline_steps, cross_sections, extra_views, overview_map) used to
each redo them from scratch. None of that depends on how a figure is
*drawn* — only on the coordinate and the handful of parameters.py
settings that feed the computation. Caching those intermediates lets
you re-run the report to tweak a figure (e.g. choosing which
cross-section lines to emphasise) in seconds instead of minutes.

Cache files live under report/output/<coord_name>/.cache/<name>.pkl
and are invalidated automatically whenever the inputs that produced
them change (the key is a hash of those inputs).
"""

import hashlib
import pickle
from pathlib import Path
from typing import Callable


def _key(parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]


def cached(cache_dir: Path, name: str, key_parts: tuple,
           compute_fn: Callable[[], dict], use_cache: bool = True,
           label: str = "") -> dict:
    """
    Return compute_fn() — from disk cache if key_parts match a prior
    run, otherwise computed fresh and written to disk.
    """
    path = cache_dir / f"{name}.pkl"
    tag = label or name
    digest = _key(key_parts)

    if use_cache and path.exists():
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == digest:
                print(f"    [cache] {tag}: hit")
                return blob["data"]
        except Exception:
            pass  # corrupt/stale cache file — fall through and recompute

    print(f"    [cache] {tag}: miss — computing …")
    data = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"key": digest, "data": data}, f)
    return data
