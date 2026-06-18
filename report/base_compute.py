"""
base_compute.py — Shared, cached terrain + detection intermediates.

Every figure generator needs the same expensive starting point: a DTM
window, median-smoothed and turned into a local relief model, then
(for most figures) Frangi ridge enhancement and parallel-wall pair
detection. get_terrain() / get_detection() compute this once per
coordinate and persist it via cache.py, so a re-run that only changes
how a figure is drawn (e.g. which cross-section lines to emphasise)
skips straight to drawing instead of redoing the raster work.
"""

from pathlib import Path

import numpy as np

from road_model.process_dtm import (
    getGTAndBand, get_window,
    median_smooth_windowed, local_relief_model, multiscale_ridge_enhancement,
)
from road_model.ridge_detection import detect_parallel_walls_ridge_pairing, remove_invalid_pairs
from . import cache as _cache
from . import config as cfg
from .report_utils import site_target_angle


def _cache_dir(coord: dict, output_dir: Path) -> Path:
    return Path(output_dir) / coord["name"] / ".cache"


def _terrain_key(coord: dict, dtm_path: str) -> tuple:
    return (
        dtm_path, coord["lat"], coord["lon"], coord.get("size_m", 200),
        cfg.MEDIAN_SMOOTH_RADIUS, cfg.LRM_RADIUS, cfg.LRM_METHOD,
    )


def _detection_key(coord: dict, dtm_path: str) -> tuple:
    _, target_angle = site_target_angle(coord)
    return _terrain_key(coord, dtm_path) + (
        tuple(cfg.RIDGE_SCALES), target_angle, cfg.SPACING_METERS,
        cfg.ANGLE_TOLERANCE, cfg.SPACING_TOLERANCE, cfg.RIDGE_THRESHOLD,
        cfg.MIN_RIDGE_LENGTH_METERS, cfg.OVERLAP_RATIO,
    )


def get_terrain(coord: dict, dtm_path: str, output_dir: Path,
                 use_cache: bool = True) -> dict | None:
    """
    Returns dict(raw, extent, nodata, smoothed, lrm) for one coordinate,
    or None if the DTM window has no data there.
    """
    def _compute():
        gt, band, nodata, _ds = getGTAndBand(dtm_path)
        size_m = coord.get("size_m", 200)
        raw, extent = get_window(coord["lat"], coord["lon"], size_m, size_m,
                                  gt, band, nodata)
        if raw is None or raw.size == 0:
            return {"empty": True}
        smoothed = median_smooth_windowed(raw, nodata=nodata, radius=cfg.MEDIAN_SMOOTH_RADIUS)
        lrm = local_relief_model(smoothed, radius=cfg.LRM_RADIUS, nodata=nodata)
        return {"empty": False, "raw": raw, "extent": extent, "nodata": nodata,
                "smoothed": smoothed, "lrm": lrm}

    data = _cache.cached(
        _cache_dir(coord, output_dir), "terrain", _terrain_key(coord, dtm_path),
        _compute, use_cache=use_cache, label=f"{coord['name']}/terrain",
    )
    return None if data.get("empty") else data


def get_detection(coord: dict, dtm_path: str, terrain: dict, output_dir: Path,
                   use_cache: bool = True) -> dict:
    """
    Returns dict(ridge, valid_ridge, pairs, pairs_clean, binary, closed,
    skeleton, labeled) built on top of an already-loaded terrain dict.

    ``pairs`` is the raw detector output; ``pairs_clean`` has
    remove_invalid_pairs() applied. Different figures historically used
    one or the other, so both are kept available.
    """
    def _compute():
        mode, target_angle = site_target_angle(coord)
        ridge = multiscale_ridge_enhancement(terrain["lrm"], scales=cfg.RIDGE_SCALES,
                                              nodata=terrain["nodata"])
        valid_ridge = np.where(terrain["raw"] == terrain["nodata"], 0.0, ridge.astype(float))
        pairs, debug = detect_parallel_walls_ridge_pairing(
            ridge_map=valid_ridge,
            target_angle=target_angle,
            extent=terrain["extent"],
            spacing_meters=cfg.SPACING_METERS,
            angle_tolerance=cfg.ANGLE_TOLERANCE,
            spacing_tolerance=cfg.SPACING_TOLERANCE,
            ridge_threshold=cfg.RIDGE_THRESHOLD,
            min_ridge_length_meters=cfg.MIN_RIDGE_LENGTH_METERS,
            overlap_ratio=cfg.OVERLAP_RATIO,
            return_debug=True,
        )
        binary, closed, skeleton, labeled, _ = debug
        pairs_clean = remove_invalid_pairs(pairs, angle_tolerance=cfg.ANGLE_TOLERANCE,
                                            target_angle=target_angle)
        return {
            "ridge": ridge, "valid_ridge": valid_ridge,
            "pairs": pairs, "pairs_clean": pairs_clean,
            "binary": binary, "closed": closed, "skeleton": skeleton, "labeled": labeled,
            "mode": mode, "target_angle": target_angle,
        }

    return _cache.cached(
        _cache_dir(coord, output_dir), "detection", _detection_key(coord, dtm_path),
        _compute, use_cache=use_cache, label=f"{coord['name']}/detection",
    )
