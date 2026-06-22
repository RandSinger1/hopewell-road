import argparse
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import parameters
from road_model.ridge_detection import search_region
from utils import (convert_meters_degrees_pixels, METERS, DEGREES,
                   make_parabola, make_line)

import json

# ── Named coordinates ─────────────────────────────────────────────────────────
# Single source of truth: parameters.py
CIRCLEVILLE = parameters.CIRCLEVILLE_COORD   # (lat, lon)
NEWARK      = parameters.NEWARK_COORD
CHILLICOTHE = parameters.CHILLICOTHE_COORD
HIGH_BANKS  = parameters.HIGH_BANKS_COORD


# ── Tile generators ───────────────────────────────────────────────────────────

def tile_polygon(polygon_vertices, size_x, size_y, overlap,
                 target_angle=None):
    """
    Cover a polygon with overlapping tiles.

    Returns
    -------
    list of (coord, angle_deg)
        coord is the tile lower-left corner (lon, lat).
        angle_deg is uniform for every tile (defaults to parameters.TARGET_ANGLE).
    """
    if target_angle is None:
        target_angle = parameters.TARGET_ANGLE

    poly = Polygon(polygon_vertices)
    minx, miny, maxx, maxy = poly.bounds
    size_x_deg, size_y_deg = convert_meters_degrees_pixels(METERS, DEGREES, size_x, size_y)
    step_x = size_x_deg * (1 - overlap)
    step_y = size_y_deg * (1 - overlap)

    tiles = []
    y = maxy
    while y > miny:
        x = minx
        while x < maxx:
            tile_rect = Polygon([
                (x,              y - size_y_deg),
                (x + size_x_deg, y - size_y_deg),
                (x + size_x_deg, y),
                (x,              y),
            ])
            if poly.intersects(tile_rect):
                tiles.append(((x, y - size_y_deg), target_angle))
            x += step_x
        y -= step_y
    return tiles


def tile_curve(curve_func, curve_deriv, x_start, x_end,
               half_width_deg, size_x, size_y, overlap):
    """
    Cover a curve corridor with overlapping tiles and compute tangent angles.

    For every tile whose center falls within *half_width_deg* of the curve
    (measured in the latitude direction), the target angle is
    arctan(curve_deriv(center_lon)) converted to degrees.

    Parameters
    ----------
    curve_func : callable
        f(lon) -> lat along the curve.
    curve_deriv : callable
        f'(lon) -> dlat/dlon at that longitude.
    x_start, x_end : float
        Longitude range to scan (degrees).
    half_width_deg : float
        Corridor half-width in degrees on each side of the curve.
    size_x, size_y : float
        Tile dimensions in metres.
    overlap : float
        Overlap fraction between adjacent tiles (e.g. 0.2 = 20 %).

    Returns
    -------
    list of (coord, angle_deg)
        coord is the tile lower-left corner (lon, lat).
    """
    size_x_deg, size_y_deg = convert_meters_degrees_pixels(METERS, DEGREES, size_x, size_y)
    step_x = size_x_deg * (1 - overlap)
    step_y = size_y_deg * (1 - overlap)

    x_samples = np.linspace(x_start, x_end, 500)
    y_samples = np.array([curve_func(x) for x in x_samples])
    miny = y_samples.min() - half_width_deg
    maxy = y_samples.max() + half_width_deg

    tiles = []
    y = maxy
    while y > miny:
        x = x_start
        while x < x_end:
            center_x = x + size_x_deg / 2
            center_y = y - size_y_deg / 2

            if abs(center_y - curve_func(center_x)) <= half_width_deg:
                tile_ll = (y - size_y_deg, x)  # (lat, lon) — consistent with tile_polygon
                # cross-ridge angle in image coords (y-down): 90 - geo_road_angle
                angle = (90.0 - np.degrees(np.arctan(curve_deriv(center_x)))) % 180
                tiles.append((tile_ll, angle))

            x += step_x
        y -= step_y

    return tiles


# ── Route presets ─────────────────────────────────────────────────────────────

def _tiles_polygon(size_x, size_y, overlap):
    """Whole study-area polygon, uniform TARGET_ANGLE."""
    polygon_vertices = [CIRCLEVILLE, NEWARK, CHILLICOTHE]
    return tile_polygon(polygon_vertices, size_x, size_y, overlap)


def _tiles_circleville(size_x, size_y, overlap, half_width_meters=5000):
    """Parabolic corridor from Circleville to Newark."""
    half_width_deg = convert_meters_degrees_pixels(METERS, DEGREES, 0, half_width_meters)[1]
    curve_func, curve_deriv = make_parabola(
        parameters.PARABOLA_A, parameters.PARABOLA_B, parameters.PARABOLA_C,
        x_offset=parameters.PARABOLA_X_OFFSET,
    )
    return tile_curve(
        curve_func, curve_deriv,
        x_start=parameters.PARABOLA_X_OFFSET, x_end=NEWARK[1],
        half_width_deg=half_width_deg,
        size_x=size_x, size_y=size_y, overlap=overlap,
    )


def _tiles_chillicothe(size_x, size_y, overlap, half_width_meters=5000):
    """Straight-line corridor from Chillicothe to Newark."""
    half_width_deg = convert_meters_degrees_pixels(METERS, DEGREES, 0, half_width_meters)[1]
    x0, y0 = CHILLICOTHE[1], CHILLICOTHE[0]
    x1, y1 = NEWARK[1], NEWARK[0]
    curve_func, curve_deriv = make_line(x0, y0, x1, y1)
    return tile_curve(
        curve_func, curve_deriv,
        x_start=x0, x_end=x1,
        half_width_deg=half_width_deg,
        size_x=size_x, size_y=size_y, overlap=overlap,
    )


def _tiles_high_banks(size_x, size_y, overlap, half_width_meters=5000):
    """Straight-line corridor from Newark to High Banks."""
    half_width_deg = convert_meters_degrees_pixels(METERS, DEGREES, 0, half_width_meters)[1]
    x0, y0 = HIGH_BANKS[1], HIGH_BANKS[0]
    x1, y1 = NEWARK[1], NEWARK[0]
    curve_func, curve_deriv = make_line(x0, y0, x1, y1)
    return tile_curve(
        curve_func, curve_deriv,
        x_start=x0, x_end=x1,
        half_width_deg=half_width_deg,
        size_x=size_x, size_y=size_y, overlap=overlap,
    )


MODES = {
    'polygon':      _tiles_polygon,
    'circleville':  _tiles_circleville,
    'chillicothe':   _tiles_chillicothe,
    'high_banks':   _tiles_high_banks,
}


# ── Preview ───────────────────────────────────────────────────────────────────

_CITIES = {
    'Circleville': CIRCLEVILLE,
    'Newark':      NEWARK,
    'Chillicothe': CHILLICOTHE,
    'High Banks':  HIGH_BANKS,
}

def preview_tiles(tiles, mode, size_x, size_y, dtm_path=None, dtm_px=400):
    """
    Show a map of tiles and named cities before the search runs, overlaid on
    a sparse readback of the actual DTM so the search range is shown against
    where data really exists — rather than against an assumed/hand-
    maintained boundary, which can drift out of sync with what was actually
    downloaded and merged.

    The DTM read is windowed to the tiles' own bounding box (not the whole
    file): this project's merged_dtm.tif files can have a corrupted overview
    pyramid (every level 100% nodata — see lidar.quick_view.read_dtm_window),
    so the background is warped directly from full-resolution data with
    overviews bypassed; doing that for the whole multi-GB file would be far
    too slow for an interactive preview, but a windowed read is fast.
    """
    size_x_deg, size_y_deg = convert_meters_degrees_pixels(METERS, DEGREES, size_x, size_y)

    fig, ax = plt.subplots(figsize=(11, 9))

    lats = [coord[0] for coord, _ in tiles]
    lons = [coord[1] for coord, _ in tiles]

    if dtm_path is None:
        dtm_path = parameters.SCAN_DTM_PATH
    if lats and dtm_path and Path(dtm_path).exists():
        from lidar.quick_view import read_dtm_window
        margin = 0.1 * max(max(lons) - min(lons) + size_x_deg,
                           max(lats) - min(lats) + size_y_deg)
        bounds = (
            min(lons) - margin, min(lats) - margin,
            max(lons) + size_x_deg + margin, max(lats) + size_y_deg + margin,
        )
        print("Loading DTM preview for the search-range window …")
        dtm_data, dtm_extent = read_dtm_window(dtm_path, bounds, width=dtm_px, height=dtm_px)
        if dtm_data is not None:
            # Masked (nodata) cells render transparent, so areas the DTM
            # doesn't actually cover show through as blank rather than
            # implying coverage.
            ax.imshow(dtm_data, extent=dtm_extent, cmap='gray', aspect='auto',
                      alpha=0.7, zorder=0)
    elif dtm_path and not Path(dtm_path).exists():
        print(f"  [preview] DTM not found at {dtm_path!r} — skipping background.")

    # Tile footprints — rectangles for small counts, scatter for large

    if len(tiles) <= 600:
        for (lat, lon), _ in tiles:
            ax.add_patch(mpatches.Rectangle(
                (lon, lat), size_x_deg, size_y_deg,
                linewidth=0.4, edgecolor='steelblue',
                facecolor='lightblue', alpha=0.35,
            ))
        # Expand limits manually since add_patch doesn't update autoscale
        ax.set_xlim(min(lons) - size_x_deg, max(lons) + 2 * size_x_deg)
        ax.set_ylim(min(lats) - size_y_deg, max(lats) + 2 * size_y_deg)
    else:
        ax.scatter(lons, lats, s=4, color='steelblue', alpha=0.4, label=f'{len(tiles)} tiles')
        ax.autoscale()

    # Bounding box of search area
    if lats:
        bbox = mpatches.Rectangle(
            (min(lons), min(lats)),
            max(lons) + size_x_deg - min(lons),
            max(lats) + size_y_deg - min(lats),
            linewidth=1.5, edgecolor='navy', facecolor='none', linestyle='--',
        )
        ax.add_patch(bbox)

    # Straight line Newark → Chillicothe
    ax.plot(
        [NEWARK[1], CHILLICOTHE[1]],
        [NEWARK[0], CHILLICOTHE[0]],
        color='darkorange', linewidth=1.5, linestyle='--', zorder=5,
        label='Newark–Chillicothe',
    )

    # Straight line Newark → High Banks
    ax.plot(
        [NEWARK[1], HIGH_BANKS[1]],
        [NEWARK[0], HIGH_BANKS[0]],
        color='mediumorchid', linewidth=1.5, linestyle='--', zorder=5,
        label='Newark–High Banks',
    )

    # Named cities
    for name, (clat, clon) in _CITIES.items():
        ax.plot(clon, clat, 'r*', markersize=14, zorder=6)
        ax.annotate(
            name, (clon, clat),
            textcoords='offset points', xytext=(7, 4),
            fontsize=11, fontweight='bold', color='darkred',
        )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(
        f'Search preview  ·  mode={mode}  ·  {len(tiles)} tiles  ·  {size_x}×{size_y} m each',
        fontsize=12,
    )
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tiles(tiles, filepath, size_x, size_y, output_dir, max_workers):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(tiles)
    print(f"Tiles to search: {total}")

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args_list = [
                (filepath, coord, size_x, size_y, output_dir, 0, False, angle)
                for coord, angle in tiles
            ]
            futures = [executor.submit(search_region, *args) for args in args_list]
            for i, future in enumerate(futures, 1):
                future.result()
                print(f"Progress: {i}/{total} ({100*i//total}%)")
    else:
        for i, (coord, angle) in enumerate(tiles, 1):
            search_region(filepath, coord, size_x, size_y, output_dir,
                          picture=False, target_angle=angle)
            print(f"Progress: {i}/{total} ({100*i//total}%)")


def main(mode, max_workers, size_x, size_y, overlap, half_width_meters, preview=False, reverse=False):
    filepath = parameters.SCAN_DTM_PATH

    tile_fn = MODES[mode]

    # corridor routes accept half_width_deg; polygon doesn't
    if mode == 'polygon':
        tiles = tile_fn(size_x, size_y, overlap)
    else:
        tiles = tile_fn(size_x, size_y, overlap, half_width_meters)

    if reverse:
        tiles = tiles[::-1]

    if preview:
        preview_tiles(tiles, mode, size_x, size_y, dtm_path=filepath)

    output_dir = Path('data/locations') / f'{mode}_{size_x}_{size_y}'
    run_tiles(tiles, filepath, size_x, size_y, output_dir, max_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tile-scan the LiDAR DTM for Hopewell Road earthworks.')
    parser.add_argument(
        '--mode', '-m',
        choices=list(MODES),
        default='polygon',
        help='Search region: polygon (default), circleville, chillicothe, or high_banks')
    parser.add_argument(
        '--max_workers', '-w',
        type=int, default=parameters.SCAN_MAX_WORKERS,
        help=f'Parallel workers (default: {parameters.SCAN_MAX_WORKERS})')
    parser.add_argument(
        '--size_x', type=int, default=parameters.SCAN_TILE_SIZE_M,
        help=f'Tile width in metres (default: {parameters.SCAN_TILE_SIZE_M})')
    parser.add_argument(
        '--size_y', type=int, default=parameters.SCAN_TILE_SIZE_M,
        help=f'Tile height in metres (default: {parameters.SCAN_TILE_SIZE_M})')
    parser.add_argument(
        '--overlap', type=float,
        default=parameters.SCAN_OVERLAP_M / parameters.SCAN_TILE_SIZE_M,
        help='Overlap fraction 0–1 (default derived from SCAN_OVERLAP_M)')
    parser.add_argument(
        '--half_width_meters', type=int, default=5000,
        help='Corridor half-width in metres for curve modes (default: 5000 m)')
    parser.add_argument(
        '--preview', action='store_true',
        help='Show a map of tiles and cities before running the search')
    parser.add_argument(
        '--reverse', action='store_true',
        help='Process tiles in reverse order (south-to-north, east-to-west)')

    args = parser.parse_args()
    main(
        mode=args.mode,
        max_workers=args.max_workers,
        size_x=args.size_x,
        size_y=args.size_y,
        overlap=args.overlap,
        half_width_meters=args.half_width_meters,
        preview=args.preview,
        reverse=args.reverse,
    )
