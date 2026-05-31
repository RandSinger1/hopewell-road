from pathlib import Path
import os

import numpy as np
from shapely.geometry import Polygon
from shapely.prepared import prep
from concurrent.futures import ProcessPoolExecutor, as_completed

from ridge_detection import search_region
from utils import convert_meters_degrees_pixels, METERS, DEGREES
import params

import json

def tile_polygon(filepath, polygon_vertices, size_x, size_y, overlap):
    """
    Cover a polygon with overlapping axis‑aligned tiles and call search_region for each.

    Parameters
    ----------
    filepath : str
        Path to the source DTM/lidar file.
    polygon_vertices : list of (x, y) tuples
        Vertices of the polygon (in the same CRS as the DTM).
    size_x : float
        Tile width in map units (e.g., meters).
    size_y : float
        Tile height in map units.
    overlap : float
        Overlap fraction between adjacent tiles, e.g., 0.2 = 20% overlap.
    output_dir : str
        Directory where search_region will save outputs.
    """
    # polygon_vertices is expected as [(lat, lon), ...]
    poly = Polygon([(lon, lat) for lat, lon in polygon_vertices])
    poly_prep = prep(poly)

    # Get bounding box of the polygon in lon/lat
    minx, miny, maxx, maxy = poly.bounds

    size_x_deg, size_y_deg = convert_meters_degrees_pixels(METERS, DEGREES, size_x, size_y)

    # Compute step sizes accounting for overlap
    step_x = size_x_deg * (1 - overlap)
    step_y = size_y_deg * (1 - overlap)

    # Iterate from top to bottom: start at maxy, go down
    y = maxy
    coords = []
    while y > miny:
        x = minx
        while x < maxx:
            tile_rect = Polygon([
                (x, y - size_y_deg),
                (x + size_x_deg, y - size_y_deg),
                (x + size_x_deg, y),
                (x, y)
            ])
            if poly_prep.intersects(tile_rect):
                # Return coordinates as (lat, lon) for get_window
                tile_center = (y - size_y_deg * 0.5, x + size_x_deg * 0.5)
                coords.append(tile_center)

            x += step_x
        y -= step_y
    return coords
def main(multicore = True):
    filepath = params.SCAN_DTM_PATH
    polygon_vertices = params.SCAN_POLYGON
    size_x = params.SCAN_TILE_SIZE_M
    size_y = params.SCAN_TILE_SIZE_M
    overlap = params.SCAN_OVERLAP_M
    metadata = {
        'coords': polygon_vertices,
        'size_x': size_x,
        'size_y': size_y,
        'overlap': overlap
    }
    output_dir = Path('locations') / f'{size_x}_{size_y}'
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    coords = tile_polygon(filepath, polygon_vertices, size_x, size_y, overlap / size_x)
    print(f"Tiles to search: {len(coords)}")
    max_workers = min(params.SCAN_MAX_WORKERS, os.cpu_count() or 1)
    total = len(coords)
    if multicore:
        with ProcessPoolExecutor(max_workers=max_workers) as executer:
            args_list = [(filepath, coord, size_x, size_y, output_dir) for coord in coords]
            futures = [executer.submit(search_region, *args) for args in args_list]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"Progress: {i}/{total} ({100*i//total}%)")

    else:
        for i, coord in enumerate(coords, 1):
            search_region(filepath, coord, size_x, size_y, output_dir, picture=False)
            print(f"Progress: {i}/{total} ({100*i//total}%)")
if __name__ == '__main__':
    main()