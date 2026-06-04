import argparse
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, Point
from concurrent.futures import ProcessPoolExecutor

from ridge_detection import search_region
from utils import convert_meters_degrees_pixels, METERS, DEGREES

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
    hopewell_road_polygon = json.load(open('polygons.json'))['Hopewell_Road']
    # flip x and y
    hopewell_road_polygon = Polygon([(y, x) for x, y in hopewell_road_polygon])

    # Create Shapely polygon
    poly = Polygon(hopewell_road_polygon)

    # Get bounding box of the polygon
    minx, miny, maxx, maxy = poly.bounds

    size_x_deg, size_y_deg = convert_meters_degrees_pixels(METERS, DEGREES, size_x, size_y)

    # Compute step sizes accounting for overlap
    step_x = size_x_deg * (1 - overlap)
    step_y = size_y_deg * (1 - overlap)

    # Iterate from top to bottom: start at maxy, go down
    y = maxy
    tile_index = 0
    coords = []
    while y > miny:
        x = minx
        while x < maxx:
            # Lower‑left corner of the tile
            tile_ll = (x, y - size_y_deg)   # because y is top edge, subtract height

            # OPTION: If search_region expects centre, use:
            # tile_centre = (x + size_x/2, y - size_y/2)

            # Check if this tile (as a rectangle) intersects the polygon
            tile_rect = Polygon([(x, y - size_y_deg), (x + size_x_deg, y - size_y_deg),
                                 (x + size_x_deg, y), (x, y)])
            if poly.intersects(tile_rect) and poly.intersects(hopewell_road_polygon):
                coords.append(tile_ll)
                # Call your existing function
                tile_index += 1

            x += step_x
        y -= step_y
    return coords
def main(max_workers):
    
    filepath ='loaded_tiles/Whole_Road/merged_dtm.tif'
    circleville =(39.6,-82.9)
    newark = (40.0, -82.4)
    chilicothe = (39.3,-82.9)
    polygon_vertices = [circleville, newark, chilicothe]
    
    size_x = 600
    size_y = 600
    overlap = 100
    metadata = {
        'coords': polygon_vertices,
        'size_x': size_x,
        'size_y': size_y,
        'overlap': overlap
    }
    output_dir = Path('locations') / f'{size_x}_{size_y}'
    metadata_path = Path(output_dir) / 'metadata.json'
    #output_dir.mkdir(exist_ok=False)
    json.dump(metadata, metadata_path.open('w'))
    coords = tile_polygon(filepath, polygon_vertices, size_x, size_y, overlap/size_x)
    print(f"Tiles to search: {len(coords)}")
    total = len(coords)
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executer:
            args_list = [(filepath, coord, size_x, size_y, output_dir) for coord in coords]
            futures = [executer.submit(search_region, *args) for args in args_list]
            for i, future in enumerate(futures, 1):
                future.result()
                print(f"Progress: {i}/{total} ({100*i//total}%)")

    elif max_workers == 1:
        for i, coord in enumerate(coords, 1):
            search_region(filepath, coord, size_x, size_y, output_dir, picture=False)
            print(f"Progress: {i}/{total} ({100*i//total}%)")
if __name__ == '__main__':
    # Get parameters from command line or config file as needed with --max_workers or -w
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_workers', '-w', type=int, default=1,
                        help='Number of parallel workers to use (default: 1)')
    args = parser.parse_args()
    main(max_workers=args.max_workers)