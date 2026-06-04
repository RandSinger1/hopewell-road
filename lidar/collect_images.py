from pathlib import Path
from AOI2List.src.aoi2list import query_polygon
import json
import argparse


def get_polygons(AREA, n_splits=10):
    with open("data/polygons.json", "r") as f:
        polygons = json.load(f)
    polygon_segment = polygons[AREA]
    for i in range(n_splits):
        polygon_segment_new = []
        polygon_segment_new.append((
            polygon_segment[0][0] + (polygon_segment[3][0] - polygon_segment[0][0]) * i / n_splits,
            polygon_segment[0][1] + (polygon_segment[3][1] - polygon_segment[0][1]) * i / n_splits
        ))
        polygon_segment_new.append((
            polygon_segment[1][0] + (polygon_segment[2][0] - polygon_segment[1][0]) * i / n_splits,
            polygon_segment[1][1] + (polygon_segment[2][1] - polygon_segment[1][1]) * i / n_splits
        ))
        polygon_segment_new.append((
            polygon_segment[2][0] + (polygon_segment[1][0] - polygon_segment[2][0]) * (n_splits - 1 - i) / n_splits,
            polygon_segment[2][1] + (polygon_segment[1][1] - polygon_segment[2][1]) * (n_splits - 1 - i) / n_splits
        ))
        polygon_segment_new.append((
            polygon_segment[3][0] + (polygon_segment[0][0] - polygon_segment[3][0]) * (n_splits - 1 - i) / n_splits,
            polygon_segment[3][1] + (polygon_segment[0][1] - polygon_segment[3][1]) * (n_splits - 1 - i) / n_splits
        ))
        polygon_segment_new.append(polygon_segment_new[0])
        area_temp = f"{AREA}_{i}"
        polygons[area_temp] = polygon_segment_new
    return polygons


def array_to_wkt_string(coordinates: list) -> str:
    """Convert a list of (lon, lat) tuples to a WKT POLYGON string."""
    wkt_string = "POLYGON (("
    for coord in coordinates:
        wkt_string += f"{coord[0]} {coord[1]}, "
    wkt_string = wkt_string[:-2] + "))"
    return wkt_string


def main(AREA="Hart_Island", n_splits=10) -> list[str]:
    """Fetch tile metadata and save JSON file lists. Returns the sub-area names written."""
    output_dir = Path("data/saved_file_lists")
    output_dir.mkdir(exist_ok=True)
    polygons = get_polygons(AREA, n_splits=n_splits)
    created: list[str] = []
    for i in range(n_splits):
        area_temp = f"{AREA}_{i}"
        wkt = array_to_wkt_string(polygons[area_temp])
        metadata = None
        for _ in range(10):
            try:
                metadata = query_polygon(wkt)[0]
                break
            except Exception as e:
                print(f"Error fetching data for {area_temp}: {e}. Retrying...")
        if metadata:
            with open(output_dir / f"{area_temp}.json", "w") as f:
                json.dump(metadata, f)
            created.append(area_temp)
    return created

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch LiDAR tile metadata for a specified area.")
    parser.add_argument("area", type=str, help="Area to fetch (e.g., Hopewell_Road, Circleville_Road, Hart_Island)")
    parser.add_argument("--area-splits", type=int, default=10, metavar="N",
                        help="Number of sub-segments for road-type polygons (default 10)")
    args = parser.parse_args()
    main(args.area, n_splits=args.area_splits)
