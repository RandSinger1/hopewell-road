from pathlib import Path
from AOI2List.src.aoi2list import query_polygon
import json


def get_polygons(AREA):
    with open("polygons.json", "r") as f:
        polygons = json.load(f)
    polygon_segment = polygons[AREA]
    for i in range(10):
        polygon_segment_new = []
        polygon_segment_new.append((
            polygon_segment[0][0] + (polygon_segment[3][0] - polygon_segment[0][0]) * i / 10,
            polygon_segment[0][1] + (polygon_segment[3][1] - polygon_segment[0][1]) * i / 10
        ))
        polygon_segment_new.append((
            polygon_segment[1][0] + (polygon_segment[2][0] - polygon_segment[1][0]) * i / 10,
            polygon_segment[1][1] + (polygon_segment[2][1] - polygon_segment[1][1]) * i / 10
        ))
        polygon_segment_new.append((
            polygon_segment[2][0] + (polygon_segment[1][0] - polygon_segment[2][0]) * (10 - 1 - i) / 10,
            polygon_segment[2][1] + (polygon_segment[1][1] - polygon_segment[2][1]) * (10 - 1 - i) / 10
        ))
        polygon_segment_new.append((
            polygon_segment[3][0] + (polygon_segment[0][0] - polygon_segment[3][0]) * (10 - 1 - i) / 10,
            polygon_segment[3][1] + (polygon_segment[0][1] - polygon_segment[3][1]) * (10 - 1 - i) / 10
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


def main(AREA="Hart_Island"):
    output_dir = Path("saved_file_lists")
    output_dir.mkdir(exist_ok=True)
    polygons = get_polygons(AREA)
    metadata = None
    if AREA in ["Hopewell_Road", "Circleville_Road"]:
        for i in range(10):
            wkt = array_to_wkt_string(polygons[f"{AREA}_{i}"])
            for _ in range(10):
                try:
                    metadata = query_polygon(wkt)[0]
                    break
                except Exception as e:
                    print(f"Error fetching data: {e}. Retrying...")
            with open(output_dir / f"{AREA}_{i}.json", "w") as f:
                if metadata:
                    json.dump(metadata, f)
    else:
        wkt = array_to_wkt_string(polygons[AREA])
        for _ in range(10):
            try:
                metadata = query_polygon(wkt)[0]
                break
            except Exception as e:
                print(f"Error fetching data: {e}. Retrying...")
        if AREA == "Hart_Island":
            for year in [2015]:#[2014, 2015, 2021, 2023]:
                metadata_year = [t for t in metadata if str(year) in t["flight_date"]] if metadata else None
                with open(output_dir / f"{AREA}_{year}.json", "w") as f:
                    if metadata_year:
                        json.dump(metadata_year, f)


if __name__ == "__main__":
    main()
