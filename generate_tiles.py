import json

import laspy
import numpy as np
import requests
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate tiles from LAZ files")
    parser.add_argument("area", type=str, help="Area name to process")
    return parser.parse_args()


def is_url(path):
    try:
        parsed = urlparse(str(path))
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def locate_file(path):
    path_obj = Path(path)
    if path_obj.exists():
        return str(path_obj)
    alt_path = Path.cwd() / path
    if alt_path.exists():
        return str(alt_path)
    return None


def open_image_source(path):
    if is_url(path):
        resp = requests.get(path, timeout=10)
        resp.raise_for_status()
        laz_data = laspy.read(BytesIO(resp.content))
    else:
        tile_file = locate_file(path)
        if tile_file is None:
            raise FileNotFoundError(f"Could not locate tile file: {path}")
        laz_data = laspy.read(tile_file)
    return laz_data


def download_and_cache(tile_url, local_path):
    resp = requests.get(tile_url, timeout=60)
    resp.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(resp.content)


def generate_tiles(tiles, area):
    output_dir = Path(f'loaded_tiles/{area}/raw_tiles')
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, t in enumerate(tiles):
        local_file = output_dir / f'tile_{idx}.laz'
        if not local_file.exists():
            try:
                download_and_cache(t, local_file)
            except Exception as e:
                print(f"Error downloading tile {idx}: {e}")


def main():
    args = parse_args()
    area = args.area
    with open(Path("saved_file_lists") / f"{area}.json", "r") as f:
        tile_array = json.load(f)
        tiles = [t['url'] for t in tile_array]
    generate_tiles(tiles, area)


if __name__ == "__main__":
    main()
