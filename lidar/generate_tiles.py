import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import laspy
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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


def _make_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://",  HTTPAdapter(max_retries=retry))
    return session


def download_and_cache(tile_url, local_path):
    session = _make_session()
    resp = session.get(tile_url, timeout=120)
    resp.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(resp.content)


def _download_one(args):
    idx, url, local_file = args
    if local_file.exists():
        return idx, url, None
    try:
        download_and_cache(url, local_file)
        return idx, url, None
    except Exception as e:
        return idx, url, str(e)


def generate_tiles(tiles, area, max_workers=4):
    output_dir = Path(f'data/loaded_tiles/{area}/raw_tiles')
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_log = Path(f'data/loaded_tiles/{area}/failed_downloads.txt')

    work = [(idx, url, output_dir / f'tile_{idx}.laz') for idx, url in enumerate(tiles)]
    total = len(work)
    done = 0
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, item): item[0] for item in work}
        for future in as_completed(futures):
            idx, url, err = future.result()
            done += 1
            if err:
                print(f"[{done}/{total}] tile {idx} error: {err}")
                failures.append((idx, url, err))
            else:
                print(f"[{done}/{total}] tile {idx} done")

    if failures:
        with open(failed_log, 'w') as f:
            for idx, url, err in sorted(failures):
                f.write(f"tile_{idx}\t{url}\t{err}\n")
        print(f"{len(failures)} failed download(s) logged to {failed_log}")
    elif failed_log.exists():
        failed_log.unlink()  # clean up old log if everything succeeded this run


def main():
    args = parse_args()
    area = args.area
    with open(Path("data/saved_file_lists") / f"{area}.json", "r") as f:
        tile_array = json.load(f)
        tiles = [t['url'] for t in tile_array]
    generate_tiles(tiles, area)


if __name__ == "__main__":
    main()
