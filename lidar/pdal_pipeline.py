import pdal
import json
import re
from pathlib import Path
from osgeo import gdal, osr
gdal.UseExceptions()
from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
import argparse

import parameters

# ---------- configuration ----------
VERBOSE = 1
# -----------------------------------


def make_pipeline(input_laz, output_laz):
    return json.dumps([
        input_laz,
        {   # denoise (remove isolated noise points)
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": parameters.OUTLIER_MEAN_K,
            "multiplier": parameters.OUTLIER_MULTIPLIER,
        },
        {   # ground classification (SMRF — Pingel et al. 2013)
            "type": "filters.smrf",
            "cell": parameters.SMRF_CELL,
            "window": parameters.SMRF_WINDOW,
            "slope": parameters.SMRF_SLOPE,
            "threshold": parameters.SMRF_THRESHOLD,
            "scalar": parameters.SMRF_SCALAR,
            "cut": parameters.SMRF_CUT,
            "returns": parameters.SMRF_RETURNS,
        },
        {   # keep only ground points (class 2)
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {   # save ground-only tile
            "type": "writers.las",
            "filename": output_laz,
            "compression": "laszip"
        }
    ])


def make_dtm_pipeline(input_laz, output_tif):
    return json.dumps([
        input_laz,
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {
            "type": "writers.gdal",
            "filename": output_tif,
            "resolution": parameters.DTM_RESOLUTION,
            "output_type": parameters.DTM_INTERP_METHOD,
            "gdaldriver": "GTiff"
        }
    ])


def _get_tile_srs(ds: "gdal.Dataset",
                  fallback: str = "") -> "osr.SpatialReference":
    """Return an osr.SpatialReference for a GDAL dataset.

    Resolution order:
      1. ds.GetSpatialRef()  — fastest, skips WKT round-trip
      2. ds.GetProjection() + ImportFromWkt / SetFromUserInput
      3. fallback string (EPSG code, PROJ string, or WKT)
    """
    srs = ds.GetSpatialRef()
    if srs is not None:
        return srs

    wkt = ds.GetProjection()
    if wkt:
        srs = osr.SpatialReference()
        try:
            srs.ImportFromWkt(wkt)
            return srs
        except RuntimeError:
            try:
                srs.SetFromUserInput(wkt)
                return srs
            except RuntimeError:
                pass  # fall through to fallback

    if fallback:
        srs = osr.SpatialReference()
        srs.SetFromUserInput(fallback)
        return srs

    raise ValueError(
        "Tile has no embedded CRS and no SOURCE_CRS fallback is set.\n"
        "Set parameters.SOURCE_CRS to the EPSG code of your data "
        "(e.g. \"EPSG:3735\" for Ohio State Plane South)."
    )


def warp_progress(complete, message, user_data):
    """Print progress percentage every time it changes."""
    pct = int(complete * 100)
    if not hasattr(warp_progress, "last_pct") or pct != warp_progress.last_pct:
        print(f"Warp progress: {pct}% - {message}")
        warp_progress.last_pct = pct
    return 1   # 1 = continue, 0 = cancel


def warp_to_4326_auto(input_tiles, output_tif,
                       resample=parameters.WARP_RESAMPLE,
                       nodata=parameters.WARP_NODATA,
                       creation_opts=None, callback=None,
                       multithread=True, workers=1):
    """
    Reproject a list of DTM tiles (any projected CRS with linear units) to
    EPSG:4326 using the EXACT input resolution converted to degrees.
    All parameters (resolution, bounds, data type) are taken from the input files.
    """
    input_tiles = [str(p) for p in input_tiles if gdal.Open(str(p)) is not None]
    if not input_tiles:
        raise ValueError("No readable input tiles found.")

    if creation_opts is None:
        creation_opts = parameters.WARP_CREATION_OPTS

    # 1. Read metadata from first tile
    first_ds = gdal.Open(input_tiles[0])
    gt = first_ds.GetGeoTransform()
    src_xres = gt[1]                # pixel width in source units (e.g., 1.0 ft)
    src_yres = abs(gt[5])           # pixel height (positive)
    src_dtype = first_ds.GetRasterBand(1).DataType
    srs = _get_tile_srs(first_ds, fallback=parameters.SOURCE_CRS)
    first_ds = None

    auth = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)
    src_crs_str = f"{auth}:{code}" if auth and code else srs.ExportToWkt()
    print(f"Source CRS: {src_crs_str}")

    # 2. Compute union extent of all tiles in the source CRS
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for f in input_tiles:
        ds = gdal.Open(f)
        g = ds.GetGeoTransform()
        cols, rows = ds.RasterXSize, ds.RasterYSize
        tile_minx = g[0]
        tile_maxy = g[3]
        tile_maxx = tile_minx + cols * g[1]
        tile_miny = tile_maxy + rows * g[5]   # g[5] is negative
        min_x = min(min_x, tile_minx)
        max_x = max(max_x, tile_maxx)
        min_y = min(min_y, tile_miny)
        max_y = max(max_y, tile_maxy)
        ds = None

    centre_x = (min_x + max_x) / 2.0
    centre_y = (min_y + max_y) / 2.0

    # 3. Set up pyproj transformer (always_xy=True gives (lon, lat) order)
    trans = Transformer.from_crs(src_crs_str, "EPSG:4326", always_xy=True)

    # 4. Transform four corners to WGS84 → output bounds
    corners = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    lonlat = [trans.transform(x, y) for x, y in corners]
    lon_min = min(pt[0] for pt in lonlat)
    lon_max = max(pt[0] for pt in lonlat)
    lat_min = min(pt[1] for pt in lonlat)
    lat_max = max(pt[1] for pt in lonlat)

    # 5. Compute degrees per pixel moving one pixel east and north from centre
    centre_lon, centre_lat = trans.transform(centre_x, centre_y)
    lon_east, _ = trans.transform(centre_x + src_xres, centre_y)
    _, lat_north = trans.transform(centre_x, centre_y + src_yres)

    deg_per_pixel_x = abs(lon_east - centre_lon)
    deg_per_pixel_y = abs(lat_north - centre_lat)

    if VERBOSE > 0:
        print(f"Centre (source): {centre_x:.1f}, {centre_y:.1f}")
        print(f"Centre (WGS84):   {centre_lon:.8f}, {centre_lat:.8f}")
        print(f"deg/pixel X = {deg_per_pixel_x:.12f}")
        print(f"deg/pixel Y = {deg_per_pixel_y:.12f}")
    cols = int((lon_max - lon_min) / deg_per_pixel_x)
    rows = int((lat_max - lat_min) / deg_per_pixel_y)
    if VERBOSE > 0:
        print(f"Output size: {cols} × {rows} pixels, ~{cols*rows*4/1e9:.2f} GB")

    # 6. Choose output data type (downgrade Float64 to Float32)
    out_dtype = gdal.GDT_Float32 if src_dtype == gdal.GDT_Float64 else src_dtype

    # 7. Execute the warp
    # multithread=True disables the per-pixel callback (not thread-safe in GDAL)
    if multithread:
        gdal_callback = None
        print("  Warping (multithreaded) …")
    else:
        if callback is None:
            def callback(complete, msg, _):
                print(f"\r  {int(complete*100)}%", end='', flush=True)
                return 1
        gdal_callback = callback

    ds = gdal.Warp(
        str(output_tif),
        input_tiles,
        dstSRS="EPSG:4326",
        outputBounds=[lon_min, lat_min, lon_max, lat_max],
        xRes=deg_per_pixel_x,
        yRes=deg_per_pixel_y,
        resampleAlg=resample,
        outputType=out_dtype,
        multithread=multithread,
        warpOptions=[f"NUM_THREADS={workers}"],
        dstNodata=nodata,
        creationOptions=creation_opts,
        callback=gdal_callback,
    )
    ds = None
    print("\nDone!")
    return output_tif


def _compute_global_warp_params(tiles_by_area: dict,
                                resample=parameters.WARP_RESAMPLE,
                                nodata=parameters.WARP_NODATA,
                                creation_opts=None) -> dict:
    """Scan all tiles once and return a JSON-serialisable params dict.

    Contains global output bounds + per-area bounds (in WGS84), the shared
    pixel resolution, output dtype, and creation options.  Stored in the
    checkpoint so a resumed run skips this expensive scan.
    """
    if creation_opts is None:
        creation_opts = parameters.WARP_CREATION_OPTS

    all_tiles = [t for tiles in tiles_by_area.values() for t in tiles]

    first_ds = gdal.Open(all_tiles[0])
    gt0 = first_ds.GetGeoTransform()
    src_xres = gt0[1]
    src_yres = abs(gt0[5])
    src_dtype = first_ds.GetRasterBand(1).DataType
    srs = _get_tile_srs(first_ds, fallback=parameters.SOURCE_CRS)
    first_ds = None

    auth = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)
    src_crs = f"{auth}:{code}" if auth and code else srs.ExportToWkt()
    trans = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    def _src_extent(tile_paths):
        mn_x, mx_x = float('inf'), float('-inf')
        mn_y, mx_y = float('inf'), float('-inf')
        for f in tile_paths:
            ds = gdal.Open(f)
            g = ds.GetGeoTransform()
            c, r = ds.RasterXSize, ds.RasterYSize
            x0, y1 = g[0], g[3]
            x1 = x0 + c * g[1]
            y0 = y1 + r * g[5]
            mn_x = min(mn_x, x0); mx_x = max(mx_x, x1)
            mn_y = min(mn_y, y0); mx_y = max(mx_y, y1)
            ds = None
        return mn_x, mx_x, mn_y, mx_y

    def _wgs84_bounds(mn_x, mx_x, mn_y, mx_y):
        corners = [(mn_x, mn_y), (mn_x, mx_y), (mx_x, mx_y), (mx_x, mn_y)]
        ll = [trans.transform(x, y) for x, y in corners]
        return [min(p[0] for p in ll), min(p[1] for p in ll),
                max(p[0] for p in ll), max(p[1] for p in ll)]

    g_mn_x, g_mx_x = float('inf'), float('-inf')
    g_mn_y, g_mx_y = float('inf'), float('-inf')
    area_bounds: dict[str, list[float]] = {}

    n = len(tiles_by_area)
    for i, (area, tiles) in enumerate(tiles_by_area.items(), 1):
        print(f"  scanning {area} ({i}/{n}) …", end='\r')
        mn_x, mx_x, mn_y, mx_y = _src_extent(tiles)
        g_mn_x = min(g_mn_x, mn_x); g_mx_x = max(g_mx_x, mx_x)
        g_mn_y = min(g_mn_y, mn_y); g_mx_y = max(g_mx_y, mx_y)
        area_bounds[area] = _wgs84_bounds(mn_x, mx_x, mn_y, mx_y)
    print()

    cx, cy = (g_mn_x + g_mx_x) / 2, (g_mn_y + g_mx_y) / 2
    clon, clat = trans.transform(cx, cy)
    lon_e, _   = trans.transform(cx + src_xres, cy)
    _, lat_n   = trans.transform(cx, cy + src_yres)

    out_dtype = gdal.GDT_Float32 if src_dtype == gdal.GDT_Float64 else src_dtype

    return {
        'xRes':          abs(lon_e - clon),
        'yRes':          abs(lat_n - clat),
        'outputType':    int(out_dtype),
        'resampleAlg':   resample,
        'nodata':        nodata,
        'creation_opts': list(creation_opts),
        'global_bounds': _wgs84_bounds(g_mn_x, g_mx_x, g_mn_y, g_mx_y),
        'area_bounds':   area_bounds,
    }


def _warp_chunk(tiles: list[str], output_tif, bounds: list[float],
                params: dict, workers: int = 1) -> None:
    """Warp tiles to output_tif using pre-computed global params and local bounds."""
    gdal.Warp(
        str(output_tif),
        tiles,
        dstSRS="EPSG:4326",
        outputBounds=bounds,
        xRes=params['xRes'],
        yRes=params['yRes'],
        resampleAlg=params['resampleAlg'],
        outputType=params['outputType'],
        multithread=True,
        warpOptions=[f"NUM_THREADS={workers}"],
        dstNodata=params['nodata'],
        creationOptions=params['creation_opts'],
    )


def _classify_tile(laz_path: Path, out_path: Path) -> None:
    """Ground-classify one LAZ tile. Top-level so it is picklable by ProcessPoolExecutor."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    pdal.Pipeline(make_pipeline(str(laz_path), str(out_path))).execute()


def _rasterize_tile(ground_file: Path, dtm_out: Path) -> None:
    """Rasterise one ground-classified LAZ to a DTM GeoTIFF. Top-level for pickling."""
    if dtm_out.exists() and dtm_out.stat().st_size > 0:
        return
    pdal.Pipeline(make_dtm_pipeline(str(ground_file), str(dtm_out))).execute()


def process_tiles(AREA: str, max_workers: int = 4) -> None:
    INPUT_DIR = Path(f"data/loaded_tiles/{AREA}")
    RAW_DIR = INPUT_DIR / "raw_tiles"
    GROUND_DIR = INPUT_DIR / "ground_tiles"
    DTM_DIR = INPUT_DIR / "dtm_tiles"
    GROUND_DIR.mkdir(parents=True, exist_ok=True)
    DTM_DIR.mkdir(parents=True, exist_ok=True)

    laz_files = sorted(RAW_DIR.glob("*.laz"))
    print(f"[{AREA}] Classifying {len(laz_files)} tiles ({max_workers} workers) …")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_classify_tile, laz, GROUND_DIR / f"{laz.stem}_ground.laz"): laz.name
            for laz in laz_files
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                if VERBOSE > 0:
                    print(f"  [{AREA}] classified {name}")
            except Exception as exc:
                print(f"  [{AREA}] ERROR classifying {name}: {exc}")

    ground_files = sorted(GROUND_DIR.glob("*.laz"))
    print(f"[{AREA}] Rasterising {len(ground_files)} tiles ({max_workers} workers) …")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _rasterize_tile, gf,
                DTM_DIR / f"{gf.stem.replace('_ground', '')}_dtm.tif",
            ): gf.name
            for gf in ground_files
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                if VERBOSE > 0:
                    print(f"  [{AREA}] rasterised {name}")
            except Exception as exc:
                print(f"  [{AREA}] ERROR rasterising {name}: {exc}")

def merge_dtms(sub_areas: list[str], output_path, workers: int = 1,
               resume: bool = True) -> "Path | None":
    """Merge DTM tiles into a single GeoTIFF with per-sub-area checkpointing.

    Each sub-area is warped to its own temp file and the result is saved to a
    checkpoint JSON.  If interrupted, re-running with resume=True (the default)
    skips already-finished sub-areas and continues from where it left off.
    """
    base = Path("data/loaded_tiles")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir         = output_path.parent / f"{output_path.stem}_tmp"
    checkpoint_path = output_path.parent / f"{output_path.stem}.checkpoint.json"

    # Collect tiles per sub-area (preserving caller order)
    tiles_by_area: dict[str, list[str]] = {}
    for sa in sub_areas:
        d = base / sa / "dtm_tiles"
        if d.exists():
            tiles = [str(t) for t in sorted(d.glob("*.tif"))]
            if tiles:
                tiles_by_area[sa] = tiles

    if not tiles_by_area:
        print("No DTM tiles found.")
        return None

    total = sum(len(t) for t in tiles_by_area.values())
    print(f"Found {total} DTM tiles across {len(tiles_by_area)} sub-areas -> {output_path}")

    # Load existing checkpoint or start fresh
    checkpoint: dict = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        n_done = len(checkpoint.get('completed', []))
        print(f"Resuming: {n_done}/{len(tiles_by_area)} sub-areas already done")

    completed: set[str] = set(checkpoint.get('completed', []))

    # Compute warp params once; skip the scan on resume if already saved
    if 'warp_params' not in checkpoint:
        print("Computing output bounds and resolution from all tiles ...")
        params = _compute_global_warp_params(tiles_by_area)
        checkpoint['warp_params'] = params
        checkpoint['completed']   = []
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
    else:
        params = checkpoint['warp_params']

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build temp-file list upfront (needed by the final mosaic regardless)
    temp_files: list[Path] = [tmp_dir / f"{sa}.tif" for sa in tiles_by_area]

    # Split into already-done and still-pending
    pending = [
        (sa, tiles_by_area[sa])
        for sa in tiles_by_area
        if not (sa in completed and (tmp_dir / f"{sa}.tif").exists())
    ]
    for sa in tiles_by_area:
        if sa not in {s for s, _ in pending}:
            print(f"  [{sa}] skipping (checkpoint)")

    # Warp each sub-area to its own temp file
    # workers > 1 → run sub-areas in parallel (one GDAL thread per process)
    # workers == 1 → sequential, single-threaded GDAL
    if workers > 1 and pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_sa = {
                executor.submit(
                    _warp_chunk,
                    tiles, tmp_dir / f"{sa}.tif",
                    params['area_bounds'][sa], params, 1,
                ): sa
                for sa, tiles in pending
            }
            for future in as_completed(future_to_sa):
                sa = future_to_sa[future]
                try:
                    future.result()
                    completed.add(sa)
                    checkpoint['completed'] = list(completed)
                    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
                    print(f"  [{sa}] done")
                except Exception as exc:
                    print(f"  [{sa}] ERROR: {exc}")
    else:
        for sa, tiles in pending:
            tmp_out = tmp_dir / f"{sa}.tif"
            print(f"  [{sa}] warping {len(tiles)} tiles ...")
            _warp_chunk(tiles, tmp_out, params['area_bounds'][sa], params, workers=1)
            completed.add(sa)
            checkpoint['completed'] = list(completed)
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
            print(f"  [{sa}] done")

    # Final mosaic: combine all local-extent temp files into the full output
    existing = [str(f) for f in temp_files if f.exists()]
    print(f"Mosaicking {len(existing)} sub-area files -> {output_path} ...")
    gdal.Warp(
        str(output_path),
        existing,
        dstSRS="EPSG:4326",
        outputBounds=params['global_bounds'],
        xRes=params['xRes'],
        yRes=params['yRes'],
        resampleAlg=params['resampleAlg'],
        outputType=params['outputType'],
        multithread=True,
        warpOptions=[f"NUM_THREADS={workers}"],
        dstNodata=params['nodata'],
        creationOptions=params['creation_opts'],
    )
    print("Done!")

    checkpoint_path.unlink(missing_ok=True)
    return output_path


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Process LiDAR tiles for a specified area.")
    parser.add_argument("area", type=str, nargs="+",
                        help="Area(s) to process (e.g., Hopewell_Road, Circleville_Road, full_road)")
    parser.add_argument("--area-splits", type=int, default=1, metavar="N")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="Parallel processes per area (default 4)")
    args = parser.parse_args()

    sub_areas: list[str] = []
    for a in args.area:
        if a == "full_road":
            sub_areas += ([f"Hopewell_Road_{i}" for i in range(args.area_splits)] +
                          [f"Circleville_Road_{i}" for i in range(args.area_splits)])
        elif args.area_splits > 1:
            sub_areas += [f"{a}_{i}" for i in range(args.area_splits)]

    for name in sub_areas:
        process_tiles(name, max_workers=args.workers)

    if len(sub_areas) > 1 or "full_road" in args.area:
        out_name = "Whole_Road" if "full_road" in args.area else "_".join(args.area)
        if "full_road" in args.area:
            # Discover every numbered split that already has DTM tiles on disk,
            # regardless of how many splits were processed this run.
            base = Path("data/loaded_tiles")
            _pattern = re.compile(r"^(Hopewell_Road|Circleville_Road)_(\d+)$")
            def _sort_key(n):
                m = _pattern.fullmatch(n)
                return (m.group(1), int(m.group(2)))  # type: ignore[union-attr]

            merge_areas = sorted(
                [d.name for d in base.iterdir()
                 if d.is_dir()
                 and _pattern.fullmatch(d.name)
                 and any((d / "dtm_tiles").glob("*.tif"))],
                key=_sort_key,
            )
            if not merge_areas:
                print("full_road: no DTM tiles found in any Hopewell_Road_* or Circleville_Road_* split.")
            else:
                print(f"full_road: merging {len(merge_areas)} splits: {merge_areas}")
                merge_dtms(merge_areas, Path(f"data/loaded_tiles/{out_name}/merged_dtm.tif"))
        else:
            merge_dtms(sub_areas, Path(f"data/loaded_tiles/{out_name}/merged_dtm.tif"))

    