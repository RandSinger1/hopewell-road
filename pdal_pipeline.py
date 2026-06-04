import pdal
import json
from pathlib import Path
from osgeo import gdal, osr
gdal.UseExceptions()
from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import params

# ---------- configuration ----------
VERBOSE = 1
# -----------------------------------


def make_pipeline(input_laz, output_laz):
    return json.dumps([
        input_laz,
        {   # denoise (remove isolated noise points)
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": params.OUTLIER_MEAN_K,
            "multiplier": params.OUTLIER_MULTIPLIER,
        },
        {   # ground classification (SMRF — Pingel et al. 2013)
            "type": "filters.smrf",
            "cell": params.SMRF_CELL,
            "window": params.SMRF_WINDOW,
            "slope": params.SMRF_SLOPE,
            "threshold": params.SMRF_THRESHOLD,
            "scalar": params.SMRF_SCALAR,
            "cut": params.SMRF_CUT,
            "returns": params.SMRF_RETURNS,
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
            "resolution": params.DTM_RESOLUTION,
            "output_type": params.DTM_INTERP_METHOD,
            "gdaldriver": "GTiff"
        }
    ])


def warp_progress(complete, message, user_data):
    """Print progress percentage every time it changes."""
    pct = int(complete * 100)
    if not hasattr(warp_progress, "last_pct") or pct != warp_progress.last_pct:
        print(f"Warp progress: {pct}% - {message}")
        warp_progress.last_pct = pct
    return 1   # 1 = continue, 0 = cancel


def warp_to_4326_auto(input_tiles, output_tif,
                       resample=params.WARP_RESAMPLE,
                       nodata=params.WARP_NODATA,
                       creation_opts=None, callback=None,
                       multithread=True):
    """
    Reproject a list of DTM tiles (any projected CRS with linear units) to
    EPSG:4326 using the EXACT input resolution converted to degrees.
    All parameters (resolution, bounds, data type) are taken from the input files.
    """
    def _has_projection(p):
        ds = gdal.Open(str(p))
        return ds is not None and bool(ds.GetProjection())

    input_tiles = [str(p) for p in input_tiles if _has_projection(p)]
    if not input_tiles:
        raise ValueError("No input tiles with a valid CRS found.")

    if creation_opts is None:
        creation_opts = params.WARP_CREATION_OPTS

    # 1. Read metadata from first tile
    first_ds = gdal.Open(input_tiles[0])
    gt = first_ds.GetGeoTransform()
    src_xres = gt[1]                # pixel width in source units (e.g., 1.0 ft)
    src_yres = abs(gt[5])           # pixel height (positive)
    src_wkt = first_ds.GetProjection()
    src_dtype = first_ds.GetRasterBand(1).DataType
    first_ds = None

    # Extract EPSG string from WKT
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_wkt)
    auth = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)
    src_crs_str = f"{auth}:{code}" if auth and code else src_wkt
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
        dstNodata=nodata,
        creationOptions=creation_opts,
        callback=gdal_callback,
    )
    ds = None
    print("\nDone!")
    return output_tif


def process_tiles(AREA):
    INPUT_DIR = Path(f"data/loaded_tiles/{AREA}")
    RAW_DIR = INPUT_DIR / "raw_tiles"
    GROUND_DIR = INPUT_DIR / "ground_tiles"
    DTM_DIR = INPUT_DIR / "dtm_tiles"
    GROUND_DIR.mkdir(parents=True, exist_ok=True)
    DTM_DIR.mkdir(parents=True, exist_ok=True)

    laz_files = sorted(RAW_DIR.glob("*.laz"))
    for i, laz_path in enumerate(laz_files):
        out_path = GROUND_DIR / f"{laz_path.stem}_ground.laz"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        if VERBOSE > 0:
            print(f"[{i}/{len(laz_files)}] Processing {laz_path.name} …")
        pipeline_json = make_pipeline(str(laz_path), str(out_path))
        pipe = pdal.Pipeline(pipeline_json)
        pipe.execute()

    for i, ground_file in enumerate(sorted(GROUND_DIR.glob("*.laz"))):
        dtm_out = DTM_DIR / f"{ground_file.stem.replace('_ground', '')}_dtm.tif"
        if dtm_out.exists() and dtm_out.stat().st_size > 0:
            continue
        if VERBOSE > 0:
            print(f"[{i}] Generating DTM for {ground_file.name} …")
        dtm_pipeline_json = make_dtm_pipeline(str(ground_file), str(dtm_out))
        pdal.Pipeline(dtm_pipeline_json).execute()

def merge_dtms():
    base = Path("data/loaded_tiles")
    dtm_tiles = []
    for area_dir in sorted(base.glob("Hopewell_Road_*/dtm_tiles")):
        dtm_tiles.extend(sorted(area_dir.glob("*.tif")))
    for area_dir in sorted(base.glob("Circleville_Road_*/dtm_tiles")):
        dtm_tiles.extend(sorted(area_dir.glob("*.tif")))

    if not dtm_tiles:
        print("No DTM tiles found.")
        return

    output_path = base / "Whole_Road" / "merged_dtm.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(dtm_tiles)} DTM tiles — warping to {output_path}")
    warp_to_4326_auto(dtm_tiles, output_path, callback=warp_progress)


if __name__ == "__main__":
    
    freeze_support()
    areas = [f"Hopewell_Road_{i}" for i in range(1, 11)] + [f"Circleville_Road_{i}" for i in range(0, 10)]
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_tiles, name) for name in areas]
        for future in as_completed(futures):
            future.result()
    merge_dtms()

    