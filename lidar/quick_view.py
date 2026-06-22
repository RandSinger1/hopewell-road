"""
quick_view_dtm.py

View a large DTM (GeoTIFF) at low resolution with proper coordinate axes.
Usage: python quick_view_dtm.py dtm_file.tif [--step STEP] [--cmap JET]
"""

import sys
import argparse
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

import parameters

def read_dtm_window(input_path, bounds, width=400, height=400,
                     resample='nearest', verbose=True):
    """
    Warp a bounded region of a DTM straight from full-resolution data into a
    small in-memory raster, bypassing any overview pyramid (overviewLevel=
    'NONE'). Use this instead of read_dtm_lowres() when only a sub-region is
    needed (e.g. a scan preview) — it's far cheaper than a whole-file
    decimated read, and it sidesteps a real failure mode observed on this
    project's merged_dtm.tif files: a thin/sparse data corridor inside a
    much larger nodata bounding box can cause the external .ovr overview
    pyramid to come out 100% nodata at every level (silently, with no
    error), which makes any decimated read that uses overviews — including
    plain GDAL ReadAsArray decimation — return an all-blank image even
    though the base raster has real data.

    Parameters
    ----------
    input_path : str
    bounds : (xmin, ymin, xmax, ymax)
        Window to read, in the raster's CRS (lon/lat for a WGS84-warped DTM).
    width, height : int
        Output pixel grid size.
    resample : 'nearest' or 'average'
        'nearest' is much faster (no need to touch every source pixel in the
        window) and is the better default for a quick preview; 'average' is
        more representative but reads every contributing full-resolution
        pixel, which is slow on a large window for an uncompressed-overview
        file like this one.
    verbose : bool

    Returns
    -------
    data : np.ma.MaskedArray (2D, float32) or None if the warp failed.
    extent : list [xmin, xmax, ymin, ymax], or None.
    """
    xmin, ymin, xmax, ymax = bounds
    ds = gdal.Open(input_path)
    if ds is None:
        raise RuntimeError(f"Could not open {input_path}")
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is None:
        nodata = -9999.0
    ds = None

    if verbose:
        print(f"  Warping window {tuple(round(b, 4) for b in bounds)} "
              f"-> {width}x{height} px (resample={resample}, overviews bypassed) …")

    opts = gdal.WarpOptions(
        format='MEM', dstSRS='EPSG:4326',
        outputBounds=[xmin, ymin, xmax, ymax],
        width=width, height=height,
        resampleAlg=gdal.GRA_NearestNeighbour if resample == 'nearest' else gdal.GRA_Average,
        dstNodata=nodata,
        overviewLevel='NONE',
    )
    mem_ds = gdal.Warp('', input_path, options=opts)
    if mem_ds is None:
        return None, None
    data = mem_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    mem_ds = None

    data = np.ma.masked_equal(data, np.float32(nodata))
    extent = [xmin, xmax, ymin, ymax]
    return data, extent


def read_dtm_lowres(input_path, sample_step=10, workers=1, verbose=True):
    """
    Read a large DTM (GeoTIFF) at low resolution.

    Parameters
    ----------
    input_path : str
        Path to the GeoTIFF file.
    sample_step : int
        Read every Nth pixel (both x and y) to reduce resolution.
    workers : int
        Parallel threads for reading (split into horizontal strips).
    verbose : bool
        Print progress/diagnostics.

    Returns
    -------
    data : np.ma.MaskedArray (2D, float32)
        Downsampled elevation array, nodata cells masked.
    extent : list [xmin, xmax, ymin, ymax]
        Geographic extent in the raster's native CRS units, matching the
        imshow(extent=...) convention used throughout this codebase.
    """
    def _p(msg):
        if verbose:
            print(msg)

    _p(f"[1] Opening: {input_path}")
    ds = gdal.Open(input_path)
    if ds is None:
        raise RuntimeError(f"Could not open {input_path}")
    _p(f"[1] OK — driver: {ds.GetDriver().ShortName}")

    band = ds.GetRasterBand(1)
    full_xsize = ds.RasterXSize
    full_ysize = ds.RasterYSize
    _p(f"[2] Full size: {full_xsize} x {full_ysize}")
    _p(f"[2] Geotransform: {ds.GetGeoTransform()}")
    _p(f"[2] CRS: {ds.GetProjection()[:80]}...")

    # Dimensions of the downsampled array
    buf_xsize = (full_xsize + sample_step - 1) // sample_step
    buf_ysize = (full_ysize + sample_step - 1) // sample_step
    _p(f"[3] Downsampled size: {buf_xsize} x {buf_ysize}  (step={sample_step})")

    # Read using positional arguments: (xoff, yoff, xsize, ysize, buf_xsize, buf_ysize)
    # workers > 1: split into horizontal strips, each opened by its own GDAL handle
    # and decompress tiles in parallel via GDAL_NUM_THREADS.
    _p(f"[4] Reading array (workers={workers}, this may take a moment)…")
    gdal.SetConfigOption('GDAL_NUM_THREADS', str(workers))

    if workers > 1:
        n_strips = workers
        strip_rows = max(1, (full_ysize + n_strips - 1) // n_strips)

        completed = [0]

        def _read_strip(strip_idx):
            y_off  = strip_idx * strip_rows
            y_size = min(strip_rows, full_ysize - y_off)
            if y_size <= 0:
                return None
            buf_y  = max(1, round(y_size / full_ysize * buf_ysize))
            ds_loc = gdal.Open(input_path)
            arr    = ds_loc.GetRasterBand(1).ReadAsArray(
                0, y_off, full_xsize, y_size, buf_xsize, buf_y)
            ds_loc = None
            completed[0] += 1
            _p(f"[4] strip {completed[0]}/{n_strips} done")
            return arr

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_read_strip, i): i for i in range(n_strips)}
            strips_by_idx = {}
            for fut in as_completed(futures):
                strips_by_idx[futures[fut]] = fut.result()
            strips = [strips_by_idx[i] for i in range(n_strips)]
        data = np.vstack([s for s in strips if s is not None])
    else:
        data = band.ReadAsArray(0, 0, full_xsize, full_ysize, buf_xsize, buf_ysize)

    gdal.SetConfigOption('GDAL_NUM_THREADS', None)
    _p(f"[4] Read complete — dtype: {data.dtype}, shape: {data.shape}")
    _p(f"[4] Raw min: {data.min():.2f}  max: {data.max():.2f}  mean: {data.mean():.2f}")

    # Optional: mask NoData values (if any)
    nodata = band.GetNoDataValue()
    _p(f"[5] Nodata value: {nodata}")
    data = data.astype(np.float32)
    if nodata is not None:
        nodata_f32 = np.float32(nodata)
        n_nodata = np.sum(data == nodata_f32)
        n_valid  = data.size - n_nodata
        _p(f"[5] Nodata pixels: {n_nodata} / {data.size}  ({100*n_nodata/data.size:.1f}%)")
        _p(f"[5] Valid pixels:  {n_valid}")
        data = np.ma.masked_equal(data, nodata_f32)
        if data.count() == 0:
            _p("[5] WARNING: all pixels masked after nodata removal!")
        else:
            _p(f"[5] Valid range: {data.min():.2f} – {data.max():.2f}")

    # Get geotransform and compute coordinate arrays for each downsampled pixel centre
    gt = ds.GetGeoTransform()
    # x coordinates: left edge + half-pixel offset for the first centre
    x_coords = np.linspace(
        gt[0] + gt[1] * (sample_step / 2.0),
        gt[0] + gt[1] * (full_xsize - sample_step / 2.0),
        buf_xsize
    )
    # y coordinates: top edge + half-pixel offset (note gt[5] is negative for north-up)
    y_coords = np.linspace(
        gt[3] + gt[5] * (sample_step / 2.0),
        gt[3] + gt[5] * (full_ysize - sample_step / 2.0),
        buf_ysize
    )

    ds = None

    # imshow(extent=...) convention: [xmin, xmax, ymin, ymax]
    extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]
    return data, extent


def quick_view_dtm(input_path, sample_step=10, cmap='viridis', figsize=(12, 8), outpath=None, workers=1, show_cities=True):
    """
    Display a low‑resolution version of a DTM with coordinate axes.

    Parameters
    ----------
    input_path : str
        Path to the GeoTIFF file.
    sample_step : int
        Read every Nth pixel (both x and y) to reduce resolution.
    cmap : str
        Matplotlib colormap name (e.g., 'viridis', 'terrain', 'jet').
    figsize : tuple
        Figure size in inches.
    """
    data, extent = read_dtm_lowres(input_path, sample_step=sample_step, workers=workers)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # imshow expects extent = [left, right, bottom, top]
    im = ax.imshow(data, extent=extent, cmap=cmap, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('X coordinate (map units)')
    ax.set_ylabel('Y coordinate (map units)')
    ax.set_title(f"DTM: {input_path} (low-res, step={sample_step})")
    if show_cities:
        circleville = parameters.CIRCLEVILLE_COORD
        newark      = parameters.NEWARK_COORD
        chillicothe = parameters.CHILLICOTHE_COORD
        high_banks  = parameters.HIGH_BANKS_COORD
        plt.plot(circleville[1], circleville[0], 'ro')
        plt.plot(newark[1],      newark[0],      'ro')
        plt.plot(chillicothe[1],  chillicothe[0],  'ro')
        plt.plot(high_banks[1],  high_banks[0],   'ro')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickly view a large DTM at low resolution")
    parser.add_argument("--input", type = str, default = parameters.REPORT_DTM_PATH,
                        help="Input GeoTIFF file")
    parser.add_argument("--step", type=int, default=1000,
                        help="Sampling step (every Nth pixel), default=10")
    parser.add_argument("--cmap", type=str, default="terrain",
                        help="Colormap, e.g., terrain, jet, viridis")
    parser.add_argument("--figsize", type=str, default="12,8",
                        help="Figure size in inches: width,height")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel threads for reading (default 4)")
    args = parser.parse_args()

    figsize = tuple(map(int, args.figsize.split(',')))
    quick_view_dtm(args.input, sample_step=args.step, cmap=args.cmap,
                   figsize=figsize, outpath="data/quick_view.png", workers=args.workers)