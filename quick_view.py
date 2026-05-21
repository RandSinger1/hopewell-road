#!/usr/bin/env python3
"""
quick_view_dtm.py

View a large DTM (GeoTIFF) at low resolution with proper coordinate axes.
Usage: python quick_view_dtm.py dtm_file.tif [--step STEP] [--cmap JET]
"""

import sys
import argparse
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

def quick_view_dtm(input_path, sample_step=10, cmap='viridis', figsize=(12, 8)):
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
    ds = gdal.Open(input_path)
    if ds is None:
        raise RuntimeError(f"Could not open {input_path}")

    band = ds.GetRasterBand(1)
    full_xsize = ds.RasterXSize
    full_ysize = ds.RasterYSize

    # Dimensions of the downsampled array
    buf_xsize = (full_xsize + sample_step - 1) // sample_step
    buf_ysize = (full_ysize + sample_step - 1) // sample_step

    # Read using positional arguments: (xoff, yoff, xsize, ysize, buf_xsize, buf_ysize)
    data = band.ReadAsArray(0, 0, full_xsize, full_ysize, buf_xsize, buf_ysize)
    data = data.astype(np.float32)

    # Optional: mask NoData values (if any)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        data = np.ma.masked_equal(data, nodata)

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

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # imshow expects extent = [left, right, bottom, top]
    # Note: y_coords[0] is the top (largest Y), y_coords[-1] is the bottom (smallest Y)
    im = ax.imshow(data, extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]],
                   cmap=cmap, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('X coordinate (map units)')
    ax.set_ylabel('Y coordinate (map units)')
    ax.set_title(f"DTM: {input_path} (low-res, step={sample_step})")
    circleville =(39.6,-82.9)
    newark = (40.0,-82.4)
    chilicothe = (39.3,-83.0)
    plt.plot(circleville[1], circleville[0], 'ro')
    plt.plot(newark[1], newark[0], 'ro')
    plt.plot(chilicothe[1], chilicothe[0], 'ro')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickly view a large DTM at low resolution")
    parser.add_argument("--input", type = str, default = "loaded_tiles/Circleville_Road/merged_dtm.tif",
                        help="Input GeoTIFF file")
    parser.add_argument("--step", type=int, default=1000,
                        help="Sampling step (every Nth pixel), default=10")
    parser.add_argument("--cmap", type=str, default="terrain",
                        help="Colormap, e.g., terrain, jet, viridis")
    parser.add_argument("--figsize", type=str, default="12,8",
                        help="Figure size in inches: width,height")
    args = parser.parse_args()

    figsize = tuple(map(int, args.figsize.split(',')))
    quick_view_dtm(args.input, sample_step=args.step, cmap=args.cmap, figsize=figsize)