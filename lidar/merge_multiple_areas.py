#!/usr/bin/env python3
"""
merge_dtms.py

Merge multiple DTM GeoTIFF files into one large mosaic.
No resampling, reprojection, or value modification is applied.
Uses GDAL's Warp functionality (faster than gdal_merge.py for many files).
"""

from pathlib import Path
from osgeo import gdal
gdal.UseExceptions()
from utils import make_gdal_progress_callback


def merge_dtms(input_files, output_path, merge_alg="last", nodata=None, verbose=True):
    """
    Merge a list of raster files into a single mosaic.

    Parameters
    ----------
    input_files : list of str or Path
        Paths to input TIFF files.
    output_path : str or Path
        Path for the output merged file.
    merge_alg : str, optional
        Algorithm for overlapping cells: "last" (default), "first", "min", "max",
        "mean", "blend". "last" means the value from the last file in the list
        will be used; for non-overlapping tiles it doesn't matter.
    nodata : float, optional
        NoData value to use for the output. If None, GDAL will determine it
        from the input files.
    verbose : bool, optional
        Print progress messages.
    """
    input_files = [str(p) for p in input_files]
    output_path = str(output_path)

    if verbose:
        print(f"Merging {len(input_files)} files into {output_path}")

    ds = gdal.Warp(
        output_path,
        input_files,
        resampleAlg=gdal.GRIORA_NearestNeighbour,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
        callback=make_gdal_progress_callback("Merge")
    )
    ds = None
    if verbose:
        print("Merge complete.")


if __name__ == "__main__":
    input_tifs = [Path("data/loaded_tiles/Circleville_Road/merged_dtm.tif"), Path("data/loaded_tiles/Hopewell_Road/merged_dtm.tif")]#[Path(f"data/loaded_tiles/Circleville_Road_{i}/merged_dtm.tif") for i in range(10)]
    output_file = Path("data/loaded_tiles/Whole_Road/merged_dtm.tif")
    existing = [f for f in input_tifs if f.exists()]
    if len(existing) != len(input_tifs):
        print(f"Warning: {len(input_tifs) - len(existing)} file(s) not found, skipping.")

    if existing:
        merge_dtms(existing, output_file, verbose=True)
    else:
        print("No valid input files found.")
