"""
process_dtm.py

Raster helpers for LiDAR DTM processing: windowed reads, local relief model,
sky-view factor, Frangi multi-scale ridge enhancement, and display utilities.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
gdal.UseExceptions()
from scipy.ndimage import gaussian_filter, generic_filter, median_filter, distance_transform_edt
from scipy.signal import convolve2d
from xrspatial import sky_view_factor
from utils import meters_to_degrees, create_parallel_ridge_kernel_compact
import xarray as xr
import params


def get_window(lat, lon, size_x_meters, size_y_meters, gt, band, nodata):
    delta_x, delta_y = meters_to_degrees(size_x_meters, size_y_meters, lat)
    min_lon = lon - delta_x
    max_lon = lon + delta_x
    min_lat = lat - delta_y
    max_lat = lat + delta_y
    col_start = int((min_lon - gt[0]) / gt[1])
    col_end = int((max_lon - gt[0]) / gt[1]) + 1
    row_start = int((gt[3] - max_lat) / abs(gt[5]))
    row_end = int((gt[3] - min_lat) / abs(gt[5])) + 1
    xmin_win = gt[0] + col_start * gt[1]
    ymax_win = gt[3] + row_start * gt[5]
    xmax_win = gt[0] + col_end * gt[1]
    ymin_win = gt[3] + row_end * gt[5]
    extent = [xmin_win, xmax_win, ymin_win, ymax_win]
    window = band.ReadAsArray(col_start, row_start, col_end - col_start, row_end - row_start)
    return window, extent


def getGTAndBand(filepath):
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open {filepath}")
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = -9999
    return gt, band, nodata, ds


def filter_image(window, extent, nodata=params.NODATA, contours=False, gaussian_smoothing=False,
                  median_smoothing=False, lrm=False, lrm_radius=params.LRM_RADIUS,
                  svf=False, svf_radius=params.SVF_RADIUS, svf_directions=params.SVF_DIRECTIONS,
                  ridge=False, ridge_scales=None,
                  smoothing_radius=params.MEDIAN_SMOOTH_RADIUS, smoothing_sigma=1.0, title=None):
    """
    Display a LiDAR window with optional archaeological pre-processing.

    Parameters
    ----------
    window : 2D numpy array
        Elevation raster (rows = y, cols = x). North is the first row.
    extent : list [xmin, xmax, ymin, ymax]
        Geographic extent of the raster.
    nodata : float
        Value to treat as no data. Masked during computations.
    contours : bool
        Overlay contour lines.
    gaussian_smoothing : bool
        Apply Gaussian smoothing to the displayed raster.
    median_smoothing : bool
        Apply median smoothing to the displayed raster.
    lrm : bool
        Compute and display Local Relief Model.
    lrm_radius : int
        Radius (in cells) of the mean filter for LRM.
    svf : bool
        Compute and display Sky-View Factor.
    svf_radius : int
        Search radius (in cells) for horizon calculation.
    svf_directions : int
        Number of azimuth directions (e.g. 16).
    ridge : bool
        Compute and display multi-scale ridge enhancement (Frangi vesselness).
    ridge_scales : list of float
        Sigma values (in cells) for the ridge filter.
    smoothing_radius : int
        Radius for initial smoothing.
    smoothing_sigma : float
        Sigma for Gaussian kernel.
    """
    if ridge_scales is None:
        ridge_scales = params.RIDGE_SCALES

    if gaussian_smoothing:
        window = gaussian_smooth_windowed(window, nodata=nodata,
                                          radius=smoothing_radius,
                                          sigma=smoothing_sigma)
    elif median_smoothing:
        window = median_smooth_windowed(window, nodata=nodata, radius=smoothing_radius)

    if lrm:
        window = local_relief_model(window, radius=lrm_radius, nodata=nodata)
        print("Displaying Local Relief Model")
    elif svf:
        window = sky_view_factor_xr(window, radius=svf_radius, directions=svf_directions, nodata=nodata)
        print("Displaying Sky-View Factor")
    elif ridge:
        window = multiscale_ridge_enhancement(window, scales=ridge_scales, nodata=nodata)
        print("Displaying Ridge Enhancement")

    plt.figure(figsize=(10, 8))
    print("Display range: max =", window.max(), "min =", window.min())

    cmap = 'terrain'
    label = 'Elevation (meters)'
    if ridge:
        cmap = 'hot'
        label = 'Ridgeness'
    elif svf:
        cmap = 'gray'
        label = 'Sky-View Factor'

    window = np.ma.masked_equal(window.astype(float), nodata)

    return window, extent, cmap, label


def displayWindow(window, extent, cmap, label=None, contours=False, lines=None,
                  peaks=None, title=None, filepath=None, save_metadata=True, picture=True):
    cmap = cmap or 'terrain'
    label = label or 'Elevation (meters)'

    plt.imshow(window, extent=extent, origin='upper', cmap=cmap, aspect='auto')
    plt.colorbar(label=label)

    if filepath is not None and picture:
        plt.title(filepath)
        plt.savefig(filepath / 'window.png')

    rows, cols = window.shape
    dx_geo = (extent[1] - extent[0]) / cols
    dy_geo = (extent[3] - extent[2]) / rows

    if contours:
        plt.contour(window, levels=10, colors='k', linewidths=0.5, extent=extent)

    if peaks is not None and len(peaks) > 0:
        peaks_sorted = sorted(peaks, key=lambda p: p['score'], reverse=True)
        max_score = peaks_sorted[0]['score']

        for peak in peaks_sorted[:10]:
            col, row = peak['x'], peak['y']
            angle_deg = peak['angle']
            length_px = peak['length']
            spacing_px = peak['spacing']
            width_px = peak['width']

            kernel = create_parallel_ridge_kernel_compact(
                length_px, spacing_px, width_px, angle_deg
            )

            kh, kw = kernel.shape
            half_kh, half_kw = kh // 2, kw // 2

            x_center = extent[0] + (col + 0.5) * dx_geo
            y_center = extent[2] + (rows - row - 0.5) * dy_geo

            kern_extent = (
                x_center - half_kw * dx_geo,
                x_center + half_kw * dx_geo,
                y_center - half_kh * dy_geo,
                y_center + half_kh * dy_geo
            )

            alpha = 0.2 + 0.1 * (peak['score'] / max_score)
            plt.imshow(kernel,
                       extent=kern_extent,
                       alpha=alpha,
                       vmin=0,
                       vmax=kernel.max(),
                       origin='upper',
                       interpolation='nearest',
                       cmap='Reds')

    geo_lines = []
    if lines is not None:
        for pair in lines:
            geo_lines.append(dict())
            for i, (line_segment, color) in enumerate([(pair['line1'], 'red'), (pair['line2'], 'blue')]):
                (xs, ys), (xe, ye) = line_segment

                xs_geo = extent[0] + (xs + 0.5) * dx_geo
                ys_geo = extent[2] + (rows - ys - 0.5) * dy_geo
                xe_geo = extent[0] + (xe + 0.5) * dx_geo
                ye_geo = extent[2] + (rows - ye - 0.5) * dy_geo

                geo_lines[-1][f'line_{i}_angle'] = np.arctan((ye_geo - ys_geo) / (xe_geo - xs_geo)).item()
                geo_lines[-1][f'line_{i}_length'] = np.sqrt((ye_geo - ys_geo)**2 + (xe_geo - xs_geo)**2).item()
                geo_lines[-1][f'line_{i}_xs'] = xs_geo.item()
                geo_lines[-1][f'line_{i}_ys'] = ys_geo.item()
                geo_lines[-1][f'line_{i}_xe'] = xe_geo.item()
                geo_lines[-1][f'line_{i}_ye'] = ye_geo.item()
                plt.plot([xs_geo, xe_geo], [ys_geo, ye_geo],
                         color=color, linewidth=2, alpha=0.8)

    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    if title:
        plt.title(title)
    if filepath and picture:
        plt.savefig(filepath / "line_overlay.png")
    elif picture:
        plt.show()
    if save_metadata and filepath:
        with open(filepath / "line_overlay.json", 'w') as f:
            json.dump(geo_lines, f)


# ----------------------------------------------------------------------
# Raster processing helpers
# ----------------------------------------------------------------------

def gaussian_smooth_windowed(data, nodata=params.NODATA, radius=params.MEDIAN_SMOOTH_RADIUS, sigma=1.0):
    """Smooth a raster while ignoring nodata values."""
    mask = data == nodata
    valid = np.where(mask, 0, data)
    filtered = gaussian_filter(valid, sigma=sigma, radius=radius)
    weights = gaussian_filter((~mask).astype(float), sigma=sigma, radius=radius)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(weights > 0, filtered / weights, nodata)
    result[mask] = nodata
    return result


def median_smooth_windowed(data, nodata=params.NODATA, radius=params.MEDIAN_SMOOTH_RADIUS):
    """Median filter a raster while ignoring nodata values."""
    mask = data == nodata
    arr = data.astype(float)
    if mask.any():
        # Fill nodata with nearest valid neighbor so the C median_filter
        # doesn't see sentinel values bleeding into valid windows.
        _, indices = distance_transform_edt(mask, return_indices=True)
        arr = arr[tuple(indices)]
    result = median_filter(arr, size=2 * radius + 1, mode='nearest')
    result[mask] = nodata
    return result


def local_relief_model(data, radius=params.LRM_RADIUS, nodata=params.NODATA):
    """
    Local Relief Model: subtract a large-scale mean surface.
    Uses a square box filter of radius `radius` cells, masking nodata.
    """
    mask = data == nodata
    valid = np.where(mask, 0, data)
    size = 2 * radius + 1
    kernel = np.ones((size, size)) / (size * size)
    local_mean = convolve2d(valid, kernel, mode='same', boundary='symm')
    weight = convolve2d((~mask).astype(float), kernel, mode='same', boundary='symm')
    with np.errstate(divide='ignore', invalid='ignore'):
        local_mean = np.where(weight > 0, local_mean / weight, 0)
    lrm = np.where(~mask, data - local_mean, 0)
    lrm[mask] = nodata
    return lrm


def sky_view_factor_xr(window, radius=params.SVF_RADIUS, directions=params.SVF_DIRECTIONS, nodata=params.NODATA):
    da = xr.DataArray(window, dims=["y", "x"])
    da = da.where(da != nodata)
    svf = sky_view_factor(agg=da, max_radius=radius, n_directions=directions)
    return svf.values


def multiscale_ridge_enhancement(data, scales=None, nodata=params.NODATA):
    """
    Multi-scale ridge enhancement using the Frangi vesselness measure.
    Returns the maximum ridge response across the given scales.
    Nodata pixels are filled with the median before processing and restored after.
    """
    if scales is None:
        scales = params.RIDGE_SCALES
    mask = data == nodata
    fill_val = np.nanmedian(np.where(mask, np.nan, data))
    working = np.where(mask, fill_val, data.astype(np.float64))

    ridge_all = np.zeros_like(working)
    beta = 0.5

    for s in scales:
        # Compute second derivatives of the original working image at scale s.
        # Avoid redundant smoothing by directly requesting derivative filters.
        Dxx = gaussian_filter(working, sigma=s, order=(0, 2))
        Dyy = gaussian_filter(working, sigma=s, order=(2, 0))
        Dxy = gaussian_filter(working, sigma=s, order=(1, 1))

        # Analytic eigenvalues of the 2x2 symmetric Hessian (vectorized)
        disc = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)
        λ1 = (Dxx + Dyy - disc) / 2   # smaller eigenvalue
        λ2 = (Dxx + Dyy + disc) / 2   # larger eigenvalue

        S = np.sqrt(λ1 ** 2 + λ2 ** 2)
        s_pos = S[S > 0]
        c_val = 0.5 * np.percentile(s_pos, 95) if s_pos.size > 0 else 1.0

        with np.errstate(divide='ignore', invalid='ignore'):
            Rb = np.where(λ1 != 0, λ2 / λ1, 0.0)

        ridge_scale = np.where(
            λ1 < 0,
            np.exp(-Rb ** 2 / (2 * beta ** 2)) *
            (1 - np.exp(-S ** 2 / (2 * c_val ** 2))),
            0.0
        )
        ridge_all = np.maximum(ridge_all, ridge_scale)

    ridge_all[mask] = nodata
    return ridge_all
def main():
    filepath = 'data/loaded_tiles/Hopewell_Road/merged_dtm.tif'
    gt, band, nodata, ds = getGTAndBand(filepath)
    window, extent = get_window(40.0, -82.4, 200, 200, gt, band, nodata)
    processed_window, extent, cmap, label = filter_image(
        window,
        extent,
        nodata=nodata,
        ridge=True,
        ridge_scales=[5, 6, 7],
        median_smoothing=True,
        title="Circleville Road - Local Relief Model"
    )
    displayWindow(processed_window, extent, cmap=cmap, label=label)
if __name__ == "__main__":
    main()