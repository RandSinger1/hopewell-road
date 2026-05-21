"""
plot_dtm_zoom.py

Zoom into a WGS84 DTM GeoTIFF around a given coordinate.
Usage:
    python plot_dtm_zoom.py final_WGS84_dtm.tif --center 40.0,-82.4 --size 1000
    python plot_dtm_zoom.py final_WGS84_dtm.tif --center 40.0,-82.4 --size 1000,2000
"""

import json
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.ndimage import gaussian_filter, generic_filter
from scipy.signal import convolve2d
from xrspatial import sky_view_factor
from utils import meters_to_degrees, create_parallel_ridge_kernel_compact
import xarray as xr
# Return the window of the GeoTIFF that contains the given center in lat/lon and size in meters
def get_window(lat, lon, size_x_meters, size_y_meters, gt, band, nodata):
    delta_x, delta_y = meters_to_degrees(size_x_meters, size_y_meters, lat)
    min_lon = lon - delta_x
    max_lon = lon + delta_x
    min_lat = lat - delta_y
    max_lat = lat + delta_y
    # get the window in pixel coordinates
    col_start = int((min_lon - gt[0]) / gt[1])
    col_end = int((max_lon - gt[0]) / gt[1]) + 1
    row_start = int((gt[3] - max_lat) / abs(gt[5]))
    row_end = int((gt[3] - min_lat) / abs(gt[5])) + 1
    # Compute the exact geographic extent of the extracted window
    # Top-left of the window in geographic coordinates
    xmin_win = gt[0] + col_start * gt[1]
    ymax_win = gt[3] + row_start * gt[5]    # gt[5] is negative
    xmax_win = gt[0] + col_end * gt[1]
    ymin_win = gt[3] + row_end * gt[5]
    extent = [xmin_win, xmax_win, ymin_win, ymax_win]
    # read the window
    window = band.ReadAsArray(col_start, row_start, col_end - col_start, row_end - row_start)
    return window, extent


def gaussian_kernel_ignore_nodata(radius, sigma):
    """Return a 2D Gaussian kernel and its flattened weights."""
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel

def median_kernel_ignore_nodata(radius):
    """Return a 2D kernel for median filtering (weights are not used)."""
    size = 2 * radius + 1
    kernel = np.ones((size, size))
    return kernel

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

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, sobel, maximum_filter
from scipy.ndimage import minimum_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def filter_image(window, extent, nodata=-9999, contours=False, gaussian_smoothing=False,
                  median_smoothing=False, lrm=False, lrm_radius=10,
                  svf=False, svf_radius=10, svf_directions=16,
                  ridge=False, ridge_scales=[1, 2, 3],
                  smoothing_radius=10, smoothing_sigma=1.0, title=None):
    """
    Display a LiDAR window with optional archaeological pre‑processing.

    Parameters
    ----------
    window : 2D numpy array
        Elevation raster (rows = y, cols = x). North is the first row.
    extent : list [xmin, xmax, ymin, ymax]
        Geographic extent of the raster (lower‑left corner of first pixel?).
        Used for axis labels.
    nodata : float
        Value to treat as no data. Masked during computations.
    contours : bool
        Overlay contour lines.
    gaussian_smoothing : bool
        Apply Gaussian smoothing to the **final** displayed raster.
    median_smoothing : bool
        Apply median smoothing to the **final** displayed raster.
    lrm : bool
        Compute and display Local Relief Model.
    lrm_radius : int
        Radius (in cells) of the mean filter for LRM.
    svf : bool
        Compute and display Sky‑View Factor.
    svf_radius : int
        Search radius (in cells) for horizon calculation.
    svf_directions : int
        Number of azimuth directions (e.g. 16).
    ridge : bool
        Compute and display multi‑scale ridge enhancement (Frangi vesselness).
    ridge_scales : list of float
        Sigma values (in cells) for the ridge filter.
    smoothing_radius : int
        Radius for initial smoothing if smoothing=True (used if no enhancement).
    smoothing_sigma : float
        Sigma for Gaussian kernel if smoothing=True.
    """
    # ----- 1. Optional smoothing on the (possibly enhanced) window -----
    if gaussian_smoothing:
        window = gaussian_smooth_windowed(window, nodata=nodata,
                                          radius=smoothing_radius,
                                          sigma=smoothing_sigma)
        #print("After Gaussian smoothing: max =", window.max(), "min =", window.min())
    elif median_smoothing:
        window = median_smooth_windowed(window, nodata=nodata, radius=smoothing_radius)
        #print("After median smoothing: max =", window.max(), "min =", window.min())
    # ----- 2. Apply enhancement (only one expected, first wins if multiple) -----
    enhanced = False
    if lrm:
        window = local_relief_model(window, radius=lrm_radius, nodata=nodata)
        enhanced = True
        print("Displaying Local Relief Model")
    elif svf:
        window = sky_view_factor_xr(window, radius=svf_radius, directions=svf_directions, nodata=nodata)
        enhanced = True
        print("Displaying Sky-View Factor")
    elif ridge:
        window = multiscale_ridge_enhancement(window, scales=ridge_scales, nodata=nodata)
        enhanced = True
        print("Displaying Ridge Enhancement")
    # If none selected, window stays raw DEM

    

    # ----- 3. Plot -----
    plt.figure(figsize=(10, 8))
    print("Display range: max =", window.max(), "min =", window.min())

    # Choose colourmap: ridge map is a probability, best in greyscale/hot
    cmap = 'terrain'
    label = 'Elevation (meters)'
    if ridge:
        cmap = 'hot'      # or 'gray'
        label = 'Ridgeness'
    elif svf:
        cmap = 'gray'
        label = 'Sky-View Factor'

    window = np.ma.masked_equal(window.astype(float), nodata)

    return window, extent, cmap, label

def displayWindow(window, extent, cmap, label = None, contours=False, lines = None,
                peaks=None, title=None, filepath=None, save_metadata = True):
    # Determine colormap: if window may be ridge probability, use 'hot', else 'terrain'
    cmap = cmap or 'terrain'
    label = label or 'Elevation (meters)'

    plt.imshow(window, extent=extent, origin='upper', cmap=cmap, aspect='auto')
    plt.colorbar(label=label)
    
    if filepath is not None:
        plt.title(filepath)
        plt.savefig(filepath/'window.png')

    # Image dimensions
    rows, cols = window.shape
    # Geographic pixel sizes
    dx_geo = (extent[1] - extent[0]) / cols
    dy_geo = (extent[3] - extent[2]) / rows
    if contours:
        plt.contour(window, levels=10, colors='k', linewidths=0.5, extent=extent)

    # ---------- Kernel overlay ----------
    if peaks is not None and len(peaks) > 0:
        peaks_sorted = sorted(peaks, key=lambda p: p['score'], reverse=True)
        max_score = peaks_sorted[0]['score']


        for peak in peaks_sorted[:10]:
            col, row = peak['x'], peak['y']             # pixel coordinates
            angle_deg = peak['angle']
            length_px = peak['length']                  # in pixels
            spacing_px = peak['spacing']
            width_px = peak['width']

            # Create the compact kernel (pixel space)
            kernel = create_parallel_ridge_kernel_compact(
                length_px, spacing_px, width_px, angle_deg
            )

            kh, kw = kernel.shape                       # kernel size in pixels
            half_kh, half_kw = kh // 2, kw // 2

            # Geographic centre of the peak pixel
            x_center = extent[0] + (col + 0.5) * dx_geo
            y_center = extent[2] + (rows - row - 0.5) * dy_geo   # origin='upper' correction

            # Kernel extent in geographic units
            kern_extent = (
                x_center - half_kw * dx_geo,   # left
                x_center + half_kw * dx_geo,   # right
                y_center - half_kh * dy_geo,   # bottom
                y_center + half_kh * dy_geo    # top
            )

            # Transparency proportional to score
            alpha = 0.2 + 0.1 * (peak['score'] / max_score)

            # Plot the kernel overlat on top of the image.  ensure it is still visible by plotting it
            # with alpha.
            plt.imshow(kernel,
                       extent=kern_extent,
                       alpha=alpha,
                       vmin=0,
                       vmax=kernel.max(),
                       origin='upper',
                       interpolation='nearest',
                       cmap='Reds')
    geo_lines = []
    # Plot lines if provided
    if lines is not None:
        for pair in lines:          # each pair is {'line1': ..., 'line2': ...}
            geo_lines.append(dict())                                              
            for i, (line_segment, color) in enumerate([(pair['line1'], 'red'), (pair['line2'], 'blue')]):
                # line_segment = ((x_start, y_start), (x_end, y_end)) in pixel coords
                (xs, ys), (xe, ye) = line_segment

                # Convert from pixel to geographic coordinates
                # (origin='upper': y increases downward, extent's ymin is bottom, ymax is top)
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
    if filepath:
        plt.savefig(filepath/"line_overlay.png")
    else:
        plt.show()
    if save_metadata and filepath:
        with open(filepath/"line_overlay.json", 'w') as f:
            json.dump(geo_lines, f)



# ----------------------------------------------------------------------
# Helper methods (used by displayWindow)
# ----------------------------------------------------------------------

def gaussian_smooth_windowed(data, nodata=-9999, radius=10, sigma=1.0):
    """ Smooth a raster while ignoring nodata values. """
    mask = data == nodata
    valid = np.where(mask, 0, data)
    # Gaussian filter on valid data, but count weights
    filtered = gaussian_filter(valid, sigma=sigma, radius=radius)
    weights = gaussian_filter((~mask).astype(float), sigma=sigma, radius=radius)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(weights > 0, filtered / weights, nodata)
    result[mask] = nodata
    return result

def median_smooth_windowed(data, nodata=-9999, radius=10):
    """ Median filter a raster while ignoring nodata values. """
    mask = data == nodata
    valid = np.where(mask, 0, data)
    # Use generic_filter with a custom function to compute median ignoring nodata
    def median_ignore_nodata(values):
        valid_values = values[values != nodata]
        return np.median(valid_values) if len(valid_values) > 0 else nodata
    result = generic_filter(valid, median_ignore_nodata, size=2*radius+1, mode='constant', cval=nodata)
    result[mask] = nodata
    return result


def local_relief_model(data, radius=10, nodata=-9999):
    """
    Local Relief Model: subtract a large‑scale mean surface.
    Uses a uniform (box) filter of radius `radius` cells, masking nodata.
    """
    mask = data == nodata
    valid = np.where(mask, 0, data)
    # Circular kernel is approximated by a square for simplicity;
    # for exact circular you need a custom kernel. Square suffices.
    size = 2 * radius + 1
    kernel = np.ones((size, size)) / (size * size)
    # Convolve with nodata masking: scale the sum by the fraction of valid pixels
    local_mean = convolve2d(valid, kernel, mode='same', boundary='symm')
    weight = convolve2d((~mask).astype(float), kernel, mode='same', boundary='symm')
    with np.errstate(divide='ignore', invalid='ignore'):
        local_mean = np.where(weight > 0, local_mean / weight, 0)
    lrm = np.where(~mask, data - local_mean, 0)
    lrm[mask] = nodata
    return lrm

def sky_view_factor_xr(window, radius=10, directions=16, nodata=-9999):
    # Convert to xarray DataArray
    # window: 2D numpy array with nodata values
    # radius: search radius in cells for horizon calculation
    # directions: number of azimuth directions to sample (e.g. 16 for every 22.5°)
    da = xr.DataArray(window, dims=["y", "x"])
    da = da.where(da != nodata) # Sets nodata to NaN
    
    svf = sky_view_factor(agg=da, max_radius=radius, n_directions=directions)
    return svf.values


def multiscale_ridge_enhancement(data, scales=[1, 2, 3], nodata=-9999):
    """
    Multi‑scale ridge enhancement using the Frangi vesselness measure.
    Returns the maximum ridge response across the given scales.
    """
    mask = data == nodata
    # Fill nodata with median to avoid edge effects; will be masked later.
    valid = np.where(mask, np.nan, data)
    fill_val = np.nanmedian(valid)
    working = np.where(mask, fill_val, data)

    from numpy.linalg import eigvalsh

    ridge_all = np.zeros_like(working, dtype=np.float64)
    for s in scales:
        # Smooth with Gaussian at scale s
        smoothed = gaussian_filter(working, sigma=s)
        # Compute Hessian
        Dxx = gaussian_filter(smoothed, sigma=s, order=(0, 2))
        Dyy = gaussian_filter(smoothed, sigma=s, order=(2, 0))
        Dxy = gaussian_filter(smoothed, sigma=s, order=(1, 1))

        rows, cols = smoothed.shape
        ridge_scale = np.zeros_like(smoothed)
        for r in range(rows):
            for c in range(cols):
                H = np.array([[Dxx[r, c], Dxy[r, c]],
                              [Dxy[r, c], Dyy[r, c]]])
                eigvals = eigvalsh(H)       # sorted ascending: λ1 ≤ λ2
                λ1, λ2 = eigvals[0], eigvals[1]
                # Frangi for bright ridges (positive second derivative across ridge → λ1 << 0)
                if λ1 < 0:
                    Rb = λ2 / λ1 if λ1 != 0 else 0
                    S = np.sqrt(λ1**2 + λ2**2)
                    # Parameters beta, c control sensitivity; here typical values
                    beta = 0.5
                    c_val = 0.5 * np.max(S) if np.max(S) > 0 else 1
                    ridge_scale[r, c] = (1 - np.exp(-(Rb**2) / (2 * beta**2))) * \
                                        (1 - np.exp(-(S**2) / (2 * c_val**2)))
                # else ridge_scale remains 0

        ridge_all = np.maximum(ridge_all, ridge_scale)

    ridge_all[mask] = nodata
    return ridge_all

# gt, band, nodata, ds = getGTAndBand("loaded_tiles/Hopewell_Road_1/merged_dtm.tif")

# window, extent = get_window(40.0,-82.4, 700, 700, gt, band, nodata)

# displayWindow(
#     window, 
#     extent, 
#     nodata=nodata, 
#     lrm=True,
#     median_smoothing=True,
# )
# displayWindow(
#     window, 
#     extent, 
#     nodata=nodata, 
#     lrm=True,
#     gaussian_smoothing=True,
# )



# ridge is bad i think
# displayWindow(
#     window, 
#     extent, 
#     nodata=nodata, 
#     ridge=True
# )

# ds = None  # close dataset