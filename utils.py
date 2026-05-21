import numpy as np
FEET = 0
METERS = 1
DEGREES = 2
PIXELS = 3
def feet_to_degrees(feet_x, feet_y, ref_lat):
    """Convert feet to approximate degrees at given latitude."""
    delta_lat = feet_y / 364000.0
    delta_lon = feet_x / (364000.0 * np.cos(np.radians(ref_lat)))
    return delta_lon, delta_lat
def meters_to_degrees(meters_x, meters_y, ref_lat):
    """Convert meters to approximate degrees at given latitude."""
    delta_lat = meters_y / 111320.0
    delta_lon = meters_x / (111320.0 * np.cos(np.radians(ref_lat)))
    return delta_lon, delta_lat
def create_parallel_ridge_kernel_compact(length_px, spacing_px, width_px, angle_deg):
    """
    Create a minimal bounding‑box kernel containing two parallel Gaussian ridges.
    The kernel is normalised (zero mean, unit variance) for NCC.
    """
    half_size = int(np.ceil(np.sqrt((length_px / 2) ** 2 +
                                    (spacing_px / 2 + 3 * width_px) ** 2))) + 2
    size = 2 * half_size + 1
    y, x = np.mgrid[-half_size : half_size + 1, -half_size : half_size + 1]

    theta = np.deg2rad(angle_deg)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)

    profile = (np.exp(-((yr - spacing_px / 2) ** 2) / (2 * width_px ** 2)) +
               np.exp(-((yr + spacing_px / 2) ** 2) / (2 * width_px ** 2)))
    length_mask = np.abs(xr) <= length_px / 2

    kernel = profile * length_mask

    # Trim to non‑zero bounding box
    rows = np.any(kernel, axis=1)
    cols = np.any(kernel, axis=0)
    if not rows.any():
        return np.zeros((1, 1), dtype=np.float32)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    kernel = kernel[rmin : rmax + 1, cmin : cmax + 1]

    # Normalise to zero mean, unit std (required for CCOEFF_NORMED)
    kernel = (kernel - kernel.mean()) / kernel.std()
    return kernel.astype(np.float32)

def convert_meters_degrees_pixels(src, out, x, y, window=None, extent=None):
    """Convert meters to degrees and pixels, and save to new GeoTIFF."""
    # src is either meters, DEGREES, or PIXELS
    # out is either meters, DEGREES, or PIXELS
    # x and y are the coordinates in the src unit
    # window and extent are needed for conversions involving PIXELS
    # window is the image array (for size), extent is (min_lon, max_lon, min_lat, max_lat)
    # METERS to DEGREES is magnitude
    # METERS to PIXELS is magnitude + scaling by extent and window size
    # DEGREES to METERS is magnitude
    # DEGREES to PIXELS is positional
    # PIXELS to DEGREES is positional
    # PIXELS to METERS is magnitude
    if src is out:
        return x, y  # No conversion needed
    if src is METERS:
        longitude, latitude = meters_to_degrees(x, y, ref_lat=40.0)
        if out is DEGREES:
            return longitude, latitude
        if out is PIXELS:
            if window is None or extent is None:
                raise ValueError("window, extent, and gt are required for meters to PIXELS conversion")
            # Convert longitude/latitude to pixel coordinates
            pixel_x = int((longitude) / (extent[1] - extent[0]) * window.shape[1])
            pixel_y = int((latitude) / (extent[3] - extent[2]) * window.shape[0])
            return pixel_x, pixel_y
    if src is PIXELS:
        if out is DEGREES:
            if window is None or extent is None:
                raise ValueError("window, extent, and gt are required for PIXELS to DEGREES conversion")
            longitude = extent[0] + (x / window.shape[1]) * (extent[1] - extent[0])
            latitude = extent[2] + (y / window.shape[0]) * (extent[3] - extent[2])
            return longitude, latitude
        if out is METERS:
            longitude, latitude = convert_meters_degrees_pixels(PIXELS, DEGREES, x, y, window, extent)
            meters_x = longitude * 364000.0 * np.cos(np.radians(latitude))
            meters_y = latitude * 364000.0
            return meters_x, meters_y
    raise ValueError(f"Unsupported conversion from {src} to {out}")