import math
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


def perp_spacing_at_overlap(line1, dir1, line2, dir2):
    """
    Perpendicular distance between two line segments measured at the midpoint
    of their overlapping region along the shared axis.

    This avoids the error introduced when full-segment midpoints sit at
    different along-wall positions (e.g., one line extends further than the
    other), which would cause midpoint-to-midpoint distance to differ from
    the true wall separation.

    line1, line2 : ((x0, y0), (x1, y1)) endpoint pairs, in any consistent
                   isotropic unit (pixels, metres, ...).
    dir1, dir2   : unit direction vectors for line1 / line2.
    """
    A = np.array(line1[0], dtype=float)
    B = np.array(line1[1], dtype=float)
    C = np.array(line2[0], dtype=float)
    D = np.array(line2[1], dtype=float)

    d2 = dir2 if np.dot(dir1, dir2) >= 0 else -dir2

    avg = dir1 + d2
    avg_len = np.linalg.norm(avg)
    avg_dir = avg / avg_len if avg_len > 1e-9 else dir1
    perp = np.array([-avg_dir[1], avg_dir[0]])

    tA = np.dot(A, avg_dir)
    tB = np.dot(B, avg_dir)
    tC = np.dot(C, avg_dir)
    tD = np.dot(D, avg_dir)

    t_lo = max(min(tA, tB), min(tC, tD))
    t_hi = min(max(tA, tB), max(tC, tD))

    if t_hi <= t_lo:
        mid1 = (A + B) / 2
        mid2 = (C + D) / 2
        return abs(np.dot(mid2 - mid1, perp))

    t_mid = (t_lo + t_hi) / 2

    dAB = tB - tA
    p1 = A + ((t_mid - tA) / dAB) * (B - A) if abs(dAB) > 1e-9 else (A + B) / 2

    dCD = tD - tC
    p2 = C + ((t_mid - tC) / dCD) * (D - C) if abs(dCD) > 1e-9 else (C + D) / 2

    return abs(np.dot(p2 - p1, perp))


def make_parabola(a, b, c, x_offset):
    """Build (curve_func, curve_deriv) for  y = a*(x-x0)^2 + b*(x-x0) + c."""
    def curve_func(x):
        dx = x - x_offset
        return a * dx**2 + b * dx + c
    def curve_deriv(x):
        dx = x - x_offset
        return 2 * a * dx + b
    return curve_func, curve_deriv


def make_line(x0, y0, x1, y1):
    """Build (curve_func, curve_deriv) for the straight line through (x0,y0)→(x1,y1)."""
    slope = (y1 - y0) / (x1 - x0)
    def curve_func(x):
        return slope * (x - x0) + y0
    def curve_deriv(_x):
        return slope
    return curve_func, curve_deriv


def corridor_angle_at(curve_deriv_fn, lon):
    """Expected wall angle at longitude (degrees, image-space y-down, CCW from +x)."""
    return (90.0 - math.degrees(math.atan(curve_deriv_fn(lon)))) % 180


def sample_corridor(curve_func, x_start, x_end, half_width_deg, n=200):
    """
    Return a closed GeoJSON LinearRing [[lon, lat], ...] for a corridor polygon.
    Left (north) edge runs x_start→x_end; right (south) edge runs x_end→x_start.
    """
    xs = np.linspace(x_start, x_end, n)
    ys = np.array([curve_func(x) for x in xs])
    left  = [[float(x), float(y + half_width_deg)] for x, y in zip(xs, ys)]
    right = [[float(x), float(y - half_width_deg)] for x, y in zip(xs[::-1], ys[::-1])]
    return left + right + [left[0]]


def parabola_from_angle_at_newark(angle_deg):
    """
    Derive corridor parabola coefficients from the wall angle at Newark.

    Constraints solved:
      1. Curve passes through Circleville  →  c = lat_circleville
      2. Curve passes through Newark       →  a*dx²  + b*dx = lat_n - lat_c
      3. Image-space angle at Newark = angle_deg  →  2a*dx + b = tan(90° - angle°)

    The angle convention is the same as corridor_angle_at():
      angle = (90 - degrees(atan(dlat/dlon))) % 180

    Returns (a, b, c, x_offset) ready for make_parabola().
    """
    import parameters as _p
    lat_c, lon_c = _p.CIRCLEVILLE_COORD
    lat_n, lon_n = _p.NEWARK_COORD
    x_offset = lon_c
    c = lat_c
    dx = lon_n - lon_c                          # positive: Newark is east of Circleville

    deriv_n = math.tan(math.radians(90.0 - angle_deg))   # dlat/dlon at Newark

    a = (deriv_n * dx - (lat_n - lat_c)) / dx ** 2
    b = deriv_n - 2.0 * a * dx
    return a, b, c, x_offset


def annotate_lines_corridor(lines, curve_func, curve_deriv_fn, lat_m=111_320.0):
    """
    Annotate line dicts in-place with 'expected_angle' (°) and 'corridor_dist_m' (m).
    Expected angle uses the same derivative formula as range_searcher.tile_curve.
    """
    for ln in lines:
        mid_lon = (ln["xs"] + ln["xe"]) / 2
        mid_lat = (ln["ys"] + ln["ye"]) / 2
        ln["expected_angle"] = corridor_angle_at(curve_deriv_fn, mid_lon)
        ln["corridor_dist_m"] = abs(mid_lat - curve_func(mid_lon)) * lat_m


def make_gdal_progress_callback(label: str = "Progress"):
    """Return a GDAL-compatible progress callback that prints percentage updates.

    Usage:
        gdal.Warp(..., callback=make_gdal_progress_callback("Merge"))
    """
    state = {"last_pct": -1}

    def callback(complete, message, user_data):
        pct = int(complete * 100)
        if pct != state["last_pct"]:
            print(f"{label}: {pct}% - {message}")
            state["last_pct"] = pct
        return 1   # 1 = continue, 0 = cancel

    return callback