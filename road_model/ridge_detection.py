from os import makedirs
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.morphology import remove_small_objects, medial_axis, disk, closing
from skimage.measure import label as label_function, regionprops
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from utils import convert_meters_degrees_pixels, METERS, PIXELS, DEGREES
from road_model.process_dtm import displayWindow, getGTAndBand, get_window, median_smooth_windowed, local_relief_model, multiscale_ridge_enhancement
import parameters

# Per-worker-process cache so each worker opens the DTM file only once.
_dtm_cache: dict = {}

def _get_dtm(filepath: str):
    if filepath not in _dtm_cache:
        _dtm_cache[filepath] = getGTAndBand(filepath)
    return _dtm_cache[filepath]

def hessian_ridge_orientation(image, sigma=1.0):
    """Return ridge orientation (radians, [0, π)) from Hessian."""
    img = image.astype(np.float64)
    Ixx = gaussian_filter(img, sigma=sigma, order=(0, 2))
    Iyy = gaussian_filter(img, sigma=sigma, order=(2, 0))
    Ixy = gaussian_filter(img, sigma=sigma, order=(1, 1))

    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)   # range [-π/2, π/2]
    orient = theta + np.pi / 2.0
    orient = np.mod(orient, np.pi)
    return orient.astype(np.float32)


def _oriented_line_se(length: int, angle_deg: float) -> np.ndarray:
    """
    Build a binary line structuring element of the given pixel length,
    oriented at angle_deg (degrees CCW from +x, image convention: y down).
    Used for directional morphological closing to bridge wall gaps without
    merging cross-cutting landscape features.
    """
    size = length | 1          # ensure odd
    half = size // 2
    se = np.zeros((size, size), dtype=bool)
    rad = np.deg2rad(angle_deg)
    for t in np.linspace(-half, half, size * 4):
        col = int(round(half + t * np.cos(rad)))
        row = int(round(half - t * np.sin(rad)))  # y-flip: image y increases downward
        if 0 <= row < size and 0 <= col < size:
            se[row, col] = True
    return se


def _perp_spacing_at_overlap(line1, dir1, line2, dir2):
    """
    Perpendicular distance between two line segments measured at the midpoint
    of their overlapping region along the shared axis.

    This avoids the error introduced when full-segment midpoints sit at
    different along-wall positions (e.g., one line extends further than the
    other), which would cause midpoint-to-midpoint distance to differ from
    the true wall separation.
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


def detect_parallel_walls_ridge_pairing(
    ridge_map,
    target_angle,               # degrees CCW from +x, 0 = horizontal
    extent,                     # [xmin, xmax, ymin, ymax] in meters
    spacing_meters,          # expected wall‑to‑wall distance
    angle_tolerance,       # ± degrees
    spacing_tolerance,     # relative, e.g. 0.15 = ±15%
    ridge_threshold,
    min_ridge_length_meters,
    overlap_ratio,
    orientation_source=None,    # DTM to use for orientation; defaults to ridge_map
    return_debug=False,
    gap_fill=True,              # False = skip along-wall integration (faster for scans)
):
    """
    Robust parallel‑wall detection from a ridge probability map.

    Steps:
        1. Compute ridge orientation via gradient and keep only pixels
           whose direction matches the target angle.
        2. Threshold & skeletonise the binary mask using medial axis.
        3. Fit a straight line to each connected skeleton component
           using cv2.fitLine (total least squares).
        4. Pair lines that are parallel (±2°), spaced correctly, and
           overlap sufficiently.

    Returns
    -------
    pairs : list of dict
        Each dict has keys: 'line1', 'line2', 'spacing', 'angle_diff', 'score'.
    """

    spacing_px = convert_meters_degrees_pixels(METERS, PIXELS, spacing_meters, 0, ridge_map, extent)[0]
    min_ridge_length_px = convert_meters_degrees_pixels(METERS, PIXELS, min_ridge_length_meters, 0, ridge_map, extent)[0]

    # ---------- 1. Orientation weighting ----------
    orient_source = orientation_source if orientation_source is not None else ridge_map
    orient_rad = hessian_ridge_orientation(orient_source, sigma=parameters.HESSIAN_SIGMA)
    target_rad = np.deg2rad(target_angle)
    angle_diff = np.abs(orient_rad - target_rad)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)  # modulo π

    orientation_mask = (angle_diff < np.deg2rad(angle_tolerance)) & np.isfinite(orient_rad)

    # ---------- 2. Threshold & create binary mask ----------
    binary = (ridge_map > ridge_threshold) & orientation_mask
    binary = remove_small_objects(binary, max_size=parameters.BINARY_MIN_OBJECT_SIZE - 1, connectivity=2)

    if not binary.any():
        if return_debug:
            return [], (binary, binary, np.zeros_like(binary, dtype=bool),
                        np.zeros_like(binary, dtype=int), orientation_mask)
        return []

    # ---------- 3. Along-wall gap filling ----------
    if gap_fill:
        from scipy.ndimage import uniform_filter1d
        from scipy.ndimage import rotate as _nd_rotate

        along_wall_image_angle = (target_angle + 90) % 180
        window_px = max(7, int(min_ridge_length_px * parameters.WALL_GAP_WINDOW_RATIO) * 2 + 1)

        b_rot  = _nd_rotate(binary.astype(np.float32), along_wall_image_angle,
                            reshape=False, order=1, cval=0.0)
        b_int  = uniform_filter1d(b_rot, size=window_px, axis=1, mode='constant', cval=0.0)
        b_back = _nd_rotate(b_int, -along_wall_image_angle,
                            reshape=False, order=1, cval=0.0)
        binary_closed = b_back > parameters.WALL_FILL_THRESHOLD
        if not binary_closed.any():
            binary_closed = binary
    else:
        binary_closed = binary

    skeleton = medial_axis(binary_closed, return_distance=False)

    # ---------- 4. Fit a straight line to each connected component ----------
    labeled = label_function(skeleton)
    lines = []   # each element: ((x1,y1), (x2,y2))

    for region in regionprops(labeled):
        coords = region.coords          # (row, col) = (y, x)
        if len(coords) < min_ridge_length_px:
            continue

        # cv2.fitLine expects (x,y) float array of shape (N,2)
        pts = np.column_stack([coords[:, 1], coords[:, 0]]).astype(np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        v = np.array([vx[0], vy[0]])
        p0 = np.array([x0[0], y0[0]])

        # Project points onto line direction to get endpoints
        t = np.dot(pts - p0, v)
        t_min, t_max = t.min(), t.max()
        pt1 = (p0[0] + t_min * v[0], p0[1] + t_min * v[1])
        pt2 = (p0[0] + t_max * v[0], p0[1] + t_max * v[1])
        lines.append((pt1, pt2))

    # ---------- 5. Find parallel pairs ----------
    # Pre-compute per-line geometry once; filter short lines before the O(n²) loop.
    line_props = []
    for l in lines:
        dx, dy = l[1][0] - l[0][0], l[1][1] - l[0][1]
        length = np.hypot(dx, dy)
        if length < min_ridge_length_px:
            continue
        direction = np.array([dx, dy]) / length
        midpoint = np.array(l[0]) + 0.5 * np.array([dx, dy])
        perp = np.array([-direction[1], direction[0]])
        line_props.append((l, length, direction, midpoint, perp))

    # Cache ridge-mean samples per line index to avoid repeated map_coordinates
    # when the same line appears in multiple candidate pairs.
    _ridge_mean_cache: dict = {}

    def _line_mean_ridge(idx, line, n=50):
        if idx not in _ridge_mean_cache:
            xs = np.linspace(line[0][0], line[1][0], n)
            ys = np.linspace(line[0][1], line[1][1], n)
            ys = np.clip(ys, 0, ridge_map.shape[0] - 1)
            xs = np.clip(xs, 0, ridge_map.shape[1] - 1)
            vals = map_coordinates(ridge_map, [ys, xs], order=1, mode='nearest')
            _ridge_mean_cache[idx] = np.mean(vals) if len(vals) else 0.0
        return _ridge_mean_cache[idx]

    pairs = []
    for i, (l1, len1, dir1, mid1, perp1) in enumerate(line_props):
        for j, (l2, len2, dir2, mid2, _) in enumerate(line_props[i + 1:], i + 1):
            # Angle difference (mod π)
            dot = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
            angle_diff_rad = np.arccos(np.abs(dot))
            if angle_diff_rad > np.deg2rad(parameters.PAIRING_ANGLE_TOLERANCE_DEG):
                continue

            # Perpendicular distance at the midpoint of the overlapping region
            dist = _perp_spacing_at_overlap(l1, dir1, l2, dir2)
            if abs(dist - spacing_px) / spacing_px > spacing_tolerance:
                continue

            # Overlap along average direction
            # Align dir2 to the same half-plane as dir1 before averaging;
            # cv2.fitLine can return either direction along the same line.
            d2 = dir2 if np.dot(dir1, dir2) >= 0 else -dir2
            avg_dir = (dir1 + d2) / np.linalg.norm(dir1 + d2)
            t1_vals = [np.dot(np.array(p) - mid1, avg_dir) for p in l1]
            t2_vals = [np.dot(np.array(p) - mid1, avg_dir) for p in l2]
            overlap = max(0.0, min(max(t1_vals), max(t2_vals)) - max(min(t1_vals), min(t2_vals)))
            if overlap < overlap_ratio * min(len1, len2):
                continue

            score = min(len1, len2) * 0.5 * (_line_mean_ridge(i, l1) + _line_mean_ridge(j, l2))

            pairs.append({
                'line1': l1,
                'line2': l2,
                'spacing': dist,
                'angle_diff': np.rad2deg(angle_diff_rad),
                'score': score,
            })

    pairs.sort(key=lambda x: x['score'], reverse=True)

    if return_debug:
        return pairs, (binary, binary_closed, skeleton, labeled, orientation_mask)
    return pairs


def remove_invalid_pairs(
    pairs,
    angle_tolerance,          # maximum allowed angle difference between two lines (degrees)
    target_angle         # desired absolute line orientation (degrees, 0‑180);
):
    """
    Filter a list of line pairs based on geometric criteria.

    Parameters
    ----------
    pairs : list of dict
        Each dict must have keys 'line1' and 'line2', each a tuple ((x1,y1), (x2,y2)).
    angle_tolerance : float
        Maximum allowed angle difference between the two lines (degrees).
    spacing_tolerance : float
        Relative tolerance for spacing (e.g., 0.15 = 15%).
    min_ridge_length_px : float
        Minimum acceptable length for each line (pixels).
    expected_spacing : float, optional
        If provided, the spacing between lines must satisfy
        |spacing - expected_spacing| / expected_spacing <= spacing_tolerance.
    target_angle : float, optional
        If provided, each line's orientation must be within `angle_tolerance`
        of this angle (modulo 180°).

    Returns
    -------
    filtered_pairs : list
        Pairs that pass all checks.
    """
    filtered = []

    for pair in pairs:
        line1 = pair['line1']
        line2 = pair['line2']

        # --- 1. Compute lengths ---
        dx1 = line1[1][0] - line1[0][0]
        dy1 = line1[1][1] - line1[0][1]
        len1 = np.hypot(dx1, dy1)

        dx2 = line2[1][0] - line2[0][0]
        dy2 = line2[1][1] - line2[0][1]
        len2 = np.hypot(dx2, dy2)

        # --- 2. Compute direction vectors (undirected) ---
        dir1 = np.array([dx1, dy1]) / len1
        dir2 = np.array([dx2, dy2]) / len2

        # --- 3. Angle difference between the two lines (0–90°) ---
        dot = np.abs(np.dot(dir1, dir2))
        # Clip to avoid numerical issues with arccos
        dot = np.clip(dot, -1.0, 1.0)
        # --- 4. Optional: enforce absolute orientation (e.g., all lines must be ~125°) ---
        # Angle of each line in [0, 180)
        angle1 = np.rad2deg(np.arctan2(dir1[1], dir1[0])) % 180
        angle2 = np.rad2deg(np.arctan2(dir2[1], dir2[0])) % 180

        def angle_difference(a, b):
            diff = abs(a - b) % 180
            return min(diff, 180 - diff)

        along_wall_angle = (target_angle + 90) % 180  # TARGET_ANGLE is cross-ridge; fitLine is along-ridge
        if angle_difference(angle1, along_wall_angle) > angle_tolerance:
            continue
        if angle_difference(angle2, along_wall_angle) > angle_tolerance:
            continue

        # --- 5. Compute spacing (perpendicular distance between lines) ---
        spacing = _perp_spacing_at_overlap(line1, dir1, line2, dir2)

        # --- 7. Keep the pair, optionally update the computed fields ---
        pair['spacing'] = spacing
        filtered.append(pair)

    return filtered    

def search_region(filepath: str, coordinates: tuple[float, float], size_x_meters: float, size_y_meters: float, output_dir: Path, verbose: int = 0, picture: bool = False, target_angle: float = None):
    if target_angle is None:
        target_angle = parameters.TARGET_ANGLE
    output_path = Path(output_dir) / f"{round(coordinates[0], 4)}_{round(coordinates[1], 4)}"
    print(f"Writing to {output_path} …")
    if (output_path / "line_overlay.json").exists():
        return
    makedirs(output_path, exist_ok=True)
    gt, band, nodata, ds = _get_dtm(filepath)

    window, extent = get_window(coordinates[0], coordinates[1], size_x_meters, size_y_meters, gt, band, nodata)
    if window is None:
        return
    window_original = window.copy()
    window_original = np.where(window_original == nodata, np.nan, window_original)
    angle_tolerance = parameters.ANGLE_TOLERANCE
    if verbose > 0: print(f"Filtering DTM with median smoothing")
    smoothed = median_smooth_windowed(window, nodata=nodata)
    lrm = local_relief_model(smoothed, nodata=nodata)
    ridge_map = multiscale_ridge_enhancement(lrm, scales=parameters.SCAN_RIDGE_SCALES, nodata=nodata)
    nd_mask = window == nodata
    ridge_map = np.where(nd_mask, 0.0, ridge_map.astype(float))
    if verbose > 0: print("Detecting parallel walls …")
    pairs = detect_parallel_walls_ridge_pairing(
        ridge_map=ridge_map,
        target_angle=target_angle,
        extent=extent,
        spacing_meters=parameters.SPACING_METERS,
        angle_tolerance=angle_tolerance,
        spacing_tolerance=parameters.SPACING_TOLERANCE,
        ridge_threshold=parameters.RIDGE_THRESHOLD,
        min_ridge_length_meters=parameters.MIN_RIDGE_LENGTH_METERS,
        overlap_ratio=parameters.OVERLAP_RATIO,
        return_debug=False,
        gap_fill=False,
    )
    if verbose > 0: print(f"Found {len(pairs)} parallel wall pairs.")

    pairs = remove_invalid_pairs(pairs, angle_tolerance=angle_tolerance, target_angle=target_angle)

    if len(pairs) > 0: 
        displayWindow(
            window_original, 
            extent, 
            cmap= 'terrain',
            title=f"Hopewell Road DTM",
            lines = pairs,
            filepath = output_path,
            picture = picture
        )
    else:
        displayWindow(
            window_original, 
            extent, 
            cmap= 'terrain',
            title=f"Hopewell Road DTM",
            filepath = output_path,
            picture = picture
        )
    print(f'Searched {size_x_meters}m x {size_y_meters}m region at {coordinates[0]}, {coordinates[1]}')