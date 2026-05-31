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
from process_dtm import displayWindow, getGTAndBand, get_window, median_smooth_windowed, local_relief_model, multiscale_ridge_enhancement
import params

def hessian_ridge_orientation(image, sigma=1.0):
    """Return ridge orientation (radians, [0, π)) from Hessian."""
    img_smooth = gaussian_filter(image.astype(np.float64), sigma=sigma)
    Iy, Ix = np.gradient(img_smooth)
    Iyy, Iyx = np.gradient(Iy)
    Ixy, Ixx = np.gradient(Ix)   # Ixy == Iyx

    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)   # range [-π/2, π/2]
    orient = theta + np.pi / 2.0
    orient = np.mod(orient, np.pi)
    return orient.astype(np.float32)


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

    # ---------- 1. Orientation mask ----------
    orient_source = orientation_source if orientation_source is not None else ridge_map
    orient_rad = hessian_ridge_orientation(orient_source, sigma=params.HESSIAN_SIGMA)
    target_rad = np.deg2rad(target_angle)
    angle_diff = np.abs(orient_rad - target_rad)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)  # modulo π
    orientation_mask = (angle_diff < np.deg2rad(angle_tolerance)) & np.isfinite(orient_rad)

    # (Optional: print diagnostics – uncomment if needed)
    # valid = np.isfinite(orient_rad)
    # print(f"Min angle: {np.rad2deg(orient_rad[valid]).min():.1f}°, Max: {np.rad2deg(orient_rad[valid]).max():.1f}°")
    # ...

    # ---------- 2. Threshold & create binary mask ----------
    binary = (ridge_map > ridge_threshold) & orientation_mask
    # Use `max_size` to match the newer skimage API (avoid FutureWarning).
    # The previous `min_size` behaviour removed objects smaller than `min_size`.
    # The new `max_size` removes objects smaller than or equal to the value,
    # so subtract 1 to preserve the original semantics.
    binary = remove_small_objects(binary, max_size=params.BINARY_MIN_OBJECT_SIZE - 1, connectivity=2)

    # ---------- 3. Robust skeletonization (works for all angles) ----------
    # `binary_closing` moved; use `closing` from skimage.morphology instead
    binary_closed = closing(binary, footprint=disk(params.MORPH_DISK_RADIUS))
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
    pairs = []

    for i in range(len(lines)):
        l1 = lines[i]
        dx1, dy1 = l1[1][0] - l1[0][0], l1[1][1] - l1[0][1]
        len1 = np.hypot(dx1, dy1)
        if len1 < min_ridge_length_px:
            continue
        dir1 = np.array([dx1, dy1]) / len1
        mid1 = np.array(l1[0]) + 0.5 * np.array([dx1, dy1])
        perp1 = np.array([-dir1[1], dir1[0]])

        for j in range(i + 1, len(lines)):
            l2 = lines[j]
            dx2, dy2 = l2[1][0] - l2[0][0], l2[1][1] - l2[0][1]
            len2 = np.hypot(dx2, dy2)
            if len2 < min_ridge_length_px:
                continue
            dir2 = np.array([dx2, dy2]) / len2
            mid2 = np.array(l2[0]) + 0.5 * np.array([dx2, dy2])

            # Angle difference (mod π)
            dot = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
            angle_diff_rad = np.arccos(np.abs(dot))
            if angle_diff_rad > np.deg2rad(params.PAIRING_ANGLE_TOLERANCE_DEG):
                continue

            # Perpendicular distance
            dist = abs(np.dot(mid2 - mid1, perp1))
            if abs(dist - spacing_px) / spacing_px > spacing_tolerance:
                continue

            # Overlap along average direction
            avg_dir = (dir1 + dir2) / np.linalg.norm(dir1 + dir2)
            t1_vals = [np.dot(np.array(p) - mid1, avg_dir) for p in l1]
            t2_vals = [np.dot(np.array(p) - mid1, avg_dir) for p in l2]
            t1_min, t1_max = min(t1_vals), max(t1_vals)
            t2_min, t2_max = min(t2_vals), max(t2_vals)
            overlap = max(0.0, min(t1_max, t2_max) - max(t1_min, t2_min))
            if overlap < overlap_ratio * min(len1, len2):
                continue

            # Score: shorter length × mean ridge strength
            def line_mean_ridge(line, n=50):
                xs = np.linspace(line[0][0], line[1][0], n)
                ys = np.linspace(line[0][1], line[1][1], n)
                ys = np.clip(ys, 0, ridge_map.shape[0] - 1)
                xs = np.clip(xs, 0, ridge_map.shape[1] - 1)
                vals = map_coordinates(ridge_map, [ys, xs], order=1, mode='nearest')
                return np.mean(vals) if len(vals) else 0.0

            score = min(len1, len2) * 0.5 * (line_mean_ridge(l1) + line_mean_ridge(l2))

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
        mid1 = np.array([(line1[0][0] + line1[1][0]) / 2,
                         (line1[0][1] + line1[1][1]) / 2])
        mid2 = np.array([(line2[0][0] + line2[1][0]) / 2,
                         (line2[0][1] + line2[1][1]) / 2])
        # Perpendicular direction to line1 (unit vector)
        perp = np.array([-dir1[1], dir1[0]])
        spacing = abs(np.dot(mid2 - mid1, perp))

        # --- 7. Keep the pair, optionally update the computed fields ---
        pair['spacing'] = spacing
        filtered.append(pair)

    return filtered    

def search_region(filepath: str, coordinates: tuple[float, float], size_x_meters: float, size_y_meters: float, output_dir: Path, verbose: int = 0, picture: bool = False):
    output_path = Path(output_dir) / f"{round(coordinates[0], 4)}_{round(coordinates[1], 4)}"
    if (output_path / "line_overlay.json").exists():
        return
    makedirs(output_path, exist_ok=True)
    gt, band, nodata, ds = getGTAndBand(filepath)

    window, extent = get_window(coordinates[0], coordinates[1], size_x_meters, size_y_meters, gt, band, nodata)
    if window is None:
        return
    window_original = window.copy()
    window_original = np.where(window_original == nodata, np.nan, window_original)
    target_angle = params.TARGET_ANGLE
    angle_tolerance = params.ANGLE_TOLERANCE
    if verbose > 0: print(f"Filtering DTM with median smoothing")
    smoothed = median_smooth_windowed(window, nodata=nodata)
    lrm = local_relief_model(smoothed, nodata=nodata)
    ridge_map = multiscale_ridge_enhancement(lrm, scales=params.SCAN_RIDGE_SCALES, nodata=nodata)
    nd_mask = window == nodata
    ridge_map = np.where(nd_mask, 0.0, ridge_map.astype(float))
    if verbose > 0: print("Detecting parallel walls …")
    pairs = detect_parallel_walls_ridge_pairing(
        ridge_map=ridge_map,
        target_angle=target_angle,
        extent=extent,
        spacing_meters=params.SPACING_METERS,
        angle_tolerance=angle_tolerance,
        spacing_tolerance=params.SPACING_TOLERANCE,
        ridge_threshold=params.RIDGE_THRESHOLD,
        min_ridge_length_meters=params.MIN_RIDGE_LENGTH_METERS,
        overlap_ratio=params.OVERLAP_RATIO,
        # orientation_source=lrm,
        return_debug=False,
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
def main():
    walls = (40.0,-82.4)
    weird_thing = (40.0, -82.4)
    search_region("loaded_tiles/Hopewell_Road/merged_dtm.tif", walls, 200, 200, Path("locations"), picture=False)

if __name__ == "__main__":
    main()