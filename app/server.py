import argparse
import glob
import io
import json
import logging
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

import parameters
from utils import (make_parabola, make_line, sample_corridor,
                   annotate_lines_corridor, perp_spacing_at_overlap)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)

app = Flask(__name__)

state = {
    "locations_folder": None,
    "dtm_path": None,
    "lines": [],
    "dtm_bounds": None,
    "dtm_elev_range": None,
    # OSM road-detection progress
    "osm_status": "idle",        # idle | pending | done | failed
    "osm_road_count": 0,
    # Recent non-fatal errors surfaced to the UI
    "load_errors": [],           # list of str, capped at 20
    # Corridor mode (None | "circleville" | "chillicothe" | "high_banks")
    "corridor_mode": None,
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "locations_folder": state["locations_folder"],
        "dtm_path": state["dtm_path"],
        "lines_loaded": len(state["lines"]),
        "dtm_loaded": state["dtm_bounds"] is not None,
        "dtm_bounds": state["dtm_bounds"],
        "corridor_mode": state["corridor_mode"],
        "angle_tolerance": parameters.ANGLE_TOLERANCE,
        "newark_coord": parameters.NEWARK_COORD,
        "chillicothe_coord": parameters.CHILLICOTHE_COORD,
        "circleville_coord": parameters.CIRCLEVILLE_COORD,
        "high_banks_coord": parameters.HIGH_BANKS_COORD,
    })


@app.route("/api/config", methods=["POST"])
def set_config():
    data = request.get_json(force=True)

    if "locations_folder" in data:
        folder = data["locations_folder"].strip()
        if not os.path.isabs(folder):
            folder = str(PROJECT_ROOT / folder)
        state["locations_folder"] = folder
        state["lines"] = _load_lines(folder)

    if "dtm_path" in data:
        path = data["dtm_path"].strip()
        if not os.path.isabs(path):
            path = str(PROJECT_ROOT / path)
        if path != state["dtm_path"]:
            state["dtm_path"] = path
            state["dtm_bounds"] = None
            state["dtm_elev_range"] = None
        _ensure_dtm()

    return jsonify({
        "lines_loaded": len(state["lines"]),
        "dtm_loaded": state["dtm_bounds"] is not None,
        "dtm_bounds": state["dtm_bounds"],
    })


@app.route("/api/status")
def get_status():
    return jsonify({
        "lines_loaded":    len(state["lines"]),
        "osm_status":      state["osm_status"],
        "osm_road_count":  state["osm_road_count"],
        "dtm_loaded":      state["dtm_bounds"] is not None,
        "errors":          state["load_errors"][-20:],
    })


@app.route("/api/lines")
def get_lines():
    zoom       = int(request.args.get("zoom", 12))
    bbox_str   = request.args.get("bbox", "")
    hide_roads = request.args.get("hide_roads", "0") == "1"

    bbox = None
    if bbox_str:
        try:
            w, s, e, n = map(float, bbox_str.split(","))
            bbox = (w, s, e, n)
        except ValueError:
            pass

    filtered = _filter_lines(state["lines"], bbox, zoom, hide_roads)
    return jsonify(_to_geojson(filtered))


@app.route("/api/corridor")
def get_corridor():
    fns = _get_corridor_fns()
    if fns is None:
        return jsonify(None)
    curve_func, _deriv, x_start, x_end, half_width_deg = fns
    ring = sample_corridor(curve_func, x_start, x_end, half_width_deg)
    return jsonify({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"mode": state["corridor_mode"]},
    })


@app.route("/api/prewarm_priority", methods=["POST"])
def prewarm_priority():
    _ensure_dtm()
    if state["dtm_bounds"] is None:
        return jsonify({"error": "No DTM loaded"}), 404

    data      = request.get_json(force=True) or {}
    corridors = data.get("corridors", [])   # [[lon1,lat1,lon2,lat2], ...]
    lo, hi    = data.get("lo", state["dtm_elev_range"][0]), \
                data.get("hi", state["dtm_elev_range"][1])

    if not corridors:
        return jsonify({"error": "No corridors provided"}), 400

    # Snapshot everything the thread needs before returning
    snap = {
        "path":    state["dtm_path"],
        "bounds":  dict(state["dtm_bounds"]),
        "range":   (float(lo), float(hi)),
        "corridors": corridors,
    }

    def _worker(snap):
        from .dtm_renderer import render_tile, iter_corridor_tiles
        # Progressively finer zoom with shrinking buffers
        zoom_levels = parameters.PREWARM_ZOOM_SCHEDULE
        total = 0
        for zoom, default_buf in zoom_levels:
            specs = [(c[0], c[1], c[2], c[3],
                      c[4] if len(c) > 4 else default_buf)
                     for c in snap["corridors"]]
            count = 0
            for x, y in iter_corridor_tiles(zoom, specs, snap["bounds"]):
                try:
                    render_tile(snap["path"], zoom, x, y,
                                snap["range"], snap["bounds"])
                    count += 1
                except Exception as e:
                    log.warning("prewarm %d/%d/%d: %s", zoom, x, y, e)
            log.info("priority prewarm zoom %d: %d tiles", zoom, count)
            total += count
        log.info("priority prewarm done: %d tiles", total)

    threading.Thread(target=_worker, args=(snap,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/dtm_stats")
def get_dtm_stats():
    _ensure_dtm()
    if state["dtm_bounds"] is None:
        return jsonify({"error": "No DTM loaded"}), 404
    bbox_str = request.args.get("bbox", "")
    if not bbox_str:
        lo, hi = state["dtm_elev_range"]
        return jsonify({"lo": lo, "hi": hi})
    try:
        w, s, e, n = map(float, bbox_str.split(","))
    except ValueError:
        return jsonify({"error": "Invalid bbox"}), 400
    try:
        from .dtm_renderer import read_region_elev_range
        lo, hi = read_region_elev_range(state["dtm_path"], w, s, e, n)
        return jsonify({"lo": round(lo, 1), "hi": round(hi, 1)})
    except Exception as ex:
        log.error("dtm_stats: %s", ex)
        lo, hi = state["dtm_elev_range"]
        return jsonify({"lo": lo, "hi": hi})


@app.route("/api/dtm_tile/<int:z>/<int:x>/<int:y>.png")
def get_dtm_tile(z, x, y):
    _ensure_dtm()
    if state["dtm_bounds"] is None:
        return ("", 204)
    try:
        lo = float(request.args.get("lo", state["dtm_elev_range"][0]))
        hi = float(request.args.get("hi", state["dtm_elev_range"][1]))
    except (ValueError, TypeError):
        lo, hi = state["dtm_elev_range"]
    try:
        from .dtm_renderer import render_tile
        png = render_tile(
            state["dtm_path"], z, x, y,
            (lo, hi),
            state["dtm_bounds"],
        )
        return send_file(io.BytesIO(png), mimetype="image/png")
    except Exception as e:
        log.error("tile %d/%d/%d: %s", z, x, y, e)
        return ("", 204)


@app.route("/api/available_folders")
def available_folders():
    loc_root = PROJECT_ROOT / "data" / "locations"
    if not loc_root.exists():
        return jsonify([])

    folders = []
    for item in sorted(loc_root.iterdir()):
        if not item.is_dir():
            continue
        count = len(list(item.rglob("line_overlay.json")))
        if count > 0:
            folders.append({
                "name": item.name,
                "path": str(item.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "count": count,
            })
    return jsonify(folders)


# ---------------------------------------------------------------------------
# Corridor geometry helpers
# ---------------------------------------------------------------------------

def _get_corridor_fns():
    """Return (curve_func, curve_deriv, x_start, x_end, half_width_deg) or None."""
    mode = state.get("corridor_mode")
    half_width_deg = parameters.CORRIDOR_HALF_WIDTH_M / parameters.METERS_PER_DEGREE_LAT
    if mode == "circleville":
        curve_func, curve_deriv = make_parabola(
            parameters.PARABOLA_A, parameters.PARABOLA_B, parameters.PARABOLA_C,
            x_offset=parameters.PARABOLA_X_OFFSET,
        )
        return curve_func, curve_deriv, parameters.PARABOLA_X_OFFSET, \
               parameters.NEWARK_COORD[1], half_width_deg
    if mode == "chillicothe":
        x0, y0 = parameters.CHILLICOTHE_COORD[1], parameters.CHILLICOTHE_COORD[0]
        x1, y1 = parameters.NEWARK_COORD[1], parameters.NEWARK_COORD[0]
        curve_func, curve_deriv = make_line(x0, y0, x1, y1)
        return curve_func, curve_deriv, x0, x1, half_width_deg
    if mode == "high_banks":
        x0, y0 = parameters.HIGH_BANKS_COORD[1], parameters.HIGH_BANKS_COORD[0]
        x1, y1 = parameters.NEWARK_COORD[1], parameters.NEWARK_COORD[0]
        curve_func, curve_deriv = make_line(x0, y0, x1, y1)
        return curve_func, curve_deriv, x0, x1, half_width_deg
    return None


# ---------------------------------------------------------------------------
# Error tracking
# ---------------------------------------------------------------------------

def _push_error(msg: str) -> None:
    """Append a non-fatal error to state, capped at 20 entries."""
    log.warning(msg)
    state["load_errors"].append(msg)
    if len(state["load_errors"]) > 20:
        state["load_errors"].pop(0)


# ---------------------------------------------------------------------------
# OSM-based road detection
# ---------------------------------------------------------------------------

_LAT_M = parameters.METERS_PER_DEGREE_LAT
_LON_M = parameters.METERS_PER_DEGREE_LON_AT_40N

_OSM_CACHE = PROJECT_ROOT / "data" / ".osm_roads_cache.json"

_ROAD_TYPES = (
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "unclassified", "road",
)


def _load_osm_cache(south, west, north, east):
    try:
        with open(_OSM_CACHE) as f:
            c = json.load(f)
        cs, cw, cn, ce = c["bbox"]
        if cs <= south and cw <= west and cn >= north and ce >= east:
            if time.time() - c["ts"] < parameters.OSM_CACHE_TTL_S:
                segs = c["segs"]
                log.info("OSM cache hit: %d road segments (%.1f h old)",
                         len(segs), (time.time() - c["ts"]) / 3600)
                return segs
    except Exception:
        pass
    return None


def _save_osm_cache(south, west, north, east, segs):
    try:
        with open(_OSM_CACHE, "w") as f:
            json.dump({"bbox": [south, west, north, east],
                       "ts": time.time(), "segs": segs}, f)
    except Exception as e:
        log.warning("OSM cache save failed: %s", e)


def _fetch_osm_roads(south, west, north, east) -> list:
    cached = _load_osm_cache(south, west, north, east)
    if cached is not None:
        return cached

    types_re = "|".join(_ROAD_TYPES)
    query = (
        f'[out:json][timeout:60];'
        f'(way["highway"~"^({types_re})$"]({south:.6f},{west:.6f},{north:.6f},{east:.6f}););'
        f'out geom;'
    )
    url  = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode()
    log.info("Querying Overpass API for roads in bbox %.4f,%.4f → %.4f,%.4f …",
             south, west, north, east)
    try:
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept":       "application/json",
                "User-Agent":   "KhthonLiDAR/1.0 (archaeological survey tool)",
            },
        )
        with urllib.request.urlopen(req, timeout=parameters.OSM_QUERY_TIMEOUT_S) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        log.warning("Overpass API unreachable: %s — road filtering disabled", e)
        return []
    except Exception as e:
        log.warning("Overpass API error: %s — road filtering disabled", e)
        return []

    segs = []
    for el in result.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry", [])
        for i in range(len(geom) - 1):
            a, b = geom[i], geom[i + 1]
            segs.append([a["lon"], a["lat"], b["lon"], b["lat"]])

    log.info("Overpass: %d road segments fetched", len(segs))
    _save_osm_cache(south, west, north, east, segs)
    return segs


def _pt_seg_dist_m(px, py, ax, ay, bx, by) -> float:
    dx = (bx - ax) * _LON_M
    dy = (by - ay) * _LAT_M
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-6:
        return math.sqrt(((px - ax) * _LON_M) ** 2 + ((py - ay) * _LAT_M) ** 2)
    t = max(0.0, min(1.0,
        ((px - ax) * _LON_M * dx + (py - ay) * _LAT_M * dy) / len_sq))
    return math.sqrt(((px - ax - t * (bx - ax)) * _LON_M) ** 2 +
                     ((py - ay - t * (by - ay)) * _LAT_M) ** 2)


def _mark_osm_road_lines(lines: list, segs: list, threshold_m: float = parameters.OSM_ROAD_THRESHOLD_M) -> int:
    if not segs:
        return 0
    buf = threshold_m / parameters.METERS_PER_DEGREE_LON_AT_40N
    marked = 0
    for ln in lines:
        mx = (ln["xs"] + ln["xe"]) / 2
        my = (ln["ys"] + ln["ye"]) / 2
        for ax, ay, bx, by in segs:
            if not (min(ax, bx) - buf <= mx <= max(ax, bx) + buf and
                    min(ay, by) - buf <= my <= max(ay, by) + buf):
                continue
            if _pt_seg_dist_m(mx, my, ax, ay, bx, by) <= threshold_m:
                ln["road"] = True
                marked += 1
                break
    return marked


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wall_angle_deg(raw_angle_rad: float) -> float:
    """Convert a line_overlay.json line angle (radians, atan(dlat/dlon) in raw
    geographic slope terms) into the same degrees / image-space, CCW-from-+x,
    mod-180 convention that utils.corridor_angle_at() returns as
    'expected_angle'. Without this conversion the two were being compared
    directly despite different units (radians vs degrees) and different
    conventions, so the angle-deviation filter was effectively comparing
    unrelated numbers rather than a true relative tolerance."""
    return (90.0 - math.degrees(raw_angle_rad)) % 180.0


def _pair_spacing_m(line_a: dict, line_b: dict) -> float:
    """Perpendicular spacing (m) between the two walls of a detected pair,
    measured at the overlap of their along-wall extent (see
    utils.perp_spacing_at_overlap) rather than naive midpoint-to-midpoint,
    which is skewed when the two lines don't extend equally far."""
    def _line_m(ln):
        a = (ln["xs"] * _LON_M, ln["ys"] * _LAT_M)
        b = (ln["xe"] * _LON_M, ln["ye"] * _LAT_M)
        return a, b

    def _unit_dir(line):
        (x0, y0), (x1, y1) = line
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-9:
            return np.array([1.0, 0.0])
        return np.array([(x1 - x0) / length, (y1 - y0) / length])

    line1 = _line_m(line_a)
    line2 = _line_m(line_b)
    return perp_spacing_at_overlap(line1, _unit_dir(line1), line2, _unit_dir(line2))


def _load_lines(folder: str) -> list:
    state["osm_status"] = "idle"
    state["osm_road_count"] = 0

    lines = []
    pattern = os.path.join(folder, "**", "line_overlay.json")
    for json_path in glob.glob(pattern, recursive=True):
        try:
            with open(json_path) as f:
                pairs = json.load(f)
            for pair in pairs:
                extracted = []
                for n in (0, 1):
                    prefix = f"line_{n}_"
                    xs = pair.get(f"{prefix}xs")
                    ys = pair.get(f"{prefix}ys")
                    xe = pair.get(f"{prefix}xe")
                    ye = pair.get(f"{prefix}ye")
                    if None in (xs, ys, xe, ye):
                        extracted.append(None)
                        continue
                    extracted.append({
                        "xs": xs, "ys": ys,
                        "xe": xe, "ye": ye,
                        "length": pair.get(f"{prefix}length", 0.0),
                        "angle":  _wall_angle_deg(pair.get(f"{prefix}angle", 0.0)),
                        "road": False,
                    })

                # Spacing between the two walls of this pair (None if only one
                # of the two lines was detected).
                pair_distance_m = None
                if extracted[0] is not None and extracted[1] is not None:
                    pair_distance_m = _pair_spacing_m(extracted[0], extracted[1])

                for ln in extracted:
                    if ln is not None:
                        ln["pair_distance_m"] = pair_distance_m
                        lines.append(ln)
        except Exception as e:
            _push_error(f"Failed to load {os.path.basename(json_path)}: {e}")

    log.info("loaded %d lines from %s — OSM road detection starting in background", len(lines), folder)

    # Annotate with corridor geometry (expected angle + distance from centreline)
    fns = _get_corridor_fns()
    if fns is not None and lines:
        curve_func, curve_deriv = fns[0], fns[1]
        annotate_lines_corridor(lines, curve_func, curve_deriv)

    # Run OSM road detection asynchronously so line loading returns immediately.
    # _mark_osm_road_lines mutates the dicts in-place; CPython's GIL makes the
    # per-key dict assignment effectively atomic, so there is no data race.
    if lines:
        all_lons = [(ln["xs"] + ln["xe"]) / 2 for ln in lines]
        all_lats = [(ln["ys"] + ln["ye"]) / 2 for ln in lines]
        margin = parameters.OSM_BBOX_MARGIN_DEG
        south = min(all_lats) - margin
        west  = min(all_lons) - margin
        north = max(all_lats) + margin
        east  = max(all_lons) + margin

        def _osm_worker(lines_ref, south, west, north, east):
            try:
                segs = _fetch_osm_roads(south, west, north, east)
                if not segs:
                    state["osm_status"] = "failed"
                    _push_error("OSM: no road segments returned (Overpass unreachable or empty bbox)")
                    return
                road_n = _mark_osm_road_lines(lines_ref, segs)
                state["osm_road_count"] = road_n
                state["osm_status"] = "done"
                log.info("OSM road detection done: %d/%d lines flagged as road",
                         road_n, len(lines_ref))
            except Exception as exc:
                state["osm_status"] = "failed"
                _push_error(f"OSM worker error: {exc}")
                log.exception("OSM worker failed")

        state["osm_status"] = "pending"
        threading.Thread(
            target=_osm_worker, args=(lines, south, west, north, east), daemon=True
        ).start()

    return lines


def _filter_lines(lines: list, bbox, zoom: int, hide_roads: bool = False) -> list:
    min_len = _min_length(zoom)
    max_n = _max_count(zoom)

    result = []
    for ln in lines:
        if ln["length"] < min_len:
            continue
        if hide_roads and ln["road"]:
            continue
        if bbox is not None:
            w, s, e, n = bbox
            mid_lon = (ln["xs"] + ln["xe"]) / 2
            mid_lat = (ln["ys"] + ln["ye"]) / 2
            if not (w <= mid_lon <= e and s <= mid_lat <= n):
                continue
        result.append(ln)

    if len(result) > max_n:
        step = max(1, len(result) // max_n)
        result = result[::step]

    return result


def _min_length(zoom: int) -> float:
    if zoom < 11: return 0.005
    if zoom < 12: return 0.002
    if zoom < 13: return 0.001
    if zoom < 14: return 0.0005
    return 0.0


def _max_count(zoom: int) -> int:
    if zoom < 11: return 50
    if zoom < 12: return 200
    if zoom < 13: return 800
    if zoom < 14: return 3000
    return 20000


def _to_geojson(lines: list) -> dict:
    features = []
    for ln in lines:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [ln["xs"], ln["ys"]],
                    [ln["xe"], ln["ye"]],
                ],
            },
            "properties": {
                "length": ln["length"],
                "angle": ln["angle"],
                "road": ln.get("road", False),
                "expected_angle": ln.get("expected_angle"),
                "corridor_dist_m": ln.get("corridor_dist_m"),
                "pair_distance_m": ln.get("pair_distance_m"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _ensure_dtm():
    if state["dtm_bounds"] is not None:
        return
    if not state["dtm_path"] or not os.path.exists(state["dtm_path"]):
        return
    try:
        from .dtm_renderer import read_dtm_info, build_overviews
        bounds, elev_range = read_dtm_info(state["dtm_path"])
        state["dtm_bounds"] = bounds
        state["dtm_elev_range"] = elev_range
        log.info("DTM info loaded, bounds=%s, elev_range=%s", bounds, elev_range)
        threading.Thread(
            target=build_overviews, args=(state["dtm_path"],), daemon=True
        ).start()
    except Exception as e:
        log.error("DTM info failed: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="KhthonLiDAR Map Viewer")
    parser.add_argument("--folder", default="", help="Locations subfolder (e.g. data/locations)")
    parser.add_argument("--search-size", type=int, default=parameters.SCAN_TILE_SIZE_M, help="Search size in pixels for line overlay matching")
    parser.add_argument("--mode", choices=["polygon", "circleville", "chillicothe", "high_banks"],
                        help="Auto-select the locations folder that matches this scan mode")
    parser.add_argument("--dtm", default="", help="Path to merged DTM GeoTIFF")
    parser.add_argument("--port", type=int, default=parameters.SERVER_PORT)
    parser.add_argument("--max-workers", type=int, default=parameters.SERVER_THREADS, help="Max workers for OSM prewarm and DTM overview generation")
    args = parser.parse_args()

    size = args.search_size
    folder = args.folder
    loc_root = PROJECT_ROOT / "data" / "locations"
    if folder:
        folder = folder if os.path.isabs(folder) else str(PROJECT_ROOT / folder)
    else:
        if args.mode:
            matches = sorted(loc_root.glob(f"{args.mode}_{size}_{size}"))
        else:
            matches = sorted(loc_root.glob(f"{size}_{size}"))
        if matches:
            folder = str(matches[0])
            log.info("mode=%s → auto-selected folder: %s", args.mode, folder)
        else:
            log.warning("No locations folder found matching mode '%s' in %s", args.mode, loc_root)
    state["corridor_mode"] = args.mode   # None | "circleville" | "chillicothe" | "high_banks"
    state["locations_folder"] = folder
    state["lines"] = _load_lines(folder)

    if args.dtm:
        path = args.dtm if os.path.isabs(args.dtm) else str(PROJECT_ROOT / args.dtm)
        state["dtm_path"] = path
        _ensure_dtm()

    log.info("Starting on http://127.0.0.1:%d", args.port)
    try:
        from waitress import serve
        serve(app, host="127.0.0.1", port=args.port, threads=args.max_workers)
    except ImportError:
        log.warning("waitress not installed, falling back to Flask dev server")
        app.run(debug=False, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
