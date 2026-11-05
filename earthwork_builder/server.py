"""
server.py — Flask app for outlining known earthworks over each site's ground grid.

    python -m earthwork_builder.server [--port 5001]

The sidebar lists sites.json; pick one to see its vendor-ground hillshade
(direction-independent views from visualize.py) and draw / edit / delete outlines.  Saving
writes labels.geojson + labels.shp in the site folder.  "Add site" appends to
sites.json and runs prepare.py for it in a background thread.  sites.json is
watched: sites added, moved or resized there are prepared automatically (one at
a time; --no-auto-prepare turns this off).
"""

from __future__ import annotations

import argparse
import logging
import threading

from flask import Flask, Response, jsonify, render_template, request

import parameters
from app import dtm_renderer
from earthwork_builder import prepare, visualize
from earthwork_builder.labels import (REJECT_REASONS, Site, get_site, inset_ring, load_sites,
                                      read_labels, save_labels, save_sites)

log = logging.getLogger(__name__)

app = Flask(__name__)

_jobs: dict[str, dict] = {}          # slug → {"status": running|done|failed, "log": [...]}
_sites_lock = threading.Lock()
_prepare_slot = threading.Semaphore(1)


def _site_summary(s: Site) -> dict:
    fc = read_labels(s)
    job = _jobs.get(s.slug, {})
    return {"slug": s.slug, "name": s.name, "lat": s.lat, "lon": s.lon, "half_m": s.half_m,
            "notes": s.notes, "background": s.background, "prepared": s.prepared, "stale": s.stale, "labels": len(fc["features"]),
            "reviewed": fc.get("reviewed", False), "rejected": fc.get("rejected", False),
            "reject_reason": fc.get("reject_reason", ""), "job": job.get("status")}


@app.errorhandler(KeyError)
def _unknown_site(e):
    return jsonify({"error": str(e)}), 404


@app.route("/")
def index():
    return render_template("index.html", types=parameters.EW_EARTHWORK_TYPES,
                           negatives=parameters.EW_NEGATIVE_TYPES,
                           unverified=parameters.EW_UNVERIFIED_TYPE,
                           views=visualize.MODES, reject_reasons=REJECT_REASONS,
                           default_half=parameters.EW_SITE_HALF_M)


@app.route("/api/sites", methods=["GET"])
def list_sites():
    return jsonify([_site_summary(s) for s in load_sites()])


@app.route("/api/sites", methods=["POST"])
def add_site():
    d = request.get_json(force=True)
    try:
        site = Site(name=str(d["name"]).strip(), lat=float(d["lat"]), lon=float(d["lon"]),
                    half_m=float(d.get("half_m") or parameters.EW_SITE_HALF_M),
                    notes=str(d.get("notes", "")),
                    extra={"background": True} if d.get("background") else {})
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"bad site: {e}"}), 400
    if not site.slug:
        return jsonify({"error": "name is required"}), 400
    with _sites_lock:
        sites = load_sites()
        if any(s.slug == site.slug for s in sites):
            return jsonify({"error": f"a site named {site.slug!r} already exists"}), 409
        save_sites(sites + [site])
    _start_prepare(site)
    return jsonify(_site_summary(site)), 201


@app.route("/api/sites/<slug>/prepare", methods=["POST"])
def prepare_route(slug):
    _start_prepare(get_site(slug), force=bool(request.args.get("force")))
    return jsonify({"status": "running"})


def _start_prepare(site: Site, force: bool = False) -> None:
    if _jobs.get(site.slug, {}).get("status") in ("queued", "running"):
        return
    job = _jobs[site.slug] = {"status": "queued", "log": ["waiting for another site to finish"],
                              "window": _window_key(site)}

    def run():
        with _prepare_slot:                       # one at a time: big windows need GBs each
            job["status"] = "running"
            try:
                prepare.prepare_site(site, force=force, log=job["log"].append)
                job["status"] = "done"
            except Exception as e:
                log.exception("prepare %s failed", site.slug)
                job["log"].append(f"FAILED: {e}")
                job["status"] = "failed"

    threading.Thread(target=run, daemon=True).start()


def _window_key(site: Site) -> tuple:
    return site.lat, site.lon, site.half_m


def _auto_prepare() -> None:
    """Queue every site sites.json moved, resized or added but nobody has built
    yet.  A failed site is retried only after its entry changes again."""
    for s in load_sites():
        if s.prepared:
            continue
        job = _jobs.get(s.slug, {})
        if job.get("status") in ("queued", "running"):
            continue
        if job.get("status") == "failed" and job.get("window") == _window_key(s):
            continue
        log.info("sites.json: %s is %s - preparing", s.name, "stale" if s.stale else "unprepared")
        _start_prepare(s)


def _watch_sites(interval_s: float = 10.0) -> None:
    """Re-check sites.json whenever it changes (and once at startup)."""
    last = None
    while True:
        try:
            mtime = parameters.EW_SITES_FILE.stat().st_mtime_ns
            if mtime != last:
                last = mtime
                with _sites_lock:
                    _auto_prepare()
        except Exception:
            log.exception("sites.json watch failed")
        threading.Event().wait(interval_s)


@app.route("/api/sites/<slug>/job")
def job_status(slug):
    job = _jobs.get(slug)
    # A finished job is moot once the site is no longer prepared (moved / resized since)
    if job and job["status"] == "done" and not get_site(slug).prepared:
        _jobs.pop(slug, None)
        job = None
    return jsonify(job or {"status": None, "log": []})


@app.route("/api/sites/<slug>")
def site_detail(slug):
    s = get_site(slug)
    out = _site_summary(s)
    out["features"] = read_labels(s)
    if s.prepared:
        bounds, elev = dtm_renderer.read_dtm_info(str(s.ground_path))
        out["bounds"], out["elev_range"] = bounds, elev
        out["ground_token"] = visualize.ground_token(s.ground_path)   # busts browser tile caches
    return jsonify(out)


@app.route("/api/sites/<slug>/labels", methods=["PUT"])
def put_labels(slug):
    try:
        fc = save_labels(get_site(slug), request.get_json(force=True))
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(fc)


@app.route("/api/inset", methods=["POST"])
def inset_route():
    """{ring: [[lon, lat], …], inset_m} → {ring} shrunk inward (null if nothing is left)."""
    d = request.get_json(force=True)
    try:
        return jsonify({"ring": inset_ring(d["ring"], float(d["inset_m"]))})
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"bad inset request: {e}"}), 400


@app.route("/api/sites/<slug>/tile/<int:z>/<int:x>/<int:y>.png")
def site_tile(slug, z, x, y):
    """XYZ tile of one of the site's terrain views (?view=mdh|slope|lrm|combo)."""
    s = get_site(slug)
    view = request.args.get("view", "mdh")
    if view not in visualize.MODES:
        return jsonify({"error": f"unknown view {view!r}"}), 400
    if not s.prepared:
        return Response(visualize.empty_tile(), mimetype="image/png")
    tif = visualize.ensure_view(s.ground_path, view)
    png = visualize.render_tile(tif, z, x, y)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "max-age=3600"})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Earthwork outline builder")
    ap.add_argument("--port", type=int, default=parameters.EW_BUILDER_PORT)
    ap.add_argument("--no-auto-prepare", action="store_true",
                    help="Don't download/rebuild sites that sites.json added, moved or resized")
    args = ap.parse_args()
    log.info("Sites file: %s (%d sites)", parameters.EW_SITES_FILE, len(load_sites()))
    if not args.no_auto_prepare:
        threading.Thread(target=_watch_sites, daemon=True).start()
    log.info("Starting on http://127.0.0.1:%d", args.port)
    try:
        from waitress import serve
        serve(app, host="127.0.0.1", port=args.port, threads=parameters.SERVER_THREADS)
    except ImportError:
        app.run(port=args.port, threaded=True)


if __name__ == "__main__":
    main()
