"""
server.py — Flask app for outlining known earthworks over each site's ground grid.

    python -m earthwork_builder.server [--port 5001]

The sidebar lists sites.json; pick one to see its vendor-ground hillshade
(rendered by app/dtm_renderer.py) and draw / edit / delete outlines.  Saving
writes labels.geojson + labels.shp in the site folder.  "Add site" appends to
sites.json and runs prepare.py for it in a background thread.
"""

from __future__ import annotations

import argparse
import logging
import threading

from flask import Flask, Response, jsonify, render_template, request

import parameters
from app import dtm_renderer
from earthwork_builder import prepare
from earthwork_builder.labels import Site, get_site, load_sites, read_labels, save_labels, save_sites

log = logging.getLogger(__name__)

app = Flask(__name__)

_jobs: dict[str, dict] = {}          # slug → {"status": running|done|failed, "log": [...]}
_sites_lock = threading.Lock()


def _site_summary(s: Site) -> dict:
    fc = read_labels(s)
    job = _jobs.get(s.slug, {})
    return {"slug": s.slug, "name": s.name, "lat": s.lat, "lon": s.lon, "half_m": s.half_m,
            "notes": s.notes, "prepared": s.prepared, "labels": len(fc["features"]),
            "reviewed": fc.get("reviewed", False), "job": job.get("status")}


@app.errorhandler(KeyError)
def _unknown_site(e):
    return jsonify({"error": str(e)}), 404


@app.route("/")
def index():
    return render_template("index.html", types=parameters.EW_EARTHWORK_TYPES,
                           negatives=parameters.EW_NEGATIVE_TYPES,
                           unverified=parameters.EW_UNVERIFIED_TYPE,
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
                    notes=str(d.get("notes", "")))
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
    if _jobs.get(site.slug, {}).get("status") == "running":
        return
    job = _jobs[site.slug] = {"status": "running", "log": []}

    def run():
        try:
            prepare.prepare_site(site, force=force, log=job["log"].append)
            job["status"] = "done"
        except Exception as e:
            log.exception("prepare %s failed", site.slug)
            job["log"].append(f"FAILED: {e}")
            job["status"] = "failed"

    threading.Thread(target=run, daemon=True).start()


@app.route("/api/sites/<slug>/job")
def job_status(slug):
    return jsonify(_jobs.get(slug, {"status": None, "log": []}))


@app.route("/api/sites/<slug>")
def site_detail(slug):
    s = get_site(slug)
    out = _site_summary(s)
    out["features"] = read_labels(s)
    if s.prepared:
        bounds, elev = dtm_renderer.read_dtm_info(str(s.ground_path))
        out["bounds"], out["elev_range"] = bounds, elev
    return jsonify(out)


@app.route("/api/sites/<slug>/labels", methods=["PUT"])
def put_labels(slug):
    try:
        fc = save_labels(get_site(slug), request.get_json(force=True))
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(fc)


@app.route("/api/sites/<slug>/tile/<int:z>/<int:x>/<int:y>.png")
def site_tile(slug, z, x, y):
    s = get_site(slug)
    if not s.prepared:
        return Response(dtm_renderer._empty_tile(), mimetype="image/png")
    path = str(s.ground_path)
    bounds, elev = dtm_renderer.read_dtm_info(path)
    png = dtm_renderer.render_tile(path, z, x, y, elev, bounds)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "max-age=3600"})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Earthwork outline builder")
    ap.add_argument("--port", type=int, default=parameters.EW_BUILDER_PORT)
    args = ap.parse_args()
    log.info("Sites file: %s (%d sites)", parameters.EW_SITES_FILE, len(load_sites()))
    log.info("Starting on http://127.0.0.1:%d", args.port)
    try:
        from waitress import serve
        serve(app, host="127.0.0.1", port=args.port, threads=parameters.SERVER_THREADS)
    except ImportError:
        app.run(port=args.port, threaded=True)


if __name__ == "__main__":
    main()
