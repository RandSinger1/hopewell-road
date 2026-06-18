"use strict";

// ---------------------------------------------------------------------------
// Reference site coordinates (Hopewell earthwork locations, Ohio)
// Single source of truth: parameters.py, fetched via /api/config.
// ---------------------------------------------------------------------------
let NEWARK      = null;
let CHILLICOTHE = null;
let CIRCLEVILLE = null;
let HIGH_BANKS  = null;

let ROUTE_WAYPOINTS = [];   // loaded from /static/coords.json

// ---------------------------------------------------------------------------
// Map & layer state
// ---------------------------------------------------------------------------
let map;
let dtmOverlay    = null;
let linesLayer    = null;
let refLayer      = null;
let corridorLayer = null;
let targetLayer   = null;
let loadDebounce  = null;
let statsDebounce = null;
let dtmLo         = null;
let dtmHi         = null;
let _rawGeojson   = null;   // cached full server response, filtered client-side

// Target-lines overlay (screen-centred wall template)
let targetOn         = false;
let targetDebounce    = null;
const TARGET_LENGTH_M = 50;

// Cursor elevation
let cursorElevDebounce = null;

// ---------------------------------------------------------------------------
// Initialise
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  _initMap();
  _initSidebar();
  _fetchInitialConfig();
  _populateFolderDropdown();
  _loadRouteWaypoints();
  _fetchCorridor();
});

function _initMap() {
  map = L.map("map", {
    center: [39.75, -82.70],
    zoom: 10,
    zoomControl: true,
    preferCanvas: true,   // faster for many polylines
  });

  // Base: OSM with subtle dark filter applied via CSS
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
    opacity: 0.55,
  }).addTo(map);

  linesLayer  = L.layerGroup().addTo(map);
  refLayer    = L.layerGroup().addTo(map);
  targetLayer = L.layerGroup();   // not added to map until toggled on

  _addMapControls();
  _addKeyboardShortcuts();

  map.on("zoomend moveend", _scheduleLineLoad);
  map.on("zoomend moveend", _updateInfoPanel);
  map.on("zoomend moveend", _scheduleDtmStats);
  map.on("zoomend moveend", _scheduleTargetLines);
  map.on("mousemove", _onMapMouseMove);
  map.on("mouseout", _onMapMouseOut);
  _updateInfoPanel();
}

function _initSidebar() {
  const btn = document.getElementById("sidebar-toggle");
  btn.addEventListener("click", () => {
    const sb = document.getElementById("sidebar");
    const collapsed = sb.classList.toggle("collapsed");
    btn.innerHTML = collapsed ? "&#x25BA;" : "&#x25C4;";
  });
}

// ---------------------------------------------------------------------------
// Server communication
// ---------------------------------------------------------------------------
function _fetchInitialConfig() {
  fetch("/api/config")
    .then(r => r.json())
    .then(cfg => {
      NEWARK      = cfg.newark_coord;
      CHILLICOTHE = cfg.chillicothe_coord;
      CIRCLEVILLE = cfg.circleville_coord;
      HIGH_BANKS  = cfg.high_banks_coord;
      _populateSiteInputs();
      _drawReferenceLines();
      if (cfg.locations_folder) {
        document.getElementById("folder-input").value = cfg.locations_folder;
        _setStatus("lines", `${cfg.lines_loaded} lines loaded`);
        _scheduleLineLoad();
        _startStatusPoll();
      }
      if (cfg.dtm_path) {
        document.getElementById("dtm-input").value = cfg.dtm_path;
      }
      if (cfg.dtm_loaded && cfg.dtm_bounds) {
        _applyDtmOverlay(cfg.dtm_bounds);
      }
      if (cfg.angle_tolerance != null) {
        _setAngleSliderBounds(cfg.angle_tolerance);
      }
    })
    .catch(console.error);
}

// Bound the angle-deviation slider to ±ANGLE_TOLERANCE (parameters.py) instead
// of an arbitrary ±90° range: the detector already discards anything outside
// that tolerance of the per-line target angle, so the slider's full travel
// should map onto the only deviations that ever actually occur.
function _setAngleSliderBounds(tolDeg) {
  const t = Math.abs(tolDeg);
  const loEl = document.getElementById("filter-angle-lo");
  const hiEl = document.getElementById("filter-angle-hi");
  loEl.min = -t; loEl.max = t; loEl.value = -t;
  hiEl.min = -t; hiEl.max = t; hiEl.value = t;
  document.getElementById("val-angle-tol").textContent = `${-t}° to ${t}°`;
  _applyFilters();
}

function _populateFolderDropdown() {
  fetch("/api/available_folders")
    .then(r => r.json())
    .then(folders => {
      const sel = document.getElementById("folder-select");
      folders.forEach(f => {
        const opt = document.createElement("option");
        opt.value = f.path;
        opt.textContent = `${f.name}  (${f.count} tiles)`;
        sel.appendChild(opt);
      });
      // When the user picks from the dropdown, copy into the text field
      sel.addEventListener("change", () => {
        if (sel.value) document.getElementById("folder-input").value = sel.value;
      });
    })
    .catch(console.error);
}

// ---------------------------------------------------------------------------
// UI actions
// ---------------------------------------------------------------------------
function applyFolder() {
  const sel   = document.getElementById("folder-select");
  const input = document.getElementById("folder-input");
  const folder = input.value.trim() || sel.value;
  if (!folder) return;

  _setStatus("lines", "Loading…");
  fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ locations_folder: folder }),
  })
    .then(r => r.json())
    .then(data => {
      _setStatus("lines", `${data.lines_loaded} lines loaded`);
      _scheduleLineLoad();
      _startStatusPoll();
    })
    .catch(e => _setStatus("lines", "Error: " + e.message));
}

function applyDtm() {
  const path = document.getElementById("dtm-input").value.trim();
  if (!path) return;

  _setStatus("dtm", "Rendering DTM (may take a moment)…");
  document.getElementById("loading-indicator").style.display = "block";

  fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dtm_path: path }),
  })
    .then(r => r.json())
    .then(data => {
      document.getElementById("loading-indicator").style.display = "none";
      if (data.dtm_loaded && data.dtm_bounds) {
        _applyDtmOverlay(data.dtm_bounds);
        _setStatus("dtm", "DTM loaded");
      } else {
        _setStatus("dtm", "DTM not found or failed to render");
      }
    })
    .catch(e => {
      document.getElementById("loading-indicator").style.display = "none";
      _setStatus("dtm", "Error: " + e.message);
    });
}

// ---------------------------------------------------------------------------
// DTM overlay
// ---------------------------------------------------------------------------
function _applyDtmOverlay(bounds) {
  if (dtmOverlay) map.removeLayer(dtmOverlay);
  _prewarmTriggered = false;   // allow prewarm to re-fire for the new DTM

  const leafletBounds = [
    [bounds.south, bounds.west],
    [bounds.north, bounds.east],
  ];
  const opacity = document.getElementById("dtm-opacity").value / 100;

  dtmOverlay = L.tileLayer("/api/dtm_tile/{z}/{x}/{y}.png", {
    opacity,
    interactive: false,
    bounds: leafletBounds,   // Leaflet skips tile requests outside DTM extent
    tileSize: 256,
    maxZoom: 20,
    attribution: "",
  });

  if (document.getElementById("chk-dtm").checked) {
    dtmOverlay.addTo(map);
  }

  map.fitBounds(leafletBounds, { padding: [20, 20] });
  _scheduleLineLoad();
  _scheduleDtmStats();
}

function toggleDtm(show) {
  if (!dtmOverlay) return;
  if (show) dtmOverlay.addTo(map); else map.removeLayer(dtmOverlay);
}

function setDtmOpacity(val) {
  if (dtmOverlay) dtmOverlay.setOpacity(val / 100);
}

// ---------------------------------------------------------------------------
// Viewport-adaptive elevation range
// ---------------------------------------------------------------------------
function _scheduleDtmStats() {
  if (!dtmOverlay) return;
  clearTimeout(statsDebounce);
  statsDebounce = setTimeout(_fetchDtmStats, 1500);
}

let _prewarmTriggered = false;

function _fetchDtmStats() {
  if (!dtmOverlay || !map) return;
  const b = map.getBounds();
  const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(",");
  fetch(`/api/dtm_stats?bbox=${bbox}`)
    .then(r => r.json())
    .then(stats => {
      if (stats.lo === undefined) return;
      // Quantise to nearest 10 m — reduces tile-reload storms while panning
      const lo = Math.round(stats.lo / 10) * 10;
      const hi = Math.round(stats.hi / 10) * 10;
      if (lo !== dtmLo || hi !== dtmHi) {
        dtmLo = lo;
        dtmHi = hi;
        dtmOverlay.setUrl(`/api/dtm_tile/{z}/{x}/{y}.png?lo=${lo}&hi=${hi}`);
      }
      // Fire priority prewarm once per DTM load, after we have a real range
      if (!_prewarmTriggered) {
        _prewarmTriggered = true;
        _triggerPriorityPrewarm(lo, hi);
      }
    })
    .catch(console.error);
}

function _triggerPriorityPrewarm(lo, hi) {
  // Corridors: [lon1, lat1, lon2, lat2]
  // Newark → Chillicothe (main Hopewell Road axis)
  // Newark → High Banks (straight-line road)
  // Newark → Circleville (secondary Hopewell corridor)
  const corridors = [
    [NEWARK[1], NEWARK[0], CHILLICOTHE[1], CHILLICOTHE[0]],
    [NEWARK[1], NEWARK[0], HIGH_BANKS[1], HIGH_BANKS[0]],
  ];
  fetch("/api/prewarm_priority", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corridors, lo, hi }),
  }).catch(console.error);
}

// ---------------------------------------------------------------------------
// Detected lines
// ---------------------------------------------------------------------------
function _scheduleLineLoad() {
  clearTimeout(loadDebounce);
  loadDebounce = setTimeout(_loadLines, 250);
}

function _loadLines() {
  const zoom   = map.getZoom();
  const bounds = map.getBounds();
  const bbox   = [
    bounds.getWest(), bounds.getSouth(),
    bounds.getEast(), bounds.getNorth(),
  ].join(",");

  const hideRoads = document.getElementById("chk-hide-roads").checked ? "1" : "0";
  fetch(`/api/lines?zoom=${zoom}&bbox=${bbox}&hide_roads=${hideRoads}`)
    .then(r => r.json())
    .then(geojson => {
      _rawGeojson = geojson;
      _applyFilters();
      if (geojson.features && geojson.features.length > 0) {
        _prewarmLineTiles(geojson, zoom);
      }
    })
    .catch(console.error);
}

// ---------------------------------------------------------------------------
// Tile pre-warming near detected lines
// ---------------------------------------------------------------------------
function _lngLatToTileXY(lng, lat, z) {
  const n = 1 << z;
  const x = Math.floor((lng + 180) / 360 * n);
  const sinLat = Math.sin(lat * Math.PI / 180);
  const y = Math.floor((0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * n);
  return [x, y];
}

function _ptSegDistM(px, py, ax, ay, bx, by) {
  // Straight-line distance in approximate metres from point (px,py) to
  // segment (ax,ay)→(bx,by), using flat-earth scaling at ~40 °N.
  const LON_M = 85000, LAT_M = 111000;
  const dx = (bx - ax) * LON_M, dy = (by - ay) * LAT_M;
  const lenSq = dx * dx + dy * dy;
  const t = lenSq > 0
    ? Math.max(0, Math.min(1, (((px - ax) * LON_M * dx + (py - ay) * LAT_M * dy)) / lenSq))
    : 0;
  return Math.hypot((px - ax - t * (bx - ax)) * LON_M,
                    (py - ay - t * (by - ay)) * LAT_M);
}

function _prewarmLineTiles(geojson, zoom) {
  if (!dtmOverlay || dtmLo === null) return;

  // tile key → closest distance (m) of any contributing line to the N→C segment
  const tileDist = new Map();

  for (const z of [zoom, zoom + 1]) {
    for (const f of geojson.features) {
      if (f.properties.road) continue;           // skip road-detected lines
      const [s, e] = f.geometry.coordinates;
      const mx = (s[0] + e[0]) / 2;
      const my = (s[1] + e[1]) / 2;
      const dist = Math.min(
        _ptSegDistM(mx, my, NEWARK[1], NEWARK[0], CHILLICOTHE[1], CHILLICOTHE[0]),
        _ptSegDistM(mx, my, NEWARK[1], NEWARK[0], HIGH_BANKS[1], HIGH_BANKS[0]),
      );
      const [tx, ty] = _lngLatToTileXY(mx, my, z);
      const key = `${z}/${tx}/${ty}`;
      if (!tileDist.has(key) || dist < tileDist.get(key)) tileDist.set(key, dist);
    }
  }

  // Sort closest-first, cap at 60
  const sorted = [...tileDist.entries()].sort((a, b) => a[1] - b[1]).slice(0, 60);
  for (const [key] of sorted) {
    fetch(`/api/dtm_tile/${key}.png?lo=${dtmLo}&hi=${dtmHi}`).catch(() => {});
  }
}

// ---------------------------------------------------------------------------
// Corridor overlay
// ---------------------------------------------------------------------------
function _fetchCorridor() {
  fetch("/api/corridor")
    .then(r => r.json())
    .then(feature => {
      if (!feature) return;
      corridorLayer = L.geoJSON(feature, {
        style: {
          color:       "#ccaa22",
          weight:      1.5,
          opacity:     0.55,
          dashArray:   "6 4",
          fillColor:   "#ccaa22",
          fillOpacity: 0.07,
        },
        interactive: false,
      });
      if (document.getElementById("chk-corridor").checked) {
        corridorLayer.addTo(map);
        // Put corridor beneath detected-lines layer
        corridorLayer.bringToBack();
      }
    })
    .catch(console.error);
}

function toggleCorridor(show) {
  if (!corridorLayer) return;
  if (show) { corridorLayer.addTo(map); corridorLayer.bringToBack(); }
  else map.removeLayer(corridorLayer);
}

// ---------------------------------------------------------------------------
// Target-lines overlay — a screen-centred template of the two expected wall
// lines (TARGET_LENGTH_M long, SPACING_METERS apart, oriented at the
// corridor's expected angle at the current view centre, or the static
// TARGET_ANGLE when no corridor is active).
// ---------------------------------------------------------------------------

/**
 * Compute the two parallel template-wall segments centred at [centerLat,
 * centerLon]. `angleDeg` uses the codebase-wide convention (degrees CCW
 * from +x / east, mod 180 — see utils.corridor_angle_at). Degree offsets
 * are scaled by the same flat-earth metres-per-degree approximations used
 * throughout the app (_ptSegDistM, _pair_spacing_m) so the drawn lines are
 * accurate in real-world metres rather than raw degree-space.
 */
function _targetLineSegments(centerLat, centerLon, angleDeg, spacingM, lengthM) {
  const LON_M = 85000, LAT_M = 111320;
  const slope   = Math.tan((90 - angleDeg) * Math.PI / 180);   // dlat/dlon
  const bearing = Math.atan2(slope * LAT_M, LON_M);            // true ground bearing
  const ux = Math.cos(bearing), uy = Math.sin(bearing);        // unit dir along wall (metres)
  const px = -uy, py = ux;                                     // unit perpendicular (metres)

  const half = lengthM / 2;
  const halfSpacing = spacingM / 2;
  const toLatLng = (mx, my) => [centerLat + my / LAT_M, centerLon + mx / LON_M];

  return [-1, 1].map(side => {
    const ox = px * halfSpacing * side;
    const oy = py * halfSpacing * side;
    return [
      toLatLng(ux * half + ox, uy * half + oy),
      toLatLng(-ux * half + ox, -uy * half + oy),
    ];
  });
}

function _scheduleTargetLines() {
  if (!targetOn) return;
  clearTimeout(targetDebounce);
  targetDebounce = setTimeout(_updateTargetLines, 200);
}

function _updateTargetLines() {
  if (!targetOn || !map) return;
  const c = map.getCenter();
  fetch(`/api/expected_angle?lon=${c.lng}`)
    .then(r => r.json())
    .then(({ angle, spacing_m }) => {
      if (!targetOn) return;   // toggled off while the request was in flight
      targetLayer.clearLayers();
      const segs = _targetLineSegments(c.lat, c.lng, angle, spacing_m, TARGET_LENGTH_M);
      segs.forEach(([a, b]) => {
        L.polyline([a, b], { color: "#33ff99", weight: 2.5, opacity: 0.9 }).addTo(targetLayer);
      });
    })
    .catch(console.error);
}

function toggleTargetLines(show) {
  targetOn = show;
  const chk = document.getElementById("chk-target");
  if (chk) chk.checked = show;
  if (show) {
    targetLayer.addTo(map);
    _updateTargetLines();
  } else {
    targetLayer.clearLayers();
    map.removeLayer(targetLayer);
  }
  _syncTargetBtn();
}

// ---------------------------------------------------------------------------
// Client-side line filters (angle deviation, corridor distance, wall spacing)
// ---------------------------------------------------------------------------

/**
 * Signed deviation of `actual` from `target`, accounting for the
 * 180°-periodicity of undirected line orientation (a line and its reverse
 * are the same line). Range: (-90, 90].  Positive = rotated clockwise from
 * the target; negative = counter-clockwise.
 */
function _angleSignedDiff(actual, target) {
  let d = (actual - target) % 180;
  if (d <= -90) d += 180;
  else if (d > 90) d -= 180;
  return d;
}

// Keep a dual-handle slider's lo <= hi by clamping the handle that just moved.
function _clampDualSlider(loEl, hiEl, moved) {
  let lo = parseFloat(loEl.value);
  let hi = parseFloat(hiEl.value);
  if (moved === "lo" && lo > hi) { lo = hi; loEl.value = lo; }
  if (moved === "hi" && hi < lo) { hi = lo; hiEl.value = hi; }
  return [lo, hi];
}

function onAngleFilterChange(moved) {
  const loEl = document.getElementById("filter-angle-lo");
  const hiEl = document.getElementById("filter-angle-hi");
  const [lo, hi] = _clampDualSlider(loEl, hiEl, moved);
  document.getElementById("val-angle-tol").textContent = `${lo}° to ${hi}°`;
  _applyFilters();
}

function onCorridorDistChange() {
  const dist = parseInt(document.getElementById("filter-dist").value);
  document.getElementById("val-dist").textContent = dist.toLocaleString() + " m";
  _applyFilters();
}

function onPairDistFilterChange(moved) {
  const loEl = document.getElementById("filter-pairdist-lo");
  const hiEl = document.getElementById("filter-pairdist-hi");
  const [lo, hi] = _clampDualSlider(loEl, hiEl, moved);
  document.getElementById("val-pairdist").textContent = `${lo} m to ${hi} m`;
  _applyFilters();
}

function _applyFilters() {
  linesLayer.clearLayers();
  if (!_rawGeojson || !_rawGeojson.features) {
    document.getElementById("info-lines").textContent = "0";
    return;
  }

  const angleLo  = parseFloat(document.getElementById("filter-angle-lo").value);
  const angleHi  = parseFloat(document.getElementById("filter-angle-hi").value);
  const corrDist = parseFloat(document.getElementById("filter-dist").value);
  const pairLo   = parseFloat(document.getElementById("filter-pairdist-lo").value);
  const pairHi   = parseFloat(document.getElementById("filter-pairdist-hi").value);

  const features = _rawGeojson.features.filter(f => {
    const p = f.properties;
    if (p.expected_angle != null) {
      const dev = _angleSignedDiff(p.angle, p.expected_angle);
      if (dev < angleLo || dev > angleHi) return false;
    }
    if (p.corridor_dist_m != null && p.corridor_dist_m > corrDist) return false;
    if (p.pair_distance_m != null && (p.pair_distance_m < pairLo || p.pair_distance_m > pairHi)) return false;
    return true;
  });

  if (features.length > 0) {
    const zoom   = map.getZoom();
    const weight = _lineWeight(zoom);
    L.geoJSON({ type: "FeatureCollection", features }, {
      style: () => ({ color: "#ff5533", weight, opacity: 0.85, lineCap: "butt" }),
    }).addTo(linesLayer);
  }
  document.getElementById("info-lines").textContent = features.length.toLocaleString();
}

function _lineWeight(zoom) {
  if (zoom < 12) return 1;
  if (zoom < 14) return 1.5;
  if (zoom < 16) return 2;
  return 2.5;
}

function toggleLines(show) {
  if (show) linesLayer.addTo(map); else map.removeLayer(linesLayer);
}

// ---------------------------------------------------------------------------
// Reference sites – editable coordinates
// ---------------------------------------------------------------------------
function _populateSiteInputs() {
  document.getElementById("site-newark-lat").value   = NEWARK[0];
  document.getElementById("site-newark-lon").value   = NEWARK[1];
  document.getElementById("site-chilli-lat").value   = CHILLICOTHE[0];
  document.getElementById("site-chilli-lon").value   = CHILLICOTHE[1];
  document.getElementById("site-circle-lat").value   = CIRCLEVILLE[0];
  document.getElementById("site-circle-lon").value   = CIRCLEVILLE[1];
  document.getElementById("site-highbanks-lat").value = HIGH_BANKS[0];
  document.getElementById("site-highbanks-lon").value = HIGH_BANKS[1];
}

function applySites() {
  const parse = id => parseFloat(document.getElementById(id).value);
  const newark    = [parse("site-newark-lat"), parse("site-newark-lon")];
  const chilli    = [parse("site-chilli-lat"), parse("site-chilli-lon")];
  const circle    = [parse("site-circle-lat"), parse("site-circle-lon")];
  const highBanks = [parse("site-highbanks-lat"), parse("site-highbanks-lon")];

  if ([...newark, ...chilli, ...circle, ...highBanks].some(isNaN)) return;

  NEWARK      = newark;
  CHILLICOTHE = chilli;
  CIRCLEVILLE = circle;
  HIGH_BANKS  = highBanks;
  _drawReferenceLines();
}

// ---------------------------------------------------------------------------
// Route waypoints (coords.json)
// ---------------------------------------------------------------------------
function _loadRouteWaypoints() {
  fetch("/static/coords.json")
    .then(r => r.json())
    .then(pts => {
      ROUTE_WAYPOINTS = pts;
      _drawReferenceLines();
    })
    .catch(console.error);
}

// ---------------------------------------------------------------------------
// Reference lines
// ---------------------------------------------------------------------------
function _drawReferenceLines() {
  if (!NEWARK || !CHILLICOTHE || !CIRCLEVILLE || !HIGH_BANKS) return;
  refLayer.clearLayers();

  const dashOpts = { weight: 1.5, opacity: 0.9, dashArray: "10 7" };

  if (ROUTE_WAYPOINTS.length >= 2) {
    // Polyline through the coords.json waypoints
    const latLngs = ROUTE_WAYPOINTS.map(p => [p.lat, p.lon]);
    L.polyline(latLngs, { ...dashOpts, color: "#44aaff" }).addTo(refLayer);

    // Named marker at each waypoint
    ROUTE_WAYPOINTS.forEach(p => {
      L.circleMarker([p.lat, p.lon], {
        radius: 5,
        color: "#44aaff",
        fillColor: "#1a3a5c",
        fillOpacity: 0.9,
        weight: 1.5,
      })
        .bindTooltip(
          `<b>${p.name}</b>${p.note ? "<br><span style='color:#aaa'>" + p.note + "</span>" : ""}`,
          { sticky: true, className: "wp-tooltip" }
        )
        .addTo(refLayer);
    });
  } else {
    // Fallback if waypoints haven't loaded yet
    L.polyline([NEWARK, CHILLICOTHE], { ...dashOpts, color: "#44aaff" }).addTo(refLayer);
  }

  // Newark → High Banks (straight-line road, treated like Newark → Chillicothe)
  L.polyline([NEWARK, HIGH_BANKS], { ...dashOpts, color: "#cc66ff" }).addTo(refLayer);
}

/**
 * Sample a quadratic Bézier arc between two [lat, lon] points.
 * The control point is the geographic midpoint displaced northward by
 * (curveRatio × great-circle distance).
 */
function _bezierArch(p1, p2, curveRatio, nPts = 120) {
  const midLat = (p1[0] + p2[0]) / 2;
  const midLon = (p1[1] + p2[1]) / 2;

  const dLat = p2[0] - p1[0];
  const dLon = p2[1] - p1[1];
  const dist = Math.sqrt(dLat * dLat + dLon * dLon);

  // Control point displaced northward (upward on map) from midpoint
  const ctrl = [midLat + dist * curveRatio, midLon];

  const pts = [];
  for (let i = 0; i <= nPts; i++) {
    const t   = i / nPts;
    const mt  = 1 - t;
    const lat = mt * mt * p1[0] + 2 * mt * t * ctrl[0] + t * t * p2[0];
    const lon = mt * mt * p1[1] + 2 * mt * t * ctrl[1] + t * t * p2[1];
    pts.push([lat, lon]);
  }
  return pts;
}

function toggleRef(show) {
  if (show) refLayer.addTo(map); else map.removeLayer(refLayer);
}

// ---------------------------------------------------------------------------
// Info panel
// ---------------------------------------------------------------------------
function _updateInfoPanel() {
  if (!map) return;
  document.getElementById("info-zoom").textContent = map.getZoom();
  const c = map.getCenter();
  document.getElementById("info-center").textContent =
    `${c.lat.toFixed(7)}, ${c.lng.toFixed(7)}`;
}

function _onMapMouseMove(e) {
  clearTimeout(cursorElevDebounce);
  cursorElevDebounce = setTimeout(() => _fetchCursorElevation(e.latlng), 80);
}

function _onMapMouseOut() {
  clearTimeout(cursorElevDebounce);
  const el = document.getElementById("info-cursor-elev");
  if (el) el.textContent = "—";
}

function _fetchCursorElevation(latlng) {
  fetch(`/api/elevation?lon=${latlng.lng}&lat=${latlng.lat}`)
    .then(r => r.json())
    .then(d => {
      const el = document.getElementById("info-cursor-elev");
      if (!el) return;
      el.textContent = d.elevation != null ? `${d.elevation.toFixed(1)} m` : "—";
    })
    .catch(() => {});
}

function copyCenter() {
  const text = document.getElementById("info-center").textContent;
  if (!text || text === "—") return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector(".copy-btn");
    const orig = btn.textContent;
    btn.textContent = "✓";
    setTimeout(() => { btn.textContent = orig; }, 1200);
  });
}

// ---------------------------------------------------------------------------
// Map-control toggle button (floating, top-right of map)
// ---------------------------------------------------------------------------
function _addMapControls() {
  const LinesToggleControl = L.Control.extend({
    options: { position: "topright" },
    onAdd() {
      const btn = L.DomUtil.create("button", "map-toggle-btn");
      btn.id    = "map-lines-btn";
      btn.title = "Toggle detected lines  [L]";
      btn.innerHTML = "Lines";
      btn.setAttribute("data-active", "true");

      L.DomEvent.on(btn, "click", L.DomEvent.stopPropagation);
      L.DomEvent.on(btn, "click", () => {
        const chk = document.getElementById("chk-lines");
        chk.checked = !chk.checked;
        toggleLines(chk.checked);
        _syncToggleBtn();
      });
      return btn;
    },
  });

  new LinesToggleControl().addTo(map);

  const TargetToggleControl = L.Control.extend({
    options: { position: "topright" },
    onAdd() {
      const btn = L.DomUtil.create("button", "map-toggle-btn");
      btn.id    = "map-target-btn";
      btn.title = "Toggle target lines  [T]";
      btn.innerHTML = "Target";
      btn.setAttribute("data-active", "false");

      L.DomEvent.on(btn, "click", L.DomEvent.stopPropagation);
      L.DomEvent.on(btn, "click", () => toggleTargetLines(!targetOn));
      return btn;
    },
  });

  new TargetToggleControl().addTo(map);
}

function _syncToggleBtn() {
  const btn    = document.getElementById("map-lines-btn");
  const active = document.getElementById("chk-lines").checked;
  if (btn) btn.setAttribute("data-active", active ? "true" : "false");
}

function _syncTargetBtn() {
  const btn = document.getElementById("map-target-btn");
  if (btn) btn.setAttribute("data-active", targetOn ? "true" : "false");
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
function _addKeyboardShortcuts() {
  document.addEventListener("keydown", e => {
    // Ignore when typing in an input
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.key === "l" || e.key === "L") {
      const chk = document.getElementById("chk-lines");
      chk.checked = !chk.checked;
      toggleLines(chk.checked);
      _syncToggleBtn();
    }
    if (e.key === "r" || e.key === "R") {
      const chk = document.getElementById("chk-ref");
      chk.checked = !chk.checked;
      toggleRef(chk.checked);
    }
    if (e.key === "d" || e.key === "D") {
      const chk = document.getElementById("chk-dtm");
      chk.checked = !chk.checked;
      toggleDtm(chk.checked);
    }
    if (e.key === "t" || e.key === "T") {
      toggleTargetLines(!targetOn);
    }
  });
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function _setStatus(key, msg) {
  const el = document.getElementById(`${key}-status`);
  if (el) el.textContent = msg;
}

// ---------------------------------------------------------------------------
// Server status polling (OSM road detection + error reporting)
// ---------------------------------------------------------------------------
let _statusPollTimer = null;

function _startStatusPoll() {
  clearTimeout(_statusPollTimer);
  _pollStatus();
}

function _pollStatus() {
  fetch("/api/status")
    .then(r => r.json())
    .then(s => {
      _renderOsmStatus(s.osm_status, s.osm_road_count);
      _renderErrors(s.errors || []);
      // Keep polling while detection is in progress
      if (s.osm_status === "pending") {
        _statusPollTimer = setTimeout(_pollStatus, 2500);
      }
    })
    .catch(() => {});
}

function _renderOsmStatus(status, roadCount) {
  const el = document.getElementById("osm-status");
  if (!el) return;
  el.className = "status " + status;
  if (status === "idle")    { el.textContent = ""; }
  if (status === "pending") { el.textContent = "⟳ Detecting roads via OSM…"; }
  if (status === "done")    { el.textContent = `✓ ${roadCount} road segment${roadCount !== 1 ? "s" : ""} matched`; }
  if (status === "failed")  { el.textContent = "✗ OSM road detection failed — check errors"; }
}

function _renderErrors(errors) {
  const panel = document.getElementById("error-panel");
  const list  = document.getElementById("error-list");
  if (!panel || !list) return;
  if (!errors.length) {
    panel.style.display = "none";
    return;
  }
  panel.style.display = "";
  list.innerHTML = errors.map(e => `<div>${e}</div>`).join("");
}
