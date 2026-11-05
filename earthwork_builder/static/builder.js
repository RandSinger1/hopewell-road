// Earthwork Builder — site list, hillshade overlay and polygon editing.
"use strict";

const map = L.map("map", { maxZoom: 21 }).setView([40.0, -82.6], 8);

const baseLayers = {
  "OpenStreetMap": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 21, maxNativeZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }),
  "Esri imagery": L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 21, maxNativeZoom: 19, attribution: "Esri World Imagery",
    }),
};
baseLayers["OpenStreetMap"].addTo(map);
L.control.layers(baseLayers).addTo(map);

map.pm.addControls({
  position: "topleft",
  drawMarker: false, drawCircleMarker: false, drawPolyline: false, drawCircle: false,
  drawText: false, drawRectangle: false, rotateMode: false,
  drawPolygon: true, editMode: true, dragMode: true, cutPolygon: false, removalMode: true,
});
map.pm.setPathOptions({ color: "#ffb000", weight: 2, fillOpacity: 0.12 });
// Double-click finishes an outline (Geoman's default is only "click the first point")
map.pm.setGlobalOptions({ finishOn: "dblclick", continueDrawing: false });

const labelLayer = L.featureGroup().addTo(map);
let hillshade = null;
let windowRect = null;
let current = null;          // site detail from /api/sites/<slug>
let dirty = false;
let sites = [];              // last /api/sites listing, for "Next site"
let drawing = false;
let lastCreated = null;      // {layer, hole} most recently finished outline / interior (Backspace reopens it)
let reopened = null;         // {ring, props, holeOf} of an outline / interior taken back into draw mode
let interiorOf = null;       // outline layer whose interior is being drawn
let selectedLayer = null;
let inset = null;            // {layer, ring, preview, seq} shrunk-copy interior being adjusted

const $ = (id) => document.getElementById(id);

function setDirty(v) {
  dirty = v;
  $("dirty").textContent = v ? "unsaved changes" : "";
}

async function api(url, opts = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

// ── Site list ────────────────────────────────────────────────────────────────

async function refreshSites() {
  sites = await api("/api/sites");
  const ul = $("site-list");
  ul.innerHTML = "";
  for (const s of sites) {
    const li = document.createElement("li");
    li.className = current && current.slug === s.slug ? "active" : "";
    const state = s.job === "running" ? "downloading…" : s.job === "queued" ? "queued…"
      : !s.prepared ? (s.job === "failed" ? "prepare failed" : (s.stale ? "moved - re-prepare" : "not prepared"))
      : s.rejected ? "rejected ✗"
      : `${s.labels} outline${s.labels === 1 ? "" : "s"}${s.reviewed ? " ✓" : ""}`;
    if (s.rejected) li.classList.add("rejected");
    const bg = s.background ? `<span class="bg-tag" title="Background site: no known earthwork">bg</span>` : "";
    li.innerHTML = `<span>${escapeHtml(s.name)}${bg}</span><span class="muted">${state}</span>`;
    li.onclick = () => selectSite(s.slug);
    ul.appendChild(li);
  }
  return sites;
}

async function selectSite(slug) {
  if (drawing) cancelDraw();
  if (dirty && !confirm("Discard unsaved changes?")) return;
  closeInset();
  current = await api(`/api/sites/${slug}`);
  setDirty(false);
  $("site-panel").hidden = false;
  $("site-title").textContent = current.name;
  $("site-notes").textContent = current.notes || "";
  $("reviewed").checked = !!current.reviewed;
  showRejected(current.rejected, current.reject_reason);

  if (hillshade) map.removeLayer(hillshade);
  if (windowRect) map.removeLayer(windowRect);
  labelLayer.clearLayers();

  if (current.prepared) {
    const b = current.bounds;
    const bounds = L.latLngBounds([b.south, b.west], [b.north, b.east]);
    hillshade = L.tileLayer(tileUrl(slug), {
      maxZoom: 21, bounds, opacity: +$("opacity").value,
    }).addTo(map);
    windowRect = L.rectangle(bounds, { color: "#4cc9f0", weight: 1, fill: false, interactive: false }).addTo(map);
    map.fitBounds(bounds);
    $("job").textContent = "";
  } else {
    map.setView([current.lat, current.lon], 16);
    pollJob(slug);
  }

  L.geoJSON(current.features, {
    onEachFeature: (f, layer) => addLabelLayer(layer, f.properties || {}),
  });
  await refreshSites();
  renderLabelList();
}

async function pollJob(slug) {
  const job = await api(`/api/sites/${slug}/job`);
  if (!current || current.slug !== slug) return;
  $("job").textContent = job.status ? `${job.status}: ${job.log.slice(-1)[0] || ""}` : (current.stale ? "Coords or size changed since it was prepared." : "Not prepared.");
  if (job.status === "running" || job.status === "queued") setTimeout(() => pollJob(slug), 3000);
  else if (job.status === "done") selectSite(slug);
  else if (!job.status) {
    const btn = document.createElement("button");
    btn.textContent = current.stale ? "Re-download LiDAR" : "Download LiDAR";
    btn.onclick = async () => { await api(`/api/sites/${slug}/prepare`, { method: "POST" }); pollJob(slug); };
    $("job").appendChild(btn);
  }
}

// ── Outlines ─────────────────────────────────────────────────────────────────

// Colour by role: earthwork (target), look-alike (hard negative), unverified candidate
function typeColor(type) {
  if (type === EW_UNVERIFIED) return "#f72585";
  return EW_NEGATIVES.includes(type) ? "#4cc9f0" : "#ffb000";
}

function restyle(layer) {
  layer.setStyle({ color: typeColor(layer.feature.properties.type), weight: 2, fillOpacity: 0.12 });
}

function addLabelLayer(layer, props) {
  layer.feature = { type: "Feature", properties: { type: props.type || EW_TYPES[0], notes: props.notes || "" } };
  restyle(layer);
  layer.on("pm:edit pm:dragend", () => setDirty(true));
  layer.on("click", () => { if (!drawing) selectLabel(layer); });
  labelLayer.addLayer(layer);
}

map.on("pm:create", (e) => {
  if (!current) { map.removeLayer(e.layer); return; }
  map.removeLayer(e.layer);
  if (interiorOf) { addInterior(interiorOf, e.layer.getLatLngs()[0]); return; }
  const props = reopened ? reopened.props : {};
  reopened = null;
  addLabelLayer(e.layer, props);
  lastCreated = { layer: e.layer, hole: false };
  setDirty(true);
  renderLabelList();
  selectLabel(e.layer, { openPicker: !props.type });
  toast("Outline added - pick its type, then Save (Ctrl+S). Backspace reopens it");
});

map.on("pm:remove", (e) => {
  if (lastCreated && e.layer === lastCreated.layer) lastCreated = null;
  labelLayer.removeLayer(e.layer);
  setDirty(true);
  renderLabelList();
});

function highlight(layer) {
  labelLayer.eachLayer((l) => l.setStyle({ weight: l === layer ? 4 : 2 }));
}

// Highlight an outline and its sidebar row (optionally focusing the type picker)
function selectLabel(layer, { openPicker = false } = {}) {
  selectedLayer = layer;
  highlight(layer);
  const i = labelLayer.getLayers().indexOf(layer);
  document.querySelectorAll("#label-list li").forEach((li, j) => li.classList.toggle("selected", j === i));
  const row = document.querySelectorAll("#label-list li")[i];
  if (!row) return;
  row.scrollIntoView({ block: "nearest" });
  if (openPicker) {
    const sel = row.querySelector("select");
    sel.focus();
    if (sel.showPicker) { try { sel.showPicker(); } catch (_) { /* needs a user gesture in some browsers */ } }
  }
}

// ── Drawing ──────────────────────────────────────────────────────────────────

function startDraw() {
  if (!current || !current.prepared) { toast("Pick a prepared site first"); return; }
  if (drawing) return;
  closeInset();
  map.pm.disableGlobalEditMode();
  map.pm.enableDraw("Polygon");
}

function drawPoints() {
  const l = map.pm.Draw.Polygon._layer;
  return l ? l.getLatLngs().length : 0;
}

function finishDraw() {
  if (drawPoints() < 3) { toast("An outline needs at least 3 points"); return; }
  map.pm.Draw.Polygon._finishShape();
}

function undoPoint() {
  // Removing the only point ends drawing; that is not a cancel, so don't restore
  if (drawPoints() <= 1) reopened = null;
  if (drawPoints() > 0) map.pm.Draw.Polygon._removeLastVertex();
  updateDrawCount();
}

// Undo after an outline was finished: remove the closing edge, i.e. take the
// most recently drawn outline back into draw mode with all its vertices.
function reopenLast() {
  const last = lastCreated;
  lastCreated = null;
  if (!last || !labelLayer.hasLayer(last.layer)) return false;
  const rings = last.layer.getLatLngs();
  if (last.hole) {
    // Interior: take the last ring back off its outline and redraw it
    if (rings.length < 2) return false;
    const ring = rings.pop();
    last.layer.setLatLngs(rings);
    reopened = { ring, holeOf: last.layer };
    interiorOf = last.layer;
  } else {
    reopened = { ring: rings[0], holes: rings.slice(1), props: { ...last.layer.feature.properties } };
    labelLayer.removeLayer(last.layer);
  }
  renderLabelList();
  setDirty(true);
  map.pm.disableGlobalEditMode();
  map.pm.enableDraw("Polygon");
  const draw = map.pm.Draw.Polygon;
  for (const latlng of reopened.ring) {
    draw._hintMarker._snapped = false;   // place exactly at the stored vertex
    draw._createVertex({ latlng });
  }
  updateDrawCount();
  return true;
}

function cancelDraw() {
  map.pm.disableDraw();
}

// Drawing a reopened outline / interior ended without finishing it: put it back as it was
function restoreReopened() {
  if (!reopened) return;
  const r = reopened;
  reopened = null;
  if (r.holeOf) {
    if (labelLayer.hasLayer(r.holeOf)) {
      r.holeOf.setLatLngs([...r.holeOf.getLatLngs(), r.ring]);
      lastCreated = { layer: r.holeOf, hole: true };
    }
  } else {
    const layer = L.polygon([r.ring, ...r.holes]);
    addLabelLayer(layer, r.props);
    lastCreated = { layer, hole: false };
  }
  renderLabelList();
}

// ── Interiors (holes): an enclosure's inside, cut out of its outline ─────────

function startInterior(layer) {
  if (!layer || !labelLayer.hasLayer(layer)) { toast("Select an outline first (click it)"); return; }
  if (drawing) return;
  closeInset();
  selectLabel(layer);
  interiorOf = layer;
  map.pm.disableGlobalEditMode();
  map.pm.enableDraw("Polygon");
}

function insideRing(ll, ring) {           // ray casting in lon/lat (fine at site scale)
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const a = ring[i], b = ring[j];
    if ((a.lat > ll.lat) !== (b.lat > ll.lat)
        && ll.lng < ((b.lng - a.lng) * (ll.lat - a.lat)) / (b.lat - a.lat) + a.lng) inside = !inside;
  }
  return inside;
}

function addInterior(layer, ring) {
  const [outer, ...holes] = layer.getLatLngs();
  const r = reopened;
  reopened = null;
  const ok = ring.every((p) => insideRing(p, outer))
    && holes.every((h) => ring.every((p) => !insideRing(p, h)) && h.every((p) => !insideRing(p, ring)));
  if (!ok) {
    if (r && r.holeOf) { reopened = r; }    // pm:drawend puts the old interior back
    toast("An interior must lie inside its outline and not overlap another interior");
    return false;
  }
  layer.setLatLngs([outer, ...holes, ring]);
  lastCreated = { layer, hole: true };
  setDirty(true);
  renderLabelList();
  selectLabel(layer);
  toast("Interior added - it is cut out of the outline. Backspace reopens it");
  return true;
}

// Shrunk copy: the outline inset by a wall width (server-side buffer, in metres)
// and optionally shifted, previewed dashed until applied.  Starts identical.
const M_PER_DEG = 111320;
function shiftRing(ring, dx, dy) {
  return ring.map((p) => L.latLng(p.lat + dy / M_PER_DEG,
                                  p.lng + dx / (M_PER_DEG * Math.cos((p.lat * Math.PI) / 180))));
}

function startInset(layer) {
  if (!layer || !labelLayer.hasLayer(layer)) { toast("Select an outline first (click it)"); return; }
  if (drawing) return;
  closeInset();
  selectLabel(layer);
  map.pm.disableGlobalEditMode();
  const b = layer.getBounds();
  const w = map.distance(b.getSouthWest(), b.getSouthEast());
  const h = map.distance(b.getSouthWest(), b.getNorthWest());
  const maxInset = Math.max(1, Math.floor(Math.min(w, h) / 2));
  const maxShift = Math.max(10, Math.ceil(maxInset / 2));
  $("inset-m").max = maxInset;
  for (const id of ["shift-x", "shift-y"]) { $(id).min = -maxShift; $(id).max = maxShift; }
  for (const id of ["inset-m", "shift-x", "shift-y"]) $(id).value = 0;
  const outer = layer.getLatLngs()[0];
  inset = { layer, ring: outer.slice(), seq: 0,
            preview: L.polygon(outer, { color: "#ffffff", weight: 2, dashArray: "6 4",
                                        fill: false, interactive: false, pmIgnore: true }).addTo(map) };
  $("inset-title").textContent = `Interior of #${labelLayer.getLayers().indexOf(layer)}`;
  $("inset-panel").hidden = false;
  renderInset();
}

async function updateInset() {
  if (!inset) return;
  const cur = inset, seq = ++cur.seq;
  const d = Number($("inset-m").value);
  const outer = cur.layer.getLatLngs()[0];
  let ring = outer.slice();
  if (d > 0) {
    let res;
    try {
      res = await api("/api/inset", {
        method: "POST", body: JSON.stringify({ ring: outer.map((p) => [p.lng, p.lat]), inset_m: d }) });
    } catch (err) { toast(`Inset failed: ${err.message}`); return; }
    if (inset !== cur || seq !== cur.seq) return;   // superseded by a newer slider value
    ring = res.ring ? res.ring.slice(0, -1).map(([lng, lat]) => L.latLng(lat, lng)) : null;
  }
  cur.ring = ring;
  renderInset();
}

function renderInset() {
  if (!inset) return;
  const dx = Number($("shift-x").value), dy = Number($("shift-y").value);
  $("inset-val").textContent = `${$("inset-m").value} m`;
  $("shift-x-val").textContent = `${dx} m`;
  $("shift-y-val").textContent = `${dy} m`;
  inset.preview.setLatLngs(inset.ring ? shiftRing(inset.ring, dx, dy) : []);
  $("inset-apply").disabled = !inset.ring;
}

function applyInset() {
  if (!inset) return;
  if (!inset.ring) { toast("Wall width too large - nothing is left inside"); return; }
  const dx = Number($("shift-x").value), dy = Number($("shift-y").value);
  if (Number($("inset-m").value) === 0 && dx === 0 && dy === 0) {
    toast("Still identical to the outline - give it a wall width first"); return;
  }
  if (addInterior(inset.layer, shiftRing(inset.ring, dx, dy))) closeInset();
}

function closeInset() {
  if (!inset) return;
  map.removeLayer(inset.preview);
  inset = null;
  $("inset-panel").hidden = true;
}

function nudgeInset(id, step) {
  const el = $(id);
  el.value = Math.min(Number(el.max), Math.max(Number(el.min), Number(el.value) + step));
  if (id === "inset-m") updateInset(); else renderInset();
}

$("inset-m").oninput = updateInset;
$("shift-x").oninput = renderInset;
$("shift-y").oninput = renderInset;
$("inset-apply").onclick = applyInset;
$("inset-cancel").onclick = closeInset;

function removeLastInterior(layer) {
  const rings = layer.getLatLngs();
  if (rings.length < 2) return;
  layer.setLatLngs(rings.slice(0, -1));
  if (lastCreated && lastCreated.layer === layer) lastCreated = null;
  setDirty(true);
  renderLabelList();
}

function updateDrawCount() {
  const n = drawPoints();
  $("draw-count").textContent = `${n} point${n === 1 ? "" : "s"}${n > 0 && n < 3 ? " (need 3)" : ""}`;
  $("draw-finish").disabled = n < 3;
}

map.on("pm:drawstart", (e) => {
  drawing = true;
  $("draw-title").textContent = interiorOf
    ? `Drawing interior of #${labelLayer.getLayers().indexOf(interiorOf)}` : "Drawing outline";
  $("draw-hint").hidden = false;
  $("draw").classList.add("active");
  updateDrawCount();
  e.workingLayer.on("pm:vertexadded pm:vertexremoved", updateDrawCount);
});

map.on("pm:drawend", () => {
  drawing = false;
  restoreReopened();   // no-op after a finish: pm:create fires first and clears it
  interiorOf = null;
  $("draw-hint").hidden = true;
  $("draw").classList.remove("active");
});

$("draw").onclick = startDraw;
$("draw-finish").onclick = finishDraw;
$("draw-undo").onclick = undoPoint;
$("draw-cancel").onclick = cancelDraw;

// ── Keyboard ─────────────────────────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
  if (drawing) {
    if (e.key === "Enter") { e.preventDefault(); finishDraw(); }
    else if (e.key === "Backspace" || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z")) { e.preventDefault(); undoPoint(); }
    else if (e.key === "Escape") { e.preventDefault(); cancelDraw(); }
    return;
  }
  if (inset && !(typing && e.target.type !== "range")) {   // shrunk-copy panel: its keys win over a focused slider
    const step = e.shiftKey ? 5 : 1;
    const keys = { ArrowRight: ["shift-x", step], ArrowLeft: ["shift-x", -step],
                   ArrowUp: ["shift-y", step], ArrowDown: ["shift-y", -step],
                   "+": ["inset-m", step / 2], "=": ["inset-m", step / 2], "-": ["inset-m", -step / 2] };
    if (e.key === "Enter") { e.preventDefault(); applyInset(); }
    else if (e.key === "Escape") { e.preventDefault(); closeInset(); }
    else if (keys[e.key]) { e.preventDefault(); nudgeInset(...keys[e.key]); }
    return;
  }
  const undoKey =e.key === "Backspace" || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z");
  // The type picker is focused right after finishing, so allow undo from a SELECT
  if (undoKey && (!typing || e.target.tagName === "SELECT") && reopenLast()) {
    e.preventDefault();
    if (typing) e.target.blur();
    return;
  }
  if (typing) { if (e.key === "Escape") e.target.blur(); return; }
  if (e.key === "d" || e.key === "D") { e.preventDefault(); startDraw(); }
  else if (e.key === "i" || e.key === "I") { e.preventDefault(); startInterior(selectedLayer); }
  else if (e.key === "o" || e.key === "O") { e.preventDefault(); startInset(selectedLayer); }
  else if (e.key === "n" || e.key === "N") { e.preventDefault(); nextSite(); }
  else if (e.key === "v" || e.key === "V") { e.preventDefault(); cycleView(e.shiftKey ? -1 : 1); }
  else if (e.key === "r" || e.key === "R") { e.preventDefault(); openReject(); }
});

// Next prepared site that is not yet reviewed (wrapping around the list)
function nextSite() {
  const start = current ? sites.findIndex((s) => s.slug === current.slug) : -1;
  for (let k = 1; k <= sites.length; k++) {
    const s = sites[(start + k) % sites.length];
    if (s.prepared && !s.reviewed && !s.rejected) { selectSite(s.slug); return; }
  }
  toast("No prepared, unreviewed sites left");
}
$("next").onclick = nextSite;

let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2500);
}

function renderLabelList() {
  const ul = $("label-list");
  ul.innerHTML = "";
  labelLayer.getLayers().forEach((layer, i) => {
    const p = layer.feature.properties;
    const li = document.createElement("li");
    const opt = (t) => `<option ${t === p.type ? "selected" : ""}>${escapeHtml(t)}</option>`;
    const opts = `<optgroup label="Earthwork">${EW_TYPES.map(opt).join("")}</optgroup>`
      + `<optgroup label="Look-alike (not an earthwork)">${EW_NEGATIVES.map(opt).join("")}</optgroup>`
      + (p.type === EW_UNVERIFIED ? `<optgroup label="Candidate">${opt(EW_UNVERIFIED)}</optgroup>` : "");
    const nHoles = layer.getLatLngs().length - 1;
    li.innerHTML = `<b>#${i}</b> <select>${opts}</select>
      <input placeholder="notes" value="${escapeHtml(p.notes)}" />
      <button title="Zoom">⌖</button><button title="Delete">✕</button>
      <button title="Interior from a shrunk copy of the outline (O)">⊚</button>
      <button title="Draw interior (I) - cut out of the outline">◎</button>
      ${nHoles ? `<button title="Remove last interior">◎✕</button>
        <span class="muted">${nHoles} interior${nHoles === 1 ? "" : "s"}</span>` : ""}`;
    const [sel, notes] = li.querySelectorAll("select, input");
    const [zoomBtn, delBtn, insetBtn, holeBtn, unholeBtn] = li.querySelectorAll("button");
    insetBtn.onclick = () => startInset(layer);
    holeBtn.onclick = () => startInterior(layer);
    if (unholeBtn) unholeBtn.onclick = () => removeLastInterior(layer);
    sel.onchange = () => { p.type = sel.value; restyle(layer); setDirty(true); renderLabelList(); };
    notes.oninput = () => { p.notes = notes.value; setDirty(true); };
    zoomBtn.onclick = () => { map.fitBounds(layer.getBounds().pad(0.5)); selectLabel(layer); };
    li.onclick = (ev) => { if (ev.target === li || ev.target.tagName === "B") selectLabel(layer); };
    delBtn.onclick = () => { if (inset && inset.layer === layer) closeInset(); labelLayer.removeLayer(layer); map.removeLayer(layer); setDirty(true); renderLabelList(); };
    ul.appendChild(li);
  });
}

$("reviewed").onchange = () => setDirty(true);

async function save() {
  if (!current || drawing) return;
  const fc = labelLayer.toGeoJSON();
  fc.reviewed = $("reviewed").checked;
  fc.rejected = !!current.rejected;
  fc.reject_reason = current.reject_reason || "";
  try {
    await api(`/api/sites/${current.slug}/labels`, { method: "PUT", body: JSON.stringify(fc) });
    setDirty(false);
    toast(fc.reviewed ? "Saved - site marked reviewed" : "Saved");
    refreshSites();
  } catch (err) {
    alert(`Save failed: ${err.message}`);
  }
}
$("save").onclick = save;

// ── Terrain view ─────────────────────────────────────────────────────────────

let view = localStorageGet("ew_view") || EW_VIEWS[0];

function localStorageGet(k) { try { return localStorage.getItem(k); } catch (_) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } }

function tileUrl(slug) {
  // The ground.tif version is part of the URL: a rebuilt site never shows cached tiles
  const v = current && current.slug === slug ? current.ground_token || "" : "";
  return `/api/sites/${slug}/tile/{z}/{x}/{y}.png?view=${view}&g=${v}`;
}

function setView(v) {
  view = v;
  localStorageSet("ew_view", v);
  document.querySelectorAll('input[name="view"]').forEach((r) => { r.checked = r.value === v; });
  if (hillshade && current) hillshade.setUrl(tileUrl(current.slug));
}

function cycleView(step) {
  const i = EW_VIEWS.indexOf(view);
  const v = EW_VIEWS[(i + step + EW_VIEWS.length) % EW_VIEWS.length];
  setView(v);
  toast(document.querySelector(`input[name="view"][value="${v}"]`).parentElement.textContent.trim());
}

document.querySelectorAll('input[name="view"]').forEach((r) => { r.onchange = () => setView(r.value); });
setView(EW_VIEWS.includes(view) ? view : EW_VIEWS[0]);

// ── Reject a site (sample not good enough) ───────────────────────────────────

function showRejected(rejected, reason) {
  $("rejected-banner").hidden = !rejected;
  $("rejected-reason").textContent = reason || "";
  $("reviewed").disabled = !!rejected;
  $("reject-box").hidden = !!rejected;
  if (rejected) $("reject-box").open = false;
}

function openReject() {
  if (!current || current.rejected) return;
  $("reject-box").open = true;
  $("reject-reason").focus();
}

async function setRejected(rejected) {
  if (!current) return;
  current.rejected = rejected;
  current.reject_reason = rejected ? $("reject-reason").value : "";
  if (rejected) $("reviewed").checked = false;
  showRejected(current.rejected, current.reject_reason);
  await save();                                 // rejecting is saved immediately
  if (rejected) toast("Site rejected - left out of training. Press N for the next site");
}

$("reject").onclick = () => setRejected(true);
$("unreject").onclick = () => setRejected(false);

$("opacity").oninput = () => hillshade && hillshade.setOpacity(+$("opacity").value);

$("add-site-form").onsubmit = async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());
  try {
    const s = await api("/api/sites", { method: "POST", body: JSON.stringify(data) });
    e.target.reset();
    await refreshSites();
    selectSite(s.slug);
  } catch (err) {
    alert(err.message);
  }
};

window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

refreshSites();

// The server rebuilds sites that sites.json moved / resized on its own: keep the
// list current, and reload the open site when it is taken down for a rebuild
setInterval(async () => {
  if (drawing || inset) return;
  try { await refreshSites(); } catch (_) { return; }
  const s = current && sites.find((x) => x.slug === current.slug);
  if (s && current.prepared && !s.prepared && !dirty) selectSite(current.slug);
}, 15000);
