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

const labelLayer = L.featureGroup().addTo(map);
let hillshade = null;
let windowRect = null;
let current = null;          // site detail from /api/sites/<slug>
let dirty = false;

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
  const sites = await api("/api/sites");
  const ul = $("site-list");
  ul.innerHTML = "";
  for (const s of sites) {
    const li = document.createElement("li");
    li.className = current && current.slug === s.slug ? "active" : "";
    const state = s.job === "running" ? "downloading…"
      : !s.prepared ? (s.job === "failed" ? "prepare failed" : "not prepared")
      : `${s.labels} outline${s.labels === 1 ? "" : "s"}${s.reviewed ? " ✓" : ""}`;
    li.innerHTML = `<span>${escapeHtml(s.name)}</span><span class="muted">${state}</span>`;
    li.onclick = () => selectSite(s.slug);
    ul.appendChild(li);
  }
  return sites;
}

async function selectSite(slug) {
  if (dirty && !confirm("Discard unsaved changes?")) return;
  current = await api(`/api/sites/${slug}`);
  setDirty(false);
  $("site-panel").hidden = false;
  $("site-title").textContent = current.name;
  $("site-notes").textContent = current.notes || "";
  $("reviewed").checked = !!current.reviewed;

  if (hillshade) map.removeLayer(hillshade);
  if (windowRect) map.removeLayer(windowRect);
  labelLayer.clearLayers();

  if (current.prepared) {
    const b = current.bounds;
    const bounds = L.latLngBounds([b.south, b.west], [b.north, b.east]);
    hillshade = L.tileLayer(`/api/sites/${slug}/tile/{z}/{x}/{y}.png`, {
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
  $("job").textContent = job.status ? `${job.status}: ${job.log.slice(-1)[0] || ""}` : "Not prepared.";
  if (job.status === "running") setTimeout(() => pollJob(slug), 3000);
  else if (job.status === "done") selectSite(slug);
  else if (!job.status) {
    const btn = document.createElement("button");
    btn.textContent = "Download LiDAR";
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
  layer.on("click", () => highlight(layer));
  labelLayer.addLayer(layer);
}

map.on("pm:create", (e) => {
  if (!current) { map.removeLayer(e.layer); return; }
  map.removeLayer(e.layer);
  addLabelLayer(e.layer, {});
  setDirty(true);
  renderLabelList();
});

map.on("pm:remove", (e) => {
  labelLayer.removeLayer(e.layer);
  setDirty(true);
  renderLabelList();
});

function highlight(layer) {
  labelLayer.eachLayer((l) => l.setStyle({ weight: l === layer ? 4 : 2 }));
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
    li.innerHTML = `<b>#${i}</b> <select>${opts}</select>
      <input placeholder="notes" value="${escapeHtml(p.notes)}" />
      <button title="Zoom">⌖</button><button title="Delete">✕</button>`;
    const [sel, notes] = li.querySelectorAll("select, input");
    const [zoomBtn, delBtn] = li.querySelectorAll("button");
    sel.onchange = () => { p.type = sel.value; restyle(layer); setDirty(true); renderLabelList(); };
    notes.oninput = () => { p.notes = notes.value; setDirty(true); };
    zoomBtn.onclick = () => { map.fitBounds(layer.getBounds().pad(0.5)); highlight(layer); };
    delBtn.onclick = () => { labelLayer.removeLayer(layer); map.removeLayer(layer); setDirty(true); renderLabelList(); };
    ul.appendChild(li);
  });
}

$("reviewed").onchange = () => setDirty(true);

$("save").onclick = async () => {
  if (!current) return;
  const fc = labelLayer.toGeoJSON();
  fc.reviewed = $("reviewed").checked;
  try {
    await api(`/api/sites/${current.slug}/labels`, { method: "PUT", body: JSON.stringify(fc) });
    setDirty(false);
    refreshSites();
  } catch (err) {
    alert(`Save failed: ${err.message}`);
  }
};

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
