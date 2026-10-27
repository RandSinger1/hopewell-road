"""
labels.py — Site registry and earthwork-label storage.

Layout (all under parameters.EW_ROOT, default data/earthworks/):

    sites.json                     [{"name", "lat", "lon", "half_m"?, "notes"?, "split"?}, ...]
                                   ("group": sites whose windows overlap share one, so
                                    they always land in the same train/val split)
    sites/<slug>/site.json         written by prepare.py: bounds, tiles, CRS, point counts
    sites/<slug>/points.laz        every point in the window, native CRS, all classes
    sites/<slug>/ground.tif        vendor-ground grid, EW_WORK_CRS, EW_PIXEL_M (metres)
    sites/<slug>/labels.geojson    drawn outlines (WGS84) — the source of truth
    sites/<slug>/labels.shp (+…)   the same outlines exported as a shapefile

labels.geojson is a FeatureCollection of Polygons with properties
{id, type, notes} plus a top-level "reviewed" flag.  Only reviewed sites go
into the training set: "reviewed" means *every* earthwork in the window has
been outlined, so unlabelled ground there is a trustworthy negative.

'type' is one of EW_EARTHWORK_TYPES (a target), EW_NEGATIVE_TYPES (a
look-alike: trained on as a hard negative, never a target) or
EW_UNVERIFIED_TYPE (an imported scan candidate awaiting a decision — a site
cannot be marked reviewed while any remain).

``write_vector`` is also used by earthwork_inference/scan.py, so candidates
share this schema and can be loaded back into the builder for review.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from osgeo import ogr, osr
from shapely import make_valid
from shapely.geometry import mapping, shape

import parameters

ogr.UseExceptions()

SITES_DIR = parameters.EW_ROOT / "sites"

_DRIVERS = {".shp": "ESRI Shapefile", ".gpkg": "GPKG", ".geojson": "GeoJSON"}


def label_types() -> list[str]:
    return [*parameters.EW_EARTHWORK_TYPES, *parameters.EW_NEGATIVE_TYPES, parameters.EW_UNVERIFIED_TYPE]


def is_negative(label_type: str) -> bool:
    return label_type in parameters.EW_NEGATIVE_TYPES


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


@dataclass
class Site:
    name: str
    lat: float
    lon: float
    half_m: float = parameters.EW_SITE_HALF_M
    notes: str = ""
    split: str | None = None          # "train" | "val" | None (assigned by hash)
    extra: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def dir(self) -> Path:
        return SITES_DIR / self.slug

    @property
    def ground_path(self) -> Path:
        return self.dir / "ground.tif"

    @property
    def points_path(self) -> Path:
        return self.dir / "points.laz"

    @property
    def labels_path(self) -> Path:
        return self.dir / "labels.geojson"

    @property
    def prepared(self) -> bool:
        return self.ground_path.exists() and (self.dir / "site.json").exists()

    def to_json(self) -> dict:
        d = {"name": self.name, "lat": self.lat, "lon": self.lon, "half_m": self.half_m}
        if self.notes:
            d["notes"] = self.notes
        if self.split:
            d["split"] = self.split
        return {**d, **self.extra}


def load_sites(path: Path = parameters.EW_SITES_FILE) -> list[Site]:
    if not path.exists():
        return []
    known = {"name", "lat", "lon", "half_m", "notes", "split"}
    sites = []
    for d in json.loads(path.read_text(encoding="utf-8")):
        sites.append(Site(name=d["name"], lat=float(d["lat"]), lon=float(d["lon"]),
                          half_m=float(d.get("half_m", parameters.EW_SITE_HALF_M)),
                          notes=d.get("notes", ""), split=d.get("split"),
                          extra={k: v for k, v in d.items() if k not in known}))
    slugs = [s.slug for s in sites]
    dupes = {s for s in slugs if slugs.count(s) > 1}
    if dupes:
        raise ValueError(f"{path}: site names collide after slugifying: {sorted(dupes)}")
    return sites


def save_sites(sites: list[Site], path: Path = parameters.EW_SITES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.to_json() for s in sites], indent=2), encoding="utf-8")


def get_site(slug: str, sites: list[Site] | None = None) -> Site:
    for s in sites if sites is not None else load_sites():
        if s.slug == slug:
            return s
    raise KeyError(f"No site with slug {slug!r} in {parameters.EW_SITES_FILE}")


# ─── Labels ───────────────────────────────────────────────────────────────────

def read_labels(site: Site) -> dict:
    """The site's FeatureCollection (empty, unreviewed if none saved yet)."""
    if site.labels_path.exists():
        return json.loads(site.labels_path.read_text(encoding="utf-8"))
    return {"type": "FeatureCollection", "features": [], "reviewed": False}


def label_polygons(site: Site) -> list[tuple[object, dict]]:
    """[(shapely Polygon in WGS84, properties), …] for a site's labels."""
    fc = read_labels(site)
    return [(shape(f["geometry"]), f.get("properties") or {}) for f in fc["features"]]


def save_labels(site: Site, fc: dict) -> dict:
    """Validate, normalise and store a FeatureCollection, then export it as a shapefile."""
    features = []
    allowed = label_types()
    for i, f in enumerate(fc.get("features", [])):
        geom = shape(f["geometry"])
        if geom.geom_type != "Polygon":
            raise ValueError(f"feature {i}: expected Polygon, got {geom.geom_type}")
        if not geom.is_valid:
            # make_valid (not buffer(0), which silently drops a bowtie's second lobe)
            geom = make_valid(geom)
            if geom.geom_type != "Polygon":
                raise ValueError(f"feature {i}: outline crosses itself - redraw it")
        props = f.get("properties") or {}
        kind = props.get("type", "other")
        if kind not in allowed:
            raise ValueError(f"feature {i}: unknown type {kind!r}")
        features.append({"type": "Feature", "geometry": mapping(geom),
                         "properties": {"id": i, "type": kind, "notes": props.get("notes", "")}})
    reviewed = bool(fc.get("reviewed", False))
    pending = [f["properties"]["id"] for f in features
               if f["properties"]["type"] == parameters.EW_UNVERIFIED_TYPE]
    if reviewed and pending:
        raise ValueError(f"outline(s) {pending} are still '{parameters.EW_UNVERIFIED_TYPE}' - "
                         f"give each a real or look-alike type (or delete it) before marking reviewed")
    out = {"type": "FeatureCollection", "features": features, "reviewed": reviewed}
    site.dir.mkdir(parents=True, exist_ok=True)
    site.labels_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_vector(site.dir / "labels.shp",
                 [(shape(f["geometry"]), dict(f["properties"], site=site.slug)) for f in features])
    return out


# ─── Vector export ────────────────────────────────────────────────────────────

def _ogr_type(v) -> int:
    if isinstance(v, bool) or isinstance(v, int):
        return ogr.OFTInteger64
    if isinstance(v, float):
        return ogr.OFTReal
    return ogr.OFTString


def write_vector(path: Path | str, features: list[tuple[object, dict]],
                 crs: str = "EPSG:4326", append: bool = False) -> Path:
    """Write [(shapely geometry, properties), …] as .shp / .gpkg / .geojson.

    Keep property names ≤ 10 characters — the shapefile format truncates them.
    With ``append`` an existing file gains the new features (fields must match).
    """
    path = Path(path)
    driver = ogr.GetDriverByName(_DRIVERS[path.suffix.lower()])
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        ds = ogr.Open(str(path), update=1)
        layer = ds.GetLayer(0)
    else:
        if path.exists():
            driver.DeleteDataSource(str(path))
        ds = driver.CreateDataSource(str(path))
        srs = osr.SpatialReference()
        srs.SetFromUserInput(crs)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        layer = ds.CreateLayer(path.stem, srs, ogr.wkbPolygon)
        names = {}
        for _, props in features:
            for k, v in props.items():
                names.setdefault(k, _ogr_type(v))
        for k, t in names.items():
            fd = ogr.FieldDefn(k, t)
            if t == ogr.OFTString:
                fd.SetWidth(254)
            layer.CreateField(fd)
    defn = layer.GetLayerDefn()
    fields = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}
    for geom, props in features:
        feat = ogr.Feature(defn)
        feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
        for k, v in props.items():
            if k[:10] in fields or k in fields:
                feat.SetField(k if k in fields else k[:10], v)
        layer.CreateFeature(feat)
    ds.FlushCache()
    ds = None
    return path
