"""
generate_report_doc.py
Generates report/KhthonLiDAR_Report.docx for distribution to archaeologists.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import params

OUTPUT_DIR = Path(__file__).parent / "output"
fig_counter = [0]


# ── helpers ──────────────────────────────────────────────────────────────────

def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)

def add_para(doc, text, bold=False, italic=False, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    p.paragraph_format.space_after = Pt(space_after)
    return p

def add_bullet(doc, text):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.space_after = Pt(3)

def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].paragraphs[0].runs[0].bold = True
    for row_data in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row_data):
            cells[i].text = str(val)
    return table

def add_figure(doc, img_path, caption, width=Inches(5.8)):
    """Insert an image with a centred italic caption. Skips silently if missing."""
    if not Path(img_path).exists():
        return
    fig_counter[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(img_path), width=width)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = cap.add_run(f"Figure {fig_counter[0]}.  {caption}")
    run2.italic = True
    run2.font.size = Pt(9)
    cap.paragraph_format.space_after = Pt(12)


# ── build document ────────────────────────────────────────────────────────────

doc = Document()

for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.25)
    section.right_margin  = Inches(1.25)

# Title
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Automated Detection of Hopewell Earthwork Remnants\nUsing Airborne LiDAR")
run.bold = True
run.font.size = Pt(16)
title.paragraph_format.space_after = Pt(4)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run2 = subtitle.add_run("KhthonLiDAR Project  ·  Preliminary Technical Report")
run2.italic = True
run2.font.size = Pt(11)
subtitle.paragraph_format.space_after = Pt(16)
doc.add_paragraph()


# ── 1. Introduction ───────────────────────────────────────────────────────────

add_heading(doc, "1.  Introduction", level=1)
add_para(doc, (
    "The Hopewell Road corridor in central Ohio was a primary route of the Hopewell "
    "culture (c. 100 BCE – 500 CE), one of the most sophisticated prehistoric cultures "
    "of eastern North America.  The Hopewell built extensive earthwork complexes — "
    "pairs of parallel embankments flanking ceremonial roads — much of which has been "
    "obscured by modern agriculture, vegetation, and development. "
    "Many sections of these earthworks are no longer visible at the surface and have "
    "not been systematically surveyed."
))
add_para(doc, (
    "This project applies automated analysis of high-resolution airborne LiDAR data "
    "to detect candidate remnants of these earthwork pairs along the corridor between "
    "Chillicothe and Newark, Ohio.  The approach targets the characteristic morphological "
    "signature of Hopewell road earthworks: two parallel low-relief linear ridges "
    "approximately 38–42 m apart, oriented along the road bearing, and visible as "
    "subtle positive anomalies in the ground surface when regional slope is removed."
))


# ── 2. Data ───────────────────────────────────────────────────────────────────

add_heading(doc, "2.  Data", level=1)
add_para(doc, (
    "Airborne LiDAR point-cloud data were obtained from the U.S. Geological Survey "
    "National Map (3DEP programme) for the Ohio Statewide Phase 2 collection (2020). "
    "The data have a nominal ground-point density of approximately 8 points/m² "
    "and were acquired with last-return and single-return filters to preferentially "
    "capture bare-earth returns.  Tiles were downloaded in LAZ format directly from "
    "the USGS RockyWeb delivery service."
))
add_para(doc, (
    "The study area encompasses the Hopewell Road corridor from Chillicothe "
    "(39.31°N, 82.98°W) north-east to Newark (40.05°N, 82.44°W), an extent of "
    "approximately 95 km."
))


# ── 3. Methodology ────────────────────────────────────────────────────────────

add_heading(doc, "3.  Methodology", level=1)
add_para(doc, (
    "The detection pipeline proceeds in five stages: point-cloud preparation, "
    "DTM generation, topographic pre-processing, ridge enhancement, and "
    "geometric line detection.  All parameters are listed in Section 4.  "
    "Section 5 includes the full pipeline figure for each site."
))

# 3.1
add_heading(doc, "3.1  Point-Cloud Preparation", level=2)
add_para(doc, "Each LAZ tile was processed through a PDAL pipeline comprising three steps:")
add_bullet(doc, (
    "Statistical outlier removal (filters.outlier):  Points whose distance from "
    f"the local mean of their {params.OUTLIER_MEAN_K} nearest neighbours exceeded "
    f"{params.OUTLIER_MULTIPLIER} standard deviations were flagged as noise and discarded."
))
add_bullet(doc, (
    "Ground classification (filters.smrf):  The Simple Morphological Filter "
    f"(Pingel et al., 2013) was applied with a maximum window radius of "
    f"{params.SMRF_WINDOW} m and a slope tolerance of {params.SMRF_SLOPE} m/m "
    "to classify returns as ground or non-ground.  Only last returns and single "
    "returns were considered, reducing the influence of canopy returns."
))
add_bullet(doc, (
    "Ground extraction:  Non-ground returns were discarded.  The resulting "
    "ground-only point cloud was re-saved in compressed LAS format."
))

# 3.2
add_heading(doc, "3.2  Digital Terrain Model Generation", level=2)
add_para(doc, (
    f"Ground point clouds were rasterised to Digital Terrain Models (DTMs) at "
    f"{params.DTM_RESOLUTION} m/pixel using inverse-distance weighting (IDW) "
    "interpolation via PDAL's GDAL writer.  Individual tile DTMs were then "
    "mosaicked and reprojected to WGS 84 (EPSG:4326) using GDAL Warp with "
    "bilinear resampling, preserving the native ground resolution in degrees per pixel."
))

# 3.3
add_heading(doc, "3.3  Topographic Pre-processing", level=2)
add_para(doc, (
    "Before ridge detection, each search window was pre-processed to suppress "
    "broadscale topographic trends and sensor noise:"
))
add_bullet(doc, (
    f"Median smoothing:  A {2*params.MEDIAN_SMOOTH_RADIUS + 1}×"
    f"{2*params.MEDIAN_SMOOTH_RADIUS + 1} pixel median filter was applied with a "
    "nodata-aware fill to reduce salt-and-pepper noise from residual vegetation "
    "returns without displacing sharp earthwork edges."
))
add_bullet(doc, (
    f"Local Relief Model (LRM):  The regional mean surface was estimated by a "
    f"{2*params.LRM_RADIUS + 1}×{2*params.LRM_RADIUS + 1} pixel box filter "
    f"({2*params.LRM_RADIUS + 1} m at 1 m/px) and subtracted from the smoothed DTM.  "
    "The LRM isolates micro-topographic features — such as earthwork embankments — "
    "from underlying valley slopes and broad terrain undulations."
))

# 3.4
add_heading(doc, "3.4  Multi-scale Ridge Enhancement", level=2)
add_para(doc, (
    "Ridge-like features in the LRM were enhanced using the Frangi vesselness "
    "measure (Frangi et al., 1998), originally developed for vascular segmentation "
    "in medical imaging and adapted here for elongated topographic ridges.  "
    "The measure derives from the eigenvalues of the Hessian matrix of the image "
    "intensity (here, relative elevation)."
))
add_para(doc, (
    "For each scale σ (in pixels), second-order partial derivatives are computed "
    "analytically after Gaussian smoothing.  The 2×2 Hessian yields eigenvalues "
    "λ₁ ≤ λ₂.  A ridge exists where λ₁ < 0 (concave across the ridge) and λ₂ ≈ 0 "
    "(flat along the ridge).  The vesselness score is:"
))
add_para(doc, "    V(σ) = exp(−Rᵦ² / 2β²) · (1 − exp(−S² / 2c²))", italic=True)
add_para(doc, (
    "where Rᵦ = λ₂/λ₁ penalises blob-like structures, S = √(λ₁² + λ₂²) suppresses "
    f"background noise, β = 0.5, and c is half the 95th percentile of S.  "
    f"The filter was applied at scales σ ∈ {params.RIDGE_SCALES} pixels and the "
    "maximum response across all scales was retained."
))

# 3.5
add_heading(doc, "3.5  Geometric Line Detection and Parallel-Wall Matching", level=2)
add_para(doc, (
    "Detected ridge pixels were converted to candidate straight-line segments and "
    "tested for the parallel geometry expected of Hopewell road earthworks:"
))
add_bullet(doc, (
    f"Orientation filtering:  Ridge pixel orientations were estimated from Hessian "
    f"eigenvectors (σ = {params.HESSIAN_SIGMA} px).  Pixels whose orientation deviated "
    f"more than ±{params.ANGLE_TOLERANCE}° from the expected wall bearing "
    f"({params.ROAD_BEARING_DEG}° from north) were discarded."
))
add_bullet(doc, (
    f"Thresholding and morphological closing:  The orientation-filtered ridge map was "
    f"thresholded at a Frangi response of {params.RIDGE_THRESHOLD:.2f}.  A morphological "
    f"closing with radius {int(params.MORPH_DISK_RADIUS)} px bridged small gaps; "
    f"components smaller than {params.BINARY_MIN_OBJECT_SIZE} px² were removed as speckle."
))
add_bullet(doc, (
    "Skeletonisation:  The binary mask was thinned to a one-pixel-wide skeleton "
    "using the medial-axis transform."
))
add_bullet(doc, (
    f"Line fitting:  Each skeleton component longer than {params.MIN_RIDGE_LENGTH_METERS} m "
    "was approximated by a straight line using total-least-squares regression (cv2.fitLine)."
))
add_bullet(doc, (
    f"Parallel-wall pairing:  Line pairs were accepted if: (i) angular difference "
    f"≤ {params.PAIRING_ANGLE_TOLERANCE_DEG}°; (ii) perpendicular separation within "
    f"±{int(params.SPACING_TOLERANCE*100)}% of {params.SPACING_METERS} m "
    f"({params.SPACING_METERS*(1-params.SPACING_TOLERANCE):.0f}–"
    f"{params.SPACING_METERS*(1+params.SPACING_TOLERANCE):.0f} m); "
    f"(iii) overlap ≥ {int(params.OVERLAP_RATIO*100)}% of the shorter segment."
))
add_para(doc, (
    f"The pipeline was run in a tiled scan across the full corridor using "
    f"{params.SCAN_TILE_SIZE_M} m × {params.SCAN_TILE_SIZE_M} m windows with "
    f"{params.SCAN_OVERLAP_M} m overlap."
))


# ── 4. Parameters ─────────────────────────────────────────────────────────────

add_heading(doc, "4.  Key Parameters", level=1)
add_para(doc, "All parameters used in the analysis are summarised below.")
param_rows = [
    ("DTM resolution",           f"{params.DTM_RESOLUTION} m/pixel"),
    ("Median smooth radius",     f"{params.MEDIAN_SMOOTH_RADIUS} px ({2*params.MEDIAN_SMOOTH_RADIUS+1}×{2*params.MEDIAN_SMOOTH_RADIUS+1} kernel)"),
    ("LRM radius",               f"{params.LRM_RADIUS} px ({2*params.LRM_RADIUS+1} m)"),
    ("Frangi scales (σ)",        str(params.RIDGE_SCALES) + " px"),
    ("Frangi threshold",         str(params.RIDGE_THRESHOLD)),
    ("Hessian orientation σ",    f"{params.HESSIAN_SIGMA} px"),
    ("Orientation tolerance",    f"±{params.ANGLE_TOLERANCE}°"),
    ("Expected wall bearing",    f"{params.ROAD_BEARING_DEG}° from north"),
    ("Morph. closing radius",    f"{int(params.MORPH_DISK_RADIUS)} px"),
    ("Min. component size",      f"{params.BINARY_MIN_OBJECT_SIZE} px²"),
    ("Min. ridge length",        f"{params.MIN_RIDGE_LENGTH_METERS} m"),
    ("Expected wall spacing",    f"{params.SPACING_METERS} m"),
    ("Spacing tolerance",        f"±{int(params.SPACING_TOLERANCE*100)}%"),
    ("Pairing angle tolerance",  f"±{params.PAIRING_ANGLE_TOLERANCE_DEG}°"),
    ("Min. overlap ratio",       f"{int(params.OVERLAP_RATIO*100)}%"),
]
add_table(doc, ["Parameter", "Value"], param_rows)
doc.add_paragraph()


# ── 5. Candidate Detections ───────────────────────────────────────────────────

add_heading(doc, "5.  Candidate Detections", level=1)
add_para(doc, (
    "The following locations returned positive detections and are recommended "
    "for ground-truth survey.  The Van Voorhis Wall site is a confirmed Hopewell "
    "earthwork and serves as the validation baseline.  Remaining sites are ordered "
    "by detection confidence.  For each site, multi-azimuth hillshade images are "
    "shown to reveal the earthwork topography under varying illumination, followed "
    "by perpendicular elevation profiles through the detected wall pairs."
))

# Summary table
site_rows = [(c["name"], f"{c['lat']:.4f}° N,  {c['lon']:.4f}° W", c.get("note", ""))
             for c in params.REPORT_COORDS]
add_table(doc, ["Site", "Coordinates (WGS 84)", "Notes"], site_rows)
doc.add_paragraph()

# Per-site images
for coord in params.REPORT_COORDS:
    name = coord["name"]
    site_dir = OUTPUT_DIR / name

    add_heading(doc, f"5.{params.REPORT_COORDS.index(coord)+1}  {name}", level=2)
    add_para(doc, (
        f"Location: {coord['lat']:.4f}° N, {coord['lon']:.4f}° W.  "
        f"{coord.get('note', '')}"
    ))

    add_figure(
        doc,
        site_dir / "multi_azimuth_hillshade.png",
        f"{name} — multi-azimuth hillshade.  Each panel uses a different illumination "
        "direction (315°, 45°, 135°, 225°) to reveal linear earthwork features that "
        "may be invisible under a single lighting angle.",
    )

    # Prefer detected_pair_profiles; fall back to cross_sections
    profiles_img = site_dir / "detected_pair_profiles.png"
    cross_img    = site_dir / "cross_sections.png"
    if profiles_img.exists():
        add_figure(
            doc,
            profiles_img,
            f"{name} — perpendicular elevation profiles through each detected wall pair.  "
            "Dotted vertical lines mark the actual detected wall positions.  "
            "The right panel shows de-trended profiles with regional slope removed; "
            "a twin-peak pattern at the expected spacing is characteristic of Hopewell earthworks.",
        )
    elif cross_img.exists():
        add_figure(
            doc,
            cross_img,
            f"{name} — perpendicular cross-sections through the detected wall pair.",
        )

    add_figure(
        doc,
        site_dir / "pipeline_steps.png",
        f"{name} — processing pipeline.  Top row: raw DTM, median-smoothed DTM, "
        "Local Relief Model, Frangi ridge response, Sky-View Factor.  "
        "Bottom row: labelled skeleton components, binary ridge mask, "
        "morphological closing, medial-axis skeleton, detected wall pairs.",
    )


# ── 6. Limitations ────────────────────────────────────────────────────────────

add_heading(doc, "6.  Limitations and Next Steps", level=1)
add_para(doc, "The automated method has several limitations that field survey would address:")
add_bullet(doc, (
    "Detection sensitivity depends on earthwork preservation.  Heavily ploughed "
    "or eroded sections may produce sub-threshold Frangi responses even where "
    "earthworks are known to exist."
))
add_bullet(doc, (
    f"The orientation and spacing filters are tuned to the Hopewell Road corridor "
    f"bearing.  Segments deviating more than ±{params.ANGLE_TOLERANCE}° from the "
    "expected bearing will be suppressed."
))
add_bullet(doc, (
    "Linear agricultural features (field drainage ditches, hedgerow banks) "
    "oriented near the corridor bearing may generate false positives.  "
    "Candidate sites should be cross-checked against cadastral boundaries and "
    "historical aerial photography."
))
add_bullet(doc, (
    "Ground-truth survey at each candidate site is required to confirm "
    "archaeological significance before any further interpretation."
))


# ── 7. References ─────────────────────────────────────────────────────────────

add_heading(doc, "7.  References", level=1)
refs = [
    ("Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998). "
     "Multiscale vessel enhancement filtering. "
     "Medical Image Computing and Computer-Assisted Intervention, 130–137."),
    ("Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). "
     "An improved simple morphological filter for the terrain classification of "
     "airborne LiDAR data. "
     "ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21–30."),
    ("U.S. Geological Survey (2020). "
     "Ohio Statewide LiDAR Phase 2 (3DEP).  "
     "National Map 3D Elevation Programme."),
]
for r in refs:
    p = doc.add_paragraph(r, style='List Number')
    p.paragraph_format.space_after = Pt(4)


# ── save ──────────────────────────────────────────────────────────────────────

out = Path(__file__).parent / "KhthonLiDAR_Report.docx"
doc.save(out)
print(f"Saved: {out}  ({fig_counter[0]} figures)")
