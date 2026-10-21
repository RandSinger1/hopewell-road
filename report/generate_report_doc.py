"""
generate_report_doc.py
Generates KhthonLiDAR_Results.docx — results section only.

Structure
---------
  Results
    Circleville Leg
      Table: site names + coordinates
      Overview map
      Per site (first occurrence only):
        pipeline_steps.png          (full 7×2 grid)
        pipeline_steps/12_detected_pairs.png
        pipeline_steps/13_hillshade_perp32.png
    Chillicothe Leg
      (same; duplicate sites cross-reference first occurrence)
    High Banks Leg
      (same)

Figure captions use Word SEQ Figure fields so numbering is maintained
automatically by Word when the document is opened.  Duplicate sites (mode
"all") emit a cross-reference paragraph rather than repeating the images.

Usage
-----
    # Called automatically by run_report.py after figure generation.
    # Standalone (generates from whatever is already in report/output/):
    python -m report.generate_report_doc
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import parameters
from .report_utils import site_target_angle, site_paths

_OUTPUT_DIR = Path(__file__).parent / "output"

_PATH_DISPLAY = {
    "circleville": "Circleville",
    "chillicothe": "Chillicothe",
    "high_banks":  "High Banks",
}


# ── Word XML helpers ──────────────────────────────────────────────────────────

def _seq_figure_into(paragraph) -> None:
    """Append 'Figure { SEQ Figure \\* ARABIC }' into paragraph using a fldSimple."""
    r_pre = paragraph.add_run("Figure ")
    r_pre.italic = True
    r_pre.font.size = Pt(9)

    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), " SEQ Figure \\* ARABIC ")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    i_el = OxmlElement("w:i")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "18")   # 9 pt = 18 half-points
    rpr.append(i_el)
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = "0"                # placeholder updated by Word on open
    r.append(t)
    fld.append(r)
    paragraph._p.append(fld)


def _bookmark_wrap(paragraph, bm_id: int, name: str) -> None:
    """Add bookmarkStart + bookmarkEnd around existing paragraph content."""
    bm_start = OxmlElement("w:bookmarkStart")
    bm_start.set(qn("w:id"), str(bm_id))
    bm_start.set(qn("w:name"), name)
    bm_end = OxmlElement("w:bookmarkEnd")
    bm_end.set(qn("w:id"), str(bm_id))
    paragraph._p.insert(0, bm_start)
    paragraph._p.append(bm_end)


# ── document helpers ──────────────────────────────────────────────────────────

def _add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def _add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    for row_data in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row_data):
            cells[i].text = str(val)
    return table


def _add_figure(doc, img_path, caption: str,
                width=Inches(5.8), bm_state: list | None = None,
                bm_name: str | None = None) -> bool:
    """
    Insert image + Word SEQ Figure caption.  Returns True if inserted.

    bm_state : [next_id]  — mutable counter shared across the document
    bm_name  : if given, the caption paragraph gets a bookmark with this name
                so it can be cross-referenced from duplicate-site paragraphs.
    """
    if not Path(img_path).exists():
        print(f"    [report] missing figure — skipped: {img_path}")
        return False

    # Image paragraph
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(img_path), width=width)

    # Caption paragraph
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)

    _seq_figure_into(cap)

    r_tail = cap.add_run(f".  {caption}")
    r_tail.italic = True
    r_tail.font.size = Pt(9)

    if bm_name and bm_state is not None:
        bm_id = bm_state[0]
        bm_state[0] += 1
        _bookmark_wrap(cap, bm_id, bm_name)

    return True


def _add_duplicate_note(doc, site_name: str, first_path_display: str,
                         this_path_display: str) -> None:
    """
    Emit a note that this site's figures already appear in another section.
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(
        f"{site_name.replace('_', ' ')} was first analyzed in the "
        f"{first_path_display} section; see the figures there for the "
        f"detection pipeline visualizations. "
        f"(Note: the {this_path_display} corridor uses a different target "
        f"azimuth — re-run with ––only “{site_name}” "
        f"to generate path-specific figures for this section.)"
    )
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x77, 0x77, 0x77)


# ── main entry point ──────────────────────────────────────────────────────────

def generate_report_doc(paths_and_coords: dict,
                         fig_base_dir: Path | None = None) -> None:
    """
    Generate KhthonLiDAR_Results.docx.

    Parameters
    ----------
    paths_and_coords : {path_name: [coord, ...], ...}
        Coords should already have "mode" forced to path_name so
        site_target_angle() returns the path-specific azimuth.
    fig_base_dir : root of per-path figure trees (default: report/output/)
        Layout expected: fig_base_dir / path_name / site_name / ...
    """
    fig_base_dir = Path(fig_base_dir) if fig_base_dir else _OUTPUT_DIR
    bm_state = [0]           # [next_bookmark_id] — unique across the document
    seen_sites: dict[str, str] = {}   # {site_name: first_path_display}

    doc = Document()
    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.25)
        section.right_margin  = Inches(1.25)

    # Title block
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Automated Detection of Hopewell Earthwork Remnants\nUsing Airborne LiDAR"
    )
    run.bold = True
    run.font.size = Pt(16)
    title.paragraph_format.space_after = Pt(4)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = sub.add_run("KhthonLiDAR Project  ·  Results")
    run2.italic = True
    run2.font.size = Pt(11)
    sub.paragraph_format.space_after = Pt(16)
    doc.add_paragraph()

    # ── Results ───────────────────────────────────────────────────────────────
    _add_heading(doc, "5.  Results", level=1)

    for sub_i, (path_name, coords) in enumerate(paths_and_coords.items(), start=1):
        display = _PATH_DISPLAY.get(path_name, path_name.replace("_", " ").title())
        path_dir = fig_base_dir / path_name

        _add_heading(doc, f"5.{sub_i}  {display}", level=2)

        # Coord table (north → south)
        sorted_coords = sorted(coords, key=lambda c: c["lat"], reverse=True)
        rows = [
            (c["name"],
             f"{c['lat']:.4f}° N,  {c['lon']:.4f}° W",
             c.get("note", ""))
            for c in sorted_coords
        ]
        _add_table(doc, ["Site", "Coordinates (WGS 84)", "Notes"], rows)
        doc.add_paragraph()

        # Overview map for this path
        _add_figure(
            doc,
            path_dir / "overview_map.png",
            f"{display}: corridor overview map showing detected wall bearings at each site.",
            width=Inches(5.5),
            bm_state=bm_state,
        )

        # Per-site figures
        for coord in sorted_coords:
            name = coord["name"]
            site_dir = path_dir / name
            _, target_angle = site_target_angle(coord)

            _add_heading(doc, name.replace("_", " "), level=3)

            if name in seen_sites:
                _add_duplicate_note(doc, name, seen_sites[name], display)
                continue

            # Unique slug prefix so bookmarks don't collide if the same figure
            # name appears across different paths (they're the same site but
            # different analyses; bookmarks are placed on the first occurrence).
            bm_prefix = f"fig_{name}"

            inserted_any = False
            inserted_any |= _add_figure(
                doc,
                site_dir / "pipeline_steps.png",
                f"{name}: full detection pipeline (7×2 panels). "
                f"Target bearing {target_angle:.0f}°.",
                bm_state=bm_state,
                bm_name=f"{bm_prefix}_pipeline",
            )
            inserted_any |= _add_figure(
                doc,
                site_dir / "pipeline_steps" / "12_detected_pairs.png",
                f"{name}: detected parallel wall pairs overlaid on raw DTM. "
                f"Pair spacing labelled in metres.",
                width=Inches(4.5),
                bm_state=bm_state,
                bm_name=f"{bm_prefix}_pairs",
            )
            inserted_any |= _add_figure(
                doc,
                site_dir / "pipeline_steps" / "13_hillshade_perp32.png",
                f"{name}: hillshade illuminated perpendicular to the "
                f"{target_angle:.0f}° corridor bearing — visual "
                f"confirmation of earthwork ridge orientation.",
                width=Inches(4.5),
                bm_state=bm_state,
                bm_name=f"{bm_prefix}_hillshade",
            )

            if inserted_any:
                seen_sites[name] = display

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "KhthonLiDAR_Results.docx"
    doc.save(out)
    n_bm = bm_state[0]
    print(f"Saved: {out}  ({n_bm} bookmarked figure captions, "
          f"{len(seen_sites)} unique sites)")


if __name__ == "__main__":
    # Build from whatever figures are already on disk
    all_coords = parameters.REPORT_COORDS
    paths_and_coords = {}
    for path_name in ( "chillicothe", "circleville", "high_banks"):
        path_coords = [
            {**c, "mode": path_name}
            for c in all_coords
            if path_name in site_paths(c)
        ]
        if path_coords:
            paths_and_coords[path_name] = path_coords
    generate_report_doc(paths_and_coords)
