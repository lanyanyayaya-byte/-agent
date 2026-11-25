#!/usr/bin/env python3
"""Apply user-requested formatting rules to a DOCX copy using python-docx and OOXML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

CN_FONT_SIZES: dict[str, float] = {
    "初号": 42.0,
    "小初": 36.0,
    "一号": 26.0,
    "小一": 24.0,
    "二号": 22.0,
    "小二": 18.0,
    "三号": 16.0,
    "小三": 15.0,
    "四号": 14.0,
    "小四": 12.0,
    "五号": 10.5,
    "小五": 9.0,
    "六号": 7.5,
    "小六": 6.5,
}

ALIGNMENTS: dict[str, WD_ALIGN_PARAGRAPH] = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
}


def resolve_size_pt(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text in CN_FONT_SIZES:
            return CN_FONT_SIZES[text]
        cleaned = text.lower().removesuffix("pt").strip()
        try:
            return float(cleaned)
        except ValueError:
            raise ValueError(f"Unsupported font size: {raw}") from None
    raise ValueError(f"Unsupported font size: {raw}")


def apply_run_font(run, *, east_asia: str | None, western: str | None, size_pt: float | None, bold: bool | None, italic: bool | None) -> None:
    if western:
        run.font.name = western
    elif east_asia and not run.font.name:
        run.font.name = east_asia
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic

    if east_asia or western:
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.rFonts
        if rFonts is None:
            rFonts = rPr.makeelement(qn("w:rFonts"))
            rPr.insert(0, rFonts)
        if western:
            rFonts.set(qn("w:ascii"), western)
            rFonts.set(qn("w:hAnsi"), western)
            rFonts.set(qn("w:cs"), western)
        if east_asia:
            rFonts.set(qn("w:eastAsia"), east_asia)
            if not western and not rFonts.get(qn("w:ascii")):
                rFonts.set(qn("w:ascii"), east_asia)
                rFonts.set(qn("w:hAnsi"), east_asia)


def apply_paragraph_rule(paragraph, rule: dict[str, Any]) -> None:
    east_asia = rule.get("east_asia_font")
    western = rule.get("western_font")
    size_pt = resolve_size_pt(rule.get("size") if "size" in rule else rule.get("size_pt"))
    bold = rule.get("bold")
    italic = rule.get("italic")

    pf = paragraph.paragraph_format
    if rule.get("line_spacing") is not None:
        pf.line_spacing = float(rule["line_spacing"])
    if rule.get("space_before_pt") is not None:
        pf.space_before = Pt(float(rule["space_before_pt"]))
    if rule.get("space_after_pt") is not None:
        pf.space_after = Pt(float(rule["space_after_pt"]))
    if rule.get("first_line_indent_pt") is not None:
        pf.first_line_indent = Pt(float(rule["first_line_indent_pt"]))
    elif rule.get("first_line_indent_chars") is not None:
        chars = float(rule["first_line_indent_chars"])
        base_pt = size_pt or 12.0
        pf.first_line_indent = Pt(chars * base_pt)
        pPr = paragraph._element.get_or_add_pPr()
        ind = pPr.find(qn("w:ind"))
        if ind is None:
            ind = pPr.makeelement(qn("w:ind"), {})
            pPr.append(ind)
        if chars <= 0:
            ind.attrib.pop(qn("w:firstLineChars"), None)
            ind.attrib.pop(qn("w:firstLine"), None)
            pf.first_line_indent = Pt(0)
        else:
            ind.set(qn("w:firstLineChars"), str(int(round(chars * 100))))

    align_name = rule.get("alignment")
    if isinstance(align_name, str) and align_name.lower() in ALIGNMENTS:
        paragraph.alignment = ALIGNMENTS[align_name.lower()]

    for run in paragraph.runs:
        # Preserve inline drawings/images untouched while styling text runs.
        if run._element.findall(f".//{qn('w:drawing')}") or run._element.findall(f".//{qn('w:pict')}"):
            continue
        apply_run_font(run, east_asia=east_asia, western=western, size_pt=size_pt, bold=bold, italic=italic)


def classify_paragraph(paragraph, index: int, spec: dict[str, Any]) -> str:
    if index in set(spec.get("skip_indices") or ()):
        return "skip"
    if index in set(spec.get("heading1_indices") or ()):
        return "heading1"
    if index in set(spec.get("heading2_indices") or ()):
        return "heading2"
    if index in set(spec.get("heading3_indices") or ()):
        return "heading3"
    sname = (paragraph.style.name if paragraph.style else "").strip().lower()
    if sname in ("heading 1", "标题 1", "title"):
        return "heading1"
    if sname in ("heading 2", "标题 2"):
        return "heading2"
    if sname in ("heading 3", "标题 3"):
        return "heading3"
    return "body"


def main() -> int:
    parser = argparse.ArgumentParser(description="Format a DOCX file according to a JSON spec.")
    parser.add_argument("--input", required=True, help="Input DOCX path")
    parser.add_argument("--output", required=True, help="Output DOCX path")
    parser.add_argument("--spec", required=True, help="JSON spec file path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    spec_path = Path(args.spec)
    if not in_path.is_file():
        print(f"Input DOCX not found: {in_path}", file=sys.stderr)
        return 2
    if not spec_path.is_file():
        print(f"Spec JSON not found: {spec_path}", file=sys.stderr)
        return 2

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    doc = Document(str(in_path))

    # Page margins
    margins = spec.get("margins_cm")
    if isinstance(margins, dict):
        for sec in doc.sections:
            if margins.get("top") is not None:
                sec.top_margin = Cm(float(margins["top"]))
            if margins.get("bottom") is not None:
                sec.bottom_margin = Cm(float(margins["bottom"]))
            if margins.get("left") is not None:
                sec.left_margin = Cm(float(margins["left"]))
            if margins.get("right") is not None:
                sec.right_margin = Cm(float(margins["right"]))

    no_indent_set = set(spec.get("no_indent_indices") or ())
    counts = {"heading1": 0, "heading2": 0, "heading3": 0, "body": 0, "skipped": 0, "table_cells": 0}
    for idx, paragraph in enumerate(doc.paragraphs):
        if not paragraph.text.strip():
            continue
        kind = classify_paragraph(paragraph, idx, spec)
        if kind == "skip":
            counts["skipped"] += 1
            continue
        rule = spec.get(kind)
        if isinstance(rule, dict):
            effective_rule = dict(rule)
            if idx in no_indent_set:
                effective_rule["first_line_indent_chars"] = 0
                effective_rule.pop("first_line_indent_pt", None)
            apply_paragraph_rule(paragraph, effective_rule)
            counts[kind] += 1

    table_rule = spec.get("table")
    if isinstance(table_rule, dict):
        header_bold = table_rule.get("header_bold")
        for table in doc.tables:
            for r_idx, row in enumerate(table.rows):
                for cell in row.cells:
                    cell_rule = dict(table_rule)
                    if r_idx == 0 and header_bold is not None:
                        cell_rule["bold"] = bool(header_bold)
                    for paragraph in cell.paragraphs:
                        apply_paragraph_rule(paragraph, cell_rule)
                    counts["table_cells"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(json.dumps({
        "ok": True,
        "output": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "updated": counts,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
