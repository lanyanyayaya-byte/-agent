#!/usr/bin/env python3
"""Read-back validator for formatted DOCX files: verifies text preservation and applied styles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

CN_FONT_SIZES = {
    "初号": 42.0, "小初": 36.0, "一号": 26.0, "小一": 24.0,
    "二号": 22.0, "小二": 18.0, "三号": 16.0, "小三": 15.0,
    "四号": 14.0, "小四": 12.0, "五号": 10.5, "小五": 9.0,
    "六号": 7.5, "小六": 6.5,
}


def _resolve_pt(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text in CN_FONT_SIZES:
        return CN_FONT_SIZES[text]
    return float(text.lower().removesuffix("pt").strip())


def _classify(p, idx: int, spec: dict) -> str:
    if idx in set(spec.get("skip_indices") or ()):
        return "skip"
    if idx in set(spec.get("heading1_indices") or ()):
        return "heading1"
    if idx in set(spec.get("heading2_indices") or ()):
        return "heading2"
    if idx in set(spec.get("heading3_indices") or ()):
        return "heading3"
    sname = (p.style.name if p.style else "").strip().lower()
    if sname in ("heading 1", "标题 1", "title"):
        return "heading1"
    if sname in ("heading 2", "标题 2"):
        return "heading2"
    if sname in ("heading 3", "标题 3"):
        return "heading3"
    return "body"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate formatted DOCX against original and spec.")
    parser.add_argument("--original", required=True, help="Original DOCX path")
    parser.add_argument("--formatted", required=True, help="Formatted DOCX path")
    parser.add_argument("--spec", default="", help="Optional JSON spec used for formatting")
    args = parser.parse_args()

    orig = Document(args.original)
    out = Document(args.formatted)

    orig_texts = [p.text for p in orig.paragraphs]
    out_texts = [p.text for p in out.paragraphs]
    if orig_texts != out_texts:
        print("Validation failed: paragraph text content changed during formatting.", file=sys.stderr)
        return 1

    if len(orig.tables) != len(out.tables):
        print("Validation failed: table count changed.", file=sys.stderr)
        return 1
    for t_idx, (t1, t2) in enumerate(zip(orig.tables, out.tables)):
        c1 = [[c.text for c in r.cells] for r in t1.rows]
        c2 = [[c.text for c in r.cells] for r in t2.rows]
        if c1 != c2:
            print(f"Validation failed: table {t_idx} content changed.", file=sys.stderr)
            return 1

    checks: dict[str, object] = {
        "paragraphs_preserved": len(out_texts),
        "tables_preserved": len(out.tables),
    }

    if args.spec and Path(args.spec).is_file():
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        for kind in ("heading1", "heading2", "heading3", "body"):
            rule = spec.get(kind)
            if not isinstance(rule, dict):
                continue
            sample = next(
                (p for idx, p in enumerate(out.paragraphs) if p.text.strip() and _classify(p, idx, spec) == kind and p.runs),
                None,
            )
            if sample is None:
                continue
            r0 = sample.runs[0]
            rPr = r0._element.rPr
            rFonts = rPr.rFonts if rPr is not None else None
            east_asia = rFonts.get(qn("w:eastAsia")) if rFonts is not None else None
            western = rFonts.get(qn("w:ascii")) if rFonts is not None else None
            actual_pt = round(r0.font.size.pt, 1) if r0.font.size else None

            if rule.get("east_asia_font") and east_asia != rule["east_asia_font"]:
                print(f"Validation failed for {kind}: expected eastAsia={rule['east_asia_font']}, got {east_asia}", file=sys.stderr)
                return 1
            if rule.get("western_font") and western != rule["western_font"]:
                print(f"Validation failed for {kind}: expected ascii={rule['western_font']}, got {western}", file=sys.stderr)
                return 1
            expected_pt = _resolve_pt(rule.get("size") if "size" in rule else rule.get("size_pt"))
            if expected_pt is not None and (actual_pt is None or abs(actual_pt - expected_pt) > 0.2):
                print(f"Validation failed for {kind}: expected size={expected_pt}pt, got {actual_pt}pt", file=sys.stderr)
                return 1
            if rule.get("line_spacing") is not None:
                ls = sample.paragraph_format.line_spacing
                if ls is None or abs(float(ls) - float(rule["line_spacing"])) > 0.05:
                    print(f"Validation failed for {kind}: expected line_spacing={rule['line_spacing']}, got {ls}", file=sys.stderr)
                    return 1
            checks[kind] = {
                "east_asia": east_asia,
                "western": western,
                "size_pt": actual_pt,
                "line_spacing": sample.paragraph_format.line_spacing,
                "first_line_indent_pt": round(sample.paragraph_format.first_line_indent.pt, 1)
                if sample.paragraph_format.first_line_indent else None,
            }

        margins = spec.get("margins_cm")
        if isinstance(margins, dict) and out.sections:
            sec = out.sections[0]
            actual_m = {
                "top": round(sec.top_margin.cm, 2),
                "bottom": round(sec.bottom_margin.cm, 2),
                "left": round(sec.left_margin.cm, 2),
                "right": round(sec.right_margin.cm, 2),
            }
            for side in ("top", "bottom", "left", "right"):
                if margins.get(side) is not None and abs(actual_m[side] - float(margins[side])) > 0.05:
                    print(f"Validation failed: margin {side} expected {margins[side]}cm, got {actual_m[side]}cm", file=sys.stderr)
                    return 1
            checks["margins_cm"] = actual_m

    print(json.dumps({"validated": True, **checks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
