#!/usr/bin/env python3
"""Inspect a DOCX file's structure, styles, tables, images, and special XML features."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from docx import Document


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: inspect_docx.py <path-to-docx>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"DOCX file not found: {path}", file=sys.stderr)
        return 2

    doc = Document(str(path))
    style_counts: dict[str, int] = {}
    sample_paragraphs: list[dict[str, object]] = []
    for idx, p in enumerate(doc.paragraphs):
        sname = p.style.name if p.style else "None"
        style_counts[sname] = style_counts.get(sname, 0) + 1
        if len(sample_paragraphs) < 60 and p.text.strip():
            first_run = next((r for r in p.runs if r.text.strip()), None)
            sample_paragraphs.append({
                "index": idx,
                "style": sname,
                "bold": bool(first_run.bold) if first_run else False,
                "size_pt": round(first_run.font.size.pt, 1) if (first_run and first_run.font.size) else None,
                "text": p.text[:140],
            })

    limitations: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        xml_text = zf.read("word/document.xml").decode("utf-8", errors="ignore") if "word/document.xml" in names else ""
        if "m:oMath" in xml_text:
            limitations.append("包含数学公式对象（OMML），排版脚本仅调整文本样式，不改写公式内容。")
        if "w:fldChar" in xml_text or "w:instrText" in xml_text:
            limitations.append("包含 Word 域代码（如目录/页码/交叉引用），重新打开后可能需更新域。")
        if "word/comments.xml" in names:
            limitations.append("包含批注内容。")
        if "w:ins" in xml_text or "w:del" in xml_text:
            limitations.append("包含修订痕迹（Track Changes）。")
        media_files = [n for n in names if n.startswith("word/media/")]

    margins = []
    for sec in doc.sections:
        margins.append({
            "top_cm": round(sec.top_margin.cm, 2) if sec.top_margin else None,
            "bottom_cm": round(sec.bottom_margin.cm, 2) if sec.bottom_margin else None,
            "left_cm": round(sec.left_margin.cm, 2) if sec.left_margin else None,
            "right_cm": round(sec.right_margin.cm, 2) if sec.right_margin else None,
        })

    print(json.dumps({
        "file": str(path),
        "paragraph_count": len(doc.paragraphs),
        "table_count": len(doc.tables),
        "image_count": len(media_files),
        "section_count": len(doc.sections),
        "margins": margins,
        "style_counts": style_counts,
        "sample_paragraphs": sample_paragraphs,
        "limitations": limitations,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
