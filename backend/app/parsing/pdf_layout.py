"""Extract PDF text and geometry inside the isolated PDF worker."""

import re
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from math import isfinite
from statistics import median

import pdfplumber
from pdfminer.converter import PDFPageAggregator
from pdfminer.data_structures import NumberTree
from pdfminer.layout import LTPage
from pdfminer.pdfdocument import PDFEncryptionError, PDFNoPageLabels
from pdfminer.pdfexceptions import PDFException
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.pdftypes import resolve1
from pdfminer.psparser import PSLiteral, PSSyntaxError
from pdfplumber.page import Page
from pdfplumber.utils import cluster_objects
from pdfplumber.utils.exceptions import PdfminerException

from app.parsing.models import ContentBlock, DocumentParseError, ParsedDocument, ParseWarning, PdfSourceLocation, Section
from app.parsing.pdf import ERRORS


MAX_PAGES = 1000
MAX_PAGE_GLYPHS = 50_000
MAX_DOCUMENT_GLYPHS = 2_000_000
MAX_DOCUMENT_LINES = 50_000
MAX_DOCUMENT_TEXT = 4_000_000
MAX_LABEL_LENGTH = 64
# Titles can start with a word such as "4-bit", as well as a letter.
NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(?:[0-9]+[-–])?[^\W\d_]")
ROMAN_HEADING = re.compile(r"^[IVXLCDM]{1,7}\.\s+[A-Z]")
CHINESE_HEADING = re.compile(r"^第[零〇一二三四五六七八九十百0-9]+([章篇节])\s*[^\W\d_]")
DOT_LEADER = re.compile(r"\.(?:\s*\.){2,}")
TEX_SYMBOL_FONT = re.compile(r"^(?:[A-Z]{6}\+)?(?:CMSY|CMEX|CMMI|MSAM|MSBM)\d*$")
BOLD_FONT = re.compile(r"bold|NimbusRomNo9L-Medi|\bCMBX\d|\bSimHei\b", re.I)
NAMED_HEADINGS = {"abstract", "introduction", "conclusion", "conclusions", "acknowledgments",
                  "references", "appendix", "摘要", "引言", "结论", "参考文献", "附录"}
UNMAPPED = re.compile(r"\(cid:\d+\)|[\ufffd\ue000-\uf8ff\U000f0000-\U0010fffd]")
Box = tuple[float, float, float, float]
PageBlock = tuple[str, int | None, tuple[PdfSourceLocation, ...]]


class TextAndImageAggregator(PDFPageAggregator):
    def paint_path(self, gstate, stroke, fill, evenodd, path) -> None:
        # The text layer and image boxes suffice here. Retaining vector paths
        # from figures can dominate peak memory on scientific PDFs.
        pass


class TextPage(Page):
    @property
    def layout(self) -> LTPage:
        if not hasattr(self, "_layout"):
            device = TextAndImageAggregator(self.pdf.rsrcmgr, pageno=self.page_number)
            PDFPageInterpreter(self.pdf.rsrcmgr, device).process_page(self.page_obj)
            self._layout = device.get_result()
        return self._layout


@dataclass(frozen=True)
class Line:
    text: str
    box: Box
    baseline: float
    size: float
    bold: bool
    leading_glyph_size: float

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]


def _fail(code: str) -> DocumentParseError:
    return DocumentParseError(code, ERRORS[code])


def _page_labels(pdf: pdfplumber.PDF, page_count: int) -> tuple[list[str], bool]:
    if "PageLabels" not in pdf.doc.catalog:
        return [], False
    tree = pdf.doc.catalog["PageLabels"]

    # Validate before calling pdfminer's formatter. It is deliberately lenient
    # about malformed metadata and can repeat letters for a huge /St value.
    try:
        entries = NumberTree(tree).values
        previous = -1
        for position, (index, entry) in enumerate(entries):
            entry = resolve1(entry)
            if not previous < index < page_count or not isinstance(entry, dict):
                return [], True
            previous = index
            style, prefix, start = (resolve1(entry.get(key, default)) for key, default in
                                    (("S", None), ("P", b""), ("St", 1)))
            if style is not None and (not isinstance(style, PSLiteral) or style.name not in ("D", "R", "r", "A", "a")):
                return [], True
            if not isinstance(prefix, bytes) or len(prefix) > 2 * MAX_LABEL_LENGTH + 2:
                return [], True
            if type(start) is not int or start < 1 or start.bit_length() > 200:
                return [], True
            end = entries[position + 1][0] if position + 1 < len(entries) else page_count
            last = start + end - index - 1
            if style is not None:
                if style.name in ("R", "r") and last > 3999:
                    return [], True
                if style.name in ("A", "a") and last > 26 * MAX_LABEL_LENGTH:
                    return [], True
        if not entries or entries[0][0] != 0:
            return [], True
        # Read no more than the document page limit from the unbounded iterator.
        labels = list(islice(pdf.doc.get_page_labels(), page_count))
        return (labels, False) if len(labels) == page_count else ([], True)
    except (AssertionError, KeyError, RecursionError, TypeError, ValueError, PDFNoPageLabels,
            PDFSyntaxError, PSSyntaxError):
        return [], True
    finally:
        # PDFPage's own iterator also reads labels. Keep metadata failures out of
        # page enumeration and use the separately validated labels above.
        pdf.doc.catalog.pop("PageLabels", None)


def _box_transform(box: Box, rotation: int, width: float, height: float) -> Box:
    """Rotate top-down coordinates clockwise around the page, preserving the box."""
    x0, y0, x1, y1 = box
    if rotation == 90:
        return height - y1, x0, height - y0, x1
    if rotation == 180:
        return width - x1, height - y1, width - x0, height - y0
    if rotation == 270:
        return y0, width - x1, y1, width - x0
    return box


def _intersection(box: Box, bounds: Box) -> Box | None:
    x0, y0 = max(box[0], bounds[0]), max(box[1], bounds[1])
    x1, y1 = min(box[2], bounds[2]), min(box[3], bounds[3])
    if x0 < x1 and y0 < y1:
        return x0, y0, x1, y1
    return None


def _relative_box(item: dict, x: float, y: float) -> Box:
    return item["x0"] - x, item["top"] - y, item["x1"] - x, item["bottom"] - y


def _direction(char: dict) -> int | None:
    a, b = char["matrix"][:2]
    if abs(b) < abs(a) * 0.1:
        return 0 if a > 0 else 180
    if abs(a) < abs(b) * 0.1:
        return 90 if b < 0 else 270
    return None


def _uncertain_math_fonts(page: Page) -> set[str]:
    """TeX symbol slots can look like ordinary ASCII without an explicit map."""
    fonts = resolve1(page.page_obj.resources.get("Font", {}))
    if not isinstance(fonts, dict):
        return set()
    names = set()
    for reference in fonts.values():
        font = resolve1(reference)
        if isinstance(font, dict) and not font.get("ToUnicode"):
            base = font.get("BaseFont")
            if isinstance(base, PSLiteral) and TEX_SYMBOL_FONT.fullmatch(base.name):
                names.add(base.name)
    return names


def _visible_glyphs(page: Page, crop: Box) -> tuple[list[dict], int, int]:
    count = len(page.chars)
    if count > MAX_PAGE_GLYPHS:
        raise _fail("PDF_RESOURCE_LIMIT")
    origin_x, origin_y = page.mediabox[:2]
    uncertain_fonts = _uncertain_math_fonts(page)
    visible = []
    unknown = 0
    for glyph in page.chars:
        box = _relative_box(glyph, origin_x, origin_y)
        if not all(isfinite(value) for value in box):
            raise _fail("PDF_INVALID")
        clipped = _intersection(box, crop)
        if clipped is None:
            continue
        text = glyph["text"]
        if UNMAPPED.search(text) or not (text.isprintable() or text.isspace()) or glyph["fontname"] in uncertain_fonts:
            unknown += 1
            text = " "
        visible.append({
            "box": clipped, "direction": _direction(glyph), "text": text, "size": glyph["size"],
            "fontname": glyph["fontname"], "origin": glyph["matrix"][4:],
        })
    return visible, count, unknown


def _make_line(glyphs: list[dict]) -> Line | None:
    words = []
    right = glyphs[0]["x0"]
    for glyph in glyphs:
        text = glyph["text"]
        if text.isspace() or (words and glyph["x0"] - right > max(0.8, 0.12 * glyph["size"])):
            words.append(" ")
        if not text.isspace():
            words.append(text)
        right = max(right, glyph["x1"])
    text = " ".join("".join(words).split())
    if not text:
        return None

    ink = [glyph for glyph in glyphs if glyph["text"].strip()]
    box = (min(g["x0"] for g in ink), min(g["top"] for g in ink),
           max(g["x1"] for g in ink), max(g["bottom"] for g in ink))
    bold = sum(bool(BOLD_FONT.search(g["fontname"])) for g in ink)
    return Line(
        text, box, median(g["baseline"] for g in ink), median(g["size"] for g in ink),
        bold > len(ink) / 2, ink[0]["size"],
    )


def _make_lines(glyphs: list[dict], width: float, height: float) -> tuple[list[Line], int, int]:
    orientations = Counter(g["direction"] for g in glyphs if g["text"].strip() and g["direction"] is not None)
    direction = orientations.most_common(1)[0][0] if orientations else 0
    rotation = (-direction) % 360
    rotated = {}
    excluded = 0
    for glyph in glyphs:
        if glyph["direction"] != direction:
            excluded += bool(glyph["text"].strip())
            continue
        x0, y0, x1, y1 = _box_transform(glyph["box"], rotation, width, height)
        base_x, base_y = glyph["origin"]
        base_y = height - base_y
        baseline = _box_transform((base_x, base_y, base_x, base_y), rotation, width, height)[1]
        # Text is sometimes painted twice at the same position. Keep the first
        # glyph within a quarter-point cell without scanning the page for matches.
        size, font, text = glyph["size"], glyph["fontname"], glyph["text"]
        key = (font, size, text, round(x0 * 4), round(y0 * 4))
        if key not in rotated:
            rotated[key] = {"x0": x0, "x1": x1, "top": y0, "bottom": y1, "baseline": baseline,
                            "size": size, "fontname": font, "text": text}
    if not rotated:
        return [], rotation, excluded

    body_size = median(g["size"] for g in rotated.values())
    lines = []
    for row in cluster_objects(rotated.values(), "baseline", tolerance=min(3, max(0.9, body_size * 0.21))):
        run = []
        right = 0.0
        for glyph in sorted(row, key=lambda g: g["x0"]):
            if run and glyph["x0"] - right > max(13, 1.4 * max(body_size, glyph["size"], run[-1]["size"])):
                line = _make_line(run)
                if line is not None:
                    lines.append(line)
                run = []
            right = max(right, glyph["x1"]) if run else glyph["x1"]
            run.append(glyph)
        line = _make_line(run)
        if line is not None:
            lines.append(line)
    return sorted(lines, key=lambda line: (line.baseline, line.box[0])), rotation, excluded


def _find_gutter(lines: list[Line], width: float) -> tuple[float, float] | None:
    """Require repeated, vertically overlapping text on both sides of a gutter."""
    substantial = [line for line in lines if len(line.text) >= 8 and line.width >= width * 0.14]
    best = None
    for percent in range(34, 67, 2):
        split = width * percent / 100
        left = [line for line in substantial if line.box[2] <= split - 2]
        right = [line for line in substantial if line.box[0] >= split + 2]
        if min(len(left), len(right)) < 4:
            continue
        # Compare local y coverage instead of requiring identical baselines.
        left_bands = {int(line.baseline / 18) for line in left}
        right_bands = {int(line.baseline / 18) for line in right}
        overlap = len(left_bands & right_bands)
        if overlap < max(3, min(len(left_bands), len(right_bands)) * 0.35):
            continue
        right_start = Counter(round(line.box[0] / 5) for line in right).most_common(1)[0]
        if right_start[1] < max(3, len(right) * 0.3):
            continue
        crossing = len(substantial) - len(left) - len(right)
        score = (overlap, -crossing, min(len(left), len(right)), -abs(percent - 50))
        if best is None or score > best[0]:
            best = score, split, right_start[0] * 5.0
    return (best[1], best[2]) if best else None


def _regions(lines: list[Line], width: float) -> Iterator[list[Line]]:
    gutter = _find_gutter(lines, width)
    if gutter is None:
        yield lines
        return

    split, right_start = gutter
    left, right = [], []
    for line in lines:
        x0, _, x1, _ = line.box
        centered = abs((x0 + x1) / 2 - width / 2) < 8 and x0 < right_start - 7
        if (x0 < split - 2 and x1 > split + 2) or centered:
            yield from (left, right, [line])
            left, right = [], []
        elif x0 >= split - 2:
            right.append(line)
        else:
            left.append(line)
    yield from (left, right)


def _heading_level(line: Line, body_size: float) -> int | None:
    text = line.text
    if len(text) > 105 or DOT_LEADER.search(text):
        return None
    # Small leading digits can be algorithm line numbers or reference marks.
    heading_sized_prefix = line.leading_glyph_size >= line.size * 0.85
    if (heading_sized_prefix and line.bold and line.size >= body_size * 0.83
            and text.isupper() and ROMAN_HEADING.match(text)):
        return 1
    if line.size < body_size * 0.92:
        return None
    if text.casefold() in NAMED_HEADINGS:
        return 1
    if line.bold or line.size >= body_size * 1.2:
        numbered = NUMBERED_HEADING.match(text)
        if numbered and heading_sized_prefix:
            return numbered[1].count(".") + 1
        chinese = CHINESE_HEADING.match(text)
        if chinese:
            return 2 if chinese[1] == "节" else 1
    return None


def _paragraphs(
    lines: list[Line], body_size: float, toc_rows: set[int],
) -> Iterator[tuple[list[Line], int | None]]:
    group: list[Line] = []
    for line in lines:
        level = None if round(line.baseline) in toc_rows else _heading_level(line, body_size)
        if group:
            last = group[-1]
            step = line.baseline - last.baseline
            if (level is not None or not 0.48 * last.size <= step <= 1.65 * max(last.size, line.size)
                    or abs(line.size - last.size) > max(1, 0.15 * last.size)
                    or abs(line.box[0] - last.box[0]) > max(7, 0.8 * last.size)):
                yield group, None
                group = []
        if level is None:
            group.append(line)
        else:
            yield [line], level
    if group:
        yield group, None


def _page_blocks(page: Page, label: str | None) -> tuple[list[PageBlock], list[ParseWarning], int, int]:
    width, height = page.width, page.height
    rotation = page.rotation
    unit = resolve1(page.page_obj.attrs.get("UserUnit", 1))
    if (rotation not in (0, 90, 180, 270) or type(unit) not in (int, float)
            or not 0 < unit <= 75_000 or not all(isfinite(v) and v > 0 for v in (width, height))):
        raise _fail("PDF_INVALID")

    x0, y0, x1, y1 = page.page_obj.mediabox
    c0, d0, c1, d1 = page.page_obj.cropbox
    crop_raw = (c0 - x0, y1 - d1, c1 - x0, y1 - d0)
    crop = _intersection(_box_transform(crop_raw, rotation, x1 - x0, y1 - y0), (0, 0, width, height))
    if crop is None:
        raise _fail("PDF_INVALID")

    glyphs, count, unknown = _visible_glyphs(page, crop)
    lines, analysis_rotation, excluded = _make_lines(glyphs, width, height)
    analysis_width, analysis_height = (height, width) if analysis_rotation in (90, 270) else (width, height)

    physical_page = page.page_number
    warnings = []
    chars = sum(len("".join(line.text.split())) for line in lines)
    cx0, cy0, cx1, cy1 = crop
    crop_area = (cx1 - cx0) * (cy1 - cy0)
    if chars < 220:
        for image in page.images:
            box = _intersection(_relative_box(image, *page.mediabox[:2]), crop)
            min_coverage = 0.5 if chars < 32 else 0.85
            if box and (box[2] - box[0]) * (box[3] - box[1]) >= crop_area * min_coverage:
                warnings.append(ParseWarning("PDF_SUSPECTED_SCAN", "A large image has little or no text.", physical_page))
                break
    if not lines:
        warnings.append(ParseWarning("PDF_NO_TEXT_LAYER", "No usable text was extracted from this page.", physical_page))
    if unknown:
        warnings.append(ParseWarning("PDF_UNMAPPED_GLYPHS", f"{unknown} unmapped glyphs were omitted.", physical_page))
    if excluded:
        warnings.append(ParseWarning(
            "PDF_MIXED_TEXT_DIRECTION", f"{excluded} glyphs outside the main text direction were omitted.",
            physical_page,
        ))

    blocks = []
    body_size = median(g["size"] for g in glyphs) if glyphs else 0
    toc_rows = {round(line.baseline) for line in lines if line.box[0] >= 0.70 * analysis_width
                and len(line.text) <= 4 and line.text.isascii() and line.text.isdigit()}
    for region in _regions(lines, analysis_width):
        for group, level in _paragraphs(region, body_size, toc_rows):
            locations = []
            for line in group:
                displayed = _box_transform(line.box, -analysis_rotation % 360, analysis_width, analysis_height)
                raw = _box_transform(displayed, -rotation % 360, width, height)
                box = tuple(round(max(0, v) * unit, 6) for v in raw)
                if box[0] >= box[2] or box[1] >= box[3]:
                    raise _fail("PDF_INVALID")
                locations.append(PdfSourceLocation(physical_page, label, box))
            blocks.append(("\n".join(line.text for line in group), level, tuple(locations)))
    return blocks, warnings, count, len(lines)


def read_pdf(data: bytes, document_id: str, version_id: str, *, first_page_only: bool = False) -> ParsedDocument:
    if (not re.match(rb"%PDF-(?:1\.[0-7]|2\.0)[ \t]*[\r\n]", data)
            or not data[-1024:].rstrip(b"\x00\t\n\f\r ").endswith(b"%%EOF")):
        raise _fail("PDF_INVALID")

    sections: list[Section] = []
    active_sections: list[Section] = []
    blocks: list[ContentBlock] = []
    warnings: list[ParseWarning] = []
    total_glyphs = total_lines = total_text = 0
    physical_page = 0
    try:
        with BytesIO(data) as source:
            pdf = pdfplumber.open(source)
            if pdf.doc.encryption is not None:
                raise _fail("PDF_ENCRYPTED")
            page_tree = resolve1(pdf.doc.catalog.get("Pages"))
            if not isinstance(page_tree, dict) or not isinstance(resolve1(page_tree.get("Kids")), list):
                raise _fail("PDF_INVALID")
            page_count = resolve1(page_tree.get("Count"))
            if type(page_count) is not int or page_count < 1:
                raise _fail("PDF_INVALID")
            if page_count > MAX_PAGES:
                raise _fail("PDF_RESOURCE_LIMIT")
            labels, labels_invalid = _page_labels(pdf, page_count)
            if labels_invalid:
                warnings.append(ParseWarning("PDF_PAGE_LABELS_INVALID", "Unreliable PDF page labels were omitted."))
            # Iterate lazily so the limit is applied before allocating every Page.
            for physical_page, page_obj in enumerate(PDFPage.create_pages(pdf.doc), 1):
                if physical_page > MAX_PAGES:
                    raise _fail("PDF_RESOURCE_LIMIT")
                if physical_page > page_count or "MediaBox" not in page_obj.attrs:
                    raise _fail("PDF_INVALID")
                page = TextPage(pdf, page_obj, page_number=physical_page)
                label = labels[physical_page - 1] if labels else None
                if label and (len(label) > MAX_LABEL_LENGTH or not label.isprintable()):
                    warnings.append(ParseWarning("PDF_PAGE_LABELS_INVALID", "Unreliable page label omitted.", physical_page))
                    label = None
                try:
                    extracted, page_warnings, glyph_count, line_count = _page_blocks(page, label or None)
                finally:
                    page.close()
                warnings.extend(page_warnings)
                total_glyphs += glyph_count
                total_lines += line_count
                total_text += sum(len(text) for text, _, _ in extracted)
                if (total_glyphs > MAX_DOCUMENT_GLYPHS or total_lines > MAX_DOCUMENT_LINES
                        or total_text > MAX_DOCUMENT_TEXT):
                    raise _fail("PDF_RESOURCE_LIMIT")

                for sequence, (text, level, locations) in enumerate(extracted, 1):
                    block_id = f"{version_id}:pdf:{physical_page}:{sequence}"
                    if level is not None:
                        while active_sections and active_sections[-1].level >= level:
                            active_sections.pop()
                        parent = active_sections[-1].heading_block_id if active_sections else None
                        section = Section(text, level, block_id, parent)
                        sections.append(section)
                        active_sections.append(section)
                    blocks.append(ContentBlock(
                        block_id, version_id, "heading" if level is not None else "paragraph", text, locations,
                        active_sections[-1].heading_block_id if active_sections else None,
                        tuple(section.title for section in active_sections),
                    ))
                if first_page_only:
                    break
    except (PdfminerException, PDFException, PDFSyntaxError, PSSyntaxError, TypeError, ValueError) as error:
        underlying = error.args[0] if isinstance(error, PdfminerException) and error.args else error
        if isinstance(underlying, PDFEncryptionError):
            raise _fail("PDF_ENCRYPTED") from None
        if isinstance(underlying, (MemoryError, RecursionError)):
            raise _fail("PDF_RESOURCE_LIMIT") from None
        raise _fail("PDF_INVALID") from None
    if physical_page == 0 or (not first_page_only and physical_page != page_count):
        raise _fail("PDF_INVALID")
    if (
        not first_page_only
        and sys.platform == "darwin"
        and sum(1 for w in warnings if w.code == "PDF_SUSPECTED_SCAN") >= max(2, page_count // 4)
        and sum(1 for w in warnings if w.code == "PDF_NO_TEXT_LAYER") >= max(2, page_count // 2)
    ):
        from app.parsing.darwin_ocr import ocr_scanned_pdf

        ocr_result = ocr_scanned_pdf(data, version_id, page_count, labels, warnings)
        if ocr_result is not None and ocr_result[1]:
            sections, blocks, warnings = ocr_result
    return ParsedDocument(document_id, version_id, "pdf", tuple(sections), tuple(blocks), tuple(warnings))
