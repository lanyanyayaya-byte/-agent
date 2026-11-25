"""Read the WordprocessingML body in document order without rendering it."""

import csv
import re
from collections.abc import Iterator
from dataclasses import replace
from io import StringIO

from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from lxml import etree

from app.parsing.docx import docx_error
from app.parsing.docx_package import open_docx
from app.parsing.models import BlockKind, ContentBlock, DocxSourceLocation, ParsedDocument, ParseWarning, Section

Element = etree._Element
PARAGRAPH, TABLE, ROW, CELL = (qn(name) for name in ("w:p", "w:tbl", "w:tr", "w:tc"))
WRAPPERS = {qn(name) for name in ("w:sdt", "w:sdtContent", "w:customXml", "w:ins", "w:moveTo")}
OMITTED_BLOCKS = {qn(name) for name in ("w:altChunk", "w:subDoc")}
ALTERNATE_CONTENT = "{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent"
INLINE_WRAPPERS = WRAPPERS | {qn(name) for name in ("w:r", "w:hyperlink", "w:smartTag", "w:fldSimple", "w:dir", "w:bdo")}
MATH = {qn("m:oMath"), qn("m:oMathPara")}
INLINE_TEXT = {qn("w:tab"): "\t", qn("w:cr"): "\n", qn("w:noBreakHyphen"): "\u2011", qn("w:softHyphen"): "\u00ad"}
INLINE_WARNINGS = {
    qn("w:drawing"): "DOCX_IMAGE_OMITTED",
    qn("w:pict"): "DOCX_IMAGE_OMITTED",
    qn("w:object"): "DOCX_OBJECT_OMITTED",
    qn("w:txbxContent"): "DOCX_TEXTBOX_OMITTED",
    qn("w:footnoteReference"): "DOCX_NOTE_OMITTED",
    qn("w:endnoteReference"): "DOCX_NOTE_OMITTED",
    ALTERNATE_CONTENT: "DOCX_ALTERNATE_CONTENT_OMITTED",
}
WARNINGS = {
    "DOCX_ALT_BLOCK_OMITTED": "Imported or alternate body content was omitted.",
    "DOCX_IMAGE_OMITTED": "A drawing or image was omitted.",
    "DOCX_OBJECT_OMITTED": "An embedded object was omitted.",
    "DOCX_TEXTBOX_OMITTED": "A text box was omitted.",
    "DOCX_NOTE_OMITTED": "The content of a footnote or endnote was omitted.",
    "DOCX_ALTERNATE_CONTENT_OMITTED": "Alternate inline content was omitted.",
    "DOCX_MATH_TEXT_ONLY": "Only the text tokens of a Word equation were retained.",
    "DOCX_HEADING_UNRESOLVED": "A heading level could not be resolved from the Word properties.",
    "DOCX_LIST_UNRESOLVED": "The list numbering properties could not be fully resolved.",
    "DOCX_TABLE_GRID_UNCERTAIN": "The table has inconsistent grid or merge information.",
    "DOCX_NESTED_TABLE_FLATTENED": "A nested table was retained as text within its containing cell.",
}
STYLED_HEADING = re.compile(r"(?:heading|标题)[ _-]*([1-9])", re.IGNORECASE)
CHINESE_LEVELS = {f"{name}级标题": level for level, name in enumerate("一二三四五六七八九", 1)}
MAX_BODY_ITEMS = 40_000
MAX_TABLE_COLUMNS = 256
MAX_TABLE_GRID_SLOTS = 100_000
MAX_LOCATIONS = 45_000
MAX_TEXT_CHARACTERS = 4_000_000


def _value(node: Element | None, child_name: str) -> str | None:
    child = node.find(qn(child_name)) if node is not None else None
    return child.get(qn("w:val")) if child is not None else None


def _integer(value: str | None) -> int | None:
    # Bound numeric properties before using them as counts or grid positions.
    return int(value) if value and re.fullmatch(r"[0-9]{1,9}", value) else None


def _related_xml(document, relationship: str) -> Element | None:
    try:
        return document.part.part_related_by(relationship).element
    except KeyError:
        return None


class _ParagraphStyles:
    def __init__(self, styles: Element | None):
        self.styles = {}
        self.default = None
        self.chains = {}
        if styles is not None:
            for style in styles.iterchildren(qn("w:style")):
                style_id = style.get(qn("w:styleId"))
                if style_id and style.get(qn("w:type")) == "paragraph":
                    self.styles[style_id] = style
                    if style.get(qn("w:default")) in ("1", "true", "on"):
                        self.default = style_id

    def chain(self, properties: Element | None) -> tuple[tuple[str, ...], bool]:
        style_id = _value(properties, "w:pStyle") or self.default
        if style_id not in self.chains:
            keys = []
            current = style_id
            while current and current not in keys and len(keys) < 32:
                keys.append(current)
                current = _value(self.styles.get(current), "w:basedOn")
            self.chains[style_id] = (tuple(keys), bool(current))
        return self.chains[style_id]

    def heading_level(self, properties: Element | None, chain: tuple[str, ...]) -> int | None:
        level = self._outline_level(properties)
        if level is not None:
            return level
        for style_id in chain:
            style = self.styles.get(style_id)
            level = self._outline_level(style.find(qn("w:pPr")) if style is not None else None)
            if level is not None:
                return level
            for label in (_value(style, "w:name"), style_id):
                label = (label or "").strip()
                if match := STYLED_HEADING.fullmatch(label):
                    return int(match[1])
                if label in CHINESE_LEVELS:
                    return CHINESE_LEVELS[label]
        return None

    @staticmethod
    def _outline_level(properties: Element | None) -> int | None:
        outline = properties.find(qn("w:outlineLvl")) if properties is not None else None
        if outline is None:
            return None
        level = _integer(outline.get(qn("w:val")))
        # Level 9 is explicitly body text. Zero here signals an invalid property.
        return level + 1 if level is not None and level <= 9 else 0

    def numbering(self, properties: Element | None, chain: tuple[str, ...]) -> tuple[int | None, int | None]:
        direct = properties.find(qn("w:numPr")) if properties is not None else None
        raw_level = _value(direct, "w:ilvl")
        level = _integer(raw_level)
        num_id = _value(direct, "w:numId")
        if num_id is None:
            for style_id in chain:
                style = self.styles.get(style_id)
                if style is not None:
                    num_id = _value(style.find(f"{qn('w:pPr')}/{qn('w:numPr')}"), "w:numId")
                    if num_id is not None:
                        break
        parsed_id = _integer(num_id)
        return (
            -1 if num_id is not None and parsed_id is None else parsed_id,
            -1 if raw_level is not None and level is None else level,
        )


class _Numbering:
    def __init__(self, root: Element | None):
        self.nums = {}
        self.abstracts = {}
        if root is not None:
            for abstract in root.iterchildren(qn("w:abstractNum")):
                levels = {
                    _integer(level.get(qn("w:ilvl"))): level for level in abstract.iterchildren(qn("w:lvl"))
                }
                styles = {_value(node, "w:pStyle"): key for key, node in levels.items() if key in range(9)}
                self.abstracts[_integer(abstract.get(qn("w:abstractNumId")))] = (levels, styles)
            for num in root.iterchildren(qn("w:num")):
                self.nums[_integer(num.get(qn("w:numId")))] = (
                    _integer(_value(num, "w:abstractNumId")),
                    {_integer(level.get(qn("w:ilvl"))): level.find(qn("w:lvl"))
                     for level in num.iterchildren(qn("w:lvlOverride"))},
                )

    def resolve(self, num_id: int, level: int | None, chain: tuple[str, ...]) -> tuple[int | None, str | None]:
        if num_id not in self.nums:
            return level, None
        abstract, overrides = self.nums[num_id]
        levels, by_style = self.abstracts.get(abstract, ({}, {}))
        if level is None:
            level = next((by_style[key] for key in chain if key in by_style), 0)
        override = overrides.get(level)
        fmt = _value(override if override is not None else levels.get(level), "w:numFmt")
        return level, fmt


def _children(node: Element, tags: tuple[str, ...]) -> Iterator[Element]:
    for child in node:
        if child.tag in tags:
            yield child
        elif child.tag in WRAPPERS:
            yield from _children(child, tags)


class _Layout:
    def __init__(self, version_id: str):
        self.version_id = version_id
        self.blocks: list[ContentBlock] = []
        self.sections: list[Section] = []
        self.active: list[Section] = []
        self.warnings: dict[str, ParseWarning] = {}
        self.body_index = 0
        self.locations = 0
        self.text_characters = 0
        self.table_grid_slots = 0

    def warn(self, code: str) -> None:
        if code not in self.warnings:
            self.warnings[code] = ParseWarning(code, f"{WARNINGS[code]} First occurrence at body item {self.body_index}.")

    def emit(self, kind: BlockKind, text: str, locations: tuple[DocxSourceLocation, ...]) -> None:
        block_id = f"{self.version_id}:docx:{self.body_index}"
        if kind == "heading":
            level = locations[0].heading_level
            assert level is not None
            while self.active and self.active[-1].level >= level:
                self.active.pop()
            section = Section(text, level, block_id, self.active[-1].heading_block_id if self.active else None)
            self.sections.append(section)
            self.active.append(section)
        path = tuple(section.title for section in self.active)
        self.text_characters += len(text) + sum(map(len, path))
        self.locations += len(locations)
        if self.text_characters > MAX_TEXT_CHARACTERS or self.locations > MAX_LOCATIONS:
            raise docx_error("DOCX_RESOURCE_LIMIT")
        self.blocks.append(ContentBlock(
            block_id, self.version_id, kind, text, locations,
            self.active[-1].heading_block_id if self.active else None, path,
        ))

    def body_items(self, node: Element) -> Iterator[Element]:
        for child in node:
            if child.tag in (PARAGRAPH, TABLE):
                self.body_index += 1
                if self.body_index > MAX_BODY_ITEMS:
                    raise docx_error("DOCX_RESOURCE_LIMIT")
                yield child
            elif child.tag in WRAPPERS:
                yield from self.body_items(child)
            elif child.tag in OMITTED_BLOCKS or child.tag == ALTERNATE_CONTENT:
                self.body_index += 1
                self.warn("DOCX_ALT_BLOCK_OMITTED")

    def paragraph_text(self, paragraph: Element) -> str:
        return "".join(self._inline_text(paragraph))

    def _inline_text(self, node: Element) -> Iterator[str]:
        for child in node:
            tag = child.tag
            if tag in INLINE_WRAPPERS:
                yield from self._inline_text(child)
            elif tag == qn("w:t"):
                yield child.text or ""
            elif tag in INLINE_TEXT:
                yield INLINE_TEXT[tag]
            elif tag == qn("w:br") and child.get(qn("w:type"), "textWrapping") == "textWrapping":
                yield "\n"
            elif tag in MATH:
                self.warn("DOCX_MATH_TEXT_ONLY")
                yield "".join(token.text or "" for token in child.iter(qn("m:t")))
            elif tag in INLINE_WARNINGS:
                if tag != qn("w:txbxContent") and next(child.iter(qn("w:txbxContent")), None) is not None:
                    self.warn("DOCX_TEXTBOX_OMITTED")
                else:
                    self.warn(INLINE_WARNINGS[tag])

    def cell_text(self, cell: Element) -> str:
        parts = []
        for item in _children(cell, (PARAGRAPH, TABLE)):
            if item.tag == PARAGRAPH:
                text = self.paragraph_text(item)
            else:
                self.warn("DOCX_NESTED_TABLE_FLATTENED")
                text = "\n".join(
                    "\t".join(self.cell_text(nested) for nested in _children(row, (CELL,)))
                    for row in _children(item, (ROW,))
                )
            if text.strip():
                parts.append(text)
        return "\n".join(parts)

    def table(self, table: Element) -> tuple[str, tuple[DocxSourceLocation, ...]]:
        grid = table.find(qn("w:tblGrid"))
        grid_width = len(grid.findall(qn("w:gridCol"))) if grid is not None else 0
        if not grid_width:
            self.warn("DOCX_TABLE_GRID_UNCERTAIN")
        width = grid_width
        cells: list[DocxSourceLocation] = []
        rows: list[dict[int, str]] = []
        merges: dict[tuple[int, int], int] = {}
        for row in _children(table, (ROW,)):
            row_properties = row.find(qn("w:trPr"))
            column = self._grid_number(row_properties, "w:gridBefore", 0) + 1
            values: dict[int, str] = {}
            next_merges: dict[tuple[int, int], int] = {}
            rows.append(values)
            for cell in _children(row, (CELL,)):
                props = cell.find(qn("w:tcPr"))
                span = self._grid_number(props, "w:gridSpan", 1)
                merge = props.find(qn("w:vMerge")) if props is not None else None
                mode = merge.get(qn("w:val"), "continue") if merge is not None else None
                anchor = merges.get((column, span)) if mode == "continue" else None
                text = self.cell_text(cell)
                if anchor is not None and not text.strip():
                    previous = cells[anchor]
                    cells[anchor] = replace(previous, row_span=previous.row_span + 1)
                    next_merges[column, span] = anchor
                else:
                    if mode not in (None, "restart") or (props is not None and props.find(qn("w:hMerge")) is not None):
                        self.warn("DOCX_TABLE_GRID_UNCERTAIN")
                    if mode == "restart":
                        next_merges[column, span] = len(cells)
                    cells.append(DocxSourceLocation(
                        body_index=self.body_index, table_row=len(rows), table_column=column,
                        row_span=1, column_span=span,
                    ))
                    values[column] = text
                column += span
            merges = next_merges
            row_width = column - 1 + self._grid_number(row_properties, "w:gridAfter", 0)
            if grid_width and row_width != grid_width:
                self.warn("DOCX_TABLE_GRID_UNCERTAIN")
            width = max(width, row_width)
            if width > MAX_TABLE_COLUMNS or self.table_grid_slots + width * len(rows) > MAX_TABLE_GRID_SLOTS:
                raise docx_error("DOCX_RESOURCE_LIMIT")
            if self.locations + len(cells) > MAX_LOCATIONS:
                raise docx_error("DOCX_RESOURCE_LIMIT")
        self.table_grid_slots += width * len(rows)
        if not cells:
            self.warn("DOCX_TABLE_GRID_UNCERTAIN")
        output = StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for values in rows:
            writer.writerow(values.get(column, "") for column in range(1, width + 1))
        return output.getvalue().removesuffix("\n"), tuple(cells)

    def _grid_number(self, props: Element | None, name: str, default: int) -> int:
        raw = _value(props, name)
        value = _integer(raw)
        if value is None or value < (1 if name == "w:gridSpan" else 0):
            if raw is not None:
                self.warn("DOCX_TABLE_GRID_UNCERTAIN")
            return default
        return value


def read_docx(data: bytes, document_id: str, version_id: str) -> ParsedDocument:
    document = open_docx(data)
    styles = _ParagraphStyles(_related_xml(document, RT.STYLES))
    numbering = _Numbering(_related_xml(document, RT.NUMBERING))
    layout = _Layout(version_id)
    paragraph_number = 0
    for item in layout.body_items(document._element.body):
        if item.tag == TABLE:
            text, locations = layout.table(item)
            if locations:
                layout.emit("table", text, locations)
            continue
        paragraph_number += 1
        text = layout.paragraph_text(item)
        if not text.strip():
            continue
        properties = item.find(qn("w:pPr"))
        chain, truncated = styles.chain(properties)
        level = styles.heading_level(properties, chain)
        if level == 0 or truncated:
            layout.warn("DOCX_HEADING_UNRESOLVED")
        heading = level if level in range(1, 10) else None
        num_id, list_level = styles.numbering(properties, chain)
        list_format = None
        if num_id == -1:
            layout.warn("DOCX_LIST_UNRESOLVED")
            num_id = None
        if num_id:
            list_level, list_format = numbering.resolve(num_id, list_level, chain)
            if list_format is None or (list_level is not None and list_level not in range(9)):
                layout.warn("DOCX_LIST_UNRESOLVED")
            list_level = list_level + 1 if list_level in range(9) else None
        else:
            list_level = None
        kind: BlockKind = "heading" if heading else "list" if num_id else "paragraph"
        layout.emit(kind, text, (DocxSourceLocation(
            paragraph_number=paragraph_number, heading_level=heading, body_index=layout.body_index,
            numbering_id=num_id or None, list_level=list_level, list_format=list_format,
        ),))

    return ParsedDocument(document_id, version_id, "docx", tuple(layout.sections), tuple(layout.blocks),
                          tuple(layout.warnings.values()))
