from dataclasses import dataclass
from typing import Literal

from app.storage.library import DocumentFormat

BlockKind = Literal["heading", "paragraph", "list", "code", "table", "blockquote", "thematic_break", "html"]


@dataclass(frozen=True)
class PdfSourceLocation:
    """Physical pages are one-based.

    A known box uses points as (x0, y0, x1, y1) from the unrotated MediaBox's top left.
    """

    physical_page: int | None = None
    page_label: str | None = None
    page_bbox: tuple[float, float, float, float] | None = None
    coordinate_system: Literal["unrotated_mediabox_top_left_pt"] = "unrotated_mediabox_top_left_pt"


@dataclass(frozen=True)
class DocxSourceLocation:
    """One-based body, paragraph and table grid coordinates, without pagination."""

    paragraph_number: int | None = None
    heading_level: int | None = None
    body_index: int | None = None
    table_row: int | None = None
    table_column: int | None = None
    row_span: int | None = None
    column_span: int | None = None
    numbering_id: int | None = None
    list_level: int | None = None
    list_format: str | None = None


@dataclass(frozen=True)
class MarkdownSourceLocation:
    """Source lines are one-based, with both endpoints included."""

    start_line: int | None = None
    end_line: int | None = None
    heading_level: int | None = None


SourceLocation = PdfSourceLocation | DocxSourceLocation | MarkdownSourceLocation


@dataclass(frozen=True)
class Section:
    title: str
    level: int
    heading_block_id: str
    parent_heading_block_id: str | None


@dataclass(frozen=True)
class ContentBlock:
    """A structural block with any number of source locations in its version."""

    block_id: str
    document_version_id: str
    kind: BlockKind
    text: str
    source_locations: tuple[SourceLocation, ...]
    section_heading_block_id: str | None = None
    section_path: tuple[str, ...] = ()
    code_language: str | None = None


@dataclass(frozen=True)
class ParseWarning:
    code: str
    message: str
    physical_page: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    document_version_id: str
    format: DocumentFormat
    sections: tuple[Section, ...]
    blocks: tuple[ContentBlock, ...]
    warnings: tuple[ParseWarning, ...] = ()


class DocumentParseError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
