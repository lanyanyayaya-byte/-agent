import json
from dataclasses import asdict, dataclass
from typing import ClassVar, Literal

from app.parsing.models import BlockKind, ParseWarning, SourceLocation
from app.storage.library import DocumentFormat
from app.storage.parsed_documents import ParseSpec


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Limits on len(Chunk.text), including inter-block separators."""

    length_unit: ClassVar[str] = "unicode_codepoints"
    chunk_size: int = 1000
    chunk_overlap: int = 120
    long_paragraph_strategy: Literal["sentence", "line", "character"] = "sentence"
    title_context_strategy: Literal["none", "leaf", "path"] = "path"

    def __post_init__(self) -> None:
        if (type(self.chunk_size) is not int or type(self.chunk_overlap) is not int
                or not 0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError("Expected 0 <= chunk_overlap < chunk_size with integer lengths.")
        if self.long_paragraph_strategy not in ("sentence", "line", "character"):
            raise ValueError("Invalid long paragraph strategy.")
        if self.title_context_strategy not in ("none", "leaf", "path"):
            raise ValueError("Invalid title context strategy.")

    @property
    def key(self) -> str:
        return json.dumps({"length_unit": self.length_unit, **asdict(self)}, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ChunkSource:
    """Identity shared by all references to this immutable parse in its knowledge library."""

    document_id: str
    document_version_id: str
    content_sha256: str
    parsed_document_id: int
    parse_spec: ParseSpec
    format: DocumentFormat
    warnings: tuple[ParseWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkSourceSpan:
    """Half-open Unicode ranges in ContentBlock.text and Chunk.text, respectively.

    Locations belong to the full parsed block, in stored order. The text offsets
    can select an exact substring of parsed text even when the original file can
    only be located at block granularity.
    """

    block_id: str
    block_kind: BlockKind
    block_start: int
    block_end: int
    chunk_start: int
    chunk_end: int
    source_locations: tuple[SourceLocation, ...]
    is_overlap: bool
    location_granularity: Literal["block"] = "block"


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    source: ChunkSource
    chunk_index: int
    text: str
    section_heading_block_id: str | None
    section_path: tuple[str, ...]
    title_context: tuple[str, ...]
    source_spans: tuple[ChunkSourceSpan, ...]
    chunker_version: str
    chunker_config_key: str

    @property
    def parsed_document_id(self) -> int:
        return self.source.parsed_document_id

    @property
    def document_version_id(self) -> str:
        return self.source.document_version_id

    @property
    def source_block_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(span.block_id for span in self.source_spans))

    @property
    def source_locations(self) -> tuple[SourceLocation, ...]:
        locations_by_block = {span.block_id: span.source_locations for span in self.source_spans}
        return tuple(location for locations in locations_by_block.values() for location in locations)
