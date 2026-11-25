import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.chunking.models import Chunk, ChunkerConfig, ChunkSource, ChunkSourceSpan
from app.parsing.models import ContentBlock, DocumentParseError

CHUNKER_VERSION = "1"
BLOCK_SEPARATOR = "\n\n"
STRUCTURED_KINDS = {"code", "table"}
SENTENCE_END = re.compile(r'''(?:[。！？!?；;]+|\.(?=\s|$|["'”’\)\]]))["'”’」』】）》）)\]]*''')
TSV_MARKER = re.compile(r'["\n]')


@dataclass(frozen=True, slots=True)
class _Part:
    block: ContentBlock
    start: int
    end: int
    is_overlap: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start


def _separator(left: _Part, right: _Part) -> str:
    return "" if left.block.block_id == right.block.block_id else BLOCK_SEPARATOR


def _prose_parts(block: ContentBlock, config: ChunkerConfig) -> Iterator[_Part]:
    text = block.text
    if len(text) <= config.chunk_size:
        yield _Part(block, 0, len(text))
        return

    # Reserve room for a source-backed overlap in the next window of a long block.
    capacity = config.chunk_size - config.chunk_overlap
    start = 0
    while start + capacity < len(text):
        end = start + capacity
        if config.long_paragraph_strategy != "character":
            end = _prose_boundary(text, start + capacity // 2, end, config.long_paragraph_strategy)
        yield _Part(block, start, end)
        start = end
    yield _Part(block, start, len(text))


def _prose_boundary(text: str, floor: int, ceiling: int, strategy: str) -> int:
    if strategy == "sentence":
        boundary = None
        for match in SENTENCE_END.finditer(text, floor, ceiling + 1):
            if match.end() <= ceiling:
                boundary = match.end()
        if boundary is not None:
            return boundary
    newline = text.rfind("\n", floor, ceiling)
    if newline >= 0:
        return newline + 1
    for index in range(ceiling - 1, floor - 1, -1):
        if text[index].isspace():
            return index + 1
    return ceiling


def _record_ends(text: str, quoted_tsv: bool) -> Iterator[int]:
    if quoted_tsv:
        # DOCX tables use csv.writer with tab delimiters and doubled quotes.
        # Even pairs of quotes leave the state unchanged, including escaped quotes.
        in_quotes = False
        for marker in TSV_MARKER.finditer(text):
            if marker[0] == '"':
                in_quotes = not in_quotes
            elif not in_quotes:
                yield marker.end()
    else:
        start = 0
        while (newline := text.find("\n", start)) >= 0:
            start = newline + 1
            yield start
    if not text.endswith("\n") or quoted_tsv:
        yield len(text)


def _structured_parts(block: ContentBlock, size: int, quoted_tsv: bool) -> Iterator[_Part]:
    if len(block.text) <= size:
        yield _Part(block, 0, len(block.text))
        return

    start = previous_end = 0
    for end in _record_ends(block.text, quoted_tsv):
        if end - start > size and previous_end > start:
            yield _Part(block, start, previous_end)
            start = previous_end
        while end - start > size:
            yield _Part(block, start, start + size)
            start += size
        previous_end = end
    if start < previous_end:
        yield _Part(block, start, previous_end)


def _overlap_suffix(parts: list[_Part], budget: int) -> tuple[list[_Part], int]:
    suffix: list[_Part] = []
    used = 0
    for part in reversed(parts):
        if part.is_overlap or part.block.kind == "heading":
            break
        gap = len(_separator(part, suffix[-1])) if suffix else 0
        length = min(part.length, budget - used - gap)
        if length <= 0:
            break
        suffix.append(_Part(part.block, part.end - length, part.end, is_overlap=True))
        used += gap + length
    suffix.reverse()
    return suffix, used


def _chunk_groups(
    blocks: Iterable[ContentBlock], source: ChunkSource, config: ChunkerConfig,
) -> Iterator[list[_Part]]:
    pending: list[_Part] = []
    length = 0
    for block in blocks:
        if block.document_version_id != source.document_version_id:
            raise DocumentParseError("PARSED_VERSION_MISMATCH", "A block belongs to another document version.")
        structured = block.kind in STRUCTURED_KINDS
        if pending:
            previous = pending[-1].block
            if (structured or block.kind == "heading" or block.section_path != previous.section_path
                    or block.section_heading_block_id != previous.section_heading_block_id):
                yield pending
                pending, length = [], 0
        if not block.text:
            continue
        if structured:
            quoted_tsv = source.format == "docx" and block.kind == "table"
            for part in _structured_parts(block, config.chunk_size, quoted_tsv):
                yield [part]
            continue

        for part in _prose_parts(block, config):
            gap = len(_separator(pending[-1], part)) if pending else 0
            if length + gap + part.length > config.chunk_size:
                yield pending
                budget = min(config.chunk_overlap, config.chunk_size - part.length - gap)
                pending, length = _overlap_suffix(pending, budget)
                if not pending:
                    gap = 0
            pending.append(part)
            length += gap + part.length
    if pending:
        yield pending


def chunk_identity(source: ChunkSource, chunker_version: str, config_key: str) -> bytes:
    return json.dumps((source.document_id, source.document_version_id, source.content_sha256,
                       source.parsed_document_id, source.parse_spec.parser_version,
                       source.parse_spec.structure_version, source.parse_spec.config_key,
                       chunker_version, config_key), separators=(",", ":")).encode("ascii")


def chunk_id_from_spans(identity: bytes, index: int, spans: Iterable[ChunkSourceSpan]) -> str:
    coordinates = [(span.block_id, span.block_start, span.block_end, span.is_overlap) for span in spans]
    digest = hashlib.sha256(identity)
    digest.update(json.dumps((index, coordinates), separators=(",", ":")).encode("ascii"))
    return f"chunk:{digest.hexdigest()}"


def iter_chunks(
    blocks: Iterable[ContentBlock], *, source: ChunkSource, config: ChunkerConfig = ChunkerConfig(),
) -> Iterator[Chunk]:
    """Chunk a parse's complete ordered blocks, independent of storage pagination."""
    config_key = config.key
    identity = chunk_identity(source, CHUNKER_VERSION, config_key)
    for index, parts in enumerate(_chunk_groups(blocks, source, config)):
        text_parts: list[str] = []
        spans: list[ChunkSourceSpan] = []
        offset = 0
        previous = None
        for part in parts:
            if previous is not None:
                separator = _separator(previous, part)
                text_parts.append(separator)
                offset += len(separator)
            block = part.block
            text_parts.append(block.text[part.start:part.end])
            spans.append(ChunkSourceSpan(block.block_id, block.kind, part.start, part.end,
                                         offset, offset + part.length, block.source_locations, part.is_overlap))
            offset += part.length
            previous = part

        first = parts[0].block
        path = first.section_path
        title_context = ()
        if config.title_context_strategy == "path":
            title_context = path
        elif config.title_context_strategy == "leaf":
            title_context = path[-1:]
        yield Chunk(
            chunk_id=chunk_id_from_spans(identity, index, spans), source=source, chunk_index=index,
            text="".join(text_parts),
            section_heading_block_id=first.section_heading_block_id, section_path=path,
            title_context=title_context, source_spans=tuple(spans), chunker_version=CHUNKER_VERSION,
            chunker_config_key=config_key,
        )
