import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.chunking.models import Chunk, ChunkerConfig, ChunkSource, ChunkSourceSpan
from app.parsing.models import ContentBlock, DocumentParseError

CHUNKER_VERSION = "2"
BLOCK_SEPARATOR = "\n\n"
STRUCTURED_KINDS = {"code", "table"}
SENTENCE_END = re.compile(r'''(?:[。！？!?；;]+|\.(?=\s|$|["'”’\)\]]))["'”’」』】）》）)\]]*''')
TSV_MARKER = re.compile(r'["\n]')
THEOREM_OPENER_RE = re.compile(
    r"^(?:Theorem|Lemma|Definition|Proposition|Corollary|Proof|Remark|Example|Algorithm|"
    r"定理|引理|定义|命题|推论|证明|注|例\s*\d|算法)\b",
    re.I,
)
TINY_PROSE_THRESHOLD = 120


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


def _align_overlap_start(text: str, raw_start: int, end: int) -> int:
    """Snap overlap start forward to a clean word or sentence boundary so English words are never cut in half."""
    if raw_start <= 0 or raw_start >= end:
        return raw_start
    # If there is a clean sentence boundary in the first 45% of the overlap window, prefer starting a fresh sentence!
    search_limit = raw_start + max(0, int((end - raw_start) * 0.45))
    for match in SENTENCE_END.finditer(text, raw_start, search_limit):
        cand = match.end()
        while cand < end and text[cand].isspace():
            cand += 1
        if cand < end and (end - cand) >= 24:
            return cand
    # Otherwise never start inside an ASCII word or number
    if text[raw_start - 1].isascii() and text[raw_start - 1].isalnum() and text[raw_start].isascii() and text[raw_start].isalnum():
        cand = raw_start
        while cand < end and not text[cand].isspace():
            cand += 1
        while cand < end and text[cand].isspace():
            cand += 1
        if cand < end:
            return cand
        return end  # single giant word; drop overlap rather than mid-word fragment
    # Skip leading whitespace
    cand = raw_start
    while cand < end and text[cand].isspace():
        cand += 1
    return cand if cand < end else raw_start


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
        raw_start = part.end - length
        aligned_start = _align_overlap_start(part.block.text, raw_start, part.end)
        if aligned_start >= part.end:
            continue
        actual_len = part.end - aligned_start
        suffix.append(_Part(part.block, aligned_start, part.end, is_overlap=True))
        used += gap + actual_len
    suffix.reverse()
    return suffix, used


def _parts_len(parts: list[_Part]) -> int:
    if not parts:
        return 0
    total = parts[0].length
    for idx in range(1, len(parts)):
        total += len(_separator(parts[idx - 1], parts[idx])) + parts[idx].length
    return total


def _is_theorem_keep_together(pending: list[_Part], next_part: _Part, max_size: int) -> bool:
    if not pending:
        return False
    cur_len = _parts_len(pending)
    if cur_len > 360:
        return False
    first_txt = pending[0].block.text[pending[0].start : pending[0].end].lstrip()
    last_txt = pending[-1].block.text[pending[-1].start : pending[-1].end].rstrip()
    gap = len(_separator(pending[-1], next_part))
    if cur_len + gap + next_part.length > int(max_size * 1.15):
        return False
    return bool(THEOREM_OPENER_RE.match(first_txt) or last_txt.endswith((":", "：", "如下", "follows")))


def _raw_chunk_groups(
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
            if length + gap + part.length > config.chunk_size and not _is_theorem_keep_together(pending, part, config.chunk_size):
                yield pending
                budget = min(config.chunk_overlap, config.chunk_size - part.length - gap)
                pending, length = _overlap_suffix(pending, budget)
                if not pending:
                    gap = 0
            pending.append(part)
            length += gap + part.length
    if pending:
        yield pending


def _merge_tiny_prose_groups(groups: Iterable[list[_Part]], config: ChunkerConfig) -> Iterator[list[_Part]]:
    """Merge tiny prose slivers (<120 chars) into adjacent same-section prose chunks while preserving streaming."""
    buffered: list[_Part] | None = None
    tiny_thresh = min(TINY_PROSE_THRESHOLD, max(1, config.chunk_size // 3))
    max_merged = int(config.chunk_size * 1.18)
    for group in groups:
        if buffered is None:
            # Yield full-sized groups immediately without peeking ahead to preserve streaming memory bounds
            if _parts_len(group) >= tiny_thresh and _parts_len(group) >= int(config.chunk_size * 0.75):
                yield group
                continue
            buffered = group
            continue
        # Only merge prose groups belonging to the exact same section and without overlap conflicts
        buf_structured = any(p.block.kind in STRUCTURED_KINDS for p in buffered)
        grp_structured = any(p.block.kind in STRUCTURED_KINDS for p in group)
        same_section = (
            buffered[-1].block.section_path == group[0].block.section_path
            and buffered[-1].block.section_heading_block_id == group[0].block.section_heading_block_id
        )
        grp_has_overlap = any(p.is_overlap for p in group)
        if not buf_structured and not grp_structured and same_section and not grp_has_overlap:
            buf_len = _parts_len(buffered)
            grp_len = _parts_len(group)
            gap = len(_separator(buffered[-1], group[0]))
            if (buf_len < tiny_thresh or grp_len < tiny_thresh) and (buf_len + gap + grp_len <= max_merged):
                buffered = [*buffered, *group]
                if _parts_len(buffered) >= int(config.chunk_size * 0.85):
                    yield buffered
                    buffered = None
                continue
        yield buffered
        if _parts_len(group) >= tiny_thresh and _parts_len(group) >= int(config.chunk_size * 0.75):
            yield group
            buffered = None
        else:
            buffered = group
    if buffered is not None:
        yield buffered


def _chunk_groups(
    blocks: Iterable[ContentBlock], source: ChunkSource, config: ChunkerConfig,
) -> Iterator[list[_Part]]:
    yield from _merge_tiny_prose_groups(_raw_chunk_groups(blocks, source, config), config)


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
