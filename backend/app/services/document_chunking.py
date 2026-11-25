from collections.abc import Iterator
from threading import Event
from typing import Literal

from app.chunking.hybrid import CHUNKER_VERSION, iter_chunks
from app.chunking.models import Chunk, ChunkerConfig, ChunkSource
from app.parsing.models import ContentBlock, DocumentParseError
from app.parsing.worker import check_cancelled
from app.services.document_parsing import PARSER_SPECS
from app.storage.chunks import ChunkSet, ChunkStore
from app.storage.index_jobs import IndexJobClaim
from app.storage.library import KnowledgeStore
from app.storage.parsed_documents import MAX_PAGE_SIZE, ParsedDocumentStore, StoredParse


def iter_stored_chunks(
    results: ParsedDocumentStore, parsed: StoredParse, *,
    config: ChunkerConfig = ChunkerConfig(), page_size: int = 100,
) -> Iterator[Chunk]:
    """Read the requested published parse, including a still-available historical parse."""
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError("Invalid block page size.")
    version = results.store.get_document_version(parsed.document_version_id)
    if (version is None or version[:2] != (parsed.document_id, parsed.format)
            or results.get_reusable(parsed.document_version_id, parsed.spec) != parsed):
        raise _unavailable_parse()

    source = ChunkSource(
        document_id=parsed.document_id, document_version_id=parsed.document_version_id,
        content_sha256=version[2].content_sha256, parsed_document_id=parsed.parsed_document_id,
        parse_spec=parsed.spec, format=parsed.format, warnings=parsed.warnings,
    )
    yield from iter_chunks(_blocks(results, parsed.parsed_document_id, page_size), source=source, config=config)


def _unavailable_parse() -> DocumentParseError:
    return DocumentParseError("PARSED_DOCUMENT_UNAVAILABLE", "The requested published parse is no longer available.")


def _blocks(results: ParsedDocumentStore, parsed_id: int, page_size: int) -> Iterator[ContentBlock]:
    cursor = -1
    while True:
        page = results.list_blocks(parsed_id, limit=page_size, after_position=cursor)
        if page is None:
            raise _unavailable_parse()
        yield from page.blocks
        if page.next_after_position is None:
            return
        cursor = page.next_after_position


class DocumentChunkingService:
    """Build chunks on demand and reuse published sets for the exact parse and config."""

    def __init__(self, store: KnowledgeStore):
        self.parses = ParsedDocumentStore(store)
        self.results = ChunkStore(store)

    def ensure_chunked(
        self, parsed_document_id: int, config: ChunkerConfig = ChunkerConfig(), *,
        cancelled: Event | None = None, claim: IndexJobClaim | None = None,
    ) -> ChunkSet:
        def generate() -> Iterator[Chunk]:
            parsed = self.parses.get_by_id(parsed_document_id)
            if parsed is None:
                raise _unavailable_parse()
            for chunk in iter_stored_chunks(self.parses, parsed, config=config):
                check_cancelled(cancelled)
                yield chunk
            check_cancelled(cancelled)

        check_cancelled(cancelled)
        return self.results.ensure(parsed_document_id, CHUNKER_VERSION, config.key, generate, claim=claim)

    def get_chunk_set(self, parsed_document_id: int, config: ChunkerConfig = ChunkerConfig()) -> ChunkSet | None:
        return self.results.get_chunk_set(parsed_document_id, CHUNKER_VERSION, config.key)

    def get_parse_state(self, chunk_set: ChunkSet) -> Literal["current", "stale", "unavailable"]:
        """Compare this set's exact parse to the active parser profile for its format."""
        parsed = self.parses.get_by_id(chunk_set.parsed_document_id)
        if parsed is None or parsed.document_version_id != chunk_set.document_version_id:
            return "unavailable"
        return "current" if parsed.spec == PARSER_SPECS[parsed.format] else "stale"
