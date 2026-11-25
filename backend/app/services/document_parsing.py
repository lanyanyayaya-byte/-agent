import hashlib
import os
import resource
import sqlite3
import sys
from threading import Event
from time import perf_counter

from app.diagnostics import record
from app.parsing.docx import parse_docx
from app.parsing.markdown import parse_markdown
from app.parsing.models import DocumentParseError, ParsedDocument
from app.parsing.pdf import parse_pdf
from app.parsing.worker import check_cancelled
from app.services.library import HASH_NAME, MAX_FILE_BYTES, private_directory, private_file
from app.storage.library import (
    ORIGINALS_DIRECTORY, DocumentFormat, DocumentVersion, KnowledgeStore, KnowledgeStoreError,
)
from app.storage.parsed_documents import ParseAvailability, ParseSpec, ParsedDocumentStore, StoredParse

PARSED_STRUCTURE_VERSION = 1
# These parsers use fixed profiles. Bump the parser version for extraction-rule changes and
# change the profile key when effective options change. A model change bumps the shared structure version.
PARSER_SPECS: dict[DocumentFormat, ParseSpec] = {
    "md": ParseSpec("1", PARSED_STRUCTURE_VERSION, "commonmark-table"),
    "pdf": ParseSpec("2", PARSED_STRUCTURE_VERSION, "text-layout"),
    "docx": ParseSpec("1", PARSED_STRUCTURE_VERSION, "word-body"),
}


class DocumentParsingService:
    """Resolve the active parser profile and keep reads separate from on-demand parsing."""

    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.results = ParsedDocumentStore(store)

    def get_status(self, document_version_id: str) -> ParseAvailability:
        version = self.store.get_document_version(document_version_id)
        if version is None:
            return ParseAvailability("missing", None)
        return self.results.get_availability(document_version_id, PARSER_SPECS[version[1]])

    def get_current(self, document_version_id: str) -> StoredParse | None:
        """Prefer the configured profile and fall back to the latest complete result."""
        return self.get_status(document_version_id).result

    def ensure_parsed(self, document_version_id: str, *, cancelled: Event | None = None) -> StoredParse:
        check_cancelled(cancelled)
        version = self.store.get_document_version(document_version_id)
        if version is None:
            raise DocumentParseError("DOCUMENT_VERSION_NOT_FOUND", "The requested document version does not exist.")
        document_id, format, _ = version
        spec = PARSER_SPECS[format]
        cached = self.results.get_reusable(document_version_id, spec)
        if cached is not None:
            return cached

        parser = {"md": parse_markdown_version, "pdf": parse_pdf_version, "docx": parse_docx_version}[format]
        parsed = parser(self.store, document_id=document_id, document_version_id=document_version_id,
                        cancelled=cancelled)
        check_cancelled(cancelled)
        if (parsed.document_id, parsed.document_version_id, parsed.format) != (document_id, document_version_id, format):
            raise DocumentParseError("PARSED_VERSION_MISMATCH", "The parse does not match the requested version.")
        started = perf_counter()
        result, published = self.results.save(parsed, spec, cancelled=cancelled)
        if published:
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            record(
                "document_parse_persisted", document_version_id=document_version_id,
                parsed_document_id=result.parsed_document_id, format=format, block_count=len(parsed.blocks),
                save_duration_ms=round((perf_counter() - started) * 1000, 1),
                process_peak_rss_bytes=peak_rss if sys.platform == "darwin" else peak_rss * 1024,
            )
        return result


def parse_markdown_version(
    store: KnowledgeStore, *, document_id: str, document_version_id: str,
    cancelled: Event | None = None,
) -> ParsedDocument:
    """Parse an imported version from its verified, private copy."""
    version = _get_version(store, document_id, document_version_id, "md")
    data = _read_private_copy(store, version, "MARKDOWN", "Markdown")
    check_cancelled(cancelled)
    if b"\0" in data:
        raise DocumentParseError("MARKDOWN_SOURCE_INVALID", "The Markdown copy contains a NUL byte.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DocumentParseError("MARKDOWN_ENCODING_INVALID", "The Markdown copy is not valid UTF-8.") from None
    return parse_markdown(text, document_id=document_id, document_version_id=version.document_version_id)


def parse_pdf_version(
    store: KnowledgeStore, *, document_id: str, document_version_id: str,
    cancelled: Event | None = None,
) -> ParsedDocument:
    """Synchronously parse a verified, imported PDF version in a local child process."""
    version = _get_version(store, document_id, document_version_id, "pdf")
    data = _read_private_copy(store, version, "PDF", "PDF")
    return parse_pdf(data, document_id=document_id, document_version_id=version.document_version_id, cancelled=cancelled)


def parse_docx_version(
    store: KnowledgeStore, *, document_id: str, document_version_id: str,
    cancelled: Event | None = None,
) -> ParsedDocument:
    """Synchronously parse a verified, imported DOCX version in a local child process."""
    version = _get_version(store, document_id, document_version_id, "docx")
    data = _read_private_copy(store, version, "DOCX", "DOCX")
    return parse_docx(data, document_id=document_id, document_version_id=version.document_version_id, cancelled=cancelled)


def _get_version(
    store: KnowledgeStore, document_id: str, document_version_id: str, expected_format: DocumentFormat,
) -> DocumentVersion:
    try:
        result = store.get_document_version(document_version_id)
    except sqlite3.Error:
        raise DocumentParseError(
            "DOCUMENT_VERSION_READ_FAILED", "Unable to read the document version metadata.",
        ) from None
    if result is None or result[0] != document_id:
        raise DocumentParseError("DOCUMENT_VERSION_NOT_FOUND", "The requested document version does not exist.")
    _, format, version = result
    if format != expected_format:
        raise DocumentParseError("UNSUPPORTED_DOCUMENT_FORMAT", f"Expected an imported {expected_format} document.")
    return version


def _read_private_copy(store: KnowledgeStore, version: DocumentVersion, code: str, name: str) -> bytes:
    sha256, size = version.content_sha256, version.size_bytes
    if not HASH_NAME.fullmatch(sha256) or not 0 < size <= MAX_FILE_BYTES:
        raise DocumentParseError(f"{code}_SOURCE_INVALID", f"The private {name} copy has invalid metadata.")

    try:
        with private_directory(store.path.parent) as root, private_directory(ORIGINALS_DIRECTORY, root) as originals:
            fd = os.open(sha256, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=originals)
            try:
                metadata = os.fstat(fd)
                if not private_file(metadata):
                    raise DocumentParseError(
                        f"{code}_SOURCE_INVALID", f"The {name} copy must be a private regular file.",
                    )
                if metadata.st_size != size:
                    raise DocumentParseError(
                        f"{code}_SOURCE_MISMATCH", f"The {name} copy does not match its stored version.",
                    )
                with os.fdopen(fd, "rb", closefd=False) as source:
                    data = source.read(size + 1)
            finally:
                os.close(fd)
    except FileNotFoundError:
        raise DocumentParseError(f"{code}_SOURCE_MISSING", f"The private {name} copy is missing.") from None
    except (OSError, KnowledgeStoreError):
        raise DocumentParseError(f"{code}_SOURCE_UNREADABLE", f"Unable to read the private {name} copy.") from None

    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise DocumentParseError(f"{code}_SOURCE_MISMATCH", f"The {name} copy does not match its stored version.")
    return data
