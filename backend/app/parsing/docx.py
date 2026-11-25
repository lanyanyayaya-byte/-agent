"""Local process boundary for parsing a verified DOCX byte snapshot."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from threading import BoundedSemaphore, Event

from app.parsing.models import ContentBlock, DocumentParseError, DocxSourceLocation, ParsedDocument, ParseWarning, Section
from app.parsing.worker import communicate, open_worker, watch_owner
from app.services.library import MAX_FILE_BYTES

WORKER_TIMEOUT_SECONDS = 60
MAX_RESULT_BYTES = 16 * 1024 * 1024
WORKER_SLOT = BoundedSemaphore()
ERRORS = {
    "DOCX_ENCRYPTED": "The DOCX is encrypted.",
    "DOCX_INVALID": "The DOCX is damaged or has an invalid package structure.",
    "DOCX_UNSUPPORTED": "The DOCX uses an unsupported package, XML, or document format.",
    "DOCX_RESOURCE_LIMIT": "The DOCX exceeds a parser resource limit.",
    "DOCX_PARSE_FAILED": "The local DOCX parser could not complete the document.",
    "DOCX_PARSE_TIMEOUT": "The local DOCX parser exceeded its time limit.",
    "DOCX_PARSER_BUSY": "Another DOCX is being parsed.",
}


def docx_error(code: str) -> DocumentParseError:
    return DocumentParseError(code, ERRORS[code])


def parse_docx(
    data: bytes, *, document_id: str, document_version_id: str, cancelled: Event | None = None,
) -> ParsedDocument:
    if not data:
        raise docx_error("DOCX_INVALID")
    if len(data) > MAX_FILE_BYTES:
        raise docx_error("DOCX_RESOURCE_LIMIT")
    if not WORKER_SLOT.acquire(blocking=False):
        raise docx_error("DOCX_PARSER_BUSY")
    try:
        if getattr(sys, "frozen", False):
            command, cwd = [sys.executable, "--parse-docx-worker"], None
        else:
            command = [sys.executable, "-E", "-s", "-B", "-m", "app.parsing.docx"]
            cwd = Path(__file__).resolve().parents[2]
        with open_worker([*command, document_id, document_version_id], cwd=cwd, cancelled=cancelled) as worker:
            result = communicate(worker, data, WORKER_TIMEOUT_SECONDS, cancelled)
            if worker.returncode != 0 or not 0 < len(result) <= MAX_RESULT_BYTES:
                raise docx_error("DOCX_PARSE_FAILED")
        payload = json.loads(result)
        if "error" in payload:
            raise docx_error(payload["error"])
        sections = tuple(Section(**section) for section in payload["sections"])
        warnings = tuple(ParseWarning(**warning) for warning in payload["warnings"])
        blocks = []
        for block in payload["blocks"]:
            locations = tuple(DocxSourceLocation(**location) for location in block.pop("source_locations"))
            path = tuple(block.pop("section_path"))
            blocks.append(ContentBlock(**block, source_locations=locations, section_path=path))
        return ParsedDocument(document_id, document_version_id, "docx", sections, tuple(blocks), warnings)
    except subprocess.TimeoutExpired:
        raise docx_error("DOCX_PARSE_TIMEOUT") from None
    except (OSError, KeyError, TypeError, ValueError):
        raise docx_error("DOCX_PARSE_FAILED") from None
    finally:
        WORKER_SLOT.release()


def docx_worker_main(arguments: list[str]) -> int:
    if len(arguments) != 3:
        return 2
    document_id, version_id, owner_fd = arguments
    watch_owner(int(owner_fd))
    try:
        from app.parsing.docx_layout import read_docx

        data = sys.stdin.buffer.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise docx_error("DOCX_RESOURCE_LIMIT")
        parsed = read_docx(data, document_id, version_id)
        result = json.dumps(asdict(parsed), ensure_ascii=False, separators=(",", ":")).encode()
        if len(result) > MAX_RESULT_BYTES:
            raise docx_error("DOCX_RESOURCE_LIMIT")
    except (MemoryError, RecursionError):
        result = b'{"error":"DOCX_RESOURCE_LIMIT"}'
    except DocumentParseError as error:
        result = json.dumps({"error": error.code}).encode()
    except Exception:
        result = b'{"error":"DOCX_PARSE_FAILED"}'
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(docx_worker_main(sys.argv[1:]))
