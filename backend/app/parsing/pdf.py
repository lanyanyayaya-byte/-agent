"""Local process boundary for parsing a verified PDF byte snapshot."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from threading import BoundedSemaphore, Event, Thread

from app.parsing.models import ContentBlock, DocumentParseError, ParsedDocument, ParseWarning, PdfSourceLocation, Section
from app.parsing.worker import communicate, open_worker, watch_owner
from app.services.library import MAX_FILE_BYTES


WORKER_TIMEOUT_SECONDS = 300
WORKER_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
WORKER_SLOT = BoundedSemaphore()
ERRORS = {
    "PDF_ENCRYPTED": "The PDF is encrypted.",
    "PDF_INVALID": "The PDF is damaged or has an invalid page structure.",
    "PDF_RESOURCE_LIMIT": "The PDF exceeds a parser resource limit.",
    "PDF_PARSE_FAILED": "The local PDF parser could not complete the document.",
    "PDF_PARSE_TIMEOUT": "The local PDF parser exceeded its time limit.",
    "PDF_PARSER_BUSY": "Another PDF is being parsed.",
}


def parse_pdf(
    data: bytes, *, document_id: str, document_version_id: str, cancelled: Event | None = None,
) -> ParsedDocument:
    if not 0 < len(data) <= MAX_FILE_BYTES:
        raise DocumentParseError("PDF_RESOURCE_LIMIT", ERRORS["PDF_RESOURCE_LIMIT"])
    if not WORKER_SLOT.acquire(blocking=False):
        raise DocumentParseError("PDF_PARSER_BUSY", ERRORS["PDF_PARSER_BUSY"])
    try:
        result = _run_worker(data, document_id, document_version_id, cancelled)
        return _read_result(result, document_id, document_version_id)
    finally:
        WORKER_SLOT.release()


def _run_worker(data: bytes, document_id: str, version_id: str, cancelled: Event | None = None) -> bytes:
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--parse-pdf-worker"]
        cwd = None
    else:
        command = [sys.executable, "-E", "-s", "-B", "-m", "app.parsing.pdf"]
        cwd = Path(__file__).resolve().parents[2]

    try:
        with open_worker([*command, document_id, version_id], cwd=cwd, cancelled=cancelled) as worker:
            result = _wait_for_worker(worker, data, cancelled)
            if worker.returncode != 0 or not 0 < len(result) <= MAX_RESULT_BYTES:
                raise DocumentParseError("PDF_PARSE_FAILED", ERRORS["PDF_PARSE_FAILED"])
            return result
    except OSError:
        raise DocumentParseError("PDF_PARSE_FAILED", ERRORS["PDF_PARSE_FAILED"]) from None


def _wait_for_worker(worker: subprocess.Popen, data: bytes, cancelled: Event | None = None) -> bytes:
    finished = Event()
    monitor_errors: list[DocumentParseError] = []

    def monitor_rss() -> None:
        # Darwin does not enforce RLIMIT_AS. Supervise RSS while the parent
        # communicates, including while the child has yet to read all of stdin.
        while not finished.wait(0.2):
            try:
                if _darwin_rss(worker.pid) <= WORKER_MEMORY_BYTES:
                    continue
                raise DocumentParseError("PDF_RESOURCE_LIMIT", ERRORS["PDF_RESOURCE_LIMIT"])
            except DocumentParseError as error:
                monitor_errors.append(error)
                worker.kill()
                return

    monitor = None
    if sys.platform == "darwin":
        monitor = Thread(target=monitor_rss, name="pdf-worker-memory")
        monitor.start()
    try:
        result = communicate(worker, data, WORKER_TIMEOUT_SECONDS, cancelled)
    except subprocess.TimeoutExpired:
        raise DocumentParseError("PDF_PARSE_TIMEOUT", ERRORS["PDF_PARSE_TIMEOUT"]) from None
    finally:
        finished.set()
        if monitor is not None:
            monitor.join()
    if monitor_errors:
        raise monitor_errors[0]
    return result


def _darwin_rss(pid: int) -> int:
    try:
        rss = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)], capture_output=True, timeout=1, check=False,
        ).stdout.strip()
        return int(rss or b"0") * 1024
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise DocumentParseError("PDF_PARSE_FAILED", ERRORS["PDF_PARSE_FAILED"]) from None


def _read_result(result: bytes, document_id: str, version_id: str) -> ParsedDocument:
    try:
        payload = json.loads(result)
        if "error" in payload:
            code = payload["error"]
            raise DocumentParseError(code, ERRORS[code])
        sections = tuple(Section(**section) for section in payload["sections"])
        warnings = tuple(ParseWarning(**warning) for warning in payload["warnings"])
        blocks = []
        for block in payload["blocks"]:
            locations = tuple(
                PdfSourceLocation(location["physical_page"], location["page_label"], tuple(location["page_bbox"]))
                for location in block.pop("source_locations")
            )
            section_path = tuple(block.pop("section_path"))
            blocks.append(ContentBlock(**block, source_locations=locations, section_path=section_path))
        return ParsedDocument(document_id, version_id, "pdf", sections, tuple(blocks), warnings)
    except (KeyError, TypeError, ValueError):
        raise DocumentParseError("PDF_PARSE_FAILED", ERRORS["PDF_PARSE_FAILED"]) from None


def pdf_worker_main(arguments: list[str]) -> int:
    if len(arguments) != 3:
        return 2
    document_id, version_id, owner_fd = arguments
    watch_owner(int(owner_fd))
    try:
        if sys.platform == "linux":
            import resource

            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            limit = WORKER_MEMORY_BYTES if hard_limit == resource.RLIM_INFINITY else min(hard_limit, WORKER_MEMORY_BYTES)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

        from app.parsing.pdf_layout import read_pdf

        data = sys.stdin.buffer.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise DocumentParseError("PDF_RESOURCE_LIMIT", ERRORS["PDF_RESOURCE_LIMIT"])
        parsed = read_pdf(data, document_id, version_id)
        result = json.dumps(asdict(parsed), ensure_ascii=False, separators=(",", ":")).encode()
        if len(result) > MAX_RESULT_BYTES:
            raise DocumentParseError("PDF_RESOURCE_LIMIT", ERRORS["PDF_RESOURCE_LIMIT"])
        if sys.platform == "darwin":
            import resource

            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > WORKER_MEMORY_BYTES:
                raise DocumentParseError("PDF_RESOURCE_LIMIT", ERRORS["PDF_RESOURCE_LIMIT"])
    except (MemoryError, RecursionError):
        result = b'{"error":"PDF_RESOURCE_LIMIT"}'
    except DocumentParseError as error:
        result = json.dumps({"error": error.code}).encode()
    except Exception:
        result = b'{"error":"PDF_PARSE_FAILED"}'
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(pdf_worker_main(sys.argv[1:]))
