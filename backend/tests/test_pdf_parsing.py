import subprocess
import sys
from pathlib import Path
from secrets import token_urlsafe

import httpx
import pytest
from anyio import to_thread

from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.parsing import pdf, pdf_layout
from app.parsing.models import DocumentParseError, PdfSourceLocation
from app.services.document_parsing import parse_pdf_version
from app.services.models import ModelService
from test_chat import ProviderStream, configure, frames, new_session, payload, run_path
from test_library_management import PDF, anyio_backend, import_file, library, original


FIXTURES = Path(__file__).with_name("fixtures") / "pdf"
# Both encrypted fixtures were generated with PdfWriter.encrypt(algorithm="RC4-40")
# and contain a single blank page. Their user passwords are "user" and "".


def text(x, y, value, *, size=11, bold=False):
    font = "F2" if bold else "F1"
    return f"BT /{font} {size} Tf {x} {y} Td ({value}) Tj ET\n".encode()


def stream(data, attributes=b""):
    return b"<< /Length %d %s >>\nstream\n%s\nendstream" % (len(data), attributes, data)


def make_pdf(pages, *, labels=b"", count=None):
    """Assemble a small PDF with known drawing coordinates and an actual xref."""
    page_count = len(pages)
    references = b" ".join(b"%d 0 R" % (6 + i * 2) for i in range(page_count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + labels + b">>",
        b"<< /Type /Pages /Count %d /Kids [%s] >>" % (page_count if count is None else count, references),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        stream(b"\x80", b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray "
                        b"/BitsPerComponent 8"),
    ]
    for i, (content, attributes) in enumerate(pages):
        objects.extend([
            b"<< /Type /Page /Parent 2 0 R /MediaBox %s /Contents %d 0 R "
            b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> /XObject << /Im0 5 0 R >> >> >>"
            % (attributes, 7 + i * 2),
            stream(content),
        ])
    output = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(b"%d 0 obj\n%s\nendobj\n" % (number, obj))
    xref = len(output)
    output.extend(b"xref\n0 %d\n0000000000 65535 f \n" % (1 + len(objects)))
    for offset in offsets:
        output.extend(b"%010d 00000 n \n" % offset)
    output.extend(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (1 + len(objects), xref))
    return bytes(output)


def page(content, attributes=b"[0 0 600 800]"):
    return (content, attributes)


def parse_import(library, imported):
    return parse_pdf_version(
        library.store, document_id=imported.document_id, document_version_id=imported.document_version_id,
    )


def test_single_column_keeps_version_labels_line_boxes_and_original(library, tmp_path):
    body = (text(72, 720, "1 Introduction", size=16, bold=True)
            + text(72, 690, "First paragraph begins here.")
            + text(72, 677, "The next line shares its block.")
            + text(72, 635, "Repeated sentence.")
            + text(72, 595, "Repeated sentence."))
    data = make_pdf([page(body), page(text(72, 720, "Repeated sentence."))],
                    labels=b"/PageLabels << /Nums [0 << /S /r >> 1 << /P (A-) /S /D /St 7 >>] >> ")
    original_file = tmp_path / "textbook.pdf"
    original_file.write_bytes(data)
    imported = import_file(library, original_file.read_bytes(), original_file.name)
    private = original(library, data)
    before = private.stat()
    original_file.unlink()

    result = parse_import(library, imported)
    assert (result.format, result.document_id, result.document_version_id) == (
        "pdf", imported.document_id, imported.document_version_id,
    )
    assert result == parse_import(library, imported)
    assert [block.text for block in result.blocks] == [
        "1 Introduction", "First paragraph begins here.\nThe next line shares its block.",
        "Repeated sentence.", "Repeated sentence.", "Repeated sentence.",
    ]
    heading = result.sections[0]
    assert heading.title == "1 Introduction"
    assert result.blocks[1].section_heading_block_id == heading.heading_block_id
    paragraph = result.blocks[1]
    assert len(paragraph.source_locations) == 2
    assert paragraph.source_locations[0].page_bbox[0] == 72
    assert paragraph.source_locations[0].page_bbox[1] == pytest.approx(101.277)
    assert paragraph.source_locations[1].page_bbox[1] - paragraph.source_locations[0].page_bbox[1] == 13
    locations = [block.source_locations[0] for block in result.blocks]
    assert [(loc.physical_page, loc.page_label) for loc in locations] == [(1, "i")] * 4 + [(2, "A-7")]
    assert all(block.document_version_id == imported.document_version_id for block in result.blocks)
    assert len({b.block_id for b in result.blocks}) == len(result.blocks)
    assert result.blocks[2].source_locations != result.blocks[3].source_locations
    assert all(isinstance(loc, PdfSourceLocation) for loc in locations)
    assert all(loc.coordinate_system == "unrotated_mediabox_top_left_pt" for loc in locations)
    after = private.stat()
    assert private.read_bytes() == data
    assert (before.st_ino, before.st_mode, before.st_mtime_ns) == (after.st_ino, after.st_mode, after.st_mtime_ns)


def test_two_columns_with_full_width_content_and_interleaved_drawing_order(library):
    header = "This full width line introduces the two columns on this page."
    footer = "This full width line follows both columns on the same page."
    top = text(156, 750, header)
    left = [f"Left column line number {i}." for i in range(1, 8)]
    right = [f"Right column line number {i}." for i in range(1, 8)]
    drawing = b"".join(text(325, 706 - i * 14, r) + text(60, 710 - i * 14, l)
                       for i, (l, r) in enumerate(zip(left, right)))
    data = make_pdf([page(top + drawing + text(154, 549, footer))])
    result = parse_import(library, import_file(library, data, "columns.pdf"))
    assert [b.text for b in result.blocks] == [header, "\n".join(left), "\n".join(right), footer]
    assert all(loc.page_bbox[2] < 300 for loc in result.blocks[1].source_locations)
    assert all(loc.page_bbox[0] > 300 for loc in result.blocks[2].source_locations)
    assert {loc.page_label for b in result.blocks for loc in b.source_locations} == {None}


def test_numbered_sections_distinguish_algorithm_lines_and_titles_starting_with_a_number(library):
    body = (
        text(72, 740, "1 Previous section", size=14, bold=True)
        + text(72, 700, "Per-row algorithm steps follow.")
        + text(85, 670, "2", size=7, bold=True)
        + text(94, 670, "while the matrix is not empty", bold=True)
        + text(72, 640, "This paragraph still belongs to the first section.")
        + text(72, 600, "1.1 4-bit Quantization", size=14, bold=True)
    )
    result = parse_import(library, import_file(library, make_pdf([page(body)]), "algorithm.pdf"))
    assert [section.title for section in result.sections] == ["1 Previous section", "1.1 4-bit Quantization"]
    assert result.sections[1].parent_heading_block_id == result.sections[0].heading_block_id
    step = next(block for block in result.blocks if block.text.startswith("2 while"))
    assert step.kind == "paragraph"
    assert step.source_locations[0].physical_page == 1
    assert step.source_locations[0].page_bbox[0] == 85
    assert result.blocks[-2].section_path == ("1 Previous section",)


def test_rotated_pages_crop_userunit_and_mediabox_offset_keep_same_unrotated_bounds(library):
    source = text(30, 340, "Visible inside crop") + text(-18, 110, "i") + text(30, 423, "Above crop")
    pages = [
        page(source, b"[-20 30 280 430] /CropBox [0 50 265 410] /UserUnit 2 /Rotate %d" % angle)
        for angle in (0, 90, 180, 270)
    ]
    result = parse_import(library, import_file(library, make_pdf(pages), "rotated.pdf"))
    assert [b.text for b in result.blocks] == ["Visible inside crop"] * 4
    locations = [b.source_locations[0] for b in result.blocks]
    assert [loc.physical_page for loc in locations] == [1, 2, 3, 4]
    assert locations[0].page_bbox[:2] == pytest.approx((100, 162.554))
    assert all(loc.page_bbox == pytest.approx(locations[0].page_bbox) for loc in locations)


def test_scan_blank_page_and_bad_page_labels_report_page_diagnostics(library):
    data = make_pdf([page(text(72, 720, "First page text.")), page(b"q 600 0 0 800 0 0 cm /Im0 Do Q"), page(b""),
                     page(text(72, 720, "Page after the blank and scan."))],
                    labels=b"/PageLabels << /Nums [0 << /P (bad) /S /Broken >>] >> ")
    result = parse_import(library, import_file(library, data, "mixed.pdf"))
    assert [(b.source_locations[0].physical_page, b.source_locations[0].page_label)
            for b in result.blocks] == [(1, None), (4, None)]
    assert {(w.physical_page, w.code) for w in result.warnings} == {
        (None, "PDF_PAGE_LABELS_INVALID"), (2, "PDF_SUSPECTED_SCAN"), (2, "PDF_NO_TEXT_LAYER"),
        (3, "PDF_NO_TEXT_LAYER"),
    }


@pytest.mark.parametrize("name", ["password.pdf", "empty-password.pdf"])
def test_encrypted_pdf_reports_clear_error(library, name):
    data = (FIXTURES / name).read_bytes()
    with pytest.raises(DocumentParseError, match="PDF_ENCRYPTED"):
        parse_import(library, import_file(library, data, name))
    assert original(library, data).read_bytes() == data


def test_corruption_and_page_resource_limits(library):
    valid = make_pdf([page(b"")])
    for data in (PDF, make_pdf([page(b"")], count=2), valid.replace(b"/Root 1 0 R", b"/Root 9 0 R")):
        with pytest.raises(DocumentParseError, match="PDF_INVALID"):
            parse_import(library, import_file(library, data, "invalid.pdf"))
    with pytest.raises(DocumentParseError, match="PDF_INVALID"):
        pdf.parse_pdf(valid[:-10], document_id="doc", document_version_id="version")
    oversized_page_tree = make_pdf([page(b"")], count=pdf_layout.MAX_PAGES + 1)
    with pytest.raises(DocumentParseError, match="PDF_RESOURCE_LIMIT"):
        parse_import(library, import_file(library, oversized_page_tree, "too-many.pdf"))


def test_private_source_format_version_integrity_and_identity(library):
    md = import_file(library, b"# other")
    with pytest.raises(DocumentParseError, match="UNSUPPORTED_DOCUMENT_FORMAT"):
        parse_import(library, md)

    data = make_pdf([page(text(72, 720, "Stable content"))])
    first = import_file(library, data, "book.pdf")
    second = import_file(library, make_pdf([page(b"")]), "book.pdf", document_id=first.document_id)
    reverted = import_file(library, data, "book.pdf", document_id=first.document_id)
    assert parse_import(library, first).blocks[0].block_id != parse_import(library, reverted).blocks[0].block_id
    assert parse_import(library, second).blocks == ()
    with pytest.raises(DocumentParseError, match="DOCUMENT_VERSION_NOT_FOUND"):
        parse_pdf_version(library.store, document_id=md.document_id, document_version_id=first.document_version_id)

    blob = original(library, data)
    blob.chmod(0o600)
    tampered = data.replace(b"Stable content", b"Forged content")
    blob.write_bytes(tampered)
    with pytest.raises(DocumentParseError, match="PDF_SOURCE_MISMATCH"):
        parse_import(library, first)
    assert blob.read_bytes() == tampered


def test_worker_timeout_releases_the_parser_slot(library, monkeypatch):
    imported = import_file(library, make_pdf([page(text(72, 720, "Short page."))]), "valid.pdf")
    with monkeypatch.context() as patch:
        patch.setattr(pdf, "WORKER_TIMEOUT_SECONDS", 0)
        with pytest.raises(DocumentParseError, match="PDF_PARSE_TIMEOUT"):
            parse_import(library, imported)
    assert parse_import(library, imported).blocks[0].text == "Short page."


def test_worker_delivers_complete_input_when_pipe_is_backpressured(monkeypatch):
    monkeypatch.setattr(pdf, "WORKER_TIMEOUT_SECONDS", 4)
    command = [
        sys.executable, "-c",
        "import sys, time; time.sleep(0.35); print(len(sys.stdin.buffer.read()))",
    ]
    data = b"x" * (4 * 1024 * 1024)
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
        try:
            assert pdf._wait_for_worker(worker, data).strip() == str(len(data)).encode()
        finally:
            if worker.poll() is None:
                worker.kill()


@pytest.mark.anyio
async def test_malformed_pdf_leaves_chat_usable(tmp_path):
    ui_token, host_token = token_urlsafe(32), token_urlsafe(32)
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=ProviderStream()))
    models = ModelService(DeepSeekAdapter(DeepSeekSettings(model="fixture-model"), transport=transport))
    app = create_app(ui_token, host_token, database_path=tmp_path / "business.db", models=models)

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {ui_token}"},
    ) as client:
        library = app.state.library
        imported = import_file(library, PDF, "damaged.pdf")
        await configure(client, {"Authorization": f"Bearer {host_token}"})
        session = await new_session(client)
        with pytest.raises(DocumentParseError, match="PDF_INVALID"):
            await to_thread.run_sync(parse_import, library, imported)
        response = await client.post(run_path(session), json=payload())
        assert response.status_code == 200
        events = [item async for item in frames(response) if isinstance(item, dict)]
        assert events[-1]["type"] == "run_completed"
        assert original(library, PDF).read_bytes() == PDF
