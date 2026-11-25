import csv
import io
import stat
import struct
from secrets import token_urlsafe
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZipFile

import httpx
import pytest
from anyio import to_thread
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_BREAK
from docx.opc.constants import CONTENT_TYPE as CT, RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Pt

from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.parsing import docx, docx_package
from app.parsing.models import DocumentParseError, DocxSourceLocation
from app.services.document_parsing import parse_docx_version
from app.services.models import ModelService
from test_chat import ProviderStream, configure, frames, new_session, payload, run_path
from test_library_management import anyio_backend, import_file, library, original


def serialize(document):
    result = io.BytesIO()
    document.save(result)
    return result.getvalue()


def property_value(parent, name, value):
    element = OxmlElement(name)
    element.set(qn("w:val"), str(value))
    parent.append(element)


def add_link(paragraph):
    link = OxmlElement("w:hyperlink")
    relation = paragraph.part.relate_to("https://example.invalid/course", RT.HYPERLINK, is_external=True)
    link.set(qn("r:id"), relation)
    link.append(parse_xml(f'<w:r {nsdecls("w")}><w:t>课程链接 Course link</w:t></w:r>'))
    paragraph._p.append(link)


def parse_import(library, imported):
    return parse_docx_version(
        library.store, document_id=imported.document_id, document_version_id=imported.document_version_id,
    )


def parse_bytes(data):
    return docx.parse_docx(data, document_id="document", document_version_id="version")


@pytest.fixture
def textbook():
    book = Document()
    book.add_paragraph("前言 Intro", "Title")
    book.add_heading("第一章 Mechanics", 1)
    paragraph = book.add_paragraph("中英混排 Mixed text，见 ")
    add_link(paragraph)
    paragraph.add_run("。")
    paragraph.add_run().add_break(WD_BREAK.PAGE)
    book.add_paragraph()
    custom = book.styles.add_style("Course Topic", WD_STYLE_TYPE.PARAGRAPH)
    custom.base_style = book.styles["Heading 2"]
    book.add_paragraph("小节 Forces", custom)
    book.add_paragraph("重复 Same")
    table = book.add_table(rows=3, cols=3)
    table.cell(0, 0).merge(table.cell(1, 1)).text = "合并 Merged"
    table.cell(0, 2).text = "列值 Column"
    table.cell(1, 2).text = "单位\tunit"
    table.cell(2, 0).merge(table.cell(2, 1)).text = "另行 Another row"
    table.cell(2, 2).text = "第一行\n第二行"
    book.add_paragraph("重复 Same")
    book.add_paragraph("列表项 Bullet", "List Bullet")
    numbered = book.add_paragraph("第一步 Step")
    num = numbered._p.get_or_add_pPr().get_or_add_numPr()
    num.get_or_add_numId().val = book.styles["List Number"]._element.pPr.numPr.numId.val
    num.get_or_add_ilvl().val = 0
    book.add_paragraph("第二步 Step", "List Number")
    heading = book.add_paragraph("练习 Exercises")
    property_value(heading._p.get_or_add_pPr(), "w:outlineLvl", 2)
    body = book.add_heading("抑制标题 Still body", 3)
    property_value(body._p.get_or_add_pPr(), "w:outlineLvl", 9)
    book.add_heading("附录 Appendix", 1)
    book.add_paragraph("大号正文 Large text").runs[0].font.size = Pt(33)
    return serialize(book)


def test_imported_docx_order_locations_sections_and_immutable_copy(library, textbook, tmp_path):
    selected = tmp_path / "source.docx"
    selected.write_bytes(textbook)
    imported = import_file(library, selected.read_bytes(), "教材.docx")
    private = original(library, textbook)
    before = private.stat()
    parsed = parse_import(library, imported)
    assert parsed == parse_import(library, imported)
    assert parsed.format == "docx" and parsed.warnings == ()
    assert [b.kind for b in parsed.blocks] == [
        "paragraph", "heading", "paragraph", "heading", "paragraph", "table", "paragraph",
        "list", "list", "list", "heading", "paragraph", "heading", "paragraph",
    ]
    (
        preface, chapter, link, topic, before_table, table, after_table, bullet,
        first, second, exercise, body, appendix, large,
    ) = parsed.blocks
    assert preface.section_path == () and preface.section_heading_block_id is None
    assert link.text == "中英混排 Mixed text，见 课程链接 Course link。"
    assert before_table.text == after_table.text == "重复 Same"
    assert [b.source_locations[0].body_index for b in parsed.blocks] == [1, 2, 3, *range(5, 16)]
    assert [b.source_locations[0].paragraph_number for b in parsed.blocks] == [1, 2, 3, 5, 6, None, *range(7, 15)]
    assert table.block_id == f"{imported.document_version_id}:docx:7"
    assert list(csv.reader(io.StringIO(table.text), delimiter="\t")) == [
        ["合并 Merged", "", "列值 Column"], ["", "", "单位\tunit"], ["另行 Another row", "", "第一行\n第二行"],
    ]
    assert [(loc.table_row, loc.table_column, loc.row_span, loc.column_span) for loc in table.source_locations] == [
        (1, 1, 2, 2), (1, 3, 1, 1), (2, 3, 1, 1), (3, 1, 1, 2), (3, 3, 1, 1),
    ]
    assert all(loc.body_index == 7 and loc.paragraph_number is None for loc in table.source_locations)
    assert [(s.level, s.heading_block_id, s.parent_heading_block_id) for s in parsed.sections] == [
        (1, chapter.block_id, None), (2, topic.block_id, chapter.block_id),
        (3, exercise.block_id, topic.block_id), (1, appendix.block_id, None),
    ]
    assert table.section_path == (chapter.text, topic.text)
    assert exercise.section_path == body.section_path == (chapter.text, topic.text, exercise.text)
    assert large.section_path == (appendix.text,)
    assert body.source_locations[0].heading_level is large.source_locations[0].heading_level is None
    bullet_loc, first_loc, second_loc = (b.source_locations[0] for b in (bullet, first, second))
    assert bullet_loc.list_level == first_loc.list_level == second_loc.list_level == 1
    assert bullet_loc.list_format == "bullet" and first_loc.list_format == second_loc.list_format == "decimal"
    assert first_loc.numbering_id == second_loc.numbering_id != bullet_loc.numbering_id
    assert len({b.block_id for b in parsed.blocks}) == len(parsed.blocks)
    assert all(b.document_version_id == imported.document_version_id for b in parsed.blocks)
    assert all(not hasattr(loc, "physical_page") for b in parsed.blocks for loc in b.source_locations)
    assert private.read_bytes() == selected.read_bytes() == textbook
    assert private.stat().st_ino == before.st_ino and private.stat().st_mtime_ns == before.st_mtime_ns
    assert stat.S_IMODE(private.stat().st_mode) == 0o400


def test_special_content_keeps_text_and_returns_located_diagnostics():
    book = Document()
    paragraph = book.add_paragraph("公式 Equation ")
    paragraph._p.append(parse_xml(f'<m:oMath {nsdecls("m")}><m:r><m:t>x + y</m:t></m:r></m:oMath>'))
    paragraph.add_run(" 完成")
    note = parse_xml(f'<w:footnoteReference {nsdecls("w")} w:id="4"/>')
    paragraph.runs[-1]._r.append(note)
    notes = (f'<w:footnotes {nsdecls("w")}><w:footnote w:id="4">'
             '<w:p><w:r><w:t>Footnote text</w:t></w:r></w:p></w:footnote></w:footnotes>').encode()
    notes_part = Part(PackURI("/word/footnotes.xml"), CT.WML_FOOTNOTES, notes, book.part.package)
    book.part.relate_to(notes_part, RT.FOOTNOTES)
    # A valid one-pixel PNG through python-docx's ordinary image API.
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
    )
    book.add_paragraph().add_run().add_picture(io.BytesIO(png))
    textbox = parse_xml(
        f'<w:pict {nsdecls("w")} xmlns:v="urn:schemas-microsoft-com:vml">'
        '<v:shape><v:textbox><w:txbxContent><w:p><w:r><w:t>Box text</w:t></w:r></w:p>'
        '</w:txbxContent></v:textbox></v:shape></w:pict>'
    )
    book.add_paragraph("Outside the box").add_run()._r.append(textbox)
    result = parse_bytes(serialize(book))
    assert [b.text for b in result.blocks] == ["公式 Equation x + y 完成", "Outside the box"]
    assert result.blocks[1].source_locations == (DocxSourceLocation(paragraph_number=3, body_index=3),)
    assert [(w.code, w.message.rsplit(" ", 1)[-1]) for w in result.warnings] == [
        ("DOCX_MATH_TEXT_ONLY", "1."), ("DOCX_NOTE_OMITTED", "1."),
        ("DOCX_IMAGE_OMITTED", "2."), ("DOCX_TEXTBOX_OMITTED", "3."),
    ]
    assert all(w.physical_page is None for w in result.warnings)


def test_localized_style_or_document_outline_and_non_rectangular_table():
    book = Document()
    localized = book.styles.add_style("三级标题", WD_STYLE_TYPE.PARAGRAPH)
    book.add_paragraph("局部标题 Topic", localized)
    table = book.add_table(rows=2, cols=3)
    row = table.rows[0]._tr
    row.remove(row.tc_lst[0])
    property_value(row.get_or_add_trPr(), "w:gridBefore", 1)
    table.rows[1].cells[0].text = "First"
    table.rows[0].cells[0].text = "Starts in column 2"
    table.rows[0].cells[1].add_table(rows=1, cols=1).cell(0, 0).text = "Nested"
    result = parse_bytes(serialize(book))
    heading, extracted_table = result.blocks
    assert heading.source_locations[0].heading_level == 3
    assert extracted_table.text == "\tStarts in column 2\tNested\nFirst\t\t"
    assert extracted_table.source_locations[0].table_column == 2
    assert [w.code for w in result.warnings] == ["DOCX_NESTED_TABLE_FLATTENED"]


def repack(data, replacements, *, compression=ZIP_DEFLATED):
    result = io.BytesIO()
    with ZipFile(io.BytesIO(data)) as source, ZipFile(result, "w", compression) as target:
        for name in source.namelist():
            target.writestr(name, replacements[name] if name in replacements else source.read(name))
        for name in replacements.keys() - set(source.namelist()):
            target.writestr(name, replacements[name])
    return result.getvalue()


def test_invalid_package_and_worker_timeout_release_slot(library, textbook, monkeypatch):
    damaged = repack(textbook, {"word/document.xml": b"<document"})
    imported = import_file(library, damaged, "damaged.docx")
    with pytest.raises(DocumentParseError, match="DOCX_INVALID"):
        parse_import(library, imported)
    assert original(library, damaged).read_bytes() == damaged
    with pytest.raises(DocumentParseError, match="DOCX_INVALID"):
        parse_bytes(textbook[:-10])
    valid = import_file(library, textbook, "valid.docx")
    with monkeypatch.context() as patch:
        patch.setattr(docx, "WORKER_TIMEOUT_SECONDS", 0)
        with pytest.raises(DocumentParseError, match="DOCX_PARSE_TIMEOUT"):
            parse_import(library, valid)
    assert parse_import(library, valid).sections[0].title == "第一章 Mechanics"


def test_version_format_and_private_copy_hash_and_size(library, textbook):
    first = import_file(library, textbook, "教材.docx")
    other_data = serialize(Document())
    second = import_file(library, other_data, "教材.docx", document_id=first.document_id)
    reverted = import_file(library, textbook, "教材.docx", document_id=first.document_id)
    assert parse_import(library, second).blocks == ()
    first_parsed, reverted_parsed = (parse_import(library, item) for item in (first, reverted))
    assert {b.block_id for b in first_parsed.blocks}.isdisjoint(b.block_id for b in reverted_parsed.blocks)
    markdown = import_file(library, b"# Markdown\n")
    with pytest.raises(DocumentParseError, match="UNSUPPORTED_DOCUMENT_FORMAT"):
        parse_import(library, markdown)
    with pytest.raises(DocumentParseError, match="DOCUMENT_VERSION_NOT_FOUND"):
        parse_docx_version(library.store, document_id=markdown.document_id, document_version_id=first.document_version_id)
    saved = original(library, textbook)
    saved.chmod(0o600)
    for changed in (textbook[:-1], textbook[:-1] + bytes([textbook[-1] ^ 1])):
        saved.write_bytes(changed)
        with pytest.raises(DocumentParseError, match="DOCX_SOURCE_MISMATCH"):
            parse_import(library, first)
        assert saved.read_bytes() == changed


def test_package_preflights_compression_xml_and_actual_deflate_size(textbook, monkeypatch):
    with pytest.raises(DocumentParseError, match="DOCX_RESOURCE_LIMIT"):
        parse_bytes(repack(textbook, {"word/media/oversized.bin": b"a" * docx_package.RATIO_CHECK_BYTES}))
    with pytest.raises(DocumentParseError, match="DOCX_UNSUPPORTED"):
        parse_bytes(repack(textbook, {}, compression=ZIP_BZIP2))
    with ZipFile(io.BytesIO(textbook)) as archive:
        xml = archive.read("word/document.xml")
        manifest = archive.read("[Content_Types].xml")
        central = archive.start_dir
    doctype = b'<!DOCTYPE w:document SYSTEM "file:///does-not-exist.dtd">'
    xml = xml.replace(b"?>", b"?>" + doctype, 1)
    with pytest.raises(DocumentParseError, match="DOCX_UNSUPPORTED"):
        parse_bytes(repack(textbook, {"word/document.xml": xml}))

    typed_manifest = manifest.replace(
        b"</Types>", b'<Override PartName="/customXml/extra.data" ContentType="application/xml"/></Types>',
    )
    with pytest.raises(DocumentParseError, match="DOCX_UNSUPPORTED"):
        parse_bytes(repack(textbook, {"[Content_Types].xml": typed_manifest, "customXml/extra.data": xml}))

    forged_size = bytearray(textbook)
    struct.pack_into("<I", forged_size, central + 24, 5)
    with pytest.raises(DocumentParseError, match="DOCX_INVALID"):
        parse_bytes(bytes(forged_size))

    with monkeypatch.context() as patch:
        patch.setattr(docx_package, "MAX_EXPANDED_BYTES", 1_000)
        with pytest.raises(DocumentParseError, match="DOCX_RESOURCE_LIMIT"):
            docx_package.open_docx(textbook)
    monkeypatch.setattr(docx_package, "MAX_XML_NODES", 20)
    with pytest.raises(DocumentParseError, match="DOCX_RESOURCE_LIMIT"):
        docx_package.open_docx(textbook)


def test_encrypted_and_unsupported_word_containers_report_explicit_errors(textbook):
    with ZipFile(io.BytesIO(textbook)) as archive:
        central, first = archive.start_dir, archive.infolist()[0]
    encrypted = bytearray(textbook)
    struct.pack_into("<H", encrypted, central + 8, first.flag_bits | 1)
    with pytest.raises(DocumentParseError, match="DOCX_ENCRYPTED"):
        parse_bytes(bytes(encrypted))
    with pytest.raises(DocumentParseError, match="DOCX_ENCRYPTED"):
        parse_bytes(docx_package.COMPOUND_FILE + "EncryptionInfo\0EncryptedPackage\0".encode("utf-16-le"))
    with pytest.raises(DocumentParseError, match="DOCX_UNSUPPORTED"):
        parse_bytes(docx_package.COMPOUND_FILE + "WordDocument\0".encode("utf-16-le"))


@pytest.mark.anyio
async def test_malformed_docx_leaves_chat_usable(tmp_path, textbook):
    ui_token, host_token = token_urlsafe(32), token_urlsafe(32)
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=ProviderStream()))
    models = ModelService(DeepSeekAdapter(DeepSeekSettings(model="fixture-model"), transport=transport))
    app = create_app(ui_token, host_token, database_path=tmp_path / "business.db", models=models)

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {ui_token}"},
    ) as client:
        data = repack(textbook, {"word/document.xml": b"<damaged>"})
        book = import_file(app.state.library, data, "damaged.docx")
        await configure(client, {"Authorization": f"Bearer {host_token}"})
        session = await new_session(client)
        with pytest.raises(DocumentParseError, match="DOCX_INVALID"):
            await to_thread.run_sync(parse_import, app.state.library, book)
        response = await client.post(run_path(session), json=payload())
        assert response.status_code == 200
        events = [item async for item in frames(response) if isinstance(item, dict)]
        assert events[-1]["type"] == "run_completed"
