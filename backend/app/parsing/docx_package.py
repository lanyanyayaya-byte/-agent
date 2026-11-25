"""Validate archive and XML bounds before python-docx loads an OPC package."""

import struct
import zlib
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from docx import Document
from docx.document import Document as WordDocument
from docx.oxml.ns import qn
from lxml import etree

from app.parsing.docx import docx_error
from app.services.library import MAX_DOCX_DIRECTORY_BYTES, MAX_DOCX_ENTRIES, ZIP_END_RECORD

MAX_EXPANDED_BYTES = 160 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_XML_BYTES = 12 * 1024 * 1024
MAX_TOTAL_XML_BYTES = 24 * 1024 * 1024
MAX_XML_NODES = 350_000
MAX_COMPRESSION_RATIO = 200
RATIO_CHECK_BYTES = 1024 * 1024
LOCAL_HEADER = struct.Struct("<4s5H3I2H")
COMPOUND_FILE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
MAIN_PART = "word/document.xml"
CONTENT_TYPES_NS = "{http://schemas.openxmlformats.org/package/2006/content-types}"
REQUIRED_PARTS = {
    "[Content_Types].xml": CONTENT_TYPES_NS + "Types",
    "_rels/.rels": "{http://schemas.openxmlformats.org/package/2006/relationships}Relationships",
    MAIN_PART: qn("w:document"),
}
STRICT_DOCUMENT = "{http://purl.oclc.org/ooxml/wordprocessingml/main}document"


def open_docx(data: bytes) -> WordDocument:
    if data.startswith(COMPOUND_FILE):
        encrypted = all(name.encode("utf-16-le") in data for name in ("EncryptionInfo", "EncryptedPackage"))
        raise docx_error("DOCX_ENCRYPTED" if encrypted else "DOCX_UNSUPPORTED")
    directory_offset, entry_count = _directory_bounds(data)
    stream = BytesIO(data)
    try:
        with ZipFile(stream) as archive:
            members = archive.infolist()
            if len(members) != entry_count or archive.start_dir != directory_offset:
                raise docx_error("DOCX_INVALID")
            _validate_members(archive, members, data)
    except (BadZipFile, UnicodeDecodeError, struct.error, zlib.error, EOFError):
        raise docx_error("DOCX_INVALID") from None
    except NotImplementedError:
        raise docx_error("DOCX_UNSUPPORTED") from None

    stream.seek(0)
    try:
        document = Document(stream)
    except (BadZipFile, AttributeError, KeyError, etree.XMLSyntaxError, UnicodeDecodeError):
        raise docx_error("DOCX_INVALID") from None
    except ValueError:
        # python-docx rejected the main content type (e.g. template or DOCM).
        raise docx_error("DOCX_UNSUPPORTED") from None
    if document.part.partname != f"/{MAIN_PART}" or len(document._element.findall(qn("w:body"))) != 1:
        raise docx_error("DOCX_INVALID")
    return document


def _directory_bounds(data: bytes) -> tuple[int, int]:
    # Check the end record before ZipFile allocates the central directory.
    if not data.startswith(b"PK\x03\x04"):
        raise docx_error("DOCX_INVALID")
    size = ZIP_END_RECORD.size
    end = len(data) - size
    if not (data[end:end + 4] == b"PK\x05\x06" and data[-2:] == b"\0\0"):
        end = data.rfind(b"PK\x05\x06", max(0, end - 65_535))
    if end < 0 or end + size > len(data):
        raise docx_error("DOCX_INVALID")
    _, disk, directory_disk, disk_entries, entries, length, offset, comment = ZIP_END_RECORD.unpack_from(data, end)
    zip64 = end >= 20 and data[end - 20:end - 16] == b"PK\x06\x07"
    if disk or directory_disk or disk_entries != entries or zip64:
        raise docx_error("DOCX_UNSUPPORTED")
    if not entries or offset + length != end or end + size + comment != len(data):
        raise docx_error("DOCX_INVALID")
    if entries > MAX_DOCX_ENTRIES or length > MAX_DOCX_DIRECTORY_BYTES:
        raise docx_error("DOCX_RESOURCE_LIMIT")
    return offset, entries


def _validate_members(archive: ZipFile, members: list[ZipInfo], data: bytes) -> None:
    names: set[str] = set()
    expanded = 0
    for member in members:
        name = member.filename.rstrip("/")
        key = name.casefold()
        if (member.orig_filename != member.filename or key in names or "\\" in name
                or any(piece in ("", ".", "..") for piece in name.split("/"))):
            raise docx_error("DOCX_INVALID")
        names.add(key)
        if member.flag_bits & (1 | 0x40):
            raise docx_error("DOCX_ENCRYPTED")
        if member.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
            raise docx_error("DOCX_UNSUPPORTED")
        expanded += member.file_size
        if (expanded > MAX_EXPANDED_BYTES or member.file_size > MAX_MEMBER_BYTES
                or (member.file_size >= RATIO_CHECK_BYTES
                    and member.file_size > MAX_COMPRESSION_RATIO * member.compress_size)):
            raise docx_error("DOCX_RESOURCE_LIMIT")

    if not REQUIRED_PARTS.keys() <= archive.NameToInfo.keys():
        raise docx_error("DOCX_INVALID")
    xml = _XmlBudget()
    view = memoryview(data)
    manifest = archive.getinfo("[Content_Types].xml")
    xml_bytes = manifest.file_size
    if xml_bytes > MAX_XML_BYTES:
        raise docx_error("DOCX_RESOURCE_LIMIT")
    with archive.open(manifest):
        root = xml.parse(bytes(_checked_member(view, manifest, archive.start_dir)), manifest=True)
    if root != REQUIRED_PARTS[manifest.filename]:
        raise docx_error("DOCX_INVALID")
    for member in members:
        if member is manifest:
            continue
        is_xml = member.filename.lower().endswith((".xml", ".rels", ".vml")) or xml.declares_xml(member.filename)
        if is_xml:
            xml_bytes += member.file_size
            if member.file_size > MAX_XML_BYTES or xml_bytes > MAX_TOTAL_XML_BYTES:
                raise docx_error("DOCX_RESOURCE_LIMIT")
        # zipfile checks the local name and overlapping entries on open. Validate
        # actual DEFLATE length independently before python-docx's unbounded reads.
        with archive.open(member):
            blob = _checked_member(view, member, archive.start_dir)
        if is_xml:
            root = xml.parse(bytes(blob))
            if member.filename == MAIN_PART and root == STRICT_DOCUMENT:
                raise docx_error("DOCX_UNSUPPORTED")
            expected = REQUIRED_PARTS.get(member.filename)
            if expected is not None and root != expected:
                raise docx_error("DOCX_INVALID")


def _checked_member(data: memoryview, member: ZipInfo, directory_offset: int) -> bytes | memoryview:
    offset = member.header_offset
    signature, _, flags, method, *_, name_size, extra_size = LOCAL_HEADER.unpack_from(data, offset)
    start = offset + LOCAL_HEADER.size + name_size + extra_size
    end = start + member.compress_size
    if offset < 0 or signature != b"PK\x03\x04" or method != member.compress_type or end > directory_offset:
        raise docx_error("DOCX_INVALID")
    if flags & (1 | 0x40):
        raise docx_error("DOCX_ENCRYPTED")
    blob = data[start:end]
    if method == ZIP_DEFLATED:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        blob = inflater.decompress(blob, member.file_size + 1)
        if not inflater.eof or inflater.unused_data:
            raise docx_error("DOCX_INVALID")
    if len(blob) != member.file_size or zlib.crc32(blob) != member.CRC:
        raise docx_error("DOCX_INVALID")
    return blob


class _XmlBudget:
    """Check XML structure and count nodes without materializing a second tree."""

    def __init__(self):
        self.nodes = 0
        self.root = None
        self.manifest = False
        self.defaults: dict[str, str] = {}
        self.overrides: dict[str, str] = {}

    def parse(self, blob: bytes, *, manifest: bool = False) -> str | None:
        self.root = None
        self.manifest = manifest
        parser = etree.XMLParser(target=self, resolve_entities=False, no_network=True, load_dtd=False)
        try:
            return etree.fromstring(blob, parser)
        except etree.XMLSyntaxError:
            raise docx_error("DOCX_INVALID") from None

    def declares_xml(self, filename: str) -> bool:
        name = filename.casefold()
        content_type = self.overrides.get("/" + name, self.defaults.get(name.rpartition(".")[2], ""))
        return content_type.endswith(("+xml", "/xml"))

    def start(self, tag, attributes):
        if self.root is None:
            self.root = tag
        self.nodes += 1
        if self.nodes > MAX_XML_NODES:
            raise docx_error("DOCX_RESOURCE_LIMIT")
        if self.manifest:
            if tag == CONTENT_TYPES_NS + "Default" and "Extension" in attributes and "ContentType" in attributes:
                self.defaults[attributes["Extension"].casefold()] = attributes["ContentType"].casefold()
            elif tag == CONTENT_TYPES_NS + "Override" and "PartName" in attributes and "ContentType" in attributes:
                self.overrides[attributes["PartName"].casefold()] = attributes["ContentType"].casefold()

    def doctype(self, _name, _public_id, _system_id):
        raise docx_error("DOCX_UNSUPPORTED")

    def close(self):
        return self.root
