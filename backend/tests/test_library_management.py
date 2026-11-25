import errno
import hashlib
import io
import os
import sqlite3
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from secrets import token_urlsafe
from threading import Barrier, Event
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pytest

from app.main import DEV_ORIGIN, create_app
from app.security import ApiError
from app.services import library as service
from app.storage.library import KnowledgeStore, KnowledgeStoreError
from test_sidecar import BACKEND, SHUTDOWN, assert_clean_exit, headers, running_sidecar
from test_status import assert_api_error

HOST_IMPORT = "/_host/knowledge/imports"
HOST_IMPORT_V1 = f"/api/v1{HOST_IMPORT}"
DOCUMENTS = "/api/v1/knowledge/documents"
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
MARKDOWN = "# 材料\n\n教材内容。\n".encode()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def library(tmp_path):
    library = service.LibraryService(KnowledgeStore(tmp_path), uuid4())
    try:
        yield library
    finally:
        library.close()


def docx(*, complete=True):
    result = io.BytesIO()
    with ZipFile(result, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "<document/>")
        if complete:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("_rels/.rels", "<Relationships/>")
    return result.getvalue()


def stage(directory, content, import_id=None):
    import_id = import_id or str(uuid4())
    fd = os.open(directory / import_id, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(content)
    return import_id


def original(library, content):
    return library.store.originals_dir / hashlib.sha256(content).hexdigest()


def import_file(library, content, filename="教材.md", document_id=None):
    return library.import_textbook(stage(library.staging_dir, content), filename, document_id)


def expect_error(status, code, call, *args):
    with pytest.raises(ApiError) as raised:
        call(*args)
    assert (raised.value.status_code, raised.value.code) == (status, code)


@pytest.mark.parametrize("filename,content", [
    ("教材.PDF", PDF), ("教材.docx", docx()), ("教材.md", MARKDOWN),
])
def test_first_import_replays_and_deduplicates_without_replacing_blob(library, filename, content):
    import_id = stage(library.staging_dir, content)
    first = library.import_textbook(import_id, filename)
    assert first.outcome == "imported"
    saved = original(library, content)
    assert saved.read_bytes() == content
    inode = saved.stat().st_ino
    assert stat.S_IMODE(saved.stat().st_mode) == 0o400
    assert stat.S_IMODE(library.staging_dir.stat().st_mode) == 0o700
    assert not (library.staging_dir / import_id).exists()

    assert library.import_textbook(import_id, filename) == first
    expect_error(409, "IMPORT_ID_CONFLICT", library.import_textbook, import_id, "重用." + filename.split(".")[-1])
    duplicate = import_file(library, content, filename)
    assert duplicate.outcome == "duplicate"
    assert (duplicate.document_id, duplicate.document_version_id) == (first.document_id, first.document_version_id)
    assert list(library.store.originals_dir.iterdir()) == [saved]
    assert saved.stat().st_ino == inode

    docs = library.list_documents()
    assert [(doc.name, doc.version_count, doc.management_status) for doc in docs] == [(filename, 1, "imported")]
    detail = library.get_document(first.document_id)
    assert detail.format == filename.split(".")[-1].lower()
    assert [(v.document_version_id, v.content_sha256, v.size_bytes) for v in detail.versions] == [
        (first.document_version_id, saved.name, len(content)),
    ]


def test_updates_are_idempotent_for_latest_content_and_keep_old_versions(library):
    first = import_file(library, MARKDOWN)
    document_id = first.document_id
    unchanged = import_file(library, MARKDOWN, "改名.md", document_id)
    assert (unchanged.outcome, unchanged.document_version_id) == ("unchanged", first.document_version_id)

    second_content = b"# edition 2\n"
    second = import_file(library, second_content, document_id=document_id)
    assert second.outcome == "updated"
    assert second.document_version_id != first.document_version_id
    assert [(doc.version_count, doc.latest_size_bytes) for doc in library.list_documents()] == [
        (2, len(second_content)),
    ]
    assert library.import_textbook(second.import_id, "教材.md", document_id) == second
    direct_duplicate = import_file(library, MARKDOWN)
    assert direct_duplicate.document_version_id == first.document_version_id

    expect_error(409, "TEXTBOOK_FORMAT_MISMATCH", import_file, library, PDF, "教材.pdf", document_id)
    before = library.get_document(document_id)
    assert before.name == "教材.md"
    assert [v.document_version_id for v in before.versions] == [first.document_version_id, second.document_version_id]

    reverted = import_file(library, MARKDOWN, document_id=document_id)
    assert reverted.outcome == "updated"
    final = library.get_document(document_id)
    assert [v.document_version_id for v in final.versions] == [
        first.document_version_id, second.document_version_id, reverted.document_version_id,
    ]
    assert len(list(library.store.originals_dir.iterdir())) == 2
    assert library.import_textbook(second.import_id, "教材.md", document_id) == second


def test_delete_preserves_shared_blob_and_invalidates_receipts(library):
    first = import_file(library, MARKDOWN)
    other_content = b"# other textbook"
    other = import_file(library, other_content)
    shared = import_file(library, MARKDOWN, document_id=other.document_id)
    assert shared.outcome == "updated"
    assert shared.document_version_id != first.document_version_id

    library.delete_document(first.document_id)
    library.delete_document(first.document_id)
    expect_error(404, "TEXTBOOK_NOT_FOUND", library.get_document, first.document_id)
    expect_error(410, "TEXTBOOK_REMOVED", library.import_textbook, first.import_id, "教材.md")
    assert original(library, MARKDOWN).read_bytes() == MARKDOWN
    assert library.get_document(other.document_id).version_count == 2

    library.delete_document(other.document_id)
    assert library.list_documents() == []
    assert list(library.store.originals_dir.iterdir()) == []
    with library.store.connect() as db:
        assert db.execute("SELECT count(*) FROM document_versions").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM content_blobs").fetchone() == (0,)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert import_file(library, MARKDOWN).document_id != first.document_id


@pytest.mark.parametrize("failure_point", ["directory_fsync", "receipt_insert"])
def test_failed_publish_or_sql_transaction_keeps_old_version_and_recovers(library, monkeypatch, failure_point):
    first = import_file(library, MARKDOWN)
    new = b"# update after failure"
    import_id = stage(library.staging_dir, new)
    if failure_point == "receipt_insert":
        with library.store.connect() as db:
            db.execute("""CREATE TRIGGER fixture_reject_receipt BEFORE INSERT ON knowledge_imports
                          BEGIN SELECT raise(ABORT, 'receipt failure'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="receipt failure"):
            library.import_textbook(import_id, "教材.md", first.document_id)
        with library.store.connect() as db:
            db.execute("DROP TRIGGER fixture_reject_receipt")
    else:
        real_fsync = os.fsync

        def fail_directory_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "injected directory sync error")
            real_fsync(fd)

        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", fail_directory_fsync)
            with pytest.raises(OSError, match="injected directory sync error"):
                library.import_textbook(import_id, "教材.md", first.document_id)

    assert original(library, new).read_bytes() == new
    assert (library.staging_dir / import_id).read_bytes() == new
    details = library.get_document(first.document_id)
    assert [v.document_version_id for v in details.versions] == [first.document_version_id]
    assert original(library, MARKDOWN).read_bytes() == MARKDOWN
    with library.store.connect() as db:
        assert db.execute("SELECT count(*) FROM content_blobs").fetchone() == (1,)
        assert db.execute("SELECT count(*) FROM knowledge_imports").fetchone() == (1,)

    if failure_point == "receipt_insert":
        assert library.cleanup_orphans() == 1
        assert not original(library, new).exists()
    else:
        # Recovery must publish the verified input even if an unregistered file was altered.
        orphan = original(library, new)
        orphan.chmod(0o600)
        orphan.write_bytes(b"x" * len(new))

    second = library.import_textbook(import_id, "教材.md", first.document_id)
    assert second.outcome == "updated"
    assert original(library, new).read_bytes() == new
    assert library.get_document(first.document_id).version_count == 2

    restarted = service.LibraryService(KnowledgeStore(library.store.path.parent.parent), uuid4())
    try:
        assert restarted.staging_dir != library.staging_dir
        assert list(restarted.staging_dir.iterdir()) == []
        assert restarted.import_textbook(import_id, "教材.md", first.document_id) == second
    finally:
        restarted.close()


def test_delete_unlink_failure_leaves_safe_orphan_for_later_cleanup(library, monkeypatch):
    saved = import_file(library, MARKDOWN)
    copy = original(library, MARKDOWN)
    unlink = os.unlink

    def fail_unlink(path, *args, **kwargs):
        if path == copy.name:
            raise PermissionError(errno.EACCES, "injected unlink error")
        unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", fail_unlink)
        library.delete_document(saved.document_id)
    assert library.list_documents() == []
    assert copy.exists()
    assert library.cleanup_orphans() == 1
    assert not copy.exists()


def test_delete_cleanup_rechecks_references_after_concurrent_reimport(library, monkeypatch):
    first = import_file(library, MARKDOWN)
    detached, finish = Event(), Event()
    detach_document = library.store.detach_document

    def pause_after_detaching(*args):
        hashes = detach_document(*args)
        detached.set()
        assert finish.wait(timeout=5)
        return hashes

    monkeypatch.setattr(library.store, "detach_document", pause_after_detaching)
    with ThreadPoolExecutor(max_workers=1) as pool:
        delete = pool.submit(library.delete_document, first.document_id)
        try:
            assert detached.wait(timeout=5)
            second = import_file(library, MARKDOWN)
            assert second.document_id != first.document_id
        finally:
            finish.set()
        delete.result(timeout=5)
    assert original(library, MARKDOWN).read_bytes() == MARKDOWN
    assert [doc.document_id for doc in library.list_documents()] == [second.document_id]


def test_process_exit_between_publication_and_commit_can_be_reconciled(library, tmp_path):
    first = import_file(library, MARKDOWN)
    import_id, instance_id = str(uuid4()), str(uuid4())
    new = b"replacement after abrupt exit"
    worker = '''
import os
import stat
import sys
from pathlib import Path
from uuid import UUID

from app.services.library import LibraryService
from app.storage.library import KnowledgeStore

directory, instance_id, import_id, content, target_id = sys.argv[1:]
library = LibraryService(KnowledgeStore(Path(directory)), UUID(instance_id))
fd = os.open(library.staging_dir / import_id, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as file:
    file.write(bytes.fromhex(content))
fsync = os.fsync

def exit_after_publish(fd):
    fsync(fd)
    if stat.S_ISDIR(os.fstat(fd).st_mode):
        os._exit(23)

os.fsync = exit_after_publish
library.import_textbook(import_id, "更新.md", target_id)
'''
    child = subprocess.run(
        [sys.executable, "-E", "-s", "-B", "-c", worker, str(tmp_path), instance_id,
         import_id, new.hex(), first.document_id],
        cwd=BACKEND, capture_output=True, timeout=10,
    )
    assert child.returncode == 23, child.stderr.decode()
    assert original(library, new).read_bytes() == new
    assert library.get_document(first.document_id).version_count == 1
    with library.store.connect() as db:
        assert db.execute("SELECT 1 FROM knowledge_imports WHERE import_id = ?", (import_id,)).fetchone() is None
    assert library.cleanup_orphans() == 1
    assert not original(library, new).exists()
    stage(library.staging_dir, new, import_id)
    updated = library.import_textbook(import_id, "更新.md", first.document_id)
    assert updated.outcome == "updated"
    assert library.get_document(first.document_id).version_count == 2
    assert original(library, new).read_bytes() == new


def test_cleanup_is_bounded_and_never_deletes_locked_scratch(library, monkeypatch):
    first = import_file(library, MARKDOWN)
    abandoned = library.store.originals_dir / (service.SCRATCH_PREFIX + uuid4().hex)
    abandoned.touch(mode=0o600)
    unused = original(library, b"unreferenced")
    unused.touch(mode=0o600)
    ignored = library.store.originals_dir / "unmanaged"
    ignored.touch(mode=0o600)
    started, finish = Event(), Event()
    copy_textbook = service.copy_textbook

    def pause_copy(*args):
        started.set()
        assert finish.wait(timeout=5)
        return copy_textbook(*args)

    monkeypatch.setattr(service, "copy_textbook", pause_copy)
    new = b"# new content"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(import_file, library, new)
        try:
            assert started.wait(timeout=5)
            scratch, = [entry for entry in library.store.originals_dir.glob(".incoming-*") if entry != abandoned]
            removed = [library.cleanup_orphans(limit=1) for _ in range(5)]
            assert max(removed) == 1 and sum(removed) == 2
            assert scratch.exists()
            assert ignored.exists()
            assert original(library, MARKDOWN).exists()
        finally:
            finish.set()
        result = future.result(timeout=5)
    assert library.get_document(first.document_id).version_count == 1
    assert library.get_document(result.document_id).version_count == 1
    assert original(library, new).read_bytes() == new
    assert not scratch.exists()


def test_concurrent_same_import_id_returns_the_same_committed_result(library, monkeypatch):
    import_id = stage(library.staging_dir, MARKDOWN)
    copy_textbook = service.copy_textbook
    rendezvous = Barrier(2)

    def concurrent_copy(*args):
        result = copy_textbook(*args)
        rendezvous.wait(timeout=5)
        return result

    monkeypatch.setattr(service, "copy_textbook", concurrent_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(library.import_textbook, import_id, "教材.md")
        second = pool.submit(library.import_textbook, import_id, "教材.md")
        assert first.result(timeout=8) == second.result(timeout=8)
    assert len(library.list_documents()) == 1
    assert list(library.store.originals_dir.iterdir()) == [original(library, MARKDOWN)]


def test_import_refuses_links_special_files_and_unprivate_files(library, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_bytes(MARKDOWN)
    outside.chmod(0o600)
    for create in (
        lambda path: path.symlink_to(outside),
        lambda path: os.link(outside, path),
        lambda path: os.mkfifo(path, 0o600),
    ):
        import_id = str(uuid4())
        staged = library.staging_dir / import_id
        create(staged)
        expect_error(415, "INVALID_IMPORT_SOURCE", library.import_textbook, import_id, "教材.md")
        staged.unlink()
    import_id = stage(library.staging_dir, MARKDOWN)
    (library.staging_dir / import_id).chmod(0o644)
    expect_error(415, "INVALID_IMPORT_SOURCE", library.import_textbook, import_id, "教材.md")
    assert outside.read_bytes() == MARKDOWN
    assert library.list_documents() == []


def test_staging_is_instance_specific_and_rejects_replaced_directory(library, tmp_path):
    other = service.LibraryService(library.store, uuid4())
    try:
        import_id = stage(other.staging_dir, MARKDOWN)
        expect_error(404, "IMPORT_SOURCE_NOT_FOUND", library.import_textbook, import_id, "教材.md")
        moved = tmp_path / "displaced"
        library.staging_dir.rename(moved)
        library.staging_dir.symlink_to(other.staging_dir, target_is_directory=True)
        with pytest.raises(OSError):
            library.import_textbook(import_id, "教材.md")
        library.staging_dir.unlink()
        library.staging_dir.mkdir(mode=0o700)
        with pytest.raises(KnowledgeStoreError, match="INVALID_IMPORT_STAGING_DIRECTORY"):
            library.import_textbook(import_id, "教材.md")
        assert (other.staging_dir / import_id).read_bytes() == MARKDOWN
    finally:
        other.close()


def test_staging_file_replaced_during_copy_is_rejected(library, monkeypatch):
    import_id = stage(library.staging_dir, MARKDOWN)
    source = library.staging_dir / import_id
    alternate = library.staging_dir / stage(library.staging_dir, b"# replacement")
    copy_textbook = service.copy_textbook

    def substitute(*args):
        result = copy_textbook(*args)
        alternate.replace(source)
        return result

    monkeypatch.setattr(service, "copy_textbook", substitute)
    expect_error(409, "IMPORT_SOURCE_CHANGED", library.import_textbook, import_id, "教材.md")
    assert source.read_bytes() == b"# replacement"
    assert library.list_documents() == []
    assert list(library.store.originals_dir.iterdir()) == []


@pytest.mark.parametrize("filename,content", [
    ("教材.pdf", b"%PDF-1.7\ntruncated"),
    ("教材.docx", docx(complete=False)),
    ("教材.docx", b"PK\x03\x04damaged"),
    ("教材.md", b"non-UTF8\xff"),
    ("教材.md", b"binary\0"),
    ("教材.md", b""),
    ("教材.txt", MARKDOWN),
])
def test_invalid_or_unsupported_formats_are_not_committed(library, filename, content):
    expect_error(415, "INVALID_TEXTBOOK_FORMAT", import_file, library, content, filename)
    assert library.list_documents() == []
    assert list(library.store.originals_dir.iterdir()) == []


def test_limits_reject_oversized_source_and_docx_expansion(library, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(service, "MAX_FILE_BYTES", 8)
        expect_error(413, "TEXTBOOK_TOO_LARGE", import_file, library, MARKDOWN)
    monkeypatch.setattr(service, "MAX_DOCX_EXPANDED_BYTES", 8)
    expect_error(415, "INVALID_TEXTBOOK_FORMAT", import_file, library, docx(), "教材.docx")
    assert library.list_documents() == []


@pytest.mark.anyio
async def test_management_api_preserves_auth_cors_and_error_contract(tmp_path):
    ui_token, host_token = token_urlsafe(32), token_urlsafe(32)
    ui, host = headers(ui_token), headers(host_token)
    app = create_app(ui_token, host_token, database_path=tmp_path / "business.db")
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
    ) as client:
        import_id = stage(app.state.library.staging_dir, MARKDOWN)
        body = {"import_id": import_id, "filename": "教材.md"}
        for url in (HOST_IMPORT, HOST_IMPORT_V1):
            assert_api_error(await client.post(url, content=b"{"), 401, "AUTH_REQUIRED")
            assert_api_error(await client.post(url, json=body, headers=headers(token_urlsafe(32))), 401, "AUTH_REQUIRED")
            assert_api_error(await client.post(url, json=body, headers=ui), 403, "FORBIDDEN")
        assert_api_error(await client.post(HOST_IMPORT_V1, json=body, headers=host), 404, "NOT_FOUND")
        assert app.state.library.list_documents() == []
        assert (app.state.library.staging_dir / import_id).is_file()
        for invalid in (
            {**body, "path": "/etc/passwd"},
            {**body, "import_id": "../escape"},
            {**body, "filename": "/etc/passwd.md"},
            {**body, "filename": "..\\escape.md"},
        ):
            assert_api_error(await client.post(HOST_IMPORT, json=invalid, headers=host), 422, "VALIDATION_ERROR")
        assert_api_error(await client.get(DOCUMENTS, headers=host), 403, "FORBIDDEN")
        response = await client.post(HOST_IMPORT, json=body, headers=host)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        result = response.json()
        assert result["outcome"] == "imported"
        assert (await client.post(HOST_IMPORT, json=body, headers=host)).json() == result
        url = f"{DOCUMENTS}/{result['document_id']}"
        listing = await client.get(DOCUMENTS, headers={**ui, "Origin": DEV_ORIGIN})
        assert listing.headers["access-control-allow-origin"] == DEV_ORIGIN
        assert listing.json()[0]["version_count"] == 1
        assert listing.json()[0]["latest_size_bytes"] == len(MARKDOWN)
        detail = (await client.get(url, headers=ui)).json()
        assert detail["versions"][0]["document_version_id"] == result["document_version_id"]
        assert "active_version_id" not in detail
        assert_api_error(await client.post(f"{url}/delete", headers=host), 403, "FORBIDDEN")
        assert (await client.post(f"{url}/delete", headers=ui)).status_code == 204
        assert (await client.post(f"{url}/delete", headers=ui)).status_code == 204
        assert_api_error(await client.get(url, headers=ui), 404, "TEXTBOOK_NOT_FOUND")
        assert_api_error(await client.post(HOST_IMPORT, json=body, headers=host), 410, "TEXTBOOK_REMOVED")


def test_sidecar_ready_identifies_private_staging_and_replay_survives_restart(tmp_path):
    with running_sidecar(tmp_path) as (child, ready, payload):
        staged = tmp_path / "knowledge" / "temporary" / ready["instance_id"]
        import_id = stage(staged, MARKDOWN)
        request = {"import_id": import_id, "filename": "教材.md"}
        host = headers(payload["host_token"])
        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False) as client:
            ui = headers(payload["ui_token"])
            invalid = headers(token_urlsafe(32))
            for url in (HOST_IMPORT, HOST_IMPORT_V1):
                assert_api_error(client.post(url, json=request), 401, "AUTH_REQUIRED")
                assert_api_error(client.post(url, json=request, headers=invalid), 401, "AUTH_REQUIRED")
                assert_api_error(client.post(url, json=request, headers=ui), 403, "FORBIDDEN")
            assert_api_error(client.post(HOST_IMPORT_V1, json=request, headers=host), 404, "NOT_FOUND")
            assert client.get(DOCUMENTS, headers=ui).json() == []
            assert (staged / import_id).is_file()
            imported = client.post(HOST_IMPORT, json=request, headers=host)
            assert imported.status_code == 200
            assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, payload)
        assert not staged.exists()

    with running_sidecar(tmp_path) as (child, ready, payload):
        staged = tmp_path / "knowledge" / "temporary" / ready["instance_id"]
        assert list(staged.iterdir()) == []
        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False) as client:
            assert client.post(HOST_IMPORT, json=request, headers=host).status_code == 401
            host = headers(payload["host_token"])
            assert client.post(HOST_IMPORT, json=request, headers=host).json() == imported.json()
            assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, payload)
