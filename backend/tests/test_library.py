import hashlib
import json
import logging
import os
import sqlite3
from contextlib import closing
from secrets import token_urlsafe
from stat import S_IMODE
from uuid import uuid4

import httpx
import pytest

from app.storage import library as storage_library
from app.diagnostics import LOGGER
from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.services.index_pipeline import IndexPipeline
from app.services.models import ModelService
from app.storage.business import DATABASE_NAME
from app.storage.checkpoints import CHECKPOINT_DATABASE_NAME
from app.storage.index_jobs import PIPELINE_JOB_TYPE
from app.storage.library import SCHEMA_V1, SCHEMA_VERSION, KnowledgeStore, KnowledgeStoreError
from test_chat import ProviderStream, configure, frames, new_session, payload, rows, run_path
from test_status import assert_api_error

PARSED_TABLES = {"parsed_documents", "parsed_sections", "parsed_blocks", "source_locations"}
CHUNK_TABLES = {"chunk_sets", "chunks", "chunk_sources"}
EMBEDDING_TABLES = {"embedding_profiles", "embedding_sets", "embeddings", "chunk_embeddings"}
KEYWORD_TABLES = {
    "keyword_index_sets", "keyword_index_rows", "keyword_fts",
    "keyword_fts_data", "keyword_fts_idx", "keyword_fts_docsize", "keyword_fts_config",
}
KNOWLEDGE_TABLES = (
    {"documents", "document_versions", "content_blobs", "knowledge_imports", "index_jobs"}
    | PARSED_TABLES | CHUNK_TABLES | EMBEDDING_TABLES | KEYWORD_TABLES
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def add_document(db, name):
    return db.execute(
        "INSERT INTO documents (name, format) VALUES (?, 'pdf') RETURNING document_id", (name,),
    ).fetchone()[0]


def add_blob(db, content):
    sha256 = hashlib.sha256(content).hexdigest()
    db.execute("INSERT INTO content_blobs (sha256, size_bytes) VALUES (?, ?)", (sha256, len(content)))
    return sha256, len(content)


def add_version(db, document_id, blob):
    return db.execute(
        """INSERT INTO document_versions (document_id, content_sha256, size_bytes)
           VALUES (?, ?, ?) RETURNING document_version_id""", (document_id, *blob),
    ).fetchone()[0]


def test_first_initialization_and_restart_preserve_metadata(tmp_path):
    store = KnowledgeStore(tmp_path)
    assert store.path == tmp_path / "knowledge" / "knowledge.db"
    with store.connect() as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert db.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        document_id = add_document(db, "教材")
        blob = add_blob(db, b"fixture")
        version_id = add_version(db, document_id, blob)
        tables = db.execute("SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        assert {name for name, in tables} == KNOWLEDGE_TABLES
        if os.name == "posix":
            for directory in (store.path.parent, store.originals_dir):
                assert S_IMODE(directory.stat().st_mode) == 0o700
            for suffix in ("", "-wal", "-shm"):
                path = store.path.with_name(store.path.name + suffix)
                assert S_IMODE(path.stat().st_mode) == 0o600
        assert list(store.originals_dir.iterdir()) == []

    with KnowledgeStore(tmp_path).connect() as db:
        assert db.execute("""SELECT d.document_id, d.management_status, v.document_version_id, b.relative_path
                             FROM documents AS d JOIN document_versions AS v USING (document_id)
                             JOIN content_blobs AS b ON b.sha256 = v.content_sha256""").fetchall() == [
            (document_id, "imported", version_id, f"originals/{blob[0]}"),
        ]
        assert "active_version_id" not in {column[1] for column in db.execute("PRAGMA table_info(documents)")}
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            db.execute("UPDATE documents SET management_status = 'ready'")


def test_version_immutability_and_blob_reference_constraints(tmp_path):
    with KnowledgeStore(tmp_path).connect() as db:
        first_doc = add_document(db, "教材")
        second_doc = add_document(db, "教材")
        first_blob = add_blob(db, b"first")
        second_blob = add_blob(db, b"second")
        first_version = add_version(db, first_doc, first_blob)
        next_version = add_version(db, first_doc, second_blob)
        shared_version = add_version(db, second_doc, first_blob)
        assert first_doc != second_doc
        assert len({first_version, next_version, shared_version}) == 3
        assert db.execute("SELECT COUNT(*) FROM content_blobs").fetchone() == (2,)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            add_version(db, "missing-document", first_blob)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            add_version(db, first_doc, ("a" * 64, 5))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            add_version(db, first_doc, (first_blob[0], first_blob[1] + 1))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            db.execute("DELETE FROM documents WHERE document_id = ?", (first_doc,))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(
                "UPDATE document_versions SET content_sha256 = ?, size_bytes = ? WHERE document_version_id = ?",
                (*second_blob, first_version),
            )
        with pytest.raises(sqlite3.IntegrityError, match="already exists"):
            db.execute("""INSERT OR REPLACE INTO document_versions
                          (document_version_id, document_id, content_sha256, size_bytes) VALUES (?, ?, ?, ?)""",
                       (first_version, second_doc, *second_blob))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE documents SET document_id = 'new-id' WHERE document_id = ?", (first_doc,))
        db.execute("DELETE FROM document_versions WHERE document_version_id = ?", (first_version,))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            db.execute("DELETE FROM content_blobs WHERE sha256 = ?", (first_blob[0],))
        db.execute("DELETE FROM document_versions WHERE document_version_id = ?", (shared_version,))
        db.execute("DELETE FROM content_blobs WHERE sha256 = ?", (first_blob[0],))
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_failed_migration_rolls_back_and_can_be_retried(tmp_path):
    path = tmp_path / "knowledge" / "knowledge.db"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE content_blobs (fixture INTEGER)")
        with pytest.raises(sqlite3.OperationalError, match="already exists"):
            KnowledgeStore(tmp_path)
        assert db.execute("PRAGMA user_version").fetchone() == (0,)
        assert db.execute("SELECT name FROM sqlite_schema WHERE type = 'table'").fetchall() == [("content_blobs",)]
        db.execute("DROP TABLE content_blobs")

    with KnowledgeStore(tmp_path).connect() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        document_id = add_document(db, "保留记录")
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    with pytest.raises(KnowledgeStoreError, match="UNSUPPORTED_KNOWLEDGE_SCHEMA_VERSION"):
        KnowledgeStore(tmp_path)
    assert rows(path, "PRAGMA user_version") == [(SCHEMA_VERSION + 1,)]
    assert rows(path, "SELECT document_id FROM documents") == [(document_id,)]


def test_v2_migration_rolls_back_to_v1_and_preserves_existing_versions(tmp_path, monkeypatch):
    path = tmp_path / "knowledge" / "knowledge.db"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("BEGIN")
        for statement in SCHEMA_V1:
            db.execute(statement)
        db.execute("PRAGMA user_version=1")
        document_id = add_document(db, "旧教材")
        blob = add_blob(db, b"v1 textbook fixture")
        version = add_version(db, document_id, blob)

    with monkeypatch.context() as patch:
        migrations = storage_library.MIGRATIONS
        patch.setattr(storage_library, "MIGRATIONS", (migrations[0], migrations[1] + ("INVALID SQL",), *migrations[2:]))
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            KnowledgeStore(tmp_path)
    assert rows(path, "PRAGMA user_version") == [(1,)]
    assert rows(path, "SELECT name FROM sqlite_schema WHERE name='knowledge_imports'") == []

    store = KnowledgeStore(tmp_path)
    assert rows(path, "PRAGMA user_version") == [(SCHEMA_VERSION,)]
    details = store.get_document(document_id)
    assert details.name == "旧教材"
    assert details.versions[0].document_version_id == version
    assert details.versions[0].content_sha256 == blob[0]
    assert rows(path, "PRAGMA foreign_key_check") == []


def test_v3_migration_rolls_back_to_v2_and_keeps_versions_and_import_receipts(tmp_path, monkeypatch):
    path = tmp_path / "knowledge" / "knowledge.db"
    path.parent.mkdir()
    migrations = storage_library.MIGRATIONS
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("BEGIN")
        for migration in migrations[:2]:
            for statement in migration:
                db.execute(statement)
        db.execute("PRAGMA user_version=2")
        document_id = add_document(db, "v2 教材")
        version_id = add_version(db, document_id, add_blob(db, b"existing content"))
        receipt = ("import", "request-hash", document_id, version_id, "imported")
        db.execute("INSERT INTO knowledge_imports VALUES (?, ?, ?, ?, ?)", receipt)

    with monkeypatch.context() as patch:
        patch.setattr(storage_library, "MIGRATIONS", (*migrations[:2], migrations[2] + ("INVALID SQL",)))
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            KnowledgeStore(tmp_path)
    assert rows(path, "PRAGMA user_version") == [(2,)]
    assert PARSED_TABLES.isdisjoint(name for name, in rows(path, "SELECT name FROM sqlite_schema"))
    assert rows(path, "SELECT * FROM knowledge_imports") == [receipt]

    store = KnowledgeStore(tmp_path)
    assert rows(path, "PRAGMA user_version") == [(SCHEMA_VERSION,)]
    assert rows(path, "SELECT * FROM knowledge_imports") == [receipt]
    assert rows(path, "SELECT * FROM parsed_documents") == []
    assert store.get_document(document_id).versions[0].document_version_id == version_id
    assert rows(path, "PRAGMA foreign_key_check") == []


@pytest.mark.anyio
@pytest.mark.parametrize("knowledge_state", [
    "ready", "startup_failure", "runtime_failure", "index_jobs_failure", "pipeline_failure",
])
async def test_lifespan_keeps_chat_independent_of_knowledge(tmp_path, caplog, knowledge_state, monkeypatch):
    path = tmp_path / "knowledge" / "knowledge.db"
    if knowledge_state == "startup_failure":
        path.parent.mkdir()
        path.write_bytes(b"not a valid SQLite database")

    business_path = tmp_path / DATABASE_NAME
    ui, host = token_urlsafe(32), token_urlsafe(32)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=ProviderStream()))
    models = ModelService(DeepSeekAdapter(DeepSeekSettings(), transport=transport))
    app = create_app(ui, host, database_path=business_path, models=models)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1",
            headers={"Authorization": f"Bearer {ui}"},
        ) as client:
            if knowledge_state == "startup_failure":
                assert app.state.knowledge_store is None
                assert app.state.index_jobs is None
            else:
                assert app.state.knowledge_store.path == path
                assert app.state.index_jobs.store is app.state.knowledge_store
            if knowledge_state == "runtime_failure":
                def fail_list():
                    raise sqlite3.OperationalError("fixture database failure")

                monkeypatch.setattr(app.state.knowledge_store, "list_documents", fail_list)
            if knowledge_state == "index_jobs_failure":
                with app.state.knowledge_store.write() as db:
                    db.execute("DROP TABLE index_jobs")
                with pytest.raises(sqlite3.OperationalError, match="no such table: index_jobs"):
                    app.state.index_jobs.list_jobs()
            if knowledge_state == "pipeline_failure":
                from test_library_management import import_file

                book = import_file(app.state.library, b"Temporary geometry textbook")
                embeddings = app.state.document_embeddings
                profile = embeddings.provider.profile(compatibility_epoch="pipeline-isolation-test")
                pipeline = IndexPipeline(app.state.knowledge_store, embeddings, profile)
                job = pipeline.submit(book.document_id, book.document_version_id)
                parsed = pipeline.parsing.ensure_parsed(book.document_version_id)
                pipeline.chunking.ensure_chunked(parsed.parsed_document_id)
                with app.state.knowledge_store.write() as db:
                    db.execute("DROP TABLE keyword_fts")
                claim = app.state.index_jobs.claim_next(
                    "pipeline-isolation", job_type=PIPELINE_JOB_TYPE, job_id=job.job_id,
                )
                failed = await pipeline.execute(claim)
                assert (failed.status, failed.stage, failed.error_code) == (
                    "failed", "keyword_indexing", "INDEX_PIPELINE_STORAGE_ERROR",
                )
            library_response = await client.get("/api/v1/knowledge/documents")
            if knowledge_state in ("ready", "index_jobs_failure"):
                assert library_response.json() == []
            elif knowledge_state == "pipeline_failure":
                assert library_response.status_code == 200 and len(library_response.json()) == 1
            else:
                assert_api_error(library_response, 503, "KNOWLEDGE_STORE_UNAVAILABLE")
            if knowledge_state == "startup_failure":
                import_response = await client.post(
                    "/_host/knowledge/imports", headers={"Authorization": f"Bearer {host}"},
                    json={"import_id": str(uuid4()), "filename": "教材.md"},
                )
                assert_api_error(import_response, 503, "KNOWLEDGE_STORE_UNAVAILABLE")
            assert (await client.get("/api/v1/status")).status_code == 200
            await configure(client, {"Authorization": f"Bearer {host}"})
            session = await new_session(client)
            response = await client.post(run_path(session), json=payload())
            assert response.status_code == 200
            assert [frame async for frame in frames(response)][-1]["type"] == "run_completed"

    assert app.state.knowledge_store is None
    assert app.state.index_jobs is None
    checkpoint_path = tmp_path / CHECKPOINT_DATABASE_NAME
    assert rows(business_path, "SELECT status FROM runs") == [("completed",)]
    checkpoints = rows(checkpoint_path, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (f"session:{session}",))
    assert checkpoints[0][0] > 0
    for chat_path in (business_path, checkpoint_path):
        assert KNOWLEDGE_TABLES.isdisjoint(name for name, in rows(chat_path, "SELECT name FROM sqlite_schema"))
    if knowledge_state == "startup_failure":
        assert path.read_bytes() == b"not a valid SQLite database"
        errors = [json.loads(record.message) for record in caplog.records
                  if record.name == LOGGER.name and '"knowledge_startup_error"' in record.message]
        assert len(errors) == 1
        assert errors[0]["error_code"] == "KNOWLEDGE_STORE_UNAVAILABLE"
        assert errors[0]["sqlite_error"] == "SQLITE_NOTADB"
