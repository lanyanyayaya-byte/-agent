"""P6.3.1 & P6.3.2 verification suite: SQLite Migration V6 and McpStore persistence.

Covers:
- V5 -> V6 real database migration preserving sessions, runs, messages, tool_calls, run_citations
- Fresh V6 initialization, repeated idempotent startup, and migration rollback on failure
- Older-version boundary check against newer V6 database
- SQLite CHECK constraints, foreign keys, and indexes
- Server CRUD, restart persistence, untrusted/disabled defaults, config_revision & trust invalidation
- Display-name rename preserving trust and internal tool identity
- Full untruncated tool definition_hash drift invalidation & removed-tool revocation
- Schema incompatibility & destructiveHint blocking
- Generation & config_revision fencing during dynamic catalog sync
- Historical tool_calls preservation after deleting an MCP server
- Runtime effective authorization never trusting SQLite permission_state alone
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest

from app.mcp.models import McpError, McpToolDefinition
from app.mcp.policy import compute_tool_definition_hash
from app.storage.business import MIGRATIONS, SCHEMA_VERSION, BusinessStore
from app.storage.mcp import ConfigValueSpec, McpStore


def _make_tool_def(
    server_id: str,
    name: str,
    *,
    title: str | None = None,
    description: str = "A safe read-only tool.",
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    annotations: dict | None = None,
) -> McpToolDefinition:
    schema = input_schema if input_schema is not None else {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    ann = annotations if annotations is not None else {"readOnlyHint": True}
    digest = compute_tool_definition_hash(
        name=name,
        title=title,
        description=description,
        input_schema=schema,
        output_schema=output_schema,
        annotations=ann,
    )
    return McpToolDefinition(
        server_id=server_id,
        original_tool_name=name,
        title=title,
        description=description,
        input_schema=schema,
        output_schema=output_schema,
        annotations=ann,
        definition_hash=digest,
    )


def _create_v5_database(path: Path) -> dict[str, str]:
    session_id = str(uuid4())
    run_id = str(uuid4())
    req_id = str(uuid4())
    user_msg_id = str(uuid4())
    asst_msg_id = str(uuid4())
    call_id = str(uuid4())
    cp_id = str(uuid4())
    now = "2026-11-24T12:00:00.000Z"

    with closing(sqlite3.connect(path)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        for migration in MIGRATIONS[:5]:
            for stmt in migration:
                db.execute(stmt)
        db.execute("PRAGMA user_version=5")
        db.execute(
            "INSERT INTO sessions VALUES (?, 'V5 历史会话', ?, ?, ?)",
            (session_id, cp_id, now, now),
        )
        db.execute(
            """INSERT INTO runs (run_id, session_id, client_request_id, status, base_checkpoint_id,
                                 final_checkpoint_id, started_at, finished_at, error_code, error_message)
               VALUES (?, ?, ?, 'completed', NULL, ?, ?, ?, NULL, NULL)""",
            (run_id, session_id, req_id, cp_id, now, now),
        )
        db.execute(
            "INSERT INTO messages VALUES (?, ?, ?, 'user', '原始问题', ?)",
            (user_msg_id, session_id, run_id, now),
        )
        db.execute(
            "INSERT INTO messages VALUES (?, ?, ?, 'assistant', '原始回答', ?)",
            (asst_msg_id, session_id, run_id, now),
        )
        db.execute(
            """INSERT INTO tool_calls (tool_call_id, session_id, run_id, tool_name, status,
                                       summary, error_code, started_at, finished_at)
               VALUES (?, ?, ?, 'read_text', 'completed', '读取 120 字符', NULL, ?, ?)""",
            (call_id, session_id, run_id, now, now),
        )
        db.execute(
            """INSERT INTO run_citations (
                   run_id, session_id, ordinal, evidence_id, chunk_id, chunk_set_id, chunk_index,
                   document_id, document_version_id, textbook_name, format, section_path,
                   location_summary, location_precision, location_status, warnings, snippet
               ) VALUES (?, ?, 0, 'ev1', 'ck1', 1, 0, 'doc1', 'ver1', '高等数学', 'pdf',
                         '["第一章"]', '第 1 页', 'exact', 'resolved', '[]', '导数定义')""",
            (run_id, session_id),
        )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "user_msg_id": user_msg_id,
        "asst_msg_id": asst_msg_id,
        "call_id": call_id,
    }


def test_v5_to_v6_migration_preserves_all_legacy_data_and_is_idempotent(tmp_path):
    path = tmp_path / "business.sqlite3"
    ids = _create_v5_database(path)

    # Upgrade V5 -> latest (V6+V7)
    store = BusinessStore(path)
    assert SCHEMA_VERSION >= 6
    with closing(store._connect()) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []

    # Repeated startup must be idempotent
    store_reopened = BusinessStore(path)
    assert [m.content for m in store_reopened.list_messages(ids["session_id"])] == ["原始问题", "原始回答"]

    runs = store_reopened.list_runs(ids["session_id"], 10, None)
    assert len(runs) == 1
    assert runs[0].run_id == ids["run_id"]
    assert len(runs[0].citations) == 1
    assert runs[0].citations[0].textbook_name == "高等数学"
    assert len(runs[0].tool_calls) == 1
    tc = runs[0].tool_calls[0]
    assert tc.tool_call_id == ids["call_id"]
    assert tc.tool_name == "read_text"
    assert tc.source == "native"
    assert tc.server_id is None
    assert tc.server_name is None
    assert tc.original_tool_name is None
    assert tc.duration_ms is None


def test_v5_to_v6_migration_rolls_back_safely_on_failure(tmp_path):
    path = tmp_path / "business.sqlite3"
    ids = _create_v5_database(path)

    # Inject a conflicting table so V6 migration fails mid-transaction.
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE mcp_tools (conflicting_id TEXT PRIMARY KEY)")

    with pytest.raises(sqlite3.OperationalError):
        BusinessStore(path)

    # Database must remain cleanly at V5 with zero partial V6 tables/columns or data loss.
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (5,)
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'mcp_servers'").fetchone() is None
        cols = {row[1] for row in db.execute("PRAGMA table_info(tool_calls)").fetchall()}
        assert "source" not in cols
        assert db.execute("SELECT count(*) FROM sessions WHERE session_id = ?", (ids["session_id"],)).fetchone() == (1,)
        assert db.execute("SELECT count(*) FROM messages").fetchone() == (2,)
        assert db.execute("SELECT count(*) FROM run_citations").fetchone() == (1,)


def test_fresh_v6_schema_constraints_and_future_version_rejection(tmp_path):
    path = tmp_path / "fresh.sqlite3"
    store = BusinessStore(path)

    with closing(store._connect()) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        # CHECK: stdio requires command and forbids url
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO mcp_servers (server_id, name, transport, command, url, created_at, updated_at)
                   VALUES ('s1', 'Bad', 'stdio', NULL, 'https://example.com', 't', 't')"""
            )
        # CHECK: trusted requires non-null trusted_revision and trusted_fingerprint
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO mcp_servers (server_id, name, transport, url, trust_state, created_at, updated_at)
                   VALUES ('s2', 'BadTrust', 'streamable_http', 'https://example.com', 'trusted', 't', 't')"""
            )
        # CHECK: mcp_tools allowed requires approved_definition_hash & approved_config_revision
        db.execute(
            """INSERT INTO mcp_servers (server_id, name, transport, url, created_at, updated_at)
               VALUES ('s3', 'ValidHTTP', 'streamable_http', 'https://example.com', 't', 't')"""
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO mcp_tools (tool_id, server_id, original_tool_name, input_schema_json,
                                          definition_hash, permission_state, discovered_at, updated_at)
                   VALUES ('t1', 's3', 'search', '{}', 'abc', 'allowed', 't', 't')"""
            )
        # Case-insensitive unique server name index
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO mcp_servers (server_id, name, transport, url, created_at, updated_at)
                   VALUES ('s4', 'validhttp', 'streamable_http', 'https://other.example', 't', 't')"""
            )

    # Simulate an older app binary encountering a newer database version.
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="Unsupported business database schema version"):
        BusinessStore(path)


def test_mcp_store_crud_trust_invalidation_rename_and_restart_recovery(tmp_path):
    path = tmp_path / "business.sqlite3"
    business = BusinessStore(path)
    mcp_store = McpStore(business)

    # 1. Create stdio server -> always defaults to disabled & untrusted
    created = mcp_store.create_server(
        name="Local Docs",
        transport="stdio",
        command=sys.executable,
        args=("-m", "http.server"),
        env_config={
            "MODE": ConfigValueSpec(kind="plain", value="readonly"),
            "API_KEY": ConfigValueSpec(kind="secret", secret_key="API_KEY"),
        },
    )
    assert created.enabled is False
    assert created.trust_state == "untrusted"
    assert created.config_revision == 1
    assert created.trusted_revision is None
    assert created.trusted_fingerprint is None
    assert created.credential_ref == f"mcp-server:{created.server_id}"

    # Cannot enable while untrusted
    with pytest.raises(McpError) as exc:
        mcp_store.update_server(created.server_id, enabled=True)
    assert exc.value.code == "MCP_UNTRUSTED_SERVER"

    # Trust fails with stale revision or mismatched fingerprint
    fp1 = created.compute_current_fingerprint()
    with pytest.raises(McpError) as exc:
        mcp_store.trust_server(created.server_id, expected_config_revision=2, expected_fingerprint=fp1)
    assert exc.value.code == "MCP_TRUST_STALE"
    with pytest.raises(McpError) as exc:
        mcp_store.trust_server(created.server_id, expected_config_revision=1, expected_fingerprint="0" * 64)
    assert exc.value.code == "MCP_TRUST_STALE"

    # Trust succeeds with matching revision & fingerprint
    trusted = mcp_store.trust_server(
        created.server_id,
        expected_config_revision=1,
        expected_fingerprint=fp1,
        enable=True,
    )
    assert trusted.trust_state == "trusted"
    assert trusted.trusted_revision == 1
    assert trusted.trusted_fingerprint == fp1
    assert trusted.enabled is True
    assert McpStore.is_server_trust_valid(trusted) is True

    # Discover tools -> default permission_state is 'denied'
    tool_v1 = _make_tool_def(created.server_id, "search_docs", description="Search local docs.")
    sync_res = mcp_store.sync_discovered_tools(
        created.server_id,
        [tool_v1],
        generation=1,
        catalog_version=1,
        expected_config_revision=1,
    )
    assert sync_res.synced is True
    assert len(sync_res.tools) == 1
    stored_tool = sync_res.tools[0]
    assert stored_tool.permission_state == "denied"

    # Authorize tool
    allowed_tool = mcp_store.set_tool_permission(
        created.server_id,
        stored_tool.tool_id,
        permission_state="allowed",
        expected_definition_hash=stored_tool.definition_hash,
        expected_config_revision=1,
    )
    assert allowed_tool.permission_state == "allowed"

    # Persisted SQLite permission alone is NEVER effective when disconnected or credentials missing!
    assert (
        McpStore.is_tool_effectively_authorized(
            trusted,
            allowed_tool,
            live_connected=False,
            live_catalog_stale=False,
            live_tool=tool_v1,
            credentials_complete=True,
        )
        is False
    )
    assert (
        McpStore.is_tool_effectively_authorized(
            trusted,
            allowed_tool,
            live_connected=True,
            live_catalog_stale=False,
            live_tool=tool_v1,
            credentials_complete=False,
        )
        is False
    )
    assert (
        McpStore.is_tool_effectively_authorized(
            trusted,
            allowed_tool,
            live_connected=True,
            live_catalog_stale=True,
            live_tool=tool_v1,
            credentials_complete=True,
        )
        is False
    )
    assert (
        McpStore.is_tool_effectively_authorized(
            trusted,
            allowed_tool,
            live_connected=True,
            live_catalog_stale=False,
            live_tool=tool_v1,
            credentials_complete=True,
        )
        is True
    )

    # Display name rename must NOT bump revision, revoke trust, or alter tool identity!
    rename_res = mcp_store.update_server(created.server_id, name="Renamed Docs")
    assert rename_res.security_changed is False
    assert rename_res.name_changed is True
    assert rename_res.server.config_revision == 1
    assert rename_res.server.trust_state == "trusted"
    tools_after_rename = mcp_store.list_tools(created.server_id)
    assert tools_after_rename[0].tool_id == stored_tool.tool_id
    assert tools_after_rename[0].original_tool_name == "search_docs"
    assert tools_after_rename[0].permission_state == "allowed"

    # Simulate restart by instantiating a brand new BusinessStore & McpStore on the same SQLite path
    restarted_store = McpStore(BusinessStore(path))
    reloaded = restarted_store.get_server(created.server_id)
    assert reloaded.name == "Renamed Docs"
    assert reloaded.trust_state == "trusted"
    assert reloaded.enabled is True
    assert reloaded.config_revision == 1
    assert restarted_store.list_tools(created.server_id)[0].permission_state == "allowed"

    # Security-critical config change (modifying args) revokes trust & ALL tool approvals!
    edit_res = restarted_store.update_server(created.server_id, args=("-m", "http.server", "8080"))
    assert edit_res.security_changed is True
    assert edit_res.server.config_revision == 2
    assert edit_res.server.trust_state == "untrusted"
    assert edit_res.server.trusted_revision is None
    assert edit_res.server.trusted_fingerprint is None
    assert edit_res.server.enabled is False
    assert restarted_store.list_tools(created.server_id)[0].permission_state == "denied"
    assert restarted_store.list_tools(created.server_id)[0].approved_config_revision is None


def test_tool_definition_hash_full_text_drift_removal_schema_incompat_and_generation_fencing(tmp_path):
    path = tmp_path / "business.sqlite3"
    mcp_store = McpStore(BusinessStore(path))
    server = mcp_store.create_server(
        name="HTTP Tools",
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
    )
    fp = server.compute_current_fingerprint()
    server = mcp_store.trust_server(server.server_id, expected_config_revision=1, expected_fingerprint=fp, enable=True)

    # Description longer than 1024 chars: changing tail beyond 1024 chars MUST change definition_hash!
    long_prefix = "A" * 1200
    t_long_v1 = _make_tool_def(server.server_id, "long_desc_tool", description=long_prefix + " benign tail")
    t_bad_schema = _make_tool_def(
        server.server_id,
        "circular_schema_tool",
        input_schema={
            "type": "object",
            "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        },
    )
    t_destructive = _make_tool_def(
        server.server_id,
        "wipe_data",
        annotations={"destructiveHint": True},
    )
    t_removable = _make_tool_def(server.server_id, "temp_tool")

    sync1 = mcp_store.sync_discovered_tools(
        server.server_id,
        [t_long_v1, t_bad_schema, t_destructive, t_removable],
        generation=2,
        catalog_version=1,
        expected_config_revision=1,
    )
    assert sync1.synced is True
    by_name = {t.original_tool_name: t for t in sync1.tools}

    # Incompatible schema and destructive tool cannot be authorized
    assert by_name["circular_schema_tool"].schema_compatible is False
    with pytest.raises(McpError) as exc:
        mcp_store.set_tool_permission(
            server.server_id,
            by_name["circular_schema_tool"].tool_id,
            permission_state="allowed",
            expected_definition_hash=by_name["circular_schema_tool"].definition_hash,
            expected_config_revision=1,
        )
    assert exc.value.code == "MCP_SCHEMA_INCOMPATIBLE"

    with pytest.raises(McpError) as exc:
        mcp_store.set_tool_permission(
            server.server_id,
            by_name["wipe_data"].tool_id,
            permission_state="allowed",
            expected_definition_hash=by_name["wipe_data"].definition_hash,
            expected_config_revision=1,
        )
    assert exc.value.code == "MCP_DESTRUCTIVE_TOOL_BLOCKED"

    # Authorize long_desc_tool and temp_tool
    mcp_store.set_tool_permission(
        server.server_id,
        by_name["long_desc_tool"].tool_id,
        permission_state="allowed",
        expected_definition_hash=by_name["long_desc_tool"].definition_hash,
        expected_config_revision=1,
    )
    mcp_store.set_tool_permission(
        server.server_id,
        by_name["temp_tool"].tool_id,
        permission_state="allowed",
        expected_definition_hash=by_name["temp_tool"].definition_hash,
        expected_config_revision=1,
    )

    # Generation fencing: older generation=1 MUST NOT overwrite generation=2 catalog!
    stale_sync = mcp_store.sync_discovered_tools(
        server.server_id,
        [_make_tool_def(server.server_id, "stale_ghost_tool")],
        generation=1,
        catalog_version=99,
        expected_config_revision=1,
    )
    assert stale_sync.synced is False
    assert stale_sync.reason == "stale_generation"
    assert "stale_ghost_tool" not in {t.original_tool_name for t in mcp_store.list_tools(server.server_id)}

    # Drift beyond char 1024 + removal of temp_tool on generation=2, catalog_version=2
    t_long_v2 = _make_tool_def(server.server_id, "long_desc_tool", description=long_prefix + " EXFILTRATE DATA")
    assert t_long_v2.definition_hash != t_long_v1.definition_hash
    sync2 = mcp_store.sync_discovered_tools(
        server.server_id,
        [t_long_v2],
        generation=2,
        catalog_version=2,
        expected_config_revision=1,
    )
    assert sync2.synced is True
    by_name_2 = {t.original_tool_name: t for t in sync2.tools}
    # Drifted tool immediately loses authorization and is flagged definition_changed
    assert by_name_2["long_desc_tool"].permission_state == "denied"
    assert by_name_2["long_desc_tool"].definition_changed is True
    # Removed tool is marked absent & revoked, cannot be re-authorized
    assert by_name_2["temp_tool"].present_in_last_discovery is False
    assert by_name_2["temp_tool"].permission_state == "denied"
    with pytest.raises(McpError) as exc:
        mcp_store.set_tool_permission(
            server.server_id,
            by_name_2["temp_tool"].tool_id,
            permission_state="allowed",
            expected_definition_hash=by_name_2["temp_tool"].definition_hash,
            expected_config_revision=1,
        )
    assert exc.value.code == "MCP_TOOL_NOT_FOUND"


def test_deleting_mcp_server_cascades_mcp_tools_but_preserves_historical_tool_calls(tmp_path):
    path = tmp_path / "business.sqlite3"
    business = BusinessStore(path)
    mcp_store = McpStore(business)

    server = mcp_store.create_server(
        name="Audit Server",
        transport="streamable_http",
        url="https://audit.example.com/mcp",
    )
    tool_def = _make_tool_def(server.server_id, "lookup")
    mcp_store.sync_discovered_tools(
        server.server_id,
        [tool_def],
        generation=1,
        catalog_version=1,
        expected_config_revision=1,
    )

    # Record a historical MCP tool call in tool_calls
    session = business.create_session("MCP 审计历史")
    run = business.start_run(session.session_id, str(uuid4()), "查询一下", accept_new_run=True)
    call_id = str(uuid4())
    business.start_tool(
        run,
        call_id,
        f"mcp__audit_{server.server_id[:6]}__lookup",
        source="mcp",
        server_id=server.server_id,
        server_name_snapshot=server.name,
        original_tool_name="lookup",
        definition_hash=tool_def.definition_hash,
    )
    business.finish_tool(run, call_id, "completed", "MCP 工具返回 42 字符。", None, duration_ms=135)
    business.complete_run(run, "查询完成。", str(uuid4()))

    # Delete the MCP server
    mcp_store.delete_server(server.server_id)

    # mcp_tools cascaded, but historical tool_calls remain completely intact and readable!
    with closing(business._connect()) as db:
        assert db.execute("SELECT count(*) FROM mcp_servers WHERE server_id = ?", (server.server_id,)).fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM mcp_tools WHERE server_id = ?", (server.server_id,)).fetchone() == (0,)

    runs = business.list_runs(session.session_id, 10, None)
    assert len(runs) == 1
    assert len(runs[0].tool_calls) == 1
    tc = runs[0].tool_calls[0]
    assert tc.source == "mcp"
    assert tc.server_id == server.server_id
    assert tc.server_name == "Audit Server"
    assert tc.original_tool_name == "lookup"
    assert tc.duration_ms == 135
    assert tc.summary == "MCP 工具返回 42 字符。"
