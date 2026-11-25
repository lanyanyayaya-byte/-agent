"""P9.1 verification suite: SQLite Migration V7, WorkspaceService, ArtifactService, and Workspace API."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.main import create_app
from app.storage.artifacts import WorkspaceArtifactStore
from app.storage.business import MIGRATIONS, SCHEMA_VERSION, BusinessStore
from app.workspace.artifacts import ArtifactService
from app.workspace.service import WorkspaceError, WorkspaceService

UI_TOKEN = "p9-ui-token-1234567890"
HOST_TOKEN = "p9-host-token-abcdefghij"


def _create_v6_database(path: Path) -> dict[str, str]:
    session_id = str(uuid4())
    run_id = str(uuid4())
    req_id = str(uuid4())
    user_msg_id = str(uuid4())
    asst_msg_id = str(uuid4())
    call_id = str(uuid4())
    cp_id = str(uuid4())
    now = "2026-11-25T10:00:00.000Z"

    with closing(sqlite3.connect(path)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        for migration in MIGRATIONS[:6]:
            for stmt in migration:
                db.execute(stmt)
        db.execute("PRAGMA user_version=6")
        db.execute(
            "INSERT INTO sessions VALUES (?, 'V6 历史会话', ?, ?, ?)",
            (session_id, cp_id, now, now),
        )
        db.execute(
            """INSERT INTO runs (run_id, session_id, client_request_id, status, base_checkpoint_id,
                                 final_checkpoint_id, started_at, finished_at, error_code, error_message)
               VALUES (?, ?, ?, 'completed', NULL, ?, ?, ?, NULL, NULL)""",
            (run_id, session_id, req_id, cp_id, now, now),
        )
        db.execute(
            "INSERT INTO messages VALUES (?, ?, ?, 'user', 'V6 原始问题', ?)",
            (user_msg_id, session_id, run_id, now),
        )
        db.execute(
            "INSERT INTO messages VALUES (?, ?, ?, 'assistant', 'V6 原始回答', ?)",
            (asst_msg_id, session_id, run_id, now),
        )
        db.execute(
            """INSERT INTO tool_calls (tool_call_id, session_id, run_id, tool_name, status,
                                       summary, error_code, started_at, finished_at, source, duration_ms)
               VALUES (?, ?, ?, 'mcp__fetch__get', 'completed', '抓取成功', NULL, ?, ?, 'mcp', 85)""",
            (call_id, session_id, run_id, now, now),
        )
        db.execute(
            """INSERT INTO mcp_servers (server_id, name, transport, url, created_at, updated_at)
               VALUES ('srv-v6', 'Fetch Server', 'streamable_http', 'https://example.com/mcp', ?, ?)""",
            (now, now),
        )
        db.execute(
            """INSERT INTO run_citations (
                   run_id, session_id, ordinal, evidence_id, chunk_id, chunk_set_id, chunk_index,
                   document_id, document_version_id, textbook_name, format, section_path,
                   location_summary, location_precision, location_status, warnings, snippet
               ) VALUES (?, ?, 0, 'ev1', 'ck1', 1, 0, 'doc1', 'ver1', '线性代数', 'pdf',
                         '["第一章"]', '第 3 页', 'exact', 'resolved', '[]', '矩阵乘法')""",
            (run_id, session_id),
        )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "call_id": call_id,
    }


def test_v6_to_v7_migration_preserves_all_data_and_rolls_back_on_error(tmp_path):
    path = tmp_path / "business.sqlite3"
    ids = _create_v6_database(path)

    # Verify rollback on mid-migration conflict first on a copy
    conflict_path = tmp_path / "conflict.sqlite3"
    _create_v6_database(conflict_path)
    with closing(sqlite3.connect(conflict_path)) as db, db:
        db.execute("CREATE TABLE artifacts (bad TEXT PRIMARY KEY)")
    with pytest.raises(sqlite3.OperationalError):
        BusinessStore(conflict_path)
    with closing(sqlite3.connect(conflict_path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (6,)
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'workspace_files'").fetchone() is None

    # Now perform real V6 -> V7 upgrade
    store = BusinessStore(path)
    assert SCHEMA_VERSION == 7
    with closing(store._connect()) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (7,)
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        mcp_count = db.execute("SELECT count(*) FROM mcp_servers WHERE server_id = 'srv-v6'").fetchone()[0]
        assert mcp_count == 1

    # Reopen idempotently and check all historical data
    reopened = BusinessStore(path)
    assert [m.content for m in reopened.list_messages(ids["session_id"])] == ["V6 原始问题", "V6 原始回答"]
    runs = reopened.list_runs(ids["session_id"], 10, None)
    assert len(runs) == 1
    assert runs[0].citations[0].textbook_name == "线性代数"
    assert runs[0].artifacts == ()
    assert runs[0].skill_activations == ()
    tc = runs[0].tool_calls[0]
    assert tc.source == "mcp"
    assert tc.tool_category is None
    assert tc.recoverable is False


def test_complete_run_allows_recoverable_failures_and_blocks_fatal_failures(tmp_path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    session = store.create_session("可恢复失败测试")

    # Run 1: recoverable runtime failure followed by successful tool -> complete_run MUST succeed
    run1 = store.start_run(session.session_id, str(uuid4()), "执行脚本", accept_new_run=True)
    assert run1 is not None
    c1, c2 = str(uuid4()), str(uuid4())
    store.start_tool(run1, c1, "execute_command", source="native", tool_category="runtime")
    store.finish_tool(
        run1, c1, "failed", "SyntaxError on line 1", "RUNTIME_NON_ZERO_EXIT",
        duration_ms=42, recoverable=True,
    )
    store.start_tool(run1, c2, "execute_command", source="native", tool_category="runtime")
    store.finish_tool(run1, c2, "completed", "脚本执行成功", None, duration_ms=110)

    msg_id = store.complete_run(run1, "已修正脚本并生成文件。", str(uuid4()))
    assert msg_id is not None

    summaries = store.list_runs(session.session_id, 10, None)
    assert len(summaries) == 1
    assert summaries[0].status == "completed"
    assert len(summaries[0].tool_calls) == 2
    assert summaries[0].tool_calls[0].status == "failed"
    assert summaries[0].tool_calls[0].recoverable is True
    assert summaries[0].tool_calls[0].tool_category == "runtime"
    assert summaries[0].tool_calls[1].status == "completed"

    # Run 2: legacy native file tool failure (recoverable=False) -> complete_run MUST still raise
    run2 = store.start_run(session.session_id, str(uuid4()), "读取未授权文件", accept_new_run=True)
    assert run2 is not None
    c3 = str(uuid4())
    store.start_tool(run2, c3, "read_text", source="native")
    store.finish_tool(run2, c3, "failed", "未获读取授权", "FILE_PERMISSION_DENIED", recoverable=False)
    with pytest.raises(RuntimeError, match="Run completion requires successful tool calls"):
        store.complete_run(run2, "不应提交", str(uuid4()))


def test_workspace_file_operations_artifacts_lifecycle_and_restart_recovery(tmp_path):
    app_data = tmp_path / "app_data"
    db_path = app_data / "knowledgeagent.sqlite3"
    business = BusinessStore(db_path)
    artifact_store = WorkspaceArtifactStore(business)
    workspace = WorkspaceService(app_data, artifact_store)
    artifacts = ArtifactService(workspace, artifact_store)

    session = business.create_session("工作区持久化测试")
    sid = session.session_id

    # 1. Import original local file (copy; original untouched)
    orig = tmp_path / "用户原件.txt"
    orig.write_text("第一行内容\n第二行内容\n第三行内容\n", encoding="utf-8")
    imported = workspace.import_local_file(sid, orig)
    assert imported.original_name == "用户原件.txt"
    assert imported.size_bytes == orig.stat().st_size

    # Session egress risk must detect workspace_files
    assert business.inspect_session_egress_risk(sid).has_historical_file_context is True

    # 2. Run 1: copy input, edit in work/, write output, stage artifact, complete_run -> published
    run1 = business.start_run(sid, str(uuid4()), "处理原件", accept_new_run=True)
    assert run1 is not None
    copied = workspace.copy_input_to_run(sid, run1.run_id, imported.file_id)
    assert copied.is_file()
    copied.write_text("修改后的副本内容", encoding="utf-8")
    assert orig.read_text(encoding="utf-8").startswith("第一行内容")  # original preserved!

    write_res = workspace.write_file(sid, "output/结果报告.md", "# 报告\n\n处理完成。", run_id=run1.run_id)
    assert write_res.size_bytes > 0

    read_res = workspace.read_file(sid, "output/结果报告.md", run_id=run1.run_id, offset_line=0, limit_lines=10)
    assert not read_res.is_binary
    assert "处理完成" in (read_res.content or "")

    # Traversal and symlink escape must be rejected
    with pytest.raises(WorkspaceError) as exc_info:
        workspace.read_file(sid, "../../etc/passwd", run_id=run1.run_id)
    assert exc_info.value.code == "WORKSPACE_PATH_INVALID"

    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    escape_link = workspace.run_work_dir(sid, run1.run_id) / "escape.txt"
    escape_link.symlink_to(outside)
    with pytest.raises(WorkspaceError) as exc_info:
        workspace.read_file(sid, "work/escape.txt", run_id=run1.run_id)
    assert exc_info.value.code == "WORKSPACE_PATH_INVALID"

    staged = artifacts.stage_artifact(sid, run1.run_id, "output/结果报告.md", title="处理报告")
    assert staged.status == "staged"
    # Before commit, published list must be empty
    assert artifacts.list_artifacts(sid, include_non_published=False) == []

    # Record skill activation metadata & complete run
    artifact_store.record_skill_activation(
        session_id=sid, run_id=run1.run_id, skill_id="sk-1", name_snapshot="demo-skill", revision="r1",
    )
    assert business.complete_run(run1, "已完成报告生成。", str(uuid4())) is not None

    # 3. Run 2 (failed) and Run 3 (cancelled): staged artifacts MUST be discarded, never shown as published
    run2 = business.start_run(sid, str(uuid4()), "失败任务", accept_new_run=True)
    assert run2 is not None
    workspace.write_file(sid, "output/failed_out.txt", "partial", run_id=run2.run_id)
    failed_art = artifacts.stage_artifact(sid, run2.run_id, "output/failed_out.txt")
    business.fail_run(run2, "RUNTIME_NON_ZERO_EXIT", "脚本失败")

    run3 = business.start_run(sid, str(uuid4()), "取消任务", accept_new_run=True)
    assert run3 is not None
    workspace.write_file(sid, "output/cancelled_out.txt", "partial", run_id=run3.run_id)
    cancelled_art = artifacts.stage_artifact(sid, run3.run_id, "output/cancelled_out.txt")
    business.request_cancel(run3)
    assert business.confirm_cancel(run3) is True

    # 4. Run 4 (interrupted by restart): recover_interrupted_runs MUST discard staged artifact
    run4 = business.start_run(sid, str(uuid4()), "崩溃任务", accept_new_run=True)
    assert run4 is not None
    workspace.write_file(sid, "output/crashed_out.txt", "partial", run_id=run4.run_id)
    crashed_art = artifacts.stage_artifact(sid, run4.run_id, "output/crashed_out.txt")

    # Simulate application restart
    business2 = BusinessStore(db_path)
    business2.recover_interrupted_runs()
    store2 = WorkspaceArtifactStore(business2)
    ws2 = WorkspaceService(app_data, store2)
    arts2 = ArtifactService(ws2, store2)

    published = arts2.list_artifacts(sid, include_non_published=False)
    assert [a.artifact_id for a in published] == [staged.artifact_id]
    assert published[0].file_exists is True
    assert published[0].status == "published"

    for discarded_id in (failed_art.artifact_id, cancelled_art.artifact_id, crashed_art.artifact_id):
        rec, _ = arts2.get_artifact(sid, discarded_id, include_non_published=True)
        assert rec.status == "discarded"
        with pytest.raises(WorkspaceError):
            arts2.get_artifact(sid, discarded_id, include_non_published=False)

    # Cross-turn reuse: Run 5 can copy Run 1's published artifact into its work/
    run5 = business2.start_run(sid, str(uuid4()), "基于旧产物继续编辑", accept_new_run=True)
    assert run5 is not None
    reused = ws2.copy_artifact_to_run(sid, run5.run_id, staged.artifact_id)
    assert reused.read_text(encoding="utf-8") == "# 报告\n\n处理完成。"
    assert business2.complete_run(run5, "继续编辑完成。", str(uuid4())) is not None

    # RunSummary includes published artifacts and skill_activations
    history_runs = {r.run_id: r for r in business2.list_runs(sid, 10, None)}
    assert [a.artifact_id for a in history_runs[run1.run_id].artifacts] == [staged.artifact_id]
    assert history_runs[run1.run_id].skill_activations[0].name_snapshot == "demo-skill"
    assert history_runs[run2.run_id].artifacts == ()
    assert history_runs[run3.run_id].artifacts == ()
    assert history_runs[run4.run_id].artifacts == ()


@pytest.mark.anyio
async def test_workspace_api_dual_token_and_session_deletion_cleanup(tmp_path):
    db_path = tmp_path / "knowledgeagent.sqlite3"
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=db_path)

    sample = tmp_path / "sample.md"
    sample.write_text("# 标题\n正文内容\n", encoding="utf-8")

    ui_headers = {"Authorization": f"Bearer {UI_TOKEN}", "Host": "127.0.0.1:8000"}
    host_headers = {"Authorization": f"Bearer {HOST_TOKEN}", "Host": "127.0.0.1:8000"}

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            # Create session via UI token
            resp = await client.post("/api/v1/sessions", json={"title": "API测试"}, headers=ui_headers)
            assert resp.status_code == 201
            sid = resp.json()["session_id"]

            # UI token cannot call _host import endpoint
            denied = await client.post(
                f"/api/v1/_host/sessions/{sid}/workspace/files",
                json={"path": str(sample)},
                headers=ui_headers,
            )
            assert denied.status_code == 403

            # Host token imports file cleanly
            imported = await client.post(
                f"/api/v1/_host/sessions/{sid}/workspace/files",
                json={"path": str(sample)},
                headers=host_headers,
            )
            assert imported.status_code == 201
            file_id = imported.json()["file_id"]

            # Stage and publish an artifact via app services
            store = app.state.runs._store
            run = store.start_run(sid, str(uuid4()), "生成SVG", accept_new_run=True)
            app.state.workspace_service.write_file(
                sid, "output/demo.svg", '<svg xmlns="http://www.w3.org/2000/svg"><text>测试</text></svg>',
                run_id=run.run_id,
            )
            art = app.state.artifact_service.stage_artifact(sid, run.run_id, "output/demo.svg", title="演示导图")
            store.complete_run(run, "完成", str(uuid4()))

            # UI reads workspace, artifact metadata, and artifact content
            ws_resp = await client.get(f"/api/v1/sessions/{sid}/workspace", headers=ui_headers)
            assert ws_resp.status_code == 200
            body = ws_resp.json()
            assert len(body["files"]) == 1 and body["files"][0]["file_id"] == file_id
            assert len(body["artifacts"]) == 1 and body["artifacts"][0]["artifact_id"] == art.artifact_id

            meta_resp = await client.get(
                f"/api/v1/sessions/{sid}/artifacts/{art.artifact_id}", headers=ui_headers,
            )
            assert meta_resp.status_code == 200
            assert meta_resp.json()["filename"] == "demo.svg"

            content_resp = await client.get(
                f"/api/v1/sessions/{sid}/artifacts/{art.artifact_id}/content", headers=ui_headers,
            )
            assert content_resp.status_code == 200
            assert "image/svg+xml" in content_resp.headers["content-type"]
            assert "测试" in content_resp.text

            # Host reads absolute path for native Open / Save As
            path_resp = await client.get(
                f"/api/v1/_host/sessions/{sid}/artifacts/{art.artifact_id}/path", headers=host_headers,
            )
            assert path_resp.status_code == 200
            assert Path(path_resp.json()["absolute_path"]).is_file()

            # Deleting session cleans workspace directory on disk
            session_dir = app.state.workspace_service.session_root(sid)
            assert session_dir.is_dir()
            del_resp = await client.post(f"/api/v1/sessions/{sid}/delete", headers=ui_headers)
            assert del_resp.status_code == 204
            assert not session_dir.exists()
