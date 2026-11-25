import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from uuid import uuid4

import pytest
import httpx

from app.storage.business import MIGRATIONS, SCHEMA_VERSION, BusinessStore, SCHEMA_V1
from test_chat import (
    SESSIONS, ProviderStream, configure, frames, live_api, new_session, next_event, payload, run_path,
)

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_history_auth_pagination_association_and_committed_messages(tmp_path):
    gate = asyncio.Event()

    async def provider(request):
        question = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(200, stream=ProviderStream(
            gate if question == "cancel" else None, mode="timeout" if question == "fail" else "stop",
        ))

    async with live_api(tmp_path / "business.sqlite3", provider) as (client, host):
        await configure(client, host)
        session, other = await new_session(client), await new_session(client)
        runs_path = f"{SESSIONS}/{session}/runs"
        messages_path = f"{SESSIONS}/{session}/messages"
        for path in (SESSIONS, runs_path, messages_path):
            assert (await client.get(path, headers={"Authorization": "Bearer wrong"})).status_code == 401
            assert (await client.get(path, headers=host)).status_code == 403
        for resource in ("runs", "messages"):
            assert (await client.get(f"{SESSIONS}/{uuid4()}/{resource}")).status_code == 404

        finished = []
        for question, event in (("complete", "run_completed"), ("fail", "error")):
            async with client.stream("POST", run_path(session), json=payload(question)) as response:
                events = [item async for item in frames(response) if isinstance(item, dict)]
            assert events[-1]["type"] == event
            finished.append(events[-1]["run_id"])

        async with client.stream("POST", run_path(session), json=payload("cancel")) as response:
            events = frames(response)
            started = await next_event(events)
            assert (await next_event(events))["type"] == "message_delta"
            run_id = started["run_id"]
            sessions = {row["session_id"]: row for row in (await client.get(SESSIONS)).json()}
            assert sessions[session]["active_run_id"] == run_id
            assert sessions[session]["active_run_status"] == "running"
            assert sessions[other]["active_run_id"] is sessions[other]["active_run_status"] is None
            messages = (await client.get(messages_path)).json()
            assert [m["role"] for m in messages if m["run_id"] == run_id] == ["user"]
            assert (await client.post(f"/api/v1/runs/{run_id}/cancel")).status_code == 202
            assert (await next_event(events))["type"] == "run_cancelled"

        all_rows = []
        cursor = None
        for expected in (run_id, *reversed(finished)):
            params = {"limit": 1, **({"cursor": cursor} if cursor else {})}
            page = (await client.get(runs_path, params=params)).json()
            assert len(page["items"]) == 1
            assert page["items"][0]["run_id"] == expected
            all_rows.extend(page["items"])
            cursor = page["next_cursor"]
            if len(all_rows) == 1:
                assert (await client.get(f"{SESSIONS}/{other}/runs", params={"cursor": cursor})).status_code == 422
        assert cursor is None
        assert [(item["status"], item["error_code"]) for item in all_rows] == [
            ("cancelled", None), ("failed", "MODEL_TIMEOUT"), ("completed", None),
        ]
        messages = (await client.get(messages_path)).json()
        assert all(message["status"] == "committed" and message["session_id"] == session for message in messages)
        users = {message["run_id"]: message["message_id"] for message in messages if message["role"] == "user"}
        assert all(users[row["run_id"]] == row["user_message_id"] and row["finished_at"] for row in all_rows)
        assert [message["run_id"] for message in messages if message["role"] == "assistant"] == [finished[0]]
        assert (await client.get(f"{SESSIONS}/{other}/messages")).json() == []
        assert (await client.get(f"{SESSIONS}/{other}/runs")).json() == {"items": [], "next_cursor": None}
        assert (await client.get(runs_path, params={"limit": 101})).status_code == 422
        invalid = await client.get(runs_path, params={"cursor": "not a cursor"})
        assert invalid.status_code == 422
        assert invalid.headers["x-request-id"] == invalid.json()["error"]["request_id"]


@pytest.fixture
def legacy_database(tmp_path):
    path = tmp_path / "business.sqlite3"
    session, run, user, answer, client = (str(uuid4()) for _ in range(5))
    now = "2026-11-22T08:30:00.000Z"
    with closing(sqlite3.connect(path)) as db, db:
        for statement in SCHEMA_V1:
            db.execute(statement)
        db.execute("PRAGMA user_version=1")
        db.execute("INSERT INTO sessions VALUES (?, 'old', NULL, ?, ?)", (session, now, now))
        db.execute(
            "INSERT INTO runs VALUES (?, ?, ?, 'completed', NULL, NULL, ?, ?, NULL, NULL)",
            (run, session, client, now, now),
        )
        for message, role in ((user, "user"), (answer, "assistant")):
            db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, 'saved', ?)", (message, session, run, role, now))
    return path, session, run, user, answer, client


def test_v1_migration_preserves_rows_and_pagination_with_equal_timestamps(legacy_database, monkeypatch):
    path, session, run, user, answer, client = legacy_database
    store = BusinessStore(path)
    assert [(m.message_id, m.content) for m in store.list_messages(session)] == [(user, "saved"), (answer, "saved")]
    assert store.lookup_run(session, client).run_id == run
    assert store.list_runs(session, 1, None)[0].user_message_id == user

    monkeypatch.setattr("app.storage.business.utc_now", lambda: "2026-11-22T08:30:00.000Z")
    for _ in range(3):
        added = store.start_run(session, str(uuid4()), "same timestamp", accept_new_run=True)
        store.complete_run(added, "done", str(uuid4()))
    summary = store.list_runs(session, 4, None)
    assert len({r.run_id for r in summary}) == 4
    assert [r.run_id for r in summary] == sorted((r.run_id for r in summary), reverse=True)
    assert store.list_runs(session, 3, (summary[0].started_at, summary[0].run_id)) == summary[1:]
    with closing(store._connect()) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_failure_leaves_v1_intact(legacy_database):
    path, *_ = legacy_database
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE INDEX runs_by_session_start ON runs (finished_at)")
    with pytest.raises(sqlite3.OperationalError):
        BusinessStore(path)
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (1,)
        assert db.execute("SELECT count(*) FROM messages").fetchone() == (2,)
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'sessions_by_activity'").fetchone() is None


def test_v2_upgrade_preserves_every_row_and_enforces_future_checkpointed_completion(legacy_database):
    path, session, *_ = legacy_database
    with closing(sqlite3.connect(path)) as db, db:
        for statement in MIGRATIONS[1]:
            db.execute(statement)
        db.execute("PRAGMA user_version=2")
        before = {table: db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                  for table in ("sessions", "runs", "messages")}

    store = BusinessStore(path)
    with closing(store._connect()) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        for table, old_rows in before.items():
            assert db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall() == old_rows
        with pytest.raises(sqlite3.IntegrityError, match="requires a final checkpoint"):
            db.execute("""INSERT INTO runs (run_id, session_id, client_request_id, status, started_at, finished_at)
                          VALUES (?, ?, ?, 'completed', 'now', 'now')""", (str(uuid4()), session, str(uuid4())))

        run = store.start_run(session, str(uuid4()), "new question", accept_new_run=True)
        assert run.base_checkpoint_id is None
        with pytest.raises(sqlite3.IntegrityError, match="requires a final checkpoint"):
            db.execute("UPDATE runs SET status = 'completed', finished_at = 'now' WHERE run_id = ?", (run.run_id,))
        checkpoint_id = str(uuid4())
        assert store.complete_run(run, "new answer", checkpoint_id)
        assert db.execute("SELECT last_safe_checkpoint_id FROM sessions").fetchone() == (checkpoint_id,)


def test_concurrent_history_reads_and_writes(tmp_path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    sessions = [store.create_session(f"writer {i}").session_id for i in range(6)]

    def write(session):
        for _ in range(30):
            run = store.start_run(session, str(uuid4()), "question", accept_new_run=True)
            assert store.complete_run(run, "answer", str(uuid4()))

    def read():
        for _ in range(40):
            for session in store.list_sessions():
                messages = store.list_messages(session.session_id)
                by_id = {message.message_id for message in messages}
                assert len(by_id) == len(messages)
                assert all(message.session_id == session.session_id for message in messages)
                page = store.list_runs(session.session_id, 5, None)
                assert all(run.session_id == session.session_id for run in page)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(write, session) for session in sessions] + [pool.submit(read) for _ in range(4)]
        for future in futures:
            future.result(timeout=15)
    assert sum(len(store.list_messages(session)) for session in sessions) == 360
    assert sum(len(store.list_runs(session, 100, None)) for session in sessions) == 180
    assert all(session.active_run_id is None for session in store.list_sessions())
