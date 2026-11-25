import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from pathlib import Path
from queue import Queue
from secrets import token_urlsafe
from threading import Thread
from uuid import UUID, uuid4

import httpx
import pytest

from app.main import DEV_ORIGIN
from app.storage.business import DATABASE_NAME, BusinessStore

BACKEND = Path(__file__).resolve().parents[1]
COMMAND = [sys.executable, "-E", "-s", "-B", "-m", "app.sidecar"]
STATUS = "/api/v1/status"
READY = "/api/v1/_host/ready"
SHUTDOWN = "/api/v1/_host/shutdown"


def bootstrap(data_dir):
    return {
        "type": "bootstrap",
        "protocol_version": 1,
        "parent_pid": os.getpid(),
        "app_data_dir": str(data_dir),
        "ui_token": token_urlsafe(32),
        "host_token": token_urlsafe(32),
    }


def wire_frame(payload):
    return json.dumps(payload).encode() + b"\n"


@contextmanager
def running_sidecar(data_dir, command=COMMAND, **bootstrap_options):
    payload = {**bootstrap(data_dir), **bootstrap_options}
    with subprocess.Popen(
        command, cwd=BACKEND, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as child:
        try:
            child.stdin.write(wire_frame(payload))
            child.stdin.flush()
            lines = Queue()
            Thread(target=lambda: lines.put(child.stdout.readline()), daemon=True).start()
            line = lines.get(timeout=10)
            assert line.endswith(b"\n")
            ready = json.loads(line)
            assert set(ready) == {"type", "protocol_version", "instance_id", "port"}
            assert ready["type"] == "ready"
            assert ready["protocol_version"] == 1
            assert UUID(ready["instance_id"]).version == 4
            assert 0 < ready["port"] <= 65535
            yield child, ready, payload
        finally:
            if child.poll() is None:
                child.kill()


def headers(token):
    return {"Authorization": f"Bearer {token}"}


def assert_clean_exit(child, payload):
    assert child.wait(timeout=6) == 0
    assert child.stdout.read() == b""
    stderr = child.stderr.read()
    assert b"Application shutdown complete" in stderr
    for token in (payload["ui_token"], payload["host_token"]):
        assert token.encode() not in stderr
    return stderr


def test_managed_sidecar_serves_bound_socket_and_isolates_tokens(tmp_path):
    with running_sidecar(tmp_path / "primary") as (child, ready, payload):
        with running_sidecar(tmp_path / "secondary") as (other, other_ready, other_payload):
            assert ready["port"] != other_ready["port"]
            assert ready["instance_id"] != other_ready["instance_id"]
            with httpx.Client(base_url=f"http://127.0.0.1:{other_ready['port']}", trust_env=False) as client:
                assert client.get(READY, headers=headers(other_payload["host_token"])).status_code == 200
                assert client.post(SHUTDOWN, headers=headers(other_payload["host_token"])).status_code == 204
            assert_clean_exit(other, other_payload)

        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False) as client:
            ui = headers(payload["ui_token"])
            host = headers(payload["host_token"])
            assert client.get(READY, headers=host).json() == {
                "status": "ready", "instance_id": ready["instance_id"],
            }
            assert client.get(STATUS).status_code == 401
            assert client.get(STATUS, headers=headers("invalid")).status_code == 401
            assert client.get(STATUS, headers=host).status_code == 403
            assert client.get(READY, headers=ui).status_code == 403
            assert client.get(READY).status_code == 401
            assert client.get(READY, headers=headers("invalid")).status_code == 401
            assert client.post(SHUTDOWN, headers=ui).status_code == 403
            assert client.post(SHUTDOWN).status_code == 401
            assert client.post(SHUTDOWN, headers=headers("invalid")).status_code == 401
            assert client.get(STATUS, headers={**ui, "Host": "127.0.0.1:0"}).status_code == 400
            assert client.get(STATUS, headers={**ui, "Origin": "http://localhost:1420"}).status_code == 403
            status = client.get(STATUS, headers={**ui, "Origin": DEV_ORIGIN})
            assert status.status_code == 200
            assert status.headers["access-control-allow-origin"] == DEV_ORIGIN
            assert status.json() == {
                "status": "ready", "database": "ready", "model_configured": False, "siliconflow_configured": False,
            }
            assert client.post(SHUTDOWN, headers=host).status_code == 204
            # Keep the pipe open to prove that the authenticated endpoint itself stops the server.
            assert b"Host stdin closed" not in assert_clean_exit(child, payload)
            with pytest.raises(httpx.ConnectError):
                client.get(STATUS, headers=ui)


def test_stdin_eof_stops_the_server_while_the_parent_remains_alive(tmp_path):
    with running_sidecar(tmp_path) as (child, ready, payload):
        child.stdin.close()
        assert b"Host stdin closed" in assert_clean_exit(child, payload)
        with httpx.Client(trust_env=False) as client, pytest.raises(httpx.ConnectError):
            client.get(f"http://127.0.0.1:{ready['port']}{STATUS}", headers=headers(payload["ui_token"]))


def test_diagnostics_use_request_ids_and_exclude_tokens_and_input(tmp_path):
    canary = token_urlsafe(32)
    with running_sidecar(tmp_path) as (child, ready, payload):
        ui, host = (headers(payload[key]) for key in ("ui_token", "host_token"))
        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False) as client:
            credential = client.put("/api/v1/_host/model-credentials/deepseek", headers=host, json={"api_key": canary})
            assert credential.status_code == 200
            malformed = client.get(f"/api/v1/sessions/{canary}/runs", headers=ui)
            assert malformed.status_code == 422
            assert malformed.headers["x-request-id"] == malformed.json()["error"]["request_id"]
            unauthorized = client.get("/api/v1/sessions", headers=headers(canary))
            assert unauthorized.status_code == 401
            assert client.post(SHUTDOWN, headers=host).status_code == 204
        stderr = assert_clean_exit(child, payload)
    logs = (tmp_path / "logs" / "sidecar.jsonl").read_text()
    entries = {event.get("request_id"): event for event in map(json.loads, logs.splitlines())}
    assert entries[credential.headers["x-request-id"]]["status"] == 200
    assert entries[malformed.headers["x-request-id"]]["error_code"] == "VALIDATION_ERROR"
    assert entries[unauthorized.headers["x-request-id"]]["status"] == 401
    for secret in (payload["ui_token"], payload["host_token"], canary):
        assert secret not in logs
        assert secret.encode() not in stderr


def test_unclean_restart_reconciles_active_runs_before_ready(tmp_path):
    fixture = """
import asyncio
import sys
import httpx
from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.services.models import ModelService
from app.sidecar import main

sys.path.insert(0, "tests")
from test_chat import ProviderStream

def fixture_app(*args, **kwargs):
    gate = asyncio.Event()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=ProviderStream(gate)))
    models = ModelService(DeepSeekAdapter(DeepSeekSettings(), transport=transport))
    return create_app(*args, **kwargs, models=models)

raise SystemExit(main(app_factory=fixture_app))
"""
    store = BusinessStore(tmp_path / DATABASE_NAME)
    snapshots = {}
    fixtures = {}
    for state in ("completed", "failed", "cancelled", "cancelling", "awaiting_approval"):
        session = store.create_session(state)
        fixtures[state] = session.session_id
    with running_sidecar(tmp_path, [*COMMAND[:4], "-c", fixture]) as (child, ready, payload):
        for state, session_id in fixtures.items():
            run = store.start_run(session_id, str(uuid4()), state, accept_new_run=True)
            if state == "completed":
                assert store.complete_run(run, "已保存", str(uuid4()))
            elif state == "failed":
                assert store.fail_run(run, "MODEL_TIMEOUT", "timeout")
            elif state in {"cancelling", "cancelled"}:
                assert store.request_cancel(run).status == "cancelling"
                if state == "cancelled":
                    assert store.confirm_cancel(run)
            else:
                with closing(store._connect()) as connection:
                    connection.execute("UPDATE runs SET status = 'awaiting_approval' WHERE run_id = ?", (run.run_id,))
            snapshots[state] = store.get_run(run.run_id)

        ui, host = headers(payload["ui_token"]), headers(payload["host_token"])
        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False, timeout=5) as client:
            credential = client.put("/api/v1/_host/model-credentials/deepseek", headers=host,
                                    json={"api_key": token_urlsafe(32)})
            assert credential.json() == {"configured": True}
            session = client.post("/api/v1/sessions", json={}, headers=ui).json()["session_id"]
            request_id = str(uuid4())
            with client.stream("POST", f"/api/v1/sessions/{session}/runs/stream", headers=ui,
                               json={"client_request_id": request_id, "message": "将被强退"}) as stream:
                events = (json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: "))
                started, draft = next(events), next(events)
                assert started["type"] == "run_started" and draft["type"] == "message_delta"
                run_id = started["run_id"]
                assert (awaiting := store.get_run(run_id)).status == "running"
                with closing(sqlite3.connect(tmp_path / "checkpoints.sqlite3")) as graph:
                    assert graph.execute(
                        "SELECT count(*) FROM checkpoints WHERE thread_id = ?", (f"session:{session}",),
                    ).fetchone()[0] > 0
                # A second Sidecar must not classify the live owner's tasks as interrupted.
                contender = subprocess.run(COMMAND, cwd=BACKEND, input=wire_frame(bootstrap(tmp_path)),
                                           capture_output=True, timeout=6)
                assert contender.returncode == 3 and not contender.stdout
                assert store.get_run(run_id) == awaiting
                child.kill()
                assert child.wait(timeout=5) < 0
        assert store.get_run(run_id).status == "running"

    snapshots["running"] = store.get_run(run_id)
    recovered = {}
    for _ in range(2):
        with running_sidecar(tmp_path) as (child, ready, payload):
            ui, host = headers(payload["ui_token"]), headers(payload["host_token"])
            with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False) as client:
                assert client.get(STATUS, headers=ui).json()["model_configured"] is False
                read = client.get("/api/v1/runs/lookup", headers=ui, params={
                    "session_id": session, "client_request_id": request_id,
                })
                assert read.status_code == 200 and read.json()["run_id"] == run_id
                for old_state, saved in snapshots.items():
                    current = client.get(f"/api/v1/runs/{saved.run_id}", headers=ui)
                    assert current.status_code == 200
                    status = store.get_run(saved.run_id)
                    assert current.json()["status"] == status.status
                    if old_state in {"running", "cancelling", "awaiting_approval"}:
                        assert status.status == "failed" and status.error_code == "RUN_INTERRUPTED"
                        assert status.started_at == saved.started_at and status.finished_at
                        assert recovered.setdefault(old_state, status) == status
                    else:
                        assert status == saved
                dup = client.post(f"/api/v1/sessions/{session}/runs/stream", headers=ui,
                                  json={"client_request_id": request_id, "message": "重复"})
                assert dup.status_code == 409 and dup.json()["error"]["run_id"] == run_id
                assert client.post(SHUTDOWN, headers=host).status_code == 204
            assert_clean_exit(child, payload)
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone() == (6,)
        assert connection.execute(
            """SELECT r.final_checkpoint_id, s.last_safe_checkpoint_id
               FROM runs AS r JOIN sessions AS s USING (session_id) WHERE r.run_id = ?""", (run_id,),
        ).fetchone() == (None, None)
        assert connection.execute("SELECT run_id, content FROM messages WHERE role = 'assistant'").fetchall() == [
            (snapshots["completed"].run_id, "已保存"),
        ]


@pytest.mark.parametrize("invalid", ["protocol", "type", "parent", "token", "relative", "file", "oversized", "truncated"])
def test_rejects_invalid_bootstrap_without_reflecting_secrets(tmp_path, invalid):
    payload = bootstrap(tmp_path)
    if invalid == "protocol":
        payload["protocol_version"] = 2
    elif invalid == "type":
        payload["type"] = "ready"
    elif invalid == "parent":
        payload["parent_pid"] += 1
    elif invalid == "token":
        payload["host_token"] = payload["ui_token"]
    elif invalid == "relative":
        payload["app_data_dir"] = "relative"
    elif invalid == "file":
        file = tmp_path / "file"
        file.touch()
        payload["app_data_dir"] = str(file)
    elif invalid == "oversized":
        payload["app_data_dir"] += "a" * 16_384

    frame = wire_frame(payload)
    if invalid == "truncated":
        frame = frame[:-1]
    result = subprocess.run(COMMAND, cwd=BACKEND, input=frame, capture_output=True, timeout=10)
    assert result.returncode == 1
    assert not result.stdout
    assert result.stderr == b"KnowledgeAgent Sidecar: bootstrap rejected.\n"


def test_live_owner_lock_prevents_concurrent_migration_and_logs_database_locked(tmp_path):
    newer_version = """
import app.storage.business as business
from app.sidecar import main

business.MIGRATIONS = (*business.MIGRATIONS, ("CREATE TABLE v5_canary (id TEXT PRIMARY KEY)",))
business.SCHEMA_VERSION = len(business.MIGRATIONS)
raise SystemExit(main())
"""
    with running_sidecar(tmp_path) as (child, ready, payload):
        ui, host = headers(payload["ui_token"]), headers(payload["host_token"])
        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False) as client:
            session_id = client.post("/api/v1/sessions", json={"title": "升级并发保护"}, headers=ui).json()["session_id"]
            contender = subprocess.run(
                [*COMMAND[:4], "-c", newer_version],
                cwd=BACKEND,
                input=wire_frame(bootstrap(tmp_path)),
                capture_output=True,
                timeout=6,
            )
            assert contender.returncode == 3 and not contender.stdout
            assert [row["session_id"] for row in client.get("/api/v1/sessions", headers=ui).json()] == [session_id]
            assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, payload)

    with closing(sqlite3.connect(tmp_path / DATABASE_NAME)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (5,)
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'v5_canary'").fetchone() is None
        assert db.execute("SELECT session_id, title FROM sessions").fetchall() == [(session_id, "升级并发保护")]

    logs = [json.loads(line) for line in (tmp_path / "logs" / "sidecar.jsonl").read_text().splitlines()]
    assert any(entry["event"] == "startup_error" and entry.get("error_code") == "DATABASE_LOCKED" for entry in logs)


def test_failed_or_incompatible_migration_preserves_existing_data(tmp_path):
    store = BusinessStore(tmp_path / DATABASE_NAME)
    session = store.create_session("原始会话")
    run = store.start_run(session.session_id, str(uuid4()), "升级前问题", accept_new_run=True)
    checkpoint_id = str(uuid4())
    message_id = store.complete_run(run, "升级前回答", checkpoint_id)
    assert message_id is not None

    with closing(sqlite3.connect(tmp_path / DATABASE_NAME)) as db:
        before_sessions = db.execute("SELECT * FROM sessions").fetchall()
        before_runs = db.execute("SELECT * FROM runs").fetchall()
        before_messages = db.execute("SELECT * FROM messages ORDER BY message_id").fetchall()

    failing_upgrade = """
import app.storage.business as business
from app.sidecar import main

business.MIGRATIONS = (
    *business.MIGRATIONS,
    ("CREATE TABLE transient_upgrade_table (id TEXT PRIMARY KEY)", "CREATE INDEX sessions_by_activity ON sessions (title)"),
)
business.SCHEMA_VERSION = len(business.MIGRATIONS)
raise SystemExit(main())
"""
    failed = subprocess.run(
        [*COMMAND[:4], "-c", failing_upgrade],
        cwd=BACKEND,
        input=wire_frame(bootstrap(tmp_path)),
        capture_output=True,
        timeout=6,
    )
    assert failed.returncode == 3 and not failed.stdout

    with closing(sqlite3.connect(tmp_path / DATABASE_NAME)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (5,)
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'transient_upgrade_table'").fetchone() is None
        assert db.execute("SELECT * FROM sessions").fetchall() == before_sessions
        assert db.execute("SELECT * FROM runs").fetchall() == before_runs
        assert db.execute("SELECT * FROM messages ORDER BY message_id").fetchall() == before_messages

        # Also test launching an older version against a newer schema version (user_version = 99).
        db.execute("PRAGMA user_version=99")

    downgrade = subprocess.run(COMMAND, cwd=BACKEND, input=wire_frame(bootstrap(tmp_path)), capture_output=True, timeout=6)
    assert downgrade.returncode == 3 and not downgrade.stdout
    with closing(sqlite3.connect(tmp_path / DATABASE_NAME)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (99,)
        assert db.execute("SELECT * FROM sessions").fetchall() == before_sessions
        assert db.execute("SELECT * FROM runs").fetchall() == before_runs
        assert db.execute("SELECT * FROM messages ORDER BY message_id").fetchall() == before_messages

    logs = [json.loads(line) for line in (tmp_path / "logs" / "sidecar.jsonl").read_text().splitlines()]
    migration_errors = [
        entry for entry in logs if entry["event"] == "startup_error" and entry.get("error_code") == "DATABASE_MIGRATION_FAILED"
    ]
    assert len(migration_errors) == 2
