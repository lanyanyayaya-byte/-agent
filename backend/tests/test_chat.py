import asyncio
import json
import socket
import sqlite3
from collections import Counter
from contextlib import asynccontextmanager, closing
from secrets import token_urlsafe
from threading import Event
from uuid import uuid4

import httpx
import pytest
import uvicorn

from app.api import chat
from app.main import DEV_ORIGIN, SidecarControl, create_app
from app.models.deepseek import DEEPSEEK_BASE_URL, DeepSeekAdapter, DeepSeekSettings
from app.services.models import ModelService
from app.storage.business import BUSY_TIMEOUT_MS, BusinessStore, RunConflict

SESSIONS = "/api/v1/sessions"
RUNS = "/api/v1/runs"
CREDENTIAL = "/api/v1/_host/model-credentials/deepseek"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def run_path(session_id):
    return f"{SESSIONS}/{session_id}/runs/stream"


def payload(message="测试问题", request_id=None):
    return {"client_request_id": request_id or str(uuid4()), "message": message}


def chunk(delta, finish=None):
    event = {
        "id": "fixture", "model": "fixture-model", "created": 1, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "finish_reason": finish, "delta": delta}],
    }
    return f"data: {json.dumps(event)}\n\n".encode()


class ProviderStream(httpx.AsyncByteStream):
    def __init__(self, gate=None, mode="stop"):
        self.gate = gate
        self.mode = mode

    async def __aiter__(self):
        yield chunk({"role": "assistant", "reasoning_content": "hidden"})
        yield chunk({"content": "第一段"})
        if self.gate is not None:
            await self.gate.wait()
        if self.mode == "timeout":
            raise httpx.ReadTimeout("private upstream details")
        if self.mode == "connection":
            raise httpx.ReadError("private upstream details")
        yield chunk({"content": "第二段"})
        yield chunk({}, self.mode) + b"data: [DONE]\n\n"


@asynccontextmanager
async def live_api(path, provider=None, *, transport=None, timeout_seconds=10):
    if transport is None:
        transport = httpx.MockTransport(provider)
    models = ModelService(DeepSeekAdapter(
        DeepSeekSettings(model="fixture-model", timeout_seconds=timeout_seconds), transport=transport,
    ))
    ui, host = token_urlsafe(32), token_urlsafe(32)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        control = SidecarControl(uuid4(), port, asyncio.Event())
        app = create_app(ui, host, database_path=path, control=control, models=models)
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on", access_log=False))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {ui}"},
                trust_env=False, timeout=3,
            ) as client:
                yield client, {"Authorization": f"Bearer {host}"}
        finally:
            server.should_exit = True
            await task


async def configure(client, host):
    response = await client.put(CREDENTIAL, headers=host, json={"api_key": token_urlsafe(32)})
    assert response.json() == {"configured": True}


async def new_session(client):
    response = await client.post(SESSIONS, json={})
    assert response.status_code == 201
    assert response.json()["title"] == "新对话"
    return response.json()["session_id"]


async def frames(response):
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            yield json.loads(line[6:])
        elif line.startswith(":"):
            yield line
        else:
            assert line == ""


async def next_event(events):
    async for item in events:
        if isinstance(item, dict):
            return item
    raise AssertionError("Expected an SSE business event")


def rows(path, sql, args=()):
    with closing(sqlite3.connect(path)) as db:
        return db.execute(sql, args).fetchall()


def test_real_sqlite_schema_and_constraints(tmp_path):
    path = tmp_path / "business.sqlite3"
    store = BusinessStore(path)
    with closing(store._connect()) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("PRAGMA busy_timeout").fetchone() == (BUSY_TIMEOUT_MS,)
        assert db.execute("PRAGMA user_version").fetchone() == (7,)
        assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert rows(path, "SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name") == [
        ("artifacts",), ("mcp_servers",), ("mcp_tools",), ("messages",),
        ("run_citations",), ("run_skill_activations",), ("runs",), ("sessions",),
        ("skills",), ("tool_calls",), ("workspace_files",),
    ]

    session = store.create_session("实测")
    request_id = str(uuid4())
    run = store.start_run(session.session_id, request_id, "问题", accept_new_run=True)
    insert = ("INSERT INTO runs (run_id, session_id, client_request_id, status, started_at, finished_at, "
              "final_checkpoint_id) VALUES (?, ?, ?, ?, 'now', ?, ?)")
    with closing(store._connect()) as db:
        with pytest.raises(sqlite3.IntegrityError, match="runs.session_id, runs.client_request_id"):
            db.execute(insert, (str(uuid4()), session.session_id, request_id, "completed", "now", str(uuid4())))
        with pytest.raises(sqlite3.IntegrityError, match="runs.session_id"):
            db.execute(insert, (str(uuid4()), session.session_id, str(uuid4()), "running", None, None))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            db.execute(
                "INSERT INTO messages VALUES (?, ?, ?, 'assistant', 'wrong session', 'now')",
                (str(uuid4()), str(uuid4()), run.run_id),
            )
    message_id = store.complete_run(run, "回答", str(uuid4()))
    assert store.request_cancel(run).status == "completed"
    assert not store.confirm_cancel(run)
    assert rows(path, "SELECT role, content FROM messages WHERE message_id = ?", (message_id,)) == [
        ("assistant", "回答"),
    ]
    BusinessStore(path)
    assert rows(path, "SELECT status FROM runs") == [("completed",)]


async def test_preflight_auth_validation_and_missing_model_are_http_errors(tmp_path):
    path = tmp_path / "business.sqlite3"

    def provider(_request):
        pytest.fail("preflight and rejected runs must never invoke the model")

    async with live_api(path, provider) as (client, host):
        session = await new_session(client)
        route = run_path(session)
        preflight = await client.options(route, headers={
            "Origin": DEV_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == DEV_ORIGIN

        denied = [
            (route, payload(), {"Authorization": ""}, 401, "AUTH_REQUIRED"),
            (route, payload(), host, 403, "FORBIDDEN"),
            (route, payload(), {"Origin": "http://localhost:1420"}, 403, "FORBIDDEN"),
            (run_path("invalid"), payload(), {}, 422, "VALIDATION_ERROR"),
            (run_path(str(uuid4())), payload(), {}, 404, "NOT_FOUND"),
            (route, payload("   "), {}, 422, "VALIDATION_ERROR"),
            (route, payload("字" * 20_001), {}, 422, "VALIDATION_ERROR"),
            (route, payload(request_id="invalid"), {}, 422, "VALIDATION_ERROR"),
            (route, payload(), {}, 503, "MODEL_NOT_CONFIGURED"),
        ]
        for target, body, headers, status, code in denied:
            response = await client.post(target, json=body, headers=headers)
            assert response.status_code == status
            assert response.headers["content-type"].startswith("application/json")
            error = response.json()["error"]
            assert error["code"] == code and error["run_id"] is None
        assert rows(path, "SELECT * FROM runs") == []
        assert rows(path, "SELECT * FROM messages") == []


async def test_live_sse_publishes_deltas_before_commit_and_terminal_after_commit(tmp_path, monkeypatch):
    path = tmp_path / "business.sqlite3"
    release = asyncio.Event()
    calls = []
    monkeypatch.setattr(chat, "KEEPALIVE_SECONDS", 0.05)

    def provider(request):
        assert request.url == f"{DEEPSEEK_BASE_URL}/chat/completions"
        calls.append(json.loads(request.content))
        return httpx.Response(200, stream=ProviderStream(release))

    async with live_api(path, provider) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        body = payload()
        try:
            async with client.stream("POST", run_path(session), json=body, headers={"Origin": DEV_ORIGIN}) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                events = frames(response)
                started = await next_event(events)
                run = started["run_id"]
                assert started["data"] == {"client_request_id": body["client_request_id"]}
                assert started["type"] == "run_started"
                first = await next_event(events)
                assert first["data"] == {"channel": "text", "delta": "第一段"}
                assert await anext(events) == ": keepalive"
                assert rows(path, "SELECT status FROM runs WHERE run_id = ?", (run,)) == [("running",)]
                assert rows(path, "SELECT role FROM messages WHERE run_id = ?", (run,)) == [("user",)]
                # Creating another session while the provider is blocked proves no write lock is held.
                assert await new_session(client) != session
                release.set()
                second, terminal = await next_event(events), await next_event(events)
                assert second["type"] == "message_delta"
                assert terminal["type"] == "run_completed"
                message = terminal["data"]["message_id"]
                assert rows(path, "SELECT session_id, run_id, role, content FROM messages WHERE message_id = ?", (message,)) == [
                    (session, run, "assistant", "第一段第二段"),
                ]
                assert rows(path, "SELECT status, error_code FROM runs WHERE run_id = ?", (run,)) == [("completed", None)]
                assert [item async for item in events if isinstance(item, dict)] == []
                for event in (started, first, second, terminal):
                    assert set(event) == {"version", "type", "session_id", "run_id", "ts", "data"}
                    assert event["version"] == 1 and event["ts"].endswith("Z")
                    assert event["session_id"] == session and event["run_id"] == run
                    assert "hidden" not in json.dumps(event)
        finally:
            release.set()
        assert len(calls) == 1 and calls[0]["stream"] is True


@pytest.mark.parametrize(("mode", "code"), [
    ("timeout", "MODEL_TIMEOUT"),
    ("connection", "MODEL_UPSTREAM_ERROR"),
    ("length", "MODEL_UPSTREAM_ERROR"),
    ("db-rollback", "INTERNAL_ERROR"),
])
async def test_failed_stream_keeps_only_user_message(tmp_path, mode, code):
    path = tmp_path / "business.sqlite3"

    def provider(_request):
        return httpx.Response(200, stream=ProviderStream(mode="stop" if mode == "db-rollback" else mode))

    async with live_api(path, provider) as (client, host):
        await configure(client, host)
        if mode == "db-rollback":
            with closing(sqlite3.connect(path)) as db:
                db.execute("""CREATE TRIGGER reject_assistant BEFORE INSERT ON messages WHEN NEW.role = 'assistant'
                              BEGIN SELECT RAISE(ABORT, 'test write failure'); END""")
        session = await new_session(client)
        response = await client.post(run_path(session), json=payload())
        assert response.status_code == 200
        events = [event async for event in frames(response)]
        assert events[0]["type"] == "run_started"
        assert any(event["type"] == "message_delta" for event in events)
        assert [event["type"] for event in events if event["type"] != "message_delta"] == ["run_started", "error"]
        assert events[-1]["data"]["code"] == code
        assert "private upstream details" not in response.text
        assert rows(path, "SELECT role FROM messages") == [("user",)]
        assert rows(path, "SELECT status, error_code FROM runs") == [("failed", code)]
        run_id = events[0]["run_id"]
        state = (await client.get(f"{RUNS}/{run_id}")).json()
        assert state["status"] == "failed" and state["error_code"] == code and state["finished_at"]
        cancel = await client.post(f"{RUNS}/{run_id}/cancel")
        assert cancel.status_code == 200
        assert cancel.json() == {"status": "failed", "run_id": run_id, "already_finished": True}
        assert (await client.get(f"{RUNS}/{run_id}")).json() == state


async def test_model_deadline_ends_an_open_stream_without_user_cancellation(tmp_path):
    path = tmp_path / "business.sqlite3"
    release = asyncio.Event()
    provider_stopped = asyncio.Event()

    class StalledProvider(ProviderStream):
        async def __aiter__(self):
            try:
                async for item in super().__aiter__():
                    yield item
            finally:
                provider_stopped.set()

    def provider(_request):
        return httpx.Response(200, stream=StalledProvider(release))

    async with live_api(path, provider, timeout_seconds=1.5) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        response = await client.post(run_path(session), json=payload())
        events = [item async for item in frames(response) if isinstance(item, dict)]
        assert [item["type"] for item in events] == ["run_started", "message_delta", "error"]
        assert provider_stopped.is_set() and events[-1]["data"]["code"] == "MODEL_TIMEOUT"
        state = (await client.get(f"{RUNS}/{events[0]['run_id']}")).json()
        assert state["status"] == "failed" and state["error_code"] == "MODEL_TIMEOUT"
        assert rows(path, "SELECT role FROM messages") == [("user",)]


async def test_real_concurrent_http_conflicts_obey_priority_and_execute_once(tmp_path):
    path = tmp_path / "business.sqlite3"
    release = asyncio.Event()
    calls = Counter()

    def provider(request):
        question = json.loads(request.content)["messages"][-1]["content"]
        calls[question] += 1
        gate = release if question == "active" else None
        return httpx.Response(200, stream=ProviderStream(gate))

    async with live_api(path, provider) as (client, host):
        await configure(client, host)
        session, other_session = await new_session(client), await new_session(client)
        route = run_path(session)
        old, active = payload("old"), payload("active")
        old_response = await client.post(route, json=old)
        old_events = [event async for event in frames(old_response)]
        old_run = old_events[0]["run_id"]
        assert old_events[-1]["type"] == "run_completed"
        try:
            async with client.stream("POST", route, json=active) as response:
                events = frames(response)
                run = (await anext(events))["run_id"]
                assert (await anext(events))["type"] == "message_delta"
                bodies = (
                    payload("different", old["client_request_id"]),
                    payload("different", active["client_request_id"]),
                    payload("new id, same session"),
                )
                barrier = asyncio.Barrier(4)

                async def submit(path, body):
                    await barrier.wait()
                    return await client.post(path, json=body)

                *conflicts, independent = await asyncio.gather(
                    *(submit(route, body) for body in bodies),
                    submit(run_path(other_session), payload("other", active["client_request_id"])),
                )
                assert independent.status_code == 200
                assert [item async for item in frames(independent)][-1]["type"] == "run_completed"
                for rejected, (code, existing_run) in zip(conflicts, [
                    ("DUPLICATE_CLIENT_REQUEST", old_run),
                    ("DUPLICATE_CLIENT_REQUEST", run),
                    ("RUN_ALREADY_ACTIVE", run),
                ], strict=True):
                    assert rejected.status_code == 409
                    assert rejected.headers["content-type"].startswith("application/json")
                    error = rejected.json()["error"]
                    assert error["code"] == code and error["run_id"] == existing_run and error["details"] is None
                assert rows(path, "SELECT count(*) FROM runs WHERE session_id = ?", (session,)) == [(2,)]
                release.set()
                assert [event async for event in events][-1]["type"] == "run_completed"
        finally:
            release.set()
        assert calls == {"old": 1, "active": 1, "other": 1}


async def test_simultaneous_sqlite_admission_with_no_existing_run(tmp_path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    session = store.create_session("并发")
    shared, other = str(uuid4()), str(uuid4())
    barrier = asyncio.Barrier(3)

    async def submit(request_id):
        await barrier.wait()
        try:
            return await asyncio.to_thread(store.start_run, session.session_id, request_id, "问题", accept_new_run=True)
        except RunConflict as error:
            return error

    results = await asyncio.gather(submit(shared), submit(shared), submit(other))
    winner, = [result for result in results if not isinstance(result, RunConflict)]
    for request_id, result in zip((shared, shared, other), results, strict=True):
        if result is not winner:
            assert result.run_id == winner.run_id
            assert result.code == ("DUPLICATE_CLIENT_REQUEST" if request_id == winner.client_request_id else "RUN_ALREADY_ACTIVE")
    assert rows(store.path, "SELECT count(*) FROM runs") == [(1,)]
    assert rows(store.path, "SELECT count(*) FROM messages") == [(1,)]


@pytest.mark.parametrize("shutdown", [False, True])
async def test_stream_disconnect_does_not_cancel_but_host_shutdown_does(tmp_path, shutdown):
    path = tmp_path / "business.sqlite3"
    release = asyncio.Event()
    calls = 0

    def provider(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=ProviderStream(release))

    async with live_api(path, provider) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        body = payload()
        route = run_path(session)
        try:
            async with client.stream("POST", route, json=body) as response:
                events = frames(response)
                run = (await anext(events))["run_id"]
                assert (await anext(events))["type"] == "message_delta"
                if shutdown:
                    assert (await client.post("/api/v1/_host/shutdown", headers=host)).status_code == 204
                    terminal, = [event async for event in events]
                    assert terminal["type"] == "run_cancelled"

            if not shutdown:
                assert rows(path, "SELECT status FROM runs") == [("running",)]
                # Reconcile using only the session and preassigned request ID.
                lookup = await client.get(f"{RUNS}/lookup", params={
                    "session_id": session, "client_request_id": body["client_request_id"],
                })
                original = lookup.json()
                assert lookup.status_code == 200
                assert original == {
                    "run_id": run, "session_id": session, "status": "running",
                    "started_at": original["started_at"], "finished_at": None,
                    "error_code": None, "pending_approval": None,
                }
                assert original["started_at"].endswith("Z")
                assert (await client.get(f"{RUNS}/{run}")).json() == original
                wrong = await client.get(f"{RUNS}/lookup", params={
                    "session_id": await new_session(client), "client_request_id": body["client_request_id"],
                })
                assert wrong.status_code == 404 and wrong.json()["error"]["code"] == "NOT_FOUND"
                repeated = await client.post(route, json=body)
                assert repeated.status_code == 409 and repeated.json()["error"]["run_id"] == run
                release.set()
                async with asyncio.timeout(2):
                    while rows(path, "SELECT status FROM runs") == [("running",)]:
                        await asyncio.sleep(0.01)

            expected = "cancelled" if shutdown else "completed"
            assert rows(path, "SELECT status FROM runs") == [(expected,)]
            assert rows(path, "SELECT count(*) FROM messages WHERE role = 'assistant'") == [(0 if shutdown else 1,)]
            assert calls == 1
        finally:
            release.set()


class LocalProviderTransport(httpx.AsyncHTTPTransport):
    """Route the official adapter's test traffic to a real loopback socket."""

    def __init__(self, port):
        super().__init__(trust_env=False)
        self.port = port

    async def handle_async_request(self, request):
        assert request.url.host == "api.deepseek.com"
        request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=self.port)
        return await super().handle_async_request(request)


async def test_cancel_closes_the_upstream_socket_and_keeps_only_the_user_message(tmp_path):
    path = tmp_path / "business.sqlite3"
    upstream_closed = asyncio.Event()
    calls = []

    async def upstream(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            size = next(int(line.split(b":", 1)[1]) for line in headers.lower().splitlines()
                        if line.startswith(b"content-length:"))
            calls.append(json.loads(await reader.readexactly(size)))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n")
            item = chunk({"role": "assistant", "content": "半截回复"})
            writer.write(f"{len(item):x}\r\n".encode() + item + b"\r\n")
            await writer.drain()
            assert await reader.read() == b""  # The client must end the live connection.
            upstream_closed.set()
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(upstream, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with live_api(path, transport=LocalProviderTransport(port)) as (client, host):
            await configure(client, host)
            session = await new_session(client)
            body = payload()
            async with client.stream("POST", run_path(session), json=body) as response:
                events = frames(response)
                started = await next_event(events)
                run_id = started["run_id"]
                run_url = f"{RUNS}/{run_id}"
                cancel_url = f"{run_url}/cancel"
                assert (await next_event(events))["data"]["delta"] == "半截回复"
                for method, url in (("GET", run_url), ("GET", f"{RUNS}/lookup"), ("POST", cancel_url)):
                    params = {"session_id": session, "client_request_id": body["client_request_id"]}
                    for auth, expected, code in (({"Authorization": ""}, 401, "AUTH_REQUIRED"),
                                                 (host, 403, "FORBIDDEN")):
                        rejected = await client.request(method, url, params=params, headers=auth)
                        assert rejected.status_code == expected and rejected.json()["error"]["code"] == code
                for method, suffix in (("GET", ""), ("POST", "/cancel")):
                    assert (await client.request(method, f"{RUNS}/{uuid4()}{suffix}")).status_code == 404

                concurrent = await asyncio.gather(*(client.post(cancel_url) for _ in range(5)))
                assert any(item.status_code == 202 for item in concurrent)
                for item in concurrent:
                    assert (item.status_code, item.json()) in (
                        (202, {"run_id": run_id, "status": "cancelling"}),
                        (200, {"run_id": run_id, "status": "cancelled", "already_finished": True}),
                    )
                await asyncio.wait_for(upstream_closed.wait(), timeout=2)
                terminal, = [event async for event in events if isinstance(event, dict)]
                assert terminal["type"] == "run_cancelled" and terminal["data"] == {"reason": "user_requested"}
                assert terminal["session_id"] == session and terminal["run_id"] == run_id
                state = (await client.get(run_url)).json()
                assert state["status"] == "cancelled" and state["finished_at"] and state["error_code"] is None
                repeat = await client.post(cancel_url)
                assert repeat.status_code == 200
                assert repeat.json() == {"run_id": run_id, "status": "cancelled", "already_finished": True}
                assert (await client.get(run_url)).json() == state
            assert rows(path, "SELECT role FROM messages WHERE run_id = ?", (run_id,)) == [("user",)]
            assert len(calls) == 1 and calls[0]["stream"] is True


async def test_cancel_waits_for_the_completed_model_to_commit(tmp_path, monkeypatch):
    path = tmp_path / "business.sqlite3"
    commit_entered, unblock_commit = Event(), Event()
    original_complete = BusinessStore.complete_run

    def delayed_commit(store, run, reply, checkpoint_id):
        commit_entered.set()
        assert unblock_commit.wait(timeout=4)
        return original_complete(store, run, reply, checkpoint_id)

    monkeypatch.setattr(BusinessStore, "complete_run", delayed_commit)
    async with live_api(path, lambda _request: httpx.Response(200, stream=ProviderStream())) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        async with client.stream("POST", run_path(session), json=payload()) as response:
            events = frames(response)
            run_id = (await next_event(events))["run_id"]
            try:
                assert await asyncio.to_thread(commit_entered.wait, 2)
                run_url = f"{RUNS}/{run_id}"
                assert (await client.get(run_url)).json()["status"] == "running"
                cancel = asyncio.create_task(client.post(f"{run_url}/cancel"))
                done, _ = await asyncio.wait({cancel}, timeout=0.05)
                assert not done
            finally:
                unblock_commit.set()
            result = await cancel
            assert result.status_code == 200
            assert result.json() == {"run_id": run_id, "status": "completed", "already_finished": True}
            terminal = [item async for item in events if isinstance(item, dict)][-1]
            assert terminal["type"] == "run_completed"
            assert rows(path, "SELECT role FROM messages WHERE message_id = ?", (terminal["data"]["message_id"],)) == [
                ("assistant",),
            ]


async def test_sqlite_races_admission_of_cancel_and_completion_on_distinct_connections(tmp_path):
    store = BusinessStore(tmp_path / "business.sqlite3")
    for _ in range(12):
        session = store.create_session("竞态")
        run = store.start_run(session.session_id, str(uuid4()), "问题", accept_new_run=True)
        assert run is not None
        barrier = asyncio.Barrier(4)

        async def race(function, *args):
            await barrier.wait()
            return await asyncio.to_thread(function, *args)

        *requests, message = await asyncio.gather(
            *(race(store.request_cancel, run) for _ in range(3)),
            race(store.complete_run, run, "完整回答", str(uuid4())),
        )
        assert store.confirm_cancel(run) == (message is None)
        snapshot = store.get_run(run.run_id)
        expected = "cancelled" if message is None else "completed"
        assert snapshot.status == expected and snapshot.finished_at
        assert not store.fail_run(run, "INTERNAL_ERROR", "late failure")
        assert not store.confirm_cancel(run)
        assert store.complete_run(run, "late reply", str(uuid4())) is None
        assert store.request_cancel(run) == snapshot
        assert {item.status for item in requests} == {"cancelling" if message is None else "completed"}
        assert rows(store.path, "SELECT content FROM messages WHERE run_id = ? AND role = 'assistant'",
                    (run.run_id,)) == ([] if message is None else [("完整回答",)])
