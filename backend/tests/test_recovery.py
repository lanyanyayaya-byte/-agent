"""P3.3 fault injection against the real Sidecar, LangGraph and two SQLite files."""

import asyncio
import json
import signal
import sqlite3
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import SecretStr

from app.agent.answer import EMPTY_CHECKPOINT_ID, _complete_text, answer_with_agent
from app.main import create_app
from app.models import ModelError
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.services.models import ModelService
from app.services.runs import RunService
from app.storage.business import DATABASE_NAME, BusinessStore, Run
from app.storage.checkpoints import CHECKPOINT_DATABASE_NAME, open_checkpointer
from test_chat import CREDENTIAL, RUNS, LocalProviderTransport, chunk, configure, frames, live_api, new_session
from test_chat import next_event, payload, rows, run_path
from test_checkpoints import graph_rows, run_ids, safe_id, submit
from test_sidecar import COMMAND, SHUTDOWN, assert_clean_exit, headers, running_sidecar

pytestmark = pytest.mark.anyio

SEED = "remember P3 marker"
SEED_REPLY = "P3 marker stored"
FOLLOWUP = "recall P3 marker"
RECALL = "recalled P3 marker"
ATTEMPT = "call fixture tool"
TOOL_ID = "call-p3-fixture"
ORPHAN = "uncommitted graph answer"
RACE = "race completion"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pending_calls(messages):
    pending = set()
    for message in messages:
        if message["role"] == "tool":
            pending.remove(message["tool_call_id"])
        else:
            if pending:
                return pending
            pending.update(call["id"] for call in message.get("tool_calls", ()))
    return pending


@pytest.fixture
def provider():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            assert self.path == "/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = body["messages"]
            try:
                invalid = bool(pending_calls(messages))
            except KeyError:
                invalid = True
            status = 400 if invalid else 200
            calls.append((status, messages))
            if invalid:
                data = b'{"error":{"message":"Unpaired tool call","type":"invalid_request_error"}}'
                content_type = "application/json"
            else:
                last = messages[-1]
                if last["role"] == "tool":
                    delta, finish = {"content": ORPHAN}, "stop"
                elif last["content"] == ATTEMPT:
                    assert [tool["function"]["name"] for tool in body["tools"]] == ["fixture_tool"]
                    delta = {"tool_calls": [{"index": 0, "id": TOOL_ID, "type": "function",
                                             "function": {"name": "fixture_tool", "arguments": "{}"}}]}
                    finish = "tool_calls"
                else:
                    previous = [(m["role"], m["content"]) for m in messages[:-1]]
                    reply = RECALL if previous == [("user", SEED), ("assistant", SEED_REPLY)] else SEED_REPLY
                    delta, finish = {"content": reply}, "stop"
                data = chunk({"role": "assistant", **delta}) + chunk({}, finish) + b"data: [DONE]\n\n"
                content_type = "text/event-stream"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        worker.start()
        try:
            yield server.server_port, calls
        finally:
            server.shutdown()
            worker.join(timeout=3)


def agent_with_tool(operation):
    async def fixture_tool() -> str:
        """Run the virtual P3.3 operation."""
        await operation()
        return "fixture result"

    def build(*, tools, **kwargs):
        assert tools == []
        return create_agent(tools=[fixture_tool], **kwargs)

    return build


def config(session, checkpoint_id=None):
    configurable = {"thread_id": f"session:{session}"}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


async def checkpoint(path, session, checkpoint_id=None):
    async with open_checkpointer(path) as saver:
        saved = await saver.aget_tuple(config(session, checkpoint_id))
    assert saved is not None
    assert checkpoint_id is None or saved.checkpoint["id"] == checkpoint_id
    return saved


def messages(saved):
    return saved.checkpoint["channel_values"]["messages"]


def assistant_rows(path, run):
    return rows(path, "SELECT message_id, content FROM messages WHERE run_id=? AND role='assistant'", (run,))


def completed_checkpoint(path, session, terminal, base, reply):
    run = terminal["run_id"]
    final = safe_id(path, session)
    assert final and final != base
    assert run_ids(path, run) == ("completed", base, final)
    assert assistant_rows(path, run) == [(terminal["data"]["message_id"], reply)]
    snapshots = graph_rows(path.with_name(CHECKPOINT_DATABASE_NAME), run)
    assert snapshots[0][1] == (base or EMPTY_CHECKPOINT_ID) and snapshots[-1][0] == final
    return final


def compare_databases(business, graph, run):
    # Read-only diagnosis by run metadata within the session's root checkpoint namespace.
    with closing(sqlite3.connect(business.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("ATTACH DATABASE ? AS graph", (graph.as_uri() + "?mode=ro",))
        connection.row_factory = sqlite3.Row
        return dict(connection.execute("""
            SELECT r.status, r.base_checkpoint_id, r.final_checkpoint_id, s.last_safe_checkpoint_id,
                   (SELECT count(*) FROM messages WHERE run_id = r.run_id AND role = 'assistant') AS assistant_rows,
                   (SELECT max(c.checkpoint_id) FROM graph.checkpoints AS c
                    WHERE c.thread_id = 'session:' || r.session_id AND c.checkpoint_ns = ''
                      AND json_extract(c.metadata, '$.knowledgeagent_run_id') = r.run_id) AS graph_checkpoint_id
            FROM runs AS r JOIN sessions AS s USING (session_id)
            WHERE r.run_id = ?
        """, (run,)).fetchone())


@pytest.mark.parametrize("fault", ["cancel", "tool_error"])
@pytest.mark.parametrize("seed_base", [True, False], ids=["historical-base", "empty-base"])
async def test_unpaired_tool_checkpoint_is_excluded_from_the_next_turn(tmp_path, monkeypatch, provider, fault, seed_base):
    port, calls = provider
    business, graph = tmp_path / DATABASE_NAME, tmp_path / CHECKPOINT_DATABASE_NAME
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    async def operation():
        effects.append("fixture operation")
        entered.set()
        await release.wait()
        raise RuntimeError("injected tool failure before its ToolMessage")

    monkeypatch.setattr("langchain.agents.create_agent", agent_with_tool(operation))
    async with live_api(business, transport=LocalProviderTransport(port)) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        base = None
        history = []
        if seed_base:
            base = completed_checkpoint(business, session, await submit(client, session, SEED), None, SEED_REPLY)
            history = [("user", SEED), ("assistant", SEED_REPLY)]
        body = payload(ATTEMPT)
        async with client.stream("POST", run_path(session), json=body) as response:
            assert response.status_code == 200
            events = frames(response)
            started = await next_event(events)
            assert started["type"] == "run_started"
            run = started["run_id"]
            await asyncio.wait_for(entered.wait(), timeout=3)

            current = await checkpoint(graph, session)
            stale = current.checkpoint["id"]
            assert stale != base and graph_rows(graph, run)[-1][0] == stale
            stored = messages(current)
            assert [item.type for item in stored] == (["human", "ai"] if seed_base else []) + ["human", "ai"]
            assert isinstance(stored[-1], AIMessage)
            assert stored[-1].tool_calls == [{"name": "fixture_tool", "args": {}, "id": TOOL_ID, "type": "tool_call"}]
            assert not any(isinstance(item, ToolMessage) for item in stored)
            assert run_ids(business, run) == ("running", base, None) and safe_id(business, session) == base

            if fault == "cancel":
                results = await asyncio.gather(*(client.post(f"{RUNS}/{run}/cancel") for _ in range(5)))
                assert any(result.status_code == 202 for result in results)
                for result in results:
                    assert (result.status_code, result.json()) in (
                        (202, {"run_id": run, "status": "cancelling"}),
                        (200, {"run_id": run, "status": "cancelled", "already_finished": True}),
                    )
                state, terminal_type = "cancelled", "run_cancelled"
            else:
                release.set()
                state, terminal_type = "failed", "error"
            terminal, = [event async for event in events if isinstance(event, dict)]
            assert terminal["type"] == terminal_type and terminal["run_id"] == run
            if fault == "tool_error":
                assert terminal["data"]["code"] == "MODEL_UPSTREAM_ERROR"

        assert run_ids(business, run) == (state, base, None) and safe_id(business, session) == base
        assert assistant_rows(business, run) == []
        repeated = await client.post(f"{RUNS}/{run}/cancel")
        assert repeated.status_code == 200
        assert repeated.json() == {"run_id": run, "status": state, "already_finished": True}
        assert (await checkpoint(graph, session)).checkpoint["id"] == stale
        if base:
            assert [item.text for item in messages(await checkpoint(graph, session, base))] == [SEED, SEED_REPLY]

        adapter = DeepSeekAdapter(DeepSeekSettings(), transport=LocalProviderTransport(port))
        async with open_checkpointer(graph) as saver, adapter.model(SecretStr("p3-fixture-only")) as model:
            with pytest.raises(ModelError, match="无法读取有效的对话安全检查点"):
                await answer_with_agent(model, FOLLOWUP, checkpointer=saver, config=config(session, stale))

        # Negative control: the stored incomplete history would be rejected by the same HTTP endpoint.
        invalid_history = [
            *calls[-1][1],
            {"role": "assistant", "content": stored[-1].text,
             "tool_calls": [{"id": call["id"], "type": "function", "function": {
                 "name": call["name"], "arguments": json.dumps(call["args"]),
             }} for call in stored[-1].tool_calls]},
            {"role": "user", "content": FOLLOWUP},
        ]
        async with httpx.AsyncClient(trust_env=False) as upstream:
            rejected = await upstream.post(f"http://127.0.0.1:{port}/chat/completions", json={"messages": invalid_history})
        assert rejected.status_code == 400 and rejected.json()["error"]["message"] == "Unpaired tool call"

        reply = RECALL if seed_base else SEED_REPLY
        resumed = await submit(client, session, FOLLOWUP)
        final = completed_checkpoint(business, session, resumed, base, reply)
        assert [item.text for item in messages(await checkpoint(graph, session, final))] == [
            *(text for _, text in history), FOLLOWUP, reply,
        ]
        assert [status for status, _ in calls] == [200] * (int(seed_base) + 1) + [400, 200]
        assert [(item["role"], item["content"]) for item in calls[-1][1]] == history + [("user", FOLLOWUP)]
        assert effects == ["fixture operation"]
        retained = await checkpoint(graph, session, stale)
        assert messages(retained)[-1].tool_calls == stored[-1].tool_calls
        print(json.dumps({"fault": fault, "run_id": run, "status": state, "safe_before_and_after_fault": base,
                          "stale": stale, "message_types": [item.type for item in stored],
                          "tool_calls": stored[-1].tool_calls, "tool_results": [],
                          "explicit_stale_base_rejected": True,
                          "stale_history_http": rejected.status_code, "resumed_http": calls[-1][0],
                          "resumed_base": base or EMPTY_CHECKPOINT_ID, "resumed_final": final,
                          "model_history": calls[-1][1], "tool_executions": len(effects)}))


def crash_fixture_main(data_dir: str, port: str, crash: str):
    """Test subprocess entry point. Only the pre-commit boundary is intercepted."""
    from app.sidecar import main

    directory = Path(data_dir)
    original_write = BusinessStore._write

    async def operation():
        with (directory / "fixture-effects.log").open("a", encoding="utf-8") as sink:
            sink.write("fixture operation\n")

    @contextmanager
    def before_commit(connection):
        with original_write(connection):
            yield
            candidate = connection.execute("""
                SELECT r.run_id, r.status, r.final_checkpoint_id, s.last_safe_checkpoint_id
                FROM runs AS r JOIN messages AS m USING (run_id) JOIN sessions AS s ON s.session_id = r.session_id
                WHERE m.role = 'assistant' AND m.content = ?
            """, (ORPHAN,)).fetchone()
            if candidate:
                marker = directory / "pre-commit.tmp"
                marker.write_text(json.dumps(candidate))
                marker.replace(directory / "pre-commit.json")
                Event().wait()

    def app_factory(*args, **kwargs):
        models = ModelService(DeepSeekAdapter(DeepSeekSettings(), transport=LocalProviderTransport(int(port))))
        return create_app(*args, **kwargs, models=models)

    with patch("langchain.agents.create_agent", agent_with_tool(operation)):
        if crash == "1":
            with patch.object(BusinessStore, "_write", staticmethod(before_commit)):
                return main(app_factory=app_factory)
        return main(app_factory=app_factory)


@contextmanager
def recovery_sidecar(directory, port, *, crash=False):
    entry = (
        'import sys; sys.path.insert(0, "tests"); from test_recovery import crash_fixture_main; '
        'raise SystemExit(crash_fixture_main(*sys.argv[1:]))'
    )
    command = [*COMMAND[:4], "-c", entry, str(directory), str(port), str(int(crash))]
    with running_sidecar(directory, command) as (child, ready, tokens):
        ui, host = headers(tokens["ui_token"]), headers(tokens["host_token"])
        with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", headers=ui, trust_env=False, timeout=7) as client:
            configured = client.put(CREDENTIAL, headers=host, json={"api_key": "p3-fixture-only"})
            assert configured.json() == {"configured": True}
            yield child, ready, client
            if not crash:
                assert client.post(SHUTDOWN, headers=host).status_code == 204
                assert_clean_exit(child, tokens)


def sse(response):
    return (json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: "))


def submit_sync(client, session, text):
    response = client.post(run_path(session), json=payload(text))
    assert response.status_code == 200
    events = list(sse(response))
    assert events[0]["type"] == "run_started" and events[-1]["type"] == "run_completed"
    return events[-1]


def wait_for_commit_window(path):
    deadline = monotonic() + 5
    while not path.is_file() and monotonic() < deadline:
        sleep(0.01)
    assert path.is_file(), "Sidecar never reached the pre-commit window"
    return json.loads(path.read_text())


def test_sigkill_after_final_checkpoint_before_business_commit(tmp_path, provider):
    port, calls = provider
    business, graph = tmp_path / DATABASE_NAME, tmp_path / CHECKPOINT_DATABASE_NAME
    effects = tmp_path / "fixture-effects.log"
    instances = []
    with recovery_sidecar(tmp_path, port, crash=True) as (child, ready, client):
        instances.append((child.pid, ready["instance_id"]))
        session = client.post("/api/v1/sessions", json={}).json()["session_id"]
        base = completed_checkpoint(business, session, submit_sync(client, session, SEED), None, SEED_REPLY)
        body = payload(ATTEMPT)
        with client.stream("POST", run_path(session), json=body) as response:
            assert response.status_code == 200
            stream = sse(response)
            started = next(stream)
            assert started["type"] == "run_started"
            run = started["run_id"]
            pending_run, pending_status, orphan, pending_safe = wait_for_commit_window(tmp_path / "pre-commit.json")
            assert pending_run == run and pending_status == "completed"
            assert orphan and orphan == pending_safe and orphan != base
            before_crash = compare_databases(business, graph, run)
            assert before_crash == {
                "status": "running", "base_checkpoint_id": base, "final_checkpoint_id": None,
                "last_safe_checkpoint_id": base, "assistant_rows": 0, "graph_checkpoint_id": orphan,
            }
            snapshots = graph_rows(graph, run)
            assert snapshots[0][1] == base and snapshots[-1][0] == orphan
            latest = asyncio.run(checkpoint(graph, session))
            assert latest.checkpoint["id"] == orphan and _complete_text(messages(latest)) == ORPHAN
            assert [item.type for item in messages(latest)] == ["human", "ai", "human", "ai", "tool", "ai"]
            assert messages(latest)[-2].tool_call_id == TOOL_ID
            assert next(stream)["type"] == "message_delta"
            child.kill()
            assert child.wait(timeout=5) == -signal.SIGKILL
    assert compare_databases(business, graph, run) == before_crash
    assert len(calls) == 3 and effects.read_text().splitlines() == ["fixture operation"]

    recovered = None
    for restart in (1, 2):
        with recovery_sidecar(tmp_path, port) as (child, ready, client):
            instances.append((child.pid, ready["instance_id"]))
            status = client.get(f"{RUNS}/{run}").json()
            assert status["status"] == "failed" and status["error_code"] == "RUN_INTERRUPTED"
            assert status["finished_at"] and (recovered is None or status == recovered)
            recovered = status
            after_restart = compare_databases(business, graph, run)
            assert after_restart == {**before_crash, "status": "failed"}
            assert graph_rows(graph, run) == snapshots
            assert asyncio.run(checkpoint(graph, session)).checkpoint["id"] == orphan
            assert [item.text for item in messages(asyncio.run(checkpoint(graph, session, base)))] == [SEED, SEED_REPLY]
            duplicate = client.post(run_path(session), json=body)
            assert duplicate.status_code == 409
            assert duplicate.json()["error"]["code"] == "DUPLICATE_CLIENT_REQUEST"
            assert duplicate.json()["error"]["run_id"] == run
            assert len(calls) == 3 and effects.read_text().splitlines() == ["fixture operation"]
            if restart == 2:
                resumed = submit_sync(client, session, FOLLOWUP)
                final = completed_checkpoint(business, session, resumed, base, RECALL)

    log = [json.loads(line) for line in (tmp_path / "logs" / "sidecar.jsonl").read_text().splitlines()]
    recovery_log, = [record for record in log if record.get("run_id") == run and record["event"] == "run_finished"]
    assert recovery_log["pid"] == instances[1][0] and recovery_log["error_code"] == "RUN_INTERRUPTED"
    assert len({pid for pid, _ in instances}) == len({instance for _, instance in instances}) == 3
    assert graph_rows(graph, run) == snapshots
    assert _complete_text(messages(asyncio.run(checkpoint(graph, session, orphan)))) == ORPHAN
    assert [item.text for item in messages(asyncio.run(checkpoint(graph, session, final)))] == [
        SEED, SEED_REPLY, FOLLOWUP, RECALL,
    ]
    assert [status for status, _ in calls] == [200, 200, 200, 200]
    assert [(item["role"], item["content"]) for item in calls[-1][1]] == [
        ("user", SEED), ("assistant", SEED_REPLY), ("user", FOLLOWUP),
    ]
    assert effects.read_text().splitlines() == ["fixture operation"]
    assert rows(business, "SELECT role, count(*) FROM messages GROUP BY role ORDER BY role") == [
        ("assistant", 2), ("user", 3),
    ]
    committed = compare_databases(business, graph, resumed["run_id"])
    assert committed == {
        "status": "completed", "base_checkpoint_id": base, "final_checkpoint_id": final,
        "last_safe_checkpoint_id": final, "assistant_rows": 1, "graph_checkpoint_id": final,
    }
    for path in (business, graph):
        assert rows(path, "PRAGMA integrity_check") == [("ok",)]
    print(json.dumps({"fault": "SIGKILL", "instances": instances, "crashed_run": run,
                      "uncommitted_transaction": {"status": pending_status, "final_checkpoint_id": orphan,
                                                  "last_safe_checkpoint_id": pending_safe, "assistant_rows": 1},
                      "before_restart": before_crash, "after_both_restarts": after_restart,
                      "orphan_reply": ORPHAN, "recovery_events": 1, "duplicate_http": 409,
                      "model_calls_before_and_after_restarts": [3, 3, 3],
                      "tool_executions": 1, "committed_next_run": committed,
                      "resumed_model_history": calls[-1][1]}))


@pytest.mark.parametrize("winner", ["cancelled", "completed"])
async def test_cancel_races_with_real_graph_completion(tmp_path, monkeypatch, provider, winner):
    port, calls = provider
    business, graph = tmp_path / DATABASE_NAME, tmp_path / CHECKPOINT_DATABASE_NAME
    graph_written = asyncio.get_running_loop().create_future()
    return_answer, cancel_arrived = asyncio.Event(), asyncio.Event()
    write_entered, release_write = Event(), Event()
    original_answer, original_cancel = RunService._answer, RunService._cancel

    async def hold_answer(service, run, message, on_delta):
        result = await original_answer(service, run, message, on_delta)
        if message == RACE:
            graph_written.set_result((asyncio.current_task(), result.checkpoint_id))
            await return_answer.wait()
        return result

    async def observe_cancel(service, run_id):
        cancel_arrived.set()
        return await original_cancel(service, run_id)

    def hold_write(method):
        def delayed(store, run, *args):
            write_entered.set()
            assert release_write.wait(timeout=6)
            return method(store, run, *args)

        return delayed

    monkeypatch.setattr(RunService, "_answer", hold_answer)
    monkeypatch.setattr(RunService, "_cancel", observe_cancel)
    method = "request_cancel" if winner == "cancelled" else "complete_run"
    async with live_api(business, transport=LocalProviderTransport(port)) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        base = completed_checkpoint(business, session, await submit(client, session, SEED), None, SEED_REPLY)
        monkeypatch.setattr(BusinessStore, method, hold_write(getattr(BusinessStore, method)))

        body = payload(RACE)
        async with client.stream("POST", run_path(session), json=body) as response:
            assert response.status_code == 200
            events = frames(response)
            run = (await next_event(events))["run_id"]
            task, graph_final = await asyncio.wait_for(graph_written, timeout=3)
            assert _complete_text(messages(await checkpoint(graph, session, graph_final))) == RECALL
            try:
                conflicts = await asyncio.gather(
                    client.post(run_path(session), json=body),
                    client.post(run_path(session), json=payload("competing prompt")),
                )
                for conflict, code in zip(conflicts, ("DUPLICATE_CLIENT_REQUEST", "RUN_ALREADY_ACTIVE"), strict=True):
                    assert conflict.status_code == 409
                    assert conflict.json()["error"]["code"] == code and conflict.json()["error"]["run_id"] == run
                if winner == "completed":
                    return_answer.set()
                    assert await asyncio.to_thread(write_entered.wait, 3)
                cancel = asyncio.create_task(client.post(f"{RUNS}/{run}/cancel"))
                await asyncio.wait_for(cancel_arrived.wait(), timeout=3)
                assert await asyncio.to_thread(write_entered.wait, 3)
                return_answer.set()
                assert (await asyncio.wait_for(asyncio.shield(task), timeout=3)).checkpoint_id == graph_final
                assert not cancel.done()
                assert safe_id(business, session) == base and assistant_rows(business, run) == []
            finally:
                return_answer.set()
                release_write.set()

            result = await cancel
            remaining = [event async for event in events if isinstance(event, dict)]
            assert [item["type"] for item in remaining] == [
                "message_delta", "run_cancelled" if winner == "cancelled" else "run_completed",
            ]
            if winner == "cancelled":
                assert result.status_code == 202 and result.json() == {"run_id": run, "status": "cancelling"}
                assert run_ids(business, run) == ("cancelled", base, None)
                assert safe_id(business, session) == base and assistant_rows(business, run) == []
            else:
                assert result.status_code == 200
                assert result.json() == {"run_id": run, "status": "completed", "already_finished": True}
                assert completed_checkpoint(business, session, remaining[-1], base, RECALL) == graph_final

        store = BusinessStore(business)
        old_run, before_late_event = Run(session, run, body["client_request_id"], base), store.get_run(run)
        assert not store.fail_run(old_run, "INTERNAL_ERROR", "late error")
        assert store.complete_run(old_run, "late reply", graph_final) is None
        repeated = await client.post(f"{RUNS}/{run}/cancel")
        assert repeated.status_code == 200
        assert repeated.json() == {"run_id": run, "status": winner, "already_finished": True}
        assert store.get_run(run) == before_late_event
        assert len(calls) == 2 and len(rows(business, "SELECT run_id FROM runs")) == 2
        result = compare_databases(business, graph, run)
        assert result == {
            "status": winner, "base_checkpoint_id": base,
            "final_checkpoint_id": graph_final if winner == "completed" else None,
            "last_safe_checkpoint_id": graph_final if winner == "completed" else base,
            "assistant_rows": int(winner == "completed"), "graph_checkpoint_id": graph_final,
        }
        print(json.dumps({"fault": "completion_cancel_race", "run_id": run, **result,
                          "competing_http": [409, 409], "late_transitions_applied": 0}))
