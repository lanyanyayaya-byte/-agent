import asyncio
import json
import sqlite3
from contextlib import closing

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.answer import EMPTY_CHECKPOINT_ID, _complete_text
from app.storage.business import BUSY_TIMEOUT_MS, DATABASE_NAME
from app.storage.checkpoints import CHECKPOINT_DATABASE_NAME, open_checkpointer
from test_chat import ProviderStream, configure, frames, live_api, new_session, next_event, payload, rows, run_path
from test_sidecar import COMMAND, SHUTDOWN, assert_clean_exit, headers, running_sidecar


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_safe_final_state_requires_a_result_for_each_tool_call():
    question = HumanMessage(content="question")
    calls = AIMessage(content="", tool_calls=[
        {"name": "fixture", "args": {}, "id": "first"},
        {"name": "fixture", "args": {}, "id": "second"},
    ])
    first_result = ToolMessage(content="first result", tool_call_id="first")
    second_result = ToolMessage(content="second result", tool_call_id="second")
    reply = AIMessage(content="answer", response_metadata={"finish_reason": "stop"})
    assert _complete_text([question, calls, first_result, second_result, reply]) == "answer"
    assert _complete_text([question, calls, first_result, reply]) is None
    assert _complete_text([question, calls, first_result, first_result, reply]) is None


def safe_id(path, session):
    return rows(path, "SELECT last_safe_checkpoint_id FROM sessions WHERE session_id = ?", (session,))[0][0]


def run_ids(path, run):
    return rows(path, "SELECT status, base_checkpoint_id, final_checkpoint_id FROM runs WHERE run_id = ?", (run,))[0]


def graph_rows(path, run):
    return rows(path, """SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints
                         WHERE json_extract(metadata, '$.knowledgeagent_run_id') = ?
                         ORDER BY checkpoint_id""", (run,))


async def checkpoint_messages(path, session, checkpoint_id):
    async with open_checkpointer(path) as saver:
        checkpoint = await saver.aget({
            "configurable": {"thread_id": f"session:{session}", "checkpoint_id": checkpoint_id},
        })
    return [message.text for message in checkpoint["channel_values"]["messages"]]


async def submit(client, session, text, terminal="run_completed"):
    response = await client.post(run_path(session), json=payload(text))
    assert response.status_code == 200
    events = [event async for event in frames(response)]
    assert events[0]["type"] == "run_started"
    assert events[-1]["type"] == terminal
    return events[-1]


@pytest.mark.anyio
@pytest.mark.parametrize("seed_base", [False, True])
async def test_failed_and_cancelled_snapshots_are_excluded_from_next_turn(tmp_path, seed_base):
    business, graph = tmp_path / DATABASE_NAME, tmp_path / CHECKPOINT_DATABASE_NAME
    gate = asyncio.Event()
    model_inputs = []

    def provider(request):
        messages = [(m["role"], m["content"]) for m in json.loads(request.content)["messages"]]
        model_inputs.append(messages)
        prompt = messages[-1][1]
        return httpx.Response(200, stream=ProviderStream(
            gate if prompt == "cancel me" else None, mode="length" if prompt == "reject me" else "stop",
        ))

    async with live_api(business, provider) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        base, expected_history = None, []
        if seed_base:
            initial = await submit(client, session, "remember this")
            base = safe_id(business, session)
            assert base and run_ids(business, initial["run_id"]) == ("completed", None, base)
            assert await checkpoint_messages(graph, session, base) == ["remember this", "第一段第二段"]
            expected_history = [("user", "remember this"), ("assistant", "第一段第二段")]

        failed = await submit(client, session, "reject me", terminal="error")
        assert failed["data"]["code"] == "MODEL_UPSTREAM_ERROR"
        assert run_ids(business, failed["run_id"]) == ("failed", base, None)
        failed_snapshots = graph_rows(graph, failed["run_id"])
        assert failed_snapshots[0][1] == (base or EMPTY_CHECKPOINT_ID)
        assert await checkpoint_messages(graph, session, failed_snapshots[-1][0]) == [
            *(text for _, text in expected_history), "reject me", "第一段第二段",
        ]
        assert safe_id(business, session) == base

        async with client.stream("POST", run_path(session), json=payload("cancel me")) as response:
            stream = frames(response)
            cancelled = (await next_event(stream))["run_id"]
            assert (await next_event(stream))["type"] == "message_delta"
            assert (await client.post(f"/api/v1/runs/{cancelled}/cancel")).status_code == 202
            assert (await next_event(stream))["type"] == "run_cancelled"
        assert run_ids(business, cancelled) == ("cancelled", base, None)
        cancelled_snapshots = graph_rows(graph, cancelled)
        assert cancelled_snapshots[0][1] == (base or EMPTY_CHECKPOINT_ID)
        assert await checkpoint_messages(graph, session, cancelled_snapshots[-1][0]) == [
            *(text for _, text in expected_history), "cancel me",
        ]
        assert safe_id(business, session) == base

        retry = await submit(client, session, "next turn")
        final = safe_id(business, session)
        assert final and final != base
        assert run_ids(business, retry["run_id"]) == ("completed", base, final)
        retry_snapshots = graph_rows(graph, retry["run_id"])
        assert retry_snapshots[0][1] == (base or EMPTY_CHECKPOINT_ID)
        assert retry_snapshots[-1][0] == final
        assert rows(business, "SELECT role FROM messages WHERE run_id IN (?, ?)",
                    (failed["run_id"], cancelled)) == [("user",), ("user",)]

        other = await new_session(client)
        isolated = await submit(client, other, "new session")
        other_final = safe_id(business, other)
        assert other_final and run_ids(business, isolated["run_id"]) == ("completed", None, other_final)
        assert await checkpoint_messages(graph, other, other_final) == ["new session", "第一段第二段"]
        assert model_inputs == [
            *([[("user", "remember this")]] if seed_base else []),
            *(expected_history + [("user", prompt)] for prompt in ("reject me", "cancel me", "next turn")),
            [("user", "new session")],
        ]


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["checkpoint", "business_commit"])
async def test_storage_failure_never_publishes_completion(tmp_path, failure):
    business, graph = tmp_path / DATABASE_NAME, tmp_path / CHECKPOINT_DATABASE_NAME
    gate = asyncio.Event()
    model_inputs = []

    def provider(request):
        messages = [m["content"] for m in json.loads(request.content)["messages"]]
        model_inputs.append(messages)
        return httpx.Response(200, stream=ProviderStream(gate if messages[-1] == "storage failure" else None))

    async with live_api(business, provider) as (client, host):
        await configure(client, host)
        session = await new_session(client)
        await submit(client, session, "safe")
        base = safe_id(business, session)
        async with client.stream("POST", run_path(session), json=payload("storage failure")) as response:
            events = frames(response)
            run = (await next_event(events))["run_id"]
            assert (await next_event(events))["type"] == "message_delta"
            assert run_ids(business, run) == ("running", base, None)
            before = graph_rows(graph, run)
            assert before and before[0][1] == base
            if failure == "checkpoint":
                with closing(sqlite3.connect(graph)) as db:
                    db.execute("""CREATE TRIGGER reject_checkpoint BEFORE INSERT ON checkpoints
                                  BEGIN SELECT RAISE(ROLLBACK, 'checkpoint write failure'); END""")
            else:
                with closing(sqlite3.connect(business)) as db:
                    db.executescript("""
                        CREATE TABLE commit_fault (
                            run_id TEXT REFERENCES runs(run_id) DEFERRABLE INITIALLY DEFERRED
                        );
                        CREATE TRIGGER reject_commit AFTER UPDATE OF last_safe_checkpoint_id ON sessions
                        BEGIN INSERT INTO commit_fault VALUES ('missing run'); END;
                    """)
            gate.set()
            tail = [event async for event in events]
            assert [event["type"] for event in tail] == ["message_delta", "error"]
            assert tail[-1]["data"]["code"] == "INTERNAL_ERROR"

        assert run_ids(business, run) == ("failed", base, None)
        assert safe_id(business, session) == base
        assert rows(business, "SELECT role FROM messages WHERE run_id = ?", (run,)) == [("user",)]
        after = graph_rows(graph, run)
        if failure == "checkpoint":
            assert after == before
            with closing(sqlite3.connect(graph)) as db:
                db.execute("DROP TRIGGER reject_checkpoint")
        else:
            assert len(after) > len(before)
            assert await checkpoint_messages(graph, session, after[-1][0]) == [
                "safe", "第一段第二段", "storage failure", "第一段第二段",
            ]
            assert rows(business, "SELECT * FROM commit_fault") == []
            with closing(sqlite3.connect(business)) as db:
                db.execute("DROP TRIGGER reject_commit")
        resumed = await submit(client, session, "try again")
        assert model_inputs[-1] == ["safe", "第一段第二段", "try again"]
        final = safe_id(business, session)
        assert final != base
        assert run_ids(business, resumed["run_id"]) == ("completed", base, final)
        assert graph_rows(graph, resumed["run_id"])[0][1] == base


def test_real_sidecar_restart_reads_persisted_checkpoint(tmp_path):
    fixture = """
import json
import sys
import httpx
from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.services.models import ModelService
from app.sidecar import main

sys.path.insert(0, "tests")
from test_chat import ProviderStream

def provider(request):
    messages = [m["content"] for m in json.loads(request.content)["messages"]]
    if messages[-1] == "restart recall":
        assert messages == ["store marker", "第一段第二段", "restart recall"]
    else:
        assert messages == ["store marker"]
    return httpx.Response(200, stream=ProviderStream())

def fixture_app(*args, **kwargs):
    transport = httpx.MockTransport(provider)
    return create_app(*args, **kwargs, models=ModelService(DeepSeekAdapter(DeepSeekSettings(), transport=transport)))

raise SystemExit(main(app_factory=fixture_app))
"""
    business, graph = tmp_path / DATABASE_NAME, tmp_path / CHECKPOINT_DATABASE_NAME

    def converse(prompt, session=None):
        with running_sidecar(tmp_path, [*COMMAND[:4], "-c", fixture]) as (child, ready, tokens):
            ui, host = headers(tokens["ui_token"]), headers(tokens["host_token"])
            with httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", trust_env=False, timeout=10) as client:
                assert graph.is_file() and business.is_file()
                assert client.put("/api/v1/_host/model-credentials/deepseek", headers=host,
                                  json={"api_key": "fixture-only"}).status_code == 200
                if session is None:
                    session = client.post("/api/v1/sessions", headers=ui, json={}).json()["session_id"]
                reply = client.post(run_path(session), headers=ui, json=payload(prompt))
                events = [json.loads(line[6:]) for line in reply.iter_lines() if line.startswith("data: ")]
                assert events[-1]["type"] == "run_completed"
                assert client.post(SHUTDOWN, headers=host).status_code == 204
            assert_clean_exit(child, tokens)
            return session, events[-1]["run_id"], child.pid, ready["instance_id"]

    session, first, first_pid, first_instance = converse("store marker")
    base = safe_id(business, session)
    assert base and run_ids(business, first) == ("completed", None, base)
    assert rows(graph, "PRAGMA journal_mode") == [("wal",)]
    assert not graph.with_name(f"{graph.name}-wal").exists()

    _, second, second_pid, second_instance = converse("restart recall", session)
    assert first_pid != second_pid and first_instance != second_instance
    final = safe_id(business, session)
    assert final and final != base
    assert run_ids(business, second) == ("completed", base, final)
    assert graph_rows(graph, second)[0][1] == base
    assert graph_rows(graph, second)[-1][0] == final

    async def inspect():
        async with open_checkpointer(graph) as saver:
            async with saver.conn.execute("PRAGMA busy_timeout") as cursor:
                assert await cursor.fetchone() == (BUSY_TIMEOUT_MS,)
        assert await checkpoint_messages(graph, session, final) == [
            "store marker", "第一段第二段", "restart recall", "第一段第二段",
        ]

    asyncio.run(inspect())
    print(json.dumps({"first_pid": first_pid, "second_pid": second_pid, "session_id": session,
                      "first_run": first, "second_run": second, "second_base": base, "last_safe": final}))
