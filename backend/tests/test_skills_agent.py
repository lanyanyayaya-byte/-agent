"""P9.4 verification suite: Agent autonomous Skills execution, recoverable failures,
continuous multi-turn editing, Progressive Disclosure, and MCP egress collaboration.

Uses a deterministic local OpenAI-compatible SSE provider driving the REAL LangGraph
`tool_graph`, REAL `RunService`, REAL `RunTools`, REAL `SkillManager`, REAL `WorkspaceService`,
REAL `ArtifactService`, and REAL `RuntimeExecutor`.
"""

from __future__ import annotations

import json
import shutil
import sys
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from uuid import uuid4

import httpx
import pytest
from docx import Document
from docx.oxml.ns import qn

from app.main import create_app
from app.models.deepseek import DeepSeekAdapter, DeepSeekSettings
from app.services.models import ModelService
from app.storage.checkpoints import CHECKPOINT_DATABASE_NAME, open_checkpointer

UI_TOKEN = "p94-ui-token-1234567890"
HOST_TOKEN = "p94-host-token-abcdefghij"
UI_HEADERS = {"Authorization": f"Bearer {UI_TOKEN}", "Host": "127.0.0.1:8000"}
HOST_HEADERS = {"Authorization": f"Bearer {HOST_TOKEN}", "Host": "127.0.0.1:8000"}

# Reuse the already-prepared isolated runtime venv from P9.2 if available so tests stay fast offline,
# otherwise create a fresh venv in tmp_path.
CACHED_P92_RUNTIME = Path(
    "/Users/lanyan/.gemini/antigravity/brain/1744b658-ced0-43e6-ac88-c99ebde7bd2b/scratch/p9_e2e_run/app_data/runtime"
)


def _seed_runtime_if_cached(app_data: Path) -> None:
    if not (CACHED_P92_RUNTIME / "venv" / "bin" / "python").is_file():
        return
    runtime_dir = app_data / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Symlink the pre-prepared read-only venv / npm-tools / cache into this test's isolated app_data
    for sub in ("venv", "npm-tools", "cache"):
        src = CACHED_P92_RUNTIME / sub
        dst = runtime_dir / sub
        if src.exists() and not dst.exists():
            dst.symlink_to(src)


def _sse_chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _tool_calls_sse(calls: list[tuple[str, dict]]) -> bytes:
    chunks = bytearray()
    for idx, (name, args) in enumerate(calls):
        chunks.extend(_sse_chunk({
            "role": "assistant",
            "tool_calls": [{
                "index": idx,
                "id": f"call_{idx}_{uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        }))
    chunks.extend(_sse_chunk({}, finish_reason="tool_calls"))
    chunks.extend(b"data: [DONE]\n\n")
    return bytes(chunks)


def _text_sse(text: str) -> bytes:
    mid = max(1, len(text) // 2)
    return (
        _sse_chunk({"role": "assistant", "content": text[:mid]})
        + _sse_chunk({"content": text[mid:]})
        + _sse_chunk({}, finish_reason="stop")
        + b"data: [DONE]\n\n"
    )


class ScriptedProvider:
    def __init__(self, decide):
        self.decide = decide
        self.requests: list[dict] = []

    def respond(self, request_body: dict) -> bytes:
        self.requests.append(request_body)
        action = self.decide(request_body, len(self.requests))
        if isinstance(action, str):
            return _text_sse(action)
        return _tool_calls_sse(action)


@contextmanager
def provider_server(provider: ScriptedProvider):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            content = provider.respond(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=3)


@asynccontextmanager
async def live_skills_app(app_data: Path, base_url: str):
    _seed_runtime_if_cached(app_data)
    models = ModelService(
        DeepSeekAdapter(
            DeepSeekSettings(model="deepseek-chat", base_url=base_url, timeout_seconds=30, max_tokens=8192)
        )
    )
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=app_data / "knowledgeagent.sqlite3", models=models)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            cred = await client.put(
                "/api/v1/_host/model-credentials/deepseek",
                json={"api_key": "sk-test-key-1234567890", "base_url": base_url},
                headers=HOST_HEADERS,
            )
            assert cred.status_code == 200
            yield app, client


async def _stream_run(client: httpx.AsyncClient, session_id: str, payload: dict) -> list[dict]:
    events: list[dict] = []
    async with client.stream(
        "POST",
        f"/api/v1/sessions/{session_id}/runs/stream",
        json=payload,
        headers=UI_HEADERS,
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _last_tool_payload(request_body: dict) -> dict:
    for msg in reversed(request_body["messages"]):
        if msg.get("role") == "tool":
            return json.loads(msg["content"])
    raise AssertionError("No tool message found")


@pytest.mark.anyio
async def test_recoverable_script_error_self_fix_and_repeat_limit(tmp_path):
    """Test C: Real Python syntax error -> model sees real stderr -> fixes script -> succeeds.
    Also verifies 3 consecutive identical errors stop cleanly while distinct errors do not accumulate.
    """
    app_data = tmp_path / "app_data"

    def decide_self_fix(req: dict, step: int):
        if step == 1:
            # Verify Progressive Disclosure: system prompt has catalog summary, NOT full SKILL.md body
            sys_prompt = req["messages"][0]["content"]
            assert "builtin-word-formatting" in sys_prompt
            assert "builtin-mind-map" in sys_prompt
            assert "w:firstLineChars" not in sys_prompt  # Not in catalog!
            # Write a broken Python script first
            return [("write_file", {"relative_path": "work/gen.py", "content": "def broken(:\n    pass\n"})]
        if step == 2:
            return [("execute_command", {"command": "python", "args": ["work/gen.py"]})]
        if step == 3:
            # Verify model received REAL stderr and original arguments were preserved!
            tool_res = _last_tool_payload(req)
            assert tool_res["ok"] is False
            assert "SyntaxError" in tool_res["content"]["stderr"]
            prev_assistant = [m for m in req["messages"] if m.get("role") == "assistant" and m.get("tool_calls")][-1]
            assert json.loads(prev_assistant["tool_calls"][0]["function"]["arguments"])["command"] == "python"
            # Fix script to generate real output file
            fixed = (
                "from pathlib import Path\n"
                "Path('output').mkdir(exist_ok=True)\n"
                "Path('output/fixed.txt').write_text('已自动修复并生成！\\n', encoding='utf-8')\n"
                "print('FIXED_OK')\n"
            )
            return [("write_file", {"relative_path": "work/gen.py", "content": fixed})]
        if step == 4:
            return [("execute_command", {"command": "python", "args": ["work/gen.py"]})]
        if step == 5:
            tool_res = _last_tool_payload(req)
            assert tool_res["ok"] is True
            assert "FIXED_OK" in tool_res["content"]["stdout"]
            return [("publish_artifact", {"relative_path": "output/fixed.txt", "title": "修复后产物"})]
        return "脚本首次存在语法错误，已根据 stderr 修正并重新执行，最终生成了 fixed.txt。"

    provider = ScriptedProvider(decide_self_fix)
    with provider_server(provider) as base_url:
        async with live_skills_app(app_data, base_url) as (app, client):
            sid = (await client.post("/api/v1/sessions", json={"title": "错误恢复测试"}, headers=UI_HEADERS)).json()["session_id"]
            events = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "运行脚本生成输出，如有报错请修复",
                "skills": {"enabled": True},
            })
            event_types = [e["type"] for e in events]
            assert "tool_failed" in event_types
            assert "artifact_created" in event_types
            assert event_types[-1] == "run_completed"
            assert len(events[-1]["data"]["artifacts"]) == 1

            # Verify SQLite audit preserves the failed recoverable tool call AND published artifact
            runs_resp = await client.get(f"/api/v1/sessions/{sid}/runs", headers=UI_HEADERS)
            run_item = runs_resp.json()["items"][0]
            assert run_item["status"] == "completed"
            failed_calls = [tc for tc in run_item["tool_calls"] if tc["status"] == "failed"]
            assert len(failed_calls) == 1
            assert failed_calls[0]["recoverable"] is True
            assert failed_calls[0]["tool_category"] == "runtime"
            assert len(run_item["artifacts"]) == 1
            assert run_item["artifacts"][0]["filename"] == "fixed.txt"

            # Verify LangGraph checkpoint does NOT bloat with raw script/tool messages
            cp_id = app.state.runs._store._connect().execute(
                "SELECT last_safe_checkpoint_id FROM sessions WHERE session_id = ?", (sid,),
            ).fetchone()[0]
            async with open_checkpointer(app_data / CHECKPOINT_DATABASE_NAME) as saver:
                cp = await saver.aget({"configurable": {"thread_id": f"session:{sid}", "checkpoint_id": cp_id}})
                saved_msgs = cp["channel_values"]["messages"]
                assert len(saved_msgs) == 2  # Only HumanMessage + final AIMessage!

            # Now verify 3 consecutive IDENTICAL recoverable errors terminate with SKILL_REPEAT_FAILURE_LIMIT
            def decide_repeat_fail(req: dict, step: int):
                return [("read_file", {"relative_path": "work/does_not_exist.txt"})]

            provider.decide = decide_repeat_fail
            provider.requests.clear()
            repeat_events = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "测试连续重复错误上限",
                "skills": {"enabled": True},
            })
            assert repeat_events[-1]["type"] == "error"
            assert repeat_events[-1]["data"]["code"] == "SKILL_REPEAT_FAILURE_LIMIT"


@pytest.mark.anyio
async def test_word_formatting_and_continuous_multi_turn_editing(tmp_path):
    """Test B + Test D: Real Word formatting via word-formatting skill + 2nd turn editing published artifact."""
    app_data = tmp_path / "app_data"

    # Build a real user DOCX file on disk
    user_docx = tmp_path / "学术论文原件.docx"
    doc = Document()
    doc.add_heading("深度检索增强生成综述", level=1)
    doc.add_paragraph("本文系统讨论了 RAG 与 Agent Skills 在本地桌面端结合的工程实践。")
    doc.save(str(user_docx))
    orig_bytes = user_docx.read_bytes()

    state: dict[str, str] = {}

    def decide_word(req: dict, step: int):
        if state.get("turn") == "1":
            if step == 1:
                return [("load_skill", {"skill_id": "builtin-word-formatting"})]
            if step == 2:
                loaded = _last_tool_payload(req)["content"]
                state["skill_dir"] = loaded["skill_dir"]
                # Verify deduplication by calling load_skill a second time
                return [("load_skill", {"skill_id": "word-formatting"})]
            if step == 3:
                dedup = _last_tool_payload(req)["content"]
                assert dedup["already_activated"] is True
                return [("list_workspace", {})]
            if step == 4:
                ws = _last_tool_payload(req)["content"]
                att_path = ws["attachments"][0]["relative_path"]
                return [("copy_file", {"source_path": att_path, "dest_path": "work/source.docx"})]
            if step == 5:
                spec = {
                    "heading1": {"east_asia_font": "黑体", "western_font": "Times New Roman", "size": "三号", "bold": True},
                    "body": {
                        "east_asia_font": "宋体",
                        "western_font": "Times New Roman",
                        "size": "小四",
                        "line_spacing": 1.5,
                        "first_line_indent_chars": 2,
                    },
                }
                return [("write_file", {"relative_path": "work/spec.json", "content": json.dumps(spec, ensure_ascii=False)})]
            if step == 6:
                return [(
                    "execute_command",
                    {
                        "command": "python",
                        "args": [
                            f"{state['skill_dir']}/scripts/format_docx.py",
                            "--input", "work/source.docx",
                            "--output", "output/学术论文_排版后.docx",
                            "--spec", "work/spec.json",
                        ],
                    },
                )]
            if step == 7:
                assert _last_tool_payload(req)["ok"] is True
                return [(
                    "execute_command",
                    {
                        "command": "python",
                        "args": [
                            f"{state['skill_dir']}/scripts/validate_docx.py",
                            "--original", "work/source.docx",
                            "--formatted", "output/学术论文_排版后.docx",
                            "--spec", "work/spec.json",
                        ],
                    },
                )]
            if step == 8:
                val = json.loads(_last_tool_payload(req)["content"]["stdout"])
                assert val["validated"] is True
                return [("publish_artifact", {"relative_path": "output/学术论文_排版后.docx", "title": "排版后论文 v1"})]
            return "已按照您的要求完成 Word 排版并回读验证，原文件保持不变。"

        # Turn 2: continuous editing based on Turn 1's published artifact
        if step == 1:
            return [("load_skill", {"skill_id": "word-formatting"})]
        if step == 2:
            state["skill_dir"] = _last_tool_payload(req)["content"]["skill_dir"]
            return [("list_workspace", {})]
        if step == 3:
            ws = _last_tool_payload(req)["content"]
            prev_art_path = ws["published_artifacts"][0]["relative_path"]
            return [("copy_file", {"source_path": prev_art_path, "dest_path": "work/prev.docx"})]
        if step == 4:
            spec2 = {
                "body": {"line_spacing": 2.0},
                "margins_cm": {"top": 2.0, "bottom": 2.0, "left": 2.5, "right": 2.5},
            }
            return [("write_file", {"relative_path": "work/spec_v2.json", "content": json.dumps(spec2, ensure_ascii=False)})]
        if step == 5:
            return [(
                "execute_command",
                {
                    "command": "python",
                    "args": [
                        f"{state['skill_dir']}/scripts/format_docx.py",
                        "--input", "work/prev.docx",
                        "--output", "output/学术论文_排版后_v2.docx",
                        "--spec", "work/spec_v2.json",
                    ],
                },
            )]
        if step == 6:
            assert _last_tool_payload(req)["ok"] is True
            return [("publish_artifact", {"relative_path": "output/学术论文_排版后_v2.docx", "title": "排版后论文 v2"})]
        return "已基于上一轮排版结果调整为 2.0 倍行距和指定页边距，生成 v2 文件。"

    provider = ScriptedProvider(decide_word)
    with provider_server(provider) as base_url:
        async with live_skills_app(app_data, base_url) as (app, client):
            sid = (await client.post("/api/v1/sessions", json={"title": "Word连续排版"}, headers=UI_HEADERS)).json()["session_id"]
            imp = await client.post(
                f"/api/v1/_host/sessions/{sid}/workspace/files",
                json={"path": str(user_docx)},
                headers=HOST_HEADERS,
            )
            assert imp.status_code == 201
            file_id = imp.json()["file_id"]

            # Turn 1
            state["turn"] = "1"
            ev1 = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "将一级标题改为黑体三号，正文改为宋体小四，英文 Times New Roman，1.5倍行距，首行缩进2字符",
                "skills": {"enabled": True},
                "attachment_ids": [file_id],
            })
            assert any(e["type"] == "skill_activated" and e["data"]["name"] == "word-formatting" for e in ev1)
            assert ev1[-1]["type"] == "run_completed"
            art1_id = ev1[-1]["data"]["artifacts"][0]["artifact_id"]

            # Inspect Turn 1 published DOCX on disk
            _, art1_path = app.state.artifact_service.get_artifact(sid, art1_id)
            out1 = Document(str(art1_path))
            r_body = out1.paragraphs[1].runs[0]
            assert r_body._element.rPr.rFonts.get(qn("w:eastAsia")) == "宋体"
            assert r_body._element.rPr.rFonts.get(qn("w:ascii")) == "Times New Roman"
            assert round(r_body.font.size.pt, 1) == 12.0
            assert abs(out1.paragraphs[1].paragraph_format.line_spacing - 1.5) < 0.01
            assert user_docx.read_bytes() == orig_bytes  # Original file preserved!

            # Turn 2: continuous editing
            state["turn"] = "2"
            provider.requests.clear()
            ev2 = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "把上一轮生成的 Word 行距改成 2.0 倍，上下页边距 2cm，左右 2.5cm",
                "skills": {"enabled": True},
                "artifact_ids": [art1_id],
            })
            assert ev2[-1]["type"] == "run_completed"
            art2_id = ev2[-1]["data"]["artifacts"][0]["artifact_id"]
            assert art2_id != art1_id

            _, art2_path = app.state.artifact_service.get_artifact(sid, art2_id)
            out2 = Document(str(art2_path))
            assert abs(out2.paragraphs[1].paragraph_format.line_spacing - 2.0) < 0.01
            assert round(out2.sections[0].top_margin.cm, 2) == 2.0
            assert round(out2.sections[0].left_margin.cm, 2) == 2.5
            # Preserved Turn 1 font!
            assert out2.paragraphs[1].runs[0]._element.rPr.rFonts.get(qn("w:eastAsia")) == "宋体"

            # Verify RunSummary history includes both runs' artifacts & skill activations
            runs_history = (await client.get(f"/api/v1/sessions/{sid}/runs", headers=UI_HEADERS)).json()["items"]
            assert len(runs_history) == 2
            assert all(len(r["artifacts"]) == 1 and len(r["skill_activations"]) == 1 for r in runs_history)


@pytest.mark.anyio
async def test_mind_map_skill_e2e_and_continuous_node_editing(tmp_path):
    """Test A + D: Real mind-map skill activation -> .mmd -> real SVG & PNG via mmdc+Chrome -> Turn 2 edit."""
    if not (CACHED_P92_RUNTIME / "npm-tools" / "node_modules" / ".bin" / "mmdc").is_file():
        pytest.skip("Pre-prepared mmdc not present in cached runtime")

    app_data = tmp_path / "app_data"
    state: dict[str, str] = {}

    def decide_mindmap(req: dict, step: int):
        if state.get("turn") == "1":
            if step == 1:
                return [("load_skill", {"skill_id": "builtin-mind-map"})]
            if step == 2:
                loaded = _last_tool_payload(req)["content"]
                state["skill_dir"] = loaded["skill_dir"]
                mmd = (
                    "mindmap\n"
                    "  root((AI Agent 核心架构))\n"
                    "    LangChain\n"
                    "      模型抽象与消息结构\n"
                    "      工具 Schema 绑定\n"
                    "    LangGraph\n"
                    "      状态图与循环编排\n"
                    "      Checkpoint 持久化\n"
                    "    RAG\n"
                    "      本地教材混合检索\n"
                    "      引用证据溯源\n"
                    "    MCP\n"
                    "      外部工具协议接入\n"
                    "      服务授权与隔离\n"
                    "    Skills\n"
                    "      标准 SKILL.md 渐进加载\n"
                    "      工作区与脚本执行交付\n"
                )
                return [("write_file", {"relative_path": "output/agent_relations.mmd", "content": mmd})]
            if step == 3:
                return [(
                    "execute_command",
                    {
                        "command": "python",
                        "args": [
                            f"{state['skill_dir']}/scripts/render_mindmap.py",
                            "--input", "output/agent_relations.mmd",
                            "--output-svg", "output/agent_relations.svg",
                            "--output-png", "output/agent_relations.png",
                            "--config", f"{state['skill_dir']}/references/mermaid-config.json",
                        ],
                    },
                )]
            if step == 4:
                res = _last_tool_payload(req)
                assert res["ok"] is True, res
                rendered = json.loads(res["content"]["stdout"])
                assert rendered["ok"] is True and rendered["svg_bytes"] > 1000 and rendered["png_bytes"] > 1000
                return [
                    ("publish_artifact", {"relative_path": "output/agent_relations.mmd", "title": "关系思维导图源文件"}),
                    ("publish_artifact", {"relative_path": "output/agent_relations.svg", "title": "关系思维导图 SVG"}),
                    ("publish_artifact", {"relative_path": "output/agent_relations.png", "title": "关系思维导图 PNG"}),
                ]
            return "已为您制作关于 LangChain、LangGraph、RAG、MCP 和 Skills 关系的中文思维导图，并输出了 .mmd、.svg 和 .png 产物。"

        # Turn 2: read previous published .mmd artifact, add a new branch, re-render and publish v2
        if step == 1:
            return [("load_skill", {"skill_id": "mind-map"})]
        if step == 2:
            state["skill_dir"] = _last_tool_payload(req)["content"]["skill_dir"]
            return [("list_workspace", {})]
        if step == 3:
            ws = _last_tool_payload(req)["content"]
            mmd_art = next(a for a in ws["published_artifacts"] if a["filename"].endswith(".mmd"))
            return [("read_file", {"relative_path": mmd_art["relative_path"], "scope": "session"})]
        if step == 4:
            prev_mmd = _last_tool_payload(req)["content"]["content"]
            updated_mmd = prev_mmd.rstrip() + "\n    Workspace 与 Artifact\n      会话级文件工作区\n      版本化产物交付\n"
            return [("write_file", {"relative_path": "output/agent_relations_v2.mmd", "content": updated_mmd})]
        if step == 5:
            return [(
                "execute_command",
                {
                    "command": "python",
                    "args": [
                        f"{state['skill_dir']}/scripts/render_mindmap.py",
                        "--input", "output/agent_relations_v2.mmd",
                        "--output-svg", "output/agent_relations_v2.svg",
                        "--output-png", "output/agent_relations_v2.png",
                    ],
                },
            )]
        if step == 6:
            assert _last_tool_payload(req)["ok"] is True
            return [
                ("publish_artifact", {"relative_path": "output/agent_relations_v2.mmd", "title": "思维导图源文件 v2"}),
                ("publish_artifact", {"relative_path": "output/agent_relations_v2.svg", "title": "思维导图 SVG v2"}),
                ("publish_artifact", {"relative_path": "output/agent_relations_v2.png", "title": "思维导图 PNG v2"}),
            ]
        return "已在上一轮思维导图基础上补充了 Workspace 与 Artifact 分支并生成 v2 产物。"

    provider = ScriptedProvider(decide_mindmap)
    with provider_server(provider) as base_url:
        async with live_skills_app(app_data, base_url) as (app, client):
            sid = (await client.post("/api/v1/sessions", json={"title": "思维导图测试"}, headers=UI_HEADERS)).json()["session_id"]

            state["turn"] = "1"
            ev1 = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "请帮我制作一张关于 LangChain、LangGraph、RAG、MCP 和 Skills 之间关系的中文思维导图。",
                "skills": {"enabled": True},
            })
            assert any(e["type"] == "skill_activated" and e["data"]["name"] == "mind-map" for e in ev1)
            assert ev1[-1]["type"] == "run_completed"
            arts1 = ev1[-1]["data"]["artifacts"]
            assert {Path(a["filename"]).suffix for a in arts1} == {".mmd", ".svg", ".png"}

            # Turn 2: continuous edit
            state["turn"] = "2"
            provider.requests.clear()
            ev2 = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "请在刚才的思维导图中增加 Workspace 与 Artifact 分支并重新生成。",
                "skills": {"enabled": True},
            })
            assert ev2[-1]["type"] == "run_completed"
            arts2 = ev2[-1]["data"]["artifacts"]
            assert {Path(a["filename"]).suffix for a in arts2} == {".mmd", ".svg", ".png"}
            svg2 = next(a for a in arts2 if a["filename"].endswith(".svg"))
            _, svg2_bytes = app.state.artifact_service.read_artifact_bytes(sid, svg2["artifact_id"])
            assert "Workspace 与 Artifact" in svg2_bytes.decode("utf-8")


@pytest.mark.anyio
async def test_plain_chat_isolation_and_mcp_egress_boundary(tmp_path):
    """Test E: Plain chat exposes zero Skills tools; Skills + MCP enforces existing egress confirmation
    and collaborates smoothly once external data egress is explicitly authorized."""
    app_data = tmp_path / "app_data"
    stdio_fixture = Path(__file__).resolve().parent / "fixtures" / "mcp" / "stdio_test_server.py"

    def decide(req: dict, step: int):
        if not req.get("tools"):
            return "这是普通聊天回答。"
        # When both MCP and Skills are enabled with egress authorized:
        # Step 1: call MCP echo_text tool; Step 2: write result to workspace & publish artifact!
        if step == 1:
            mcp_alias = next(t["function"]["name"] for t in req["tools"] if t["function"]["name"].startswith("mcp__"))
            return [(mcp_alias, {"text": "MCP 检索得到的知识摘要"})]
        if step == 2:
            mcp_data = _last_tool_payload(req)["content"]
            return [("write_file", {"relative_path": "output/mcp_summary.md", "content": f"# MCP 汇总\n\n{mcp_data}\n"})]
        if step == 3:
            return [("publish_artifact", {"relative_path": "output/mcp_summary.md", "title": "MCP 汇总文档"})]
        return "已通过 MCP 获取信息并生成 Markdown 产物。"

    provider = ScriptedProvider(decide)
    with provider_server(provider) as base_url:
        async with live_skills_app(app_data, base_url) as (app, client):
            sid = (await client.post("/api/v1/sessions", json={"title": "隔离与协作测试"}, headers=UI_HEADERS)).json()["session_id"]

            # 1. Ordinary chat (skills omitted) -> no tools exposed
            ev = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "你好，介绍一下你自己",
            })
            assert ev[-1]["type"] == "run_completed"
            assert "artifacts" not in ev[-1]["data"]

            # 2. Setup a trusted local stdio MCP server with echo_text allowed
            srv = (await client.post(
                "/api/v1/_host/mcp/servers",
                json={"name": "EchoMcp", "transport": "stdio", "command": sys.executable, "args": [str(stdio_fixture)]},
                headers=HOST_HEADERS,
            )).json()
            srv_id = srv["server_id"]
            await client.post(
                f"/api/v1/_host/mcp/servers/{srv_id}/trust",
                json={
                    "expected_config_revision": srv["config_revision"],
                    "expected_fingerprint": srv["current_fingerprint"],
                    "enable": True,
                },
                headers=HOST_HEADERS,
            )
            assert (await client.post(f"/api/v1/_host/mcp/servers/{srv_id}/connect", headers=HOST_HEADERS)).status_code == 200
            tools_list = (await client.get(f"/api/v1/mcp/servers/{srv_id}/tools", headers=UI_HEADERS)).json()
            echo_tool = next(t for t in tools_list if t["original_tool_name"] == "echo_text")
            await client.patch(
                f"/api/v1/_host/mcp/servers/{srv_id}/tools/{echo_tool['tool_id']}",
                json={
                    "permission_state": "allowed",
                    "expected_definition_hash": echo_tool["definition_hash"],
                    "expected_config_revision": srv["config_revision"],
                },
                headers=HOST_HEADERS,
            )

            # 3. Skills + MCP WITHOUT egress authorization -> blocked with 403 MCP_EGRESS_AUTHORIZATION_REQUIRED
            blocked = await client.post(
                f"/api/v1/sessions/{sid}/runs/stream",
                json={
                    "client_request_id": str(uuid4()),
                    "message": "用 MCP 查询并保存文件",
                    "mcp": {"enabled": True, "allow_external_data_egress": False},
                    "skills": {"enabled": True},
                },
                headers=UI_HEADERS,
            )
            assert blocked.status_code == 403
            assert blocked.json()["error"]["code"] == "MCP_EGRESS_AUTHORIZATION_REQUIRED"

            # 4. Skills + MCP WITH egress authorization -> collaborates end-to-end and publishes artifact!
            provider.requests.clear()
            collab_ev = await _stream_run(client, sid, {
                "client_request_id": str(uuid4()),
                "message": "用 MCP 查询并保存为 Markdown 文件",
                "mcp": {"enabled": True, "allow_external_data_egress": True},
                "skills": {"enabled": True},
            })
            assert collab_ev[-1]["type"] == "run_completed"
            assert collab_ev[-1]["data"]["artifacts"][0]["filename"] == "mcp_summary.md"
