"""P9.7 stability unit tests: centralized budgets, DeepSeek error classification,
retry without re-executing completed tools, length-truncation recovery, and
format_docx.py index-based paragraph styling on all-Normal Chinese Word documents.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from docx import Document
from docx.oxml.ns import qn
from langchain_core.messages import AIMessage, AIMessageChunk
from openai import APIStatusError

from app.agent.budgets import (
    LEGACY_MAX_RUN_SECONDS,
    LEGACY_MAX_TOOL_CALLS,
    SKILLS_MAX_CATEGORY_TOOL_CALLS,
    SKILLS_MAX_LEGACY_TOOL_CALLS,
    SKILLS_MAX_RUN_SECONDS,
    SKILLS_MAX_TOTAL_TOOL_CALLS,
    SKILLS_RECURSION_LIMIT,
)
from app.agent.tool_graph import build_tool_graph
from app.models import ModelError, RetryableModelError
from app.models.deepseek import _translate_deepseek_error
from app.runtime.executor import RuntimeExecutor
from app.runtime.manager import RuntimeManager
from app.services.grants import GrantRegistry
from app.skills.manager import SkillManager
from app.storage.artifacts import WorkspaceArtifactStore
from app.storage.business import BusinessStore
from app.storage.checkpoints import open_checkpointer
from app.storage.skills import SkillStore
from app.tools.registry import RunTools
from app.workspace.artifacts import ArtifactService
from app.workspace.service import WorkspaceService


def test_p97_centralized_budget_values() -> None:
    assert LEGACY_MAX_TOOL_CALLS == 8
    assert LEGACY_MAX_RUN_SECONDS == 300.0
    assert SKILLS_MAX_TOTAL_TOOL_CALLS == 100
    assert SKILLS_MAX_CATEGORY_TOOL_CALLS == 80
    assert SKILLS_MAX_LEGACY_TOOL_CALLS == 20
    assert SKILLS_RECURSION_LIMIT == 250
    assert SKILLS_MAX_RUN_SECONDS == 1800.0


def test_deepseek_error_classification_retryable_vs_permanent() -> None:
    def _status_err(status: int, body: dict | str) -> APIStatusError:
        req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
        resp = httpx.Response(status, request=req)
        return APIStatusError(str(body), response=resp, body=body)

    e429 = _translate_deepseek_error(_status_err(429, {"error": {"message": "rate limit"}}))
    assert isinstance(e429, RetryableModelError)
    assert e429.reason_kind == "rate_limit"
    assert "429" in str(e429)

    e503 = _translate_deepseek_error(_status_err(503, "service unavailable"))
    assert isinstance(e503, RetryableModelError)
    assert e503.reason_kind == "server_error"
    assert "503" in str(e503)

    e_ctx = _translate_deepseek_error(_status_err(400, {"error": {"message": "maximum context length exceeded"}}))
    assert isinstance(e_ctx, ModelError) and not isinstance(e_ctx, RetryableModelError)
    assert "上下文" in str(e_ctx)

    e_fmt = _translate_deepseek_error(_status_err(400, {"error": {"message": "invalid parameter"}}))
    assert isinstance(e_fmt, ModelError) and not isinstance(e_fmt, RetryableModelError)
    assert "请求格式" in str(e_fmt)


@pytest.mark.anyio
async def test_tool_graph_retries_transient_error_and_recovers_length_without_reexecuting_tools(tmp_path: Path) -> None:
    app_data = tmp_path / "app_data"
    db = BusinessStore(app_data / "knowledgeagent.sqlite3")
    session = db.create_session("Retry & Length Recovery Test")
    run = db.start_run(session.session_id, str(uuid4()), "test", accept_new_run=True)
    assert run is not None

    runtime = RuntimeManager(app_data)
    ws_store = WorkspaceArtifactStore(db)
    ws = WorkspaceService(app_data, ws_store)
    arts = ArtifactService(ws, ws_store)
    skill_mgr = SkillManager(app_data, SkillStore(db), runtime)
    skill_mgr.sync_builtin_skills()

    grants = GrantRegistry()
    snapshot = RunTools.build_snapshot(
        grants,
        uuid4(),
        skill_manager=skill_mgr,
        skills_enabled=True,
    )
    tools = RunTools(
        grants,
        db,
        run,
        lambda _ev, _data: None,
        skill_manager=skill_mgr,
        workspace_service=ws,
        artifact_service=arts,
        runtime_executor=RuntimeExecutor(runtime),
        snapshot=snapshot,
    )

    model_attempts = 0
    ws_list_executions = 0
    orig_exec = tools._workspace_ctx.list_workspace

    async def counting_list_workspace(call, args):
        nonlocal ws_list_executions
        ws_list_executions += 1
        return await orig_exec(call, args)

    tools._workspace_ctx.list_workspace = counting_list_workspace  # type: ignore[method-assign]

    class FakeModel:
        _per_request_timeout = 1.0  # fast test backoff

        def bind_tools(self, _schemas):
            return self

        async def astream(self, messages, config=None, **kwargs):
            nonlocal model_attempts
            model_attempts += 1
            if model_attempts == 1:
                # Step 1: call list_workspace
                yield AIMessageChunk(
                    content="",
                    tool_calls=[{"id": "call_ws_1", "name": "list_workspace", "args": {}, "type": "tool_call"}],
                    response_metadata={"finish_reason": "tool_calls"},
                )
                return
            if model_attempts == 2:
                # Step 2: transient HTTP 429 -> must retry WITHOUT re-executing list_workspace!
                raise RetryableModelError("MODEL_UPSTREAM_ERROR", "HTTP 429", reason_kind="rate_limit", status_code=429)
            if model_attempts == 3:
                # Step 3: finish_reason == "length" -> must trigger length recovery in Skills Run!
                yield AIMessageChunk(
                    content="partial truncated...",
                    response_metadata={"finish_reason": "length"},
                )
                return
            # Step 4: after recovery reminder, return clean final answer
            assert any("8192" in str(getattr(m, "content", "")) for m in messages)
            yield AIMessageChunk(content="已完成任务。", response_metadata={"finish_reason": "stop"})

        async def ainvoke(self, messages, config=None, **kwargs):
            return AIMessage(content="已完成任务。", response_metadata={"finish_reason": "stop"})

    async with open_checkpointer(app_data / "checkpoints.sqlite3") as cp:
        graph = build_tool_graph(FakeModel(), tools, cp, None)
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "测试重试和截断恢复"}]},
            config={"configurable": {"thread_id": session.session_id}},
        )

    assert ws_list_executions == 1, "Completed tool MUST NOT be re-executed on model retry!"
    assert model_attempts == 4
    assert state["messages"][-1].content == "已完成任务。"


def test_word_formatting_scripts_support_all_normal_paragraphs_via_indices(tmp_path: Path) -> None:
    src = tmp_path / "leave_form.docx"
    out = tmp_path / "leave_form_formatted.docx"
    spec_path = tmp_path / "spec.json"

    doc = Document()
    doc.add_paragraph("学院硕博离校请假条")
    doc.add_paragraph("因为科研交流原因，我需要请假离校，遵守国家法律。")
    doc.add_paragraph("审批说明")
    doc.add_paragraph("导师签字：________")
    doc.save(src)

    spec_path.write_text(
        json.dumps(
            {
                "heading1_indices": [0],
                "heading2_indices": [2],
                "no_indent_indices": [3],
                "heading1": {"east_asia_font": "黑体", "western_font": "Times New Roman", "size": "三号", "line_spacing": 1.5, "first_line_indent_chars": 0},
                "heading2": {"east_asia_font": "黑体", "western_font": "Times New Roman", "size": "四号", "line_spacing": 1.5, "first_line_indent_chars": 0},
                "body": {"east_asia_font": "宋体", "western_font": "Times New Roman", "size": "小四", "line_spacing": 1.5, "first_line_indent_chars": 2},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    scripts_dir = Path("/Users/lanyan/Documents/项目/agent_1/backend/app/skills/builtin/word-formatting/scripts")
    r_fmt = subprocess.run(
        [sys.executable, str(scripts_dir / "format_docx.py"), "--input", str(src), "--output", str(out), "--spec", str(spec_path)],
        capture_output=True, text=True, check=True,
    )
    fmt_info = json.loads(r_fmt.stdout)
    assert fmt_info["updated"]["heading1"] == 1
    assert fmt_info["updated"]["heading2"] == 1
    assert fmt_info["updated"]["body"] == 2

    r_val = subprocess.run(
        [sys.executable, str(scripts_dir / "validate_docx.py"), "--original", str(src), "--formatted", str(out), "--spec", str(spec_path)],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(r_val.stdout)["validated"] is True

    formatted = Document(out)
    assert formatted.paragraphs[0].runs[0]._element.rPr.rFonts.get(qn("w:eastAsia")) == "黑体"
    assert round(formatted.paragraphs[0].runs[0].font.size.pt, 1) == 16.0
    assert formatted.paragraphs[2].runs[0]._element.rPr.rFonts.get(qn("w:eastAsia")) == "黑体"
    assert round(formatted.paragraphs[2].runs[0].font.size.pt, 1) == 14.0
    assert formatted.paragraphs[1].runs[0]._element.rPr.rFonts.get(qn("w:eastAsia")) == "宋体"
    assert formatted.paragraphs[3].paragraph_format.first_line_indent.pt == 0
