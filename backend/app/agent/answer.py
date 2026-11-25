import asyncio
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langsmith import tracing_context

from app.agent.tool_graph import build_tool_graph, paired_tool_calls
from app.models import ModelError
from app.tools.registry import RunTools

EMPTY_CHECKPOINT_ID = "00000000-0000-0000-0000-000000000000"


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    checkpoint_id: str | None


def _complete_text(messages: Sequence[BaseMessage]) -> str | None:
    reply = messages[-1] if messages else None
    if isinstance(reply, AIMessage) and paired_tool_calls(messages):
        if reply.response_metadata.get("finish_reason") == "stop" and reply.text.strip():
            return reply.text
    return None


async def answer_with_agent(
    model,
    message: str,
    *,
    on_delta: Callable[[str], None] | None = None,
    on_reset_draft: Callable[[], None] | None = None,
    checkpointer: AsyncSqliteSaver | None = None,
    config: RunnableConfig | None = None,
    tools: RunTools | None = None,
    system_prompt: str | None = None,
) -> AgentAnswer:
    from langchain.agents import create_agent

    # The graph reducer appends this turn to the exact persisted base when configured.
    inputs = {"messages": [HumanMessage(content=message)]}
    with tracing_context(enabled=False):
        if checkpointer is not None:
            graph = (build_tool_graph(model, tools, checkpointer, on_delta, system_prompt=system_prompt,
                                      on_reset_draft=on_reset_draft) if tools is not None
                     else create_agent(model=model, tools=[], system_prompt=system_prompt, checkpointer=checkpointer))
            try:
                return await _checkpointed_answer(graph, inputs, config, on_delta, tools)
            except (asyncio.CancelledError, TimeoutError, ModelError):
                raise
            except sqlite3.Error:
                raise ModelError("INTERNAL_ERROR", "无法保存或读取对话检查点。") from None
            except Exception:
                if tools is None:
                    raise ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 调用失败或响应无效。") from None
                raise

        graph = create_agent(model=model, tools=[], system_prompt=system_prompt, checkpointer=checkpointer)
        if on_delta is None:
            state = await graph.ainvoke(inputs)
        else:
            state = {}
            async with aclosing(graph.astream(inputs, stream_mode=["messages", "values"])) as stream:
                async for mode, event in stream:
                    if mode == "values":
                        state = event
                    else:
                        _emit_delta(event, on_delta)

    return AgentAnswer(_require_reply(state.get("messages", ()), on_delta), None)


async def _checkpointed_answer(graph, inputs, config, on_delta, tools: RunTools | None) -> AgentAnswer:
    base_id = get_checkpoint_id(config) if config else None
    if not base_id:
        raise ModelError("INTERNAL_ERROR", "对话缺少明确的检查点起点。")
    if base_id != EMPTY_CHECKPOINT_ID:
        base = await graph.aget_state(config)
        if base.created_at is None or base.next or base.interrupts or not _complete_text(base.values.get("messages", ())):
            raise ModelError("INTERNAL_ERROR", "无法读取有效的对话安全检查点。")

    final_config = None
    modes = ["checkpoints"] if tools is not None else ["messages", "checkpoints"]
    async with aclosing(graph.astream(
        inputs, config, stream_mode=modes, durability="sync",
    )) as stream:
        async for mode, event in stream:
            if mode == "checkpoints":
                final_config = event["config"]
            else:
                _emit_delta(event, on_delta)

    if tools is not None and tools.failure is not None:
        raise ModelError(tools.failure.error_code, tools.failure.display_summary)
    final_id = get_checkpoint_id(final_config) if final_config else None
    if not final_id or final_id == base_id:
        raise ModelError("INTERNAL_ERROR", "本次生成缺少有效的对话检查点。")

    # Stream events provide the ID. Re-read that exact checkpoint after all writes have settled.
    final = await graph.aget_state(final_config)
    if final.created_at is None or final.next or final.interrupts:
        raise ModelError("INTERNAL_ERROR", "本次生成缺少完整的对话检查点。")
    return AgentAnswer(_require_reply(final.values.get("messages", ()), on_delta), final_id)


def _emit_delta(event, on_delta: Callable[[str], None] | None) -> None:
    chunk, _metadata = event
    if on_delta is not None and isinstance(chunk, AIMessageChunk) and chunk.text:
        on_delta(chunk.text)


def _require_reply(messages: Sequence[BaseMessage], on_delta: Callable[[str], None] | None) -> str:
    reply = _complete_text(messages)
    if reply is None:
        raise ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 未返回完整文本回答。")
    return reply.strip() if on_delta is None else reply
