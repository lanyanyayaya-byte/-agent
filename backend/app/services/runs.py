import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.answer import EMPTY_CHECKPOINT_ID, AgentAnswer, answer_with_agent
from app.agent.budgets import LEGACY_MAX_RUN_SECONDS, SKILLS_MAX_RUN_SECONDS
from app.diagnostics import record
from app.models import ModelError
from app.services.grants import GrantRegistry
from app.services.hybrid_retrieval import CoverageMode, RetrievalMode
from app.services.knowledge_qa import KnowledgeQAService
from app.services.models import ModelService
from app.storage.business import (
    INTERRUPTED_CODE, TERMINAL_STATUSES, BusinessStore, Run, RunStatus, Session, StoredCitation, utc_now,
)
from app.storage.checkpoints import open_checkpointer
from app.tools.protocol import ERROR_MESSAGES, ToolEvent
from app.tools.registry import RunToolSnapshot, RunTools

if TYPE_CHECKING:
    from app.runtime.executor import RuntimeExecutor
    from app.services.mcp import McpService
    from app.skills.manager import SkillManager
    from app.workspace.artifacts import ArtifactService
    from app.workspace.service import WorkspaceService

DEFAULT_MAX_RUN_SECONDS = LEGACY_MAX_RUN_SECONDS
DEFAULT_SKILLS_MAX_RUN_SECONDS = SKILLS_MAX_RUN_SECONDS


@dataclass(frozen=True, slots=True)
class KnowledgeRunConfig:
    document_ids: tuple[str, ...] | None = None
    retrieval_mode: RetrievalMode = "hybrid"
    coverage_mode: CoverageMode = "global"
    top_k: int = 6


@dataclass(frozen=True, slots=True)
class McpRunConfig:
    enabled: bool = False
    allow_external_data_egress: bool = False


@dataclass(frozen=True, slots=True)
class SkillsRunConfig:
    enabled: bool = False
    selected_skill_ids: tuple[str, ...] = ()
    attachment_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()


class RunServiceStopping(Exception):
    pass


class RunNotCancellable(Exception):
    pass


@dataclass(frozen=True)
class RunEvent:
    type: Literal[
        "run_started", "message_delta", "run_citations",
        "skill_activated", "artifact_created",
        "run_completed", "error", "run_cancelled",
    ] | ToolEvent
    data: dict[str, Any]
    ts: str = field(default_factory=utc_now)


@dataclass
class RunStream:
    run: Run
    events: asyncio.Queue[RunEvent | None] = field(default_factory=asyncio.Queue)
    observed_text: list[str] = field(default_factory=list)
    citations: tuple[StoredCitation, ...] = ()
    tool_snapshot: RunToolSnapshot | None = None
    _observing: bool = True

    def publish(self, event: RunEvent | None) -> None:
        if self._observing:
            self.events.put_nowait(event)

    def detach(self) -> None:
        self._observing = False

    def on_delta(self, delta: str) -> None:
        self.observed_text.append(delta)
        self.publish(RunEvent("message_delta", {"channel": "text", "delta": delta}))

    def reset_observed_text(self) -> None:
        self.observed_text.clear()

    def on_tool_event(self, event: ToolEvent, data: dict[str, Any]) -> None:
        if event == "tool_started" and self.observed_text:
            self.observed_text.clear()
        self.publish(RunEvent(event, data))

    def on_custom_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.publish(RunEvent(event_type, data))  # type: ignore[arg-type]



@dataclass
class Execution:
    run: Run
    model: asyncio.Task[AgentAnswer]
    settled: asyncio.Event = field(default_factory=asyncio.Event)


class RunService:
    def __init__(
        self,
        store: BusinessStore,
        models: ModelService,
        grants: GrantRegistry,
        *,
        max_run_seconds: float = DEFAULT_MAX_RUN_SECONDS,
        skills_max_run_seconds: float = DEFAULT_SKILLS_MAX_RUN_SECONDS,
        mcp_tool_timeout_seconds: float | None = None,
    ):
        self._store = store
        self._models = models
        self._grants = grants
        self.knowledge_qa: KnowledgeQAService | None = None
        self.mcp_service: McpService | None = None
        self.workspace_service: WorkspaceService | None = None
        self.artifact_service: ArtifactService | None = None
        self.runtime_executor: RuntimeExecutor | None = None
        self.skill_manager: SkillManager | None = None
        self._max_run_seconds = max_run_seconds
        self._skills_max_run_seconds = skills_max_run_seconds
        self._mcp_tool_timeout_seconds = mcp_tool_timeout_seconds
        self._checkpointer: AsyncSqliteSaver | None = None
        self._stopping = False
        self._workers: set[asyncio.Task] = set()
        self._executions: dict[str, Execution] = {}
        self._transitions = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(self, checkpoint_path: Path):
        async with open_checkpointer(checkpoint_path) as checkpointer:
            self._checkpointer = checkpointer
            self._stopping = False
            try:
                yield
            finally:
                await self.close()
                self._checkpointer = None

    async def create_session(self, title: str) -> Session:
        self._require_running()
        return await asyncio.to_thread(self._store.create_session, title)

    async def rename_session(self, session_id: str, title: str) -> Session:
        self._require_running()
        return await asyncio.to_thread(self._store.rename_session, session_id, title)

    async def delete_session(self, session_id: str) -> None:
        self._require_running()
        await asyncio.to_thread(self._store.delete_session, session_id)
        if self.workspace_service is not None:
            await asyncio.to_thread(self.workspace_service.delete_session_workspace, session_id)

    async def get_run(self, run_id: str) -> RunStatus:
        return await asyncio.to_thread(self._store.get_run, run_id)

    async def lookup_run(self, session_id: str, client_request_id: str) -> RunStatus:
        return await asyncio.to_thread(self._store.lookup_run, session_id, client_request_id)

    async def start(
        self, session_id: str, client_request_id: str, message: str, *, request_id: str,
        knowledge: KnowledgeRunConfig | None = None,
        mcp: McpRunConfig | None = None,
        skills: SkillsRunConfig | None = None,
    ) -> RunStream:
        self._require_running()
        accepted = asyncio.get_running_loop().create_future()
        worker = asyncio.create_task(
            self._run(session_id, client_request_id, message, accepted, request_id, knowledge, mcp, skills),
        )
        self._workers.add(worker)
        worker.add_done_callback(self._workers.discard)
        return await accepted

    async def cancel(self, run_id: str) -> RunStatus:
        self._require_running()
        # Finish dispatching cancellation even if the HTTP caller disconnects during the write.
        worker = asyncio.create_task(self._cancel(run_id))
        self._workers.add(worker)
        worker.add_done_callback(self._cancel_finished)
        return await asyncio.shield(worker)

    def _cancel_finished(self, worker: asyncio.Task) -> None:
        self._workers.discard(worker)
        if not worker.cancelled():
            worker.exception()  # Also retrieve exceptions if the HTTP caller has gone away.

    async def _cancel(self, run_id: str) -> RunStatus:
        async with self._transitions:
            execution = self._executions.get(run_id)
            if execution and not execution.model.done():
                status = await asyncio.to_thread(self._store.request_cancel, execution.run)
                if status.status == "cancelling" and not execution.model.cancelling():
                    execution.model.cancel()
                return status
        if execution:
            # A completed model task may still be committing its result.
            await execution.settled.wait()
        status = await self.get_run(run_id)
        if status.status not in TERMINAL_STATUSES:
            raise RunNotCancellable
        return status

    def _require_running(self) -> None:
        if self._stopping or self._checkpointer is None:
            raise RunServiceStopping

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        for execution in self._executions.values():
            if not execution.model.cancelling():
                execution.model.cancel()

    async def close(self) -> None:
        self.stop()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def _answer(
        self,
        run: Run,
        message: str,
        stream: RunStream,
        knowledge: KnowledgeRunConfig | None = None,
    ) -> AgentAnswer:
        timeout_s = (
            self._skills_max_run_seconds
            if (stream.tool_snapshot is not None and stream.tool_snapshot.has_skills_tools)
            else self._max_run_seconds
        )
        try:
            async with asyncio.timeout(timeout_s):
                return await self._answer_inner(run, message, stream, knowledge, stream.tool_snapshot)
        except TimeoutError:
            raise ModelError("RUN_TIMEOUT", ERROR_MESSAGES["RUN_TIMEOUT"]) from None

    async def _answer_inner(
        self,
        run: Run,
        message: str,
        stream: RunStream,
        knowledge: KnowledgeRunConfig | None = None,
        tool_snapshot: RunToolSnapshot | None = None,
    ) -> AgentAnswer:
        system_prompt: str | None = None
        if knowledge is not None:
            qa = self.knowledge_qa
            call_id = str(uuid4())
            await asyncio.to_thread(self._store.start_tool, run, call_id, "search_knowledge")
            stream.on_tool_event("tool_started", {"tool_call_id": call_id, "tool_name": "search_knowledge"})
            if qa is None:
                msg = "教材知识库暂不可用。"
                await asyncio.to_thread(
                    self._store.finish_tool, run, call_id, "failed", msg, "KNOWLEDGE_STORE_UNAVAILABLE",
                )
                stream.on_tool_event("tool_failed", {
                    "tool_call_id": call_id, "code": "KNOWLEDGE_STORE_UNAVAILABLE", "message": msg,
                })
                raise ModelError("KNOWLEDGE_STORE_UNAVAILABLE", msg)
            try:
                history_messages = await asyncio.to_thread(self._store.list_messages, run.session_id)
                recent_questions = [
                    m.content for m in history_messages
                    if m.role == "user" and m.run_id != run.run_id
                ][-3:]
                rag_ctx = await qa.prepare_chat_context(
                    message,
                    recent_questions,
                    document_ids=knowledge.document_ids,
                    retrieval_mode=knowledge.retrieval_mode,
                    coverage_mode=knowledge.coverage_mode,
                    top_k=knowledge.top_k,
                )
            except asyncio.CancelledError:
                await asyncio.to_thread(
                    self._store.finish_tool, run, call_id, "cancelled", "教材检索已取消。", "RUN_CANCELLED",
                )
                stream.on_tool_event("tool_failed", {
                    "tool_call_id": call_id, "code": "RUN_CANCELLED", "message": "教材检索已取消。",
                })
                raise
            except ModelError as error:
                await asyncio.to_thread(self._store.finish_tool, run, call_id, "failed", str(error), error.code)
                stream.on_tool_event("tool_failed", {
                    "tool_call_id": call_id, "code": error.code, "message": str(error),
                })
                raise
            except Exception:
                msg = "教材知识库检索失败。"
                await asyncio.to_thread(
                    self._store.finish_tool, run, call_id, "failed", msg, "HYBRID_QUERY_FAILED",
                )
                stream.on_tool_event("tool_failed", {
                    "tool_call_id": call_id, "code": "HYBRID_QUERY_FAILED", "message": msg,
                })
                raise ModelError("HYBRID_QUERY_FAILED", msg) from None

            await asyncio.to_thread(self._store.finish_tool, run, call_id, "completed", rag_ctx.summary, None)
            stream.on_tool_event("tool_completed", {"tool_call_id": call_id, "summary": rag_ctx.summary})
            stream.citations = rag_ctx.citations
            stream.publish(RunEvent("run_citations", {
                "citations": [asdict(c) for c in rag_ctx.citations],
                "retrieval_query": rag_ctx.retrieval_query,
                "coverage_mode": rag_ctx.coverage_mode,
                "channel_status": rag_ctx.channel_status,
            }))
            system_prompt = rag_ctx.system_prompt

        tools = RunTools(
            self._grants,
            self._store,
            run,
            stream.on_tool_event,
            mcp_service=self.mcp_service,
            skill_manager=self.skill_manager,
            workspace_service=self.workspace_service,
            artifact_service=self.artifact_service,
            runtime_executor=self.runtime_executor,
            emit_run_event=stream.on_custom_event,
            snapshot=tool_snapshot,
            mcp_tool_timeout_seconds=self._mcp_tool_timeout_seconds,
        )
        if system_prompt is None and not tools.has_tools:
            # Plain chat carries the configured assistant identity (production settings enable it).
            system_prompt = self._models.persona
        async with self._models.open_model() as model:
            return await answer_with_agent(
                model, message, on_delta=stream.on_delta, on_reset_draft=stream.reset_observed_text,
                checkpointer=self._checkpointer,
                tools=tools if tools.has_tools else None,
                system_prompt=system_prompt,
                config={
                    "configurable": {
                        "thread_id": f"session:{run.session_id}",
                        # A reserved, non-existent ID forces an empty root even after a failed first run.
                        "checkpoint_id": run.base_checkpoint_id or EMPTY_CHECKPOINT_ID,
                    },
                    "metadata": {"knowledgeagent_run_id": run.run_id},
                },
            )

    async def _run(
        self, session_id: str, client_request_id: str, message: str, accepted: asyncio.Future[RunStream],
        request_id: str, knowledge: KnowledgeRunConfig | None = None, mcp: McpRunConfig | None = None,
        skills: SkillsRunConfig | None = None,
    ) -> None:
        # The service owns admission and execution even if the requesting HTTP task disappears.
        mcp_cfg = mcp or McpRunConfig(enabled=False, allow_external_data_egress=False)
        skills_cfg = skills or SkillsRunConfig(enabled=False)
        try:
            async with self._transitions:
                self._require_running()
                snapshot: RunToolSnapshot | None = None
                precheck_error: ModelError | None = None
                if self._models.configured:
                    try:
                        session_egress_risk = (
                            await asyncio.to_thread(self._store.inspect_session_egress_risk, session_id)
                            if mcp_cfg.enabled
                            else None
                        )
                        snapshot = RunTools.build_snapshot(
                            self._grants,
                            UUID(session_id),
                            mcp_service=self.mcp_service,
                            mcp_enabled=mcp_cfg.enabled,
                            allow_external_data_egress=mcp_cfg.allow_external_data_egress,
                            knowledge_enabled=knowledge is not None,
                            session_egress_risk=session_egress_risk,
                            skill_manager=self.skill_manager,
                            skills_enabled=skills_cfg.enabled,
                            selected_skill_ids=skills_cfg.selected_skill_ids or None,
                            attachment_ids=skills_cfg.attachment_ids,
                            artifact_ids=skills_cfg.artifact_ids,
                        )
                    except ModelError as err:
                        precheck_error = err

                run = await asyncio.to_thread(
                    self._store.start_run, session_id, client_request_id, message,
                    accept_new_run=(self._models.configured and precheck_error is None),
                )
                if run is None:
                    if not self._models.configured:
                        raise ModelError("MODEL_NOT_CONFIGURED", "请先设置 DeepSeek API Key。")
                    assert precheck_error is not None
                    raise precheck_error
                stream = RunStream(run, tool_snapshot=snapshot)
                model = asyncio.create_task(
                    self._answer(run, message, stream) if knowledge is None
                    else self._answer(run, message, stream, knowledge)
                )
                execution = self._executions[run.run_id] = Execution(run, model)
                if self._stopping:
                    model.cancel()
        except Exception as error:
            if not accepted.done():
                accepted.set_exception(error)
            return

        started = monotonic()
        identifiers = dict(request_id=request_id, session_id=session_id, run_id=run.run_id,
                           client_request_id=client_request_id)
        record("run_started", **identifiers)
        if accepted.cancelled():
            stream.detach()
        else:
            stream.publish(RunEvent("run_started", {"client_request_id": client_request_id}))
            accepted.set_result(stream)
        try:
            event = await self._settle(execution, stream)
            if event:
                record("run_finished", **identifiers, status={
                    "run_completed": "completed", "error": "failed", "run_cancelled": "cancelled",
                }[event.type], error_code=event.data.get("code"),
                    elapsed_ms=round((monotonic() - started) * 1000))
            stream.publish(event)
        except Exception:
            # A storage failure may prevent recording any terminal state.
            record("run_commit_error", **identifiers, error_code="INTERNAL_ERROR")
        finally:
            del self._executions[run.run_id]
            execution.settled.set()
            stream.publish(None)

    async def _settle(
        self, execution: Execution, stream: RunStream,
    ) -> RunEvent | None:
        run = execution.run
        try:
            answer = await execution.model
            if answer.checkpoint_id is None:
                raise ModelError("INTERNAL_ERROR", "本次生成缺少对话检查点。")
            if answer.text != "".join(stream.observed_text):
                raise ModelError("MODEL_UPSTREAM_ERROR", "DeepSeek 的流式回答与最终结果不一致。")
        except asyncio.CancelledError:
            async with self._transitions:
                if await asyncio.to_thread(self._store.confirm_cancel, run):
                    return RunEvent("run_cancelled", {"reason": "user_requested"})
                if self._stopping and await asyncio.to_thread(self._store.confirm_cancel, run, shutdown=True):
                    return RunEvent("run_cancelled", {"reason": "shutdown"})
                return await self._fail(run, INTERRUPTED_CODE, "后端模型任务已中断。")
        except ModelError as error:
            async with self._transitions:
                return await self._fail(run, error.code, str(error))
        except Exception:
            async with self._transitions:
                return await self._fail(run, "INTERNAL_ERROR", "无法完成本次生成。")

        async with self._transitions:
            try:
                if stream.citations:
                    message_id = await asyncio.to_thread(
                        self._store.complete_run, run, answer.text, answer.checkpoint_id, stream.citations,
                    )
                else:
                    message_id = await asyncio.to_thread(
                        self._store.complete_run, run, answer.text, answer.checkpoint_id,
                    )
            except Exception:
                return await self._fail(run, "INTERNAL_ERROR", "无法完成本次生成。")
            if message_id is not None:
                completed_data: dict[str, Any] = {"message_id": message_id}
                if self.artifact_service is not None:
                    published_arts = await asyncio.to_thread(
                        self.artifact_service.list_artifacts,
                        run.session_id,
                        run_id=run.run_id,
                        include_non_published=False,
                    )
                    if published_arts:
                        completed_data["artifacts"] = [asdict(a) for a in published_arts]
                return RunEvent("run_completed", completed_data)
            # A cancellation accepted while the model was still active can win the database race.
            if await asyncio.to_thread(self._store.confirm_cancel, run):
                return RunEvent("run_cancelled", {"reason": "user_requested"})

    async def _fail(self, run: Run, code: str, message: str) -> RunEvent | None:
        if await asyncio.to_thread(self._store.fail_run, run, code, message):
            return RunEvent("error", {"code": code, "message": message})
