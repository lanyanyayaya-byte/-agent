"""Agent tool for executing Python, Node.js, Shell, and CLI commands via RuntimeExecutor."""

from __future__ import annotations

import asyncio
import shlex
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.runtime.executor import DEFAULT_TIMEOUT_SECONDS, ExecutionRequest, RuntimeExecutor
from app.tools.protocol import ToolCall, ToolResult
from app.workspace.service import WorkspaceError, WorkspaceService, resolve_within


class ExecuteCommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Annotated[str, Field(min_length=1, max_length=8192)]
    args: Annotated[list[Annotated[str, Field(max_length=4096)]], Field(max_length=64)] = Field(default_factory=list)
    cwd: Annotated[str, Field(max_length=512)] = "."
    timeout_seconds: Annotated[float, Field(ge=1.0, le=600.0)] = DEFAULT_TIMEOUT_SECONDS
    use_shell: bool = False


class RuntimeToolContext:
    """Executes commands inside `runs/<run_id>/` using the managed independent RuntimeExecutor."""

    def __init__(
        self,
        *,
        executor: RuntimeExecutor,
        workspace: WorkspaceService,
        session_id: str,
        run_id: str,
    ) -> None:
        self.executor = executor
        self.workspace = workspace
        self.session_id = session_id
        self.run_id = run_id

    async def execute_command(self, call: ToolCall, args: ExecuteCommandArgs) -> ToolResult:
        await asyncio.to_thread(self.workspace.ensure_run_dirs, self.session_id, self.run_id)
        run_root = self.workspace.run_root(self.session_id, self.run_id)
        try:
            cwd_path = resolve_within(run_root, args.cwd)
            if not cwd_path.is_dir():
                raise WorkspaceError("WORKSPACE_PATH_INVALID", f"工作目录不是有效目录：{args.cwd}")
        except WorkspaceError as exc:
            return ToolResult.failure(
                call.tool_call_id,
                exc.code,
                display_summary=exc.message,
                content={"error": exc.message, "cwd": args.cwd},
                tool_category="runtime",
                recoverable=True,
                error_signature=f"execute_command:{exc.code}:{args.cwd}",
            )

        command = args.command
        cmd_args = tuple(args.args)
        if not args.use_shell and not cmd_args and any(ch.isspace() for ch in command.strip()):
            try:
                parts = shlex.split(command)
                if parts:
                    command = parts[0]
                    cmd_args = tuple(parts[1:])
            except ValueError:
                pass

        res = await self.executor.execute(
            ExecutionRequest(
                run_id=self.run_id,
                command=command,
                args=cmd_args,
                cwd=cwd_path,
                timeout_seconds=args.timeout_seconds,
                use_shell=args.use_shell,
            )
        )

        payload: dict[str, Any] = {
            "execution_id": res.execution_id,
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "duration_ms": res.duration_ms,
            "timed_out": res.timed_out,
            "output_truncated": res.output_truncated,
        }

        if res.ok:
            summary = f"执行 `{args.command}` 成功（{res.duration_ms}ms）。"
            return ToolResult(
                call.tool_call_id,
                True,
                payload,
                summary,
                truncated=res.output_truncated,
                duration_ms=res.duration_ms,
                tool_category="runtime",
            )

        # Build a meaningful normalized signature so distinct errors do NOT accumulate together,
        # while the exact same repeated error stops after 3 consecutive tries.
        first_err_line = next(
            (line.strip() for line in reversed(res.stderr.splitlines()) if line.strip()),
            (res.error_message or ""),
        )[:120]
        signature = f"execute_command:{res.error_code}:{res.exit_code}:{args.command}:{first_err_line}"
        return ToolResult.failure(
            call.tool_call_id,
            res.error_code or "RUNTIME_NON_ZERO_EXIT",
            display_summary=res.error_message or f"命令 `{args.command}` 执行失败（exit={res.exit_code}）。",
            content=payload,
            duration_ms=res.duration_ms,
            tool_category="runtime",
            recoverable=not res.cancelled,
            error_signature=signature,
        )


def runtime_tool_schemas() -> tuple[dict[str, Any], ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": (
                    "在独立本地运行环境中真实执行 Python、Node.js、Mermaid CLI 或 Shell 命令。"
                    "默认工作目录为当前任务根目录（可直接访问 work/... 与 output/...），"
                    "command='python' 自动使用独立 Python 虚拟环境；仅在需要管道或 Shell 语法时将 use_shell 设为 true。"
                ),
                "parameters": ExecuteCommandArgs.model_json_schema(),
            },
        },
    )
