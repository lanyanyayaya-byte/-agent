"""Async subprocess executor for Python, Node.js, Shell, and general CLI commands.

Features:
- Uses `argv` array by default (`create_subprocess_exec`), with explicit `use_shell=True`
  opt-in when shell pipelines/syntax are needed.
- Maps `python`/`python3` to the managed independent runtime venv (never PyInstaller frozen binary).
- Isolates child environment from PyInstaller dylib vars and model/internal credentials.
- Bounded async stdout/stderr collection with guaranteed stderr headroom and non-blocking pipe drain.
- POSIX process-group teardown (`SIGTERM` -> `SIGKILL`) on timeout or cancellation.
- Global concurrency cap via `asyncio.Semaphore`.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from uuid import uuid4

from app.runtime.manager import RuntimeError_, RuntimeManager

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_OUTPUT_BYTES = 24 * 1024  # 24 KiB combined visible output
DEFAULT_MAX_CONCURRENCY = 2
SIGTERM_GRACE_SECONDS = 1.5
READ_CHUNK_BYTES = 4096


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    run_id: str | None
    command: str
    args: tuple[str, ...] = ()
    cwd: Path | str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    use_shell: bool = False
    shell_executable: str = "/bin/zsh"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    execution_id: str
    command: str
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    cancelled: bool = False
    output_truncated: bool = False
    error_code: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.cancelled
            and self.error_code is None
        )


async def _read_bounded_stream(
    stream: asyncio.StreamReader | None,
    *,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Read stream to EOF, keeping at most `max_bytes` while draining excess to avoid pipe deadlock."""
    if stream is None:
        return b"", False
    buf = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        remaining = max_bytes - len(buf)
        if remaining > 0:
            buf.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        else:
            truncated = True
    return bytes(buf), truncated


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate an entire POSIX process group cleanly (SIGTERM -> grace wait -> SIGKILL)."""
    if proc.returncode is not None:
        return
    pid = proc.pid
    if pid is not None and os.name == "posix":
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pid, signal.SIGTERM)
    else:
        with suppress(ProcessLookupError, OSError):
            proc.terminate()

    try:
        await asyncio.wait_for(proc.wait(), timeout=SIGTERM_GRACE_SECONDS)
        return
    except TimeoutError:
        pass

    if pid is not None and os.name == "posix":
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pid, signal.SIGKILL)
    else:
        with suppress(ProcessLookupError, OSError):
            proc.kill()
    with suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=2.0)


class RuntimeExecutor:
    """Executes Python, Node.js, Shell, and CLI commands inside the managed runtime."""

    def __init__(
        self,
        manager: RuntimeManager,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_timeout_seconds: float = MAX_TIMEOUT_SECONDS,
    ) -> None:
        self.manager = manager
        self.default_timeout_seconds = default_timeout_seconds
        self.max_timeout_seconds = max_timeout_seconds
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def execute(
        self,
        request: ExecutionRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionResult:
        execution_id = str(uuid4())
        started = monotonic()

        timeout_s = request.timeout_seconds if request.timeout_seconds > 0 else self.default_timeout_seconds
        timeout_s = min(timeout_s, self.max_timeout_seconds)

        cwd_path = Path(request.cwd).resolve() if request.cwd is not None else self.manager.runtime_dir.resolve()
        if not cwd_path.is_dir():
            return ExecutionResult(
                execution_id=execution_id,
                command=request.command,
                argv=(request.command, *request.args),
                cwd=str(cwd_path),
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=0,
                error_code="RUNTIME_INVALID_CWD",
                error_message=f"指定的工作目录不存在：{cwd_path}",
            )

        try:
            if request.use_shell:
                shell_exe = self.manager.resolve_command(request.shell_executable)
                argv: tuple[str, ...] = (str(shell_exe), "-c", request.command, *request.args)
            else:
                exe = self.manager.resolve_command(request.command)
                argv = (str(exe), *request.args)
        except RuntimeError_ as err:
            return ExecutionResult(
                execution_id=execution_id,
                command=request.command,
                argv=(request.command, *request.args),
                cwd=str(cwd_path),
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=max(0, round((monotonic() - started) * 1000)),
                error_code=err.code,
                error_message=err.message,
            )

        env = self.manager.build_clean_env(include_venv=True, env_overrides=request.env_overrides)
        max_total = max(2048, request.max_output_bytes)
        # Reserve at least 1/3 of the output budget for stderr so tracebacks are never hidden by verbose stdout.
        stderr_budget = max(1024, max_total // 3)
        stdout_budget = max(1024, max_total - stderr_budget)

        async with self._semaphore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(cwd_path),
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except OSError as exc:
                return ExecutionResult(
                    execution_id=execution_id,
                    command=request.command,
                    argv=argv,
                    cwd=str(cwd_path),
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration_ms=max(0, round((monotonic() - started) * 1000)),
                    error_code="RUNTIME_SPAWN_FAILED",
                    error_message=f"无法启动命令 {request.command}：{exc}",
                )

            stdout_task = asyncio.create_task(_read_bounded_stream(proc.stdout, max_bytes=stdout_budget))
            stderr_task = asyncio.create_task(_read_bounded_stream(proc.stderr, max_bytes=stderr_budget))
            wait_task = asyncio.create_task(proc.wait())
            cancel_waiter: asyncio.Task[bool] | None = (
                asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
            )

            timed_out = False
            cancelled = False
            try:
                waiters: set[asyncio.Task] = {wait_task}
                if cancel_waiter is not None:
                    waiters.add(cancel_waiter)
                done, _ = await asyncio.wait(waiters, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    timed_out = True
                    await asyncio.shield(_terminate_process_group(proc))
                elif cancel_waiter is not None and cancel_waiter in done and wait_task not in done:
                    cancelled = True
                    await asyncio.shield(_terminate_process_group(proc))
            except asyncio.CancelledError:
                # Always reap the entire child process group before propagating task cancellation.
                await asyncio.shield(_terminate_process_group(proc))
                for task in (stdout_task, stderr_task, wait_task, cancel_waiter):
                    if task is not None and not task.done():
                        task.cancel()
                with suppress(Exception):
                    await asyncio.gather(
                        *(t for t in (stdout_task, stderr_task, wait_task, cancel_waiter) if t is not None),
                        return_exceptions=True,
                    )
                raise
            finally:
                if cancel_waiter is not None and not cancel_waiter.done():
                    cancel_waiter.cancel()
                    with suppress(asyncio.CancelledError):
                        await cancel_waiter

            # Ensure process exited and drain/collect readers cleanly.
            with suppress(Exception):
                await asyncio.wait_for(wait_task, timeout=2.0)
            try:
                (out_bytes, out_trunc), (err_bytes, err_trunc) = await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task),
                    timeout=2.0,
                )
            except TimeoutError:
                stdout_task.cancel()
                stderr_task.cancel()
                out_bytes, out_trunc = b"", True
                err_bytes, err_trunc = b"", True

            elapsed_ms = max(0, round((monotonic() - started) * 1000))
            stdout_text = out_bytes.decode("utf-8", errors="replace")
            stderr_text = err_bytes.decode("utf-8", errors="replace")
            truncated = out_trunc or err_trunc

            error_code: str | None = None
            error_message: str | None = None
            if timed_out:
                error_code = "RUNTIME_TIMEOUT"
                error_message = f"命令执行超时（超过 {timeout_s:.0f} 秒），已终止进程组。"
            elif cancelled:
                error_code = "RUNTIME_CANCELLED"
                error_message = "命令执行已取消并回收子进程。"
            elif proc.returncode != 0:
                error_code = "RUNTIME_NON_ZERO_EXIT"
                first_err = next((line.strip() for line in stderr_text.splitlines() if line.strip()), "")
                error_message = (
                    f"命令以退出码 {proc.returncode} 结束：{first_err[:200]}"
                    if first_err
                    else f"命令以非零退出码 {proc.returncode} 结束。"
                )

            return ExecutionResult(
                execution_id=execution_id,
                command=request.command,
                argv=argv,
                cwd=str(cwd_path),
                exit_code=proc.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_ms=elapsed_ms,
                timed_out=timed_out,
                cancelled=cancelled,
                output_truncated=truncated,
                error_code=error_code,
                error_message=error_message,
            )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    async def run_python(
        self,
        args: Sequence[str],
        *,
        cwd: Path | str,
        run_id: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        env_overrides: Mapping[str, str] | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionResult:
        return await self.execute(
            ExecutionRequest(
                run_id=run_id,
                command="python",
                args=tuple(args),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                env_overrides=dict(env_overrides or {}),
                max_output_bytes=max_output_bytes,
                use_shell=False,
            ),
            cancel_event=cancel_event,
        )

    async def run_node(
        self,
        args: Sequence[str],
        *,
        cwd: Path | str,
        run_id: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        env_overrides: Mapping[str, str] | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionResult:
        return await self.execute(
            ExecutionRequest(
                run_id=run_id,
                command="node",
                args=tuple(args),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                env_overrides=dict(env_overrides or {}),
                max_output_bytes=max_output_bytes,
                use_shell=False,
            ),
            cancel_event=cancel_event,
        )

    async def run_shell(
        self,
        script: str,
        *,
        cwd: Path | str,
        run_id: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        env_overrides: Mapping[str, str] | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        shell_executable: str = "/bin/zsh",
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionResult:
        return await self.execute(
            ExecutionRequest(
                run_id=run_id,
                command=script,
                args=(),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                env_overrides=dict(env_overrides or {}),
                max_output_bytes=max_output_bytes,
                use_shell=True,
                shell_executable=shell_executable,
            ),
            cancel_event=cancel_event,
        )
