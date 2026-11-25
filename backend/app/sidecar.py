import asyncio
import json
import logging
import os
import socket
import sqlite3
import sys
from contextlib import suppress
from pathlib import Path
from threading import Thread
from typing import Literal
from uuid import uuid4

import uvicorn
from pydantic import BaseModel, ConfigDict

from app.diagnostics import configure_log, record
from app.main import DEV_ORIGIN, MACOS_APP_ORIGIN, SidecarControl, create_app
from app.security import validate_tokens
from app.storage.business import DATABASE_NAME

PROTOCOL_VERSION = 1
MAX_BOOTSTRAP_BYTES = 16_384
LOGGER = logging.getLogger("uvicorn.error")
FROZEN_MACOS = sys.platform == "darwin" and getattr(sys, "frozen", False)


class Bootstrap(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["bootstrap"]
    protocol_version: int
    ui_token: str
    host_token: str
    app_data_dir: str
    parent_pid: int
    restore_credentials: bool = False


def has_expected_parent(host_pid: int) -> bool:
    if not 1 < host_pid <= 0x7FFF_FFFF:
        return False
    parent_pid = os.getppid()
    if host_pid == parent_pid:
        return True
    if not FROZEN_MACOS or parent_pid <= 1:
        return False

    # A macOS onefile worker is forked by a second instance of its own executable.
    # Verify both the launcher executable and its actual parent via libproc.
    from ctypes import CDLL, c_int, create_string_buffer, sizeof

    libproc = CDLL("/usr/lib/libproc.dylib")
    path = create_string_buffer(4096)  # PROC_PIDPATHINFO_MAXSIZE
    if libproc.proc_pidpath(parent_pid, path, len(path)) <= 0:
        return False
    if not Path(os.fsdecode(path.value)).samefile(sys.executable):
        return False

    count = libproc.proc_listchildpids(host_pid, None, 0)
    if count <= 0:
        return False
    children = (c_int * (count + 16))()
    found = libproc.proc_listchildpids(host_pid, children, sizeof(children))
    return any(children[index] == parent_pid for index in range(found))


def read_bootstrap() -> Bootstrap:
    frame = sys.stdin.buffer.readline(MAX_BOOTSTRAP_BYTES + 1)
    if len(frame) > MAX_BOOTSTRAP_BYTES or not frame.endswith(b"\n"):
        raise ValueError("Invalid bootstrap frame")
    bootstrap = Bootstrap.model_validate_json(frame)
    if bootstrap.protocol_version != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version")
    if not has_expected_parent(bootstrap.parent_pid):
        raise ValueError("Invalid parent process")
    validate_tokens(bootstrap.ui_token, bootstrap.host_token)
    data_dir = Path(bootstrap.app_data_dir)
    if not data_dir.is_absolute():
        raise ValueError("Invalid application data directory")
    data_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(data_dir, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError("Application data directory is inaccessible")
    return bootstrap


def watch_parent(fd: int, loop: asyncio.AbstractEventLoop, stopping: asyncio.Event) -> None:
    def stop_after_disconnect() -> None:
        if not stopping.is_set():
            LOGGER.info("Host stdin closed, shutting down.")
            record("host_disconnected")
            stopping.set()

    # Use the raw descriptor so a blocking read cannot hold a Python stdio lock at exit.
    with suppress(OSError):
        while os.read(fd, 4096):
            pass
    with suppress(RuntimeError):  # The event loop may already have shut down by this point.
        loop.call_soon_threadsafe(stop_after_disconnect)


class SidecarServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, control: SidecarControl) -> None:
        super().__init__(config)
        self.control = control

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.control.stopping.is_set():
            self.should_exit = True
        if self.started and not self.should_exit:
            record("sidecar_ready", instance_id=str(self.control.instance_id))
            print(json.dumps({
                "type": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "instance_id": str(self.control.instance_id),
                "port": self.control.port,
            }), flush=True)

    async def stop_when_requested(self) -> None:
        await self.control.stopping.wait()
        self.should_exit = True

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        self.control.stopping.set()
        await super().shutdown(sockets)


async def serve(bootstrap: Bootstrap, app_factory=create_app) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        control = SidecarControl(
            uuid4(), listener.getsockname()[1], asyncio.Event(),
            credentials_restored=asyncio.Event() if bootstrap.restore_credentials else None,
        )
        app = app_factory(
            bootstrap.ui_token,
            bootstrap.host_token,
            database_path=Path(bootstrap.app_data_dir) / DATABASE_NAME,
            control=control,
            ui_origin=MACOS_APP_ORIGIN if FROZEN_MACOS else DEV_ORIGIN,
        )
        server = SidecarServer(uvicorn.Config(
            app, access_log=False, proxy_headers=False, lifespan="on", timeout_graceful_shutdown=3,
        ), control)
        Thread(
            target=watch_parent,
            args=(sys.stdin.fileno(), asyncio.get_running_loop(), control.stopping),
            daemon=True,
        ).start()
        stop_task = asyncio.create_task(server.stop_when_requested())
        try:
            await server.serve(sockets=[listener])
        finally:
            stop_task.cancel()
            with suppress(asyncio.CancelledError):
                await stop_task
            record("sidecar_stopped", instance_id=str(control.instance_id))


def main(*, app_factory=create_app) -> int:
    try:
        bootstrap = read_bootstrap()
    except (OSError, ValueError):
        # Validation errors can contain the original frame. Never print them.
        print("KnowledgeAgent Sidecar: bootstrap rejected.", file=sys.stderr)
        return 1
    try:
        configure_log(Path(bootstrap.app_data_dir))
        # The host supplies the provider credential and the adapter pins its destination.
        # Keep inherited SDK logging, custom headers, and gateway settings out of that channel.
        for name in tuple(os.environ):
            if name.startswith(("OPENAI_", "DEEPSEEK_", "SILICONFLOW_", "LANGSMITH_", "LANGCHAIN_")):
                del os.environ[name]
        asyncio.run(serve(bootstrap, app_factory))
    except (OSError, RuntimeError, sqlite3.Error):
        record("startup_error", error_code="SIDECAR_NOT_READY")
        LOGGER.error("KnowledgeAgent Sidecar: could not run the HTTP service.")
        return 1
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--parse-pdf-worker"]:
        from app.parsing.pdf import pdf_worker_main

        raise SystemExit(pdf_worker_main(sys.argv[2:]))
    if sys.argv[1:2] == ["--parse-docx-worker"]:
        from app.parsing.docx import docx_worker_main

        raise SystemExit(docx_worker_main(sys.argv[2:]))
    raise SystemExit(main())
