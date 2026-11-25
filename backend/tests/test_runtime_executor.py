"""P9.2 verification suite: RuntimeManager probing, RuntimeExecutor real execution, and Runtime API."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from app.main import create_app
from app.runtime.executor import ExecutionRequest, RuntimeExecutor
from app.runtime.manager import RuntimeManager

UI_TOKEN = "p92-ui-token-1234567890"
HOST_TOKEN = "p92-host-token-abcdefghij"


def test_runtime_manager_probes_host_tools_and_preserves_mcp_procs(tmp_path):
    app_data = tmp_path / "app_data"
    mcp_procs = app_data / "runtime" / "mcp-procs"
    mcp_procs.mkdir(parents=True)
    sentinel = mcp_procs / "keep.json"
    sentinel.write_text("{}", encoding="utf-8")

    manager = RuntimeManager(app_data)
    status = manager.probe_status()

    assert status.uv.available is True
    assert status.python.available is True
    assert status.python.path is not None
    # Must NOT pick the backend Sidecar's own .venv/bin/python
    assert Path(status.python.path).parent.resolve() != (Path(sys.prefix) / "bin").resolve()
    assert status.node.available is True
    assert status.npm.available is True
    assert status.git.available is True
    assert status.shell.available is True
    # Existing MCP process directory must be untouched
    assert sentinel.is_file()


@pytest.mark.anyio
async def test_executor_python_syntax_error_shell_node_truncation_timeout_and_cancel(tmp_path, monkeypatch):
    app_data = tmp_path / "app_data"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Inject sensitive / PyInstaller-like env vars into current process to verify child scrubbing
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-must-not-leak")
    monkeypatch.setenv("KNOWLEDGEAGENT_HOST_TOKEN", "host-secret-must-not-leak")
    monkeypatch.setenv("PYINSTALLER_RESET_ENVIRONMENT", "1")
    monkeypatch.setenv("PYTHONPATH", "/tmp/fake-pythonpath")

    manager = RuntimeManager(app_data)
    executor = RuntimeExecutor(manager, max_concurrency=2)

    # 1. Normal Python script + env isolation check
    ok_res = await executor.run_python(
        [
            "-c",
            (
                "import json, os, sys; "
                "print(json.dumps({"
                "'exe': sys.executable, "
                "'cwd': os.getcwd(), "
                "'has_deepseek': 'DEEPSEEK_API_KEY' in os.environ, "
                "'has_ka': 'KNOWLEDGEAGENT_HOST_TOKEN' in os.environ, "
                "'has_pyinstaller': 'PYINSTALLER_RESET_ENVIRONMENT' in os.environ, "
                "'has_pythonpath': 'PYTHONPATH' in os.environ"
                "}))"
            ),
        ],
        cwd=work_dir,
    )
    assert ok_res.ok, ok_res
    assert ok_res.exit_code == 0
    payload = json.loads(ok_res.stdout.strip())
    assert Path(payload["cwd"]).resolve() == work_dir.resolve()
    assert payload["has_deepseek"] is False
    assert payload["has_ka"] is False
    assert payload["has_pyinstaller"] is False
    assert payload["has_pythonpath"] is False

    # 2. Python script with syntax error -> non-zero exit + real stderr (no exception crash)
    bad_script = work_dir / "broken.py"
    bad_script.write_text("def oops(:\n    pass\n", encoding="utf-8")
    err_res = await executor.run_python([str(bad_script)], cwd=work_dir)
    assert not err_res.ok
    assert err_res.exit_code != 0
    assert err_res.error_code == "RUNTIME_NON_ZERO_EXIT"
    assert "SyntaxError" in err_res.stderr

    # 3. Shell command with pipeline
    shell_res = await executor.run_shell("printf 'alpha\\nbeta\\ngamma\\n' | wc -l", cwd=work_dir)
    assert shell_res.ok, shell_res
    assert shell_res.stdout.strip() == "3"

    # 4. Node.js script
    node_res = await executor.run_node(
        ["-e", "console.log(JSON.stringify({ok: true, sum: 21 + 21, msg: '中文正常'}))"],
        cwd=work_dir,
    )
    assert node_res.ok, node_res
    node_payload = json.loads(node_res.stdout.strip())
    assert node_payload == {"ok": True, "sum": 42, "msg": "中文正常"}

    # 5. Large output truncation without pipe deadlock
    trunc_res = await executor.execute(
        ExecutionRequest(
            run_id="r-trunc",
            command="python",
            args=("-c", "import sys; print('A' * 60000); print('ERR_MARKER', file=sys.stderr)"),
            cwd=work_dir,
            max_output_bytes=4096,
        )
    )
    assert trunc_res.ok
    assert trunc_res.output_truncated is True
    assert len(trunc_res.stdout.encode("utf-8")) <= 4096
    assert "ERR_MARKER" in trunc_res.stderr

    # 6. Timeout script kills process group (including grandchild)
    pid_file = work_dir / "child.pid"
    timeout_res = await executor.run_shell(
        f"sleep 30 & echo $! > {pid_file}; wait",
        cwd=work_dir,
        timeout_seconds=0.6,
    )
    assert not timeout_res.ok
    assert timeout_res.timed_out is True
    assert timeout_res.error_code == "RUNTIME_TIMEOUT"
    if pid_file.exists() and pid_file.read_text().strip().isdigit():
        grandchild_pid = int(pid_file.read_text().strip())
        await asyncio.sleep(0.15)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)

    # 7. Cancellable long task via cancel_event and asyncio.Task.cancel()
    cancel_event = asyncio.Event()
    pid_file2 = work_dir / "cancel.pid"
    task = asyncio.create_task(
        executor.run_shell(
            f"echo $$ > {pid_file2}; sleep 30",
            cwd=work_dir,
            timeout_seconds=30.0,
            cancel_event=cancel_event,
        )
    )
    for _ in range(40):
        if pid_file2.exists() and pid_file2.read_text().strip().isdigit():
            break
        await asyncio.sleep(0.05)
    cancel_event.set()
    cancel_res = await task
    assert not cancel_res.ok
    assert cancel_res.cancelled is True
    assert cancel_res.error_code == "RUNTIME_CANCELLED"
    if pid_file2.exists() and pid_file2.read_text().strip().isdigit():
        cancelled_pid = int(pid_file2.read_text().strip())
        await asyncio.sleep(0.15)
        with pytest.raises(ProcessLookupError):
            os.kill(cancelled_pid, 0)


@pytest.mark.anyio
async def test_runtime_api_status_and_host_auth_boundary(tmp_path):
    app = create_app(UI_TOKEN, HOST_TOKEN, database_path=tmp_path / "knowledgeagent.sqlite3")
    ui_headers = {"Authorization": f"Bearer {UI_TOKEN}", "Host": "127.0.0.1:8000"}
    host_headers = {"Authorization": f"Bearer {HOST_TOKEN}", "Host": "127.0.0.1:8000"}

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
            status_resp = await client.get("/api/v1/runtime/status", headers=ui_headers)
            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["uv"]["available"] is True
            assert data["node"]["available"] is True

            # UI token cannot call _host runtime prepare
            denied = await client.post(
                "/api/v1/_host/runtime/prepare",
                json={"ensure_python_venv": False},
                headers=ui_headers,
            )
            assert denied.status_code == 403

            # Invalid package name rejected with 422
            bad_pkg = await client.post(
                "/api/v1/_host/runtime/prepare",
                json={"python_packages": ["bad;rm -rf /"], "ensure_python_venv": False},
                headers=host_headers,
            )
            assert bad_pkg.status_code == 422
