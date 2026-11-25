"""Independent local execution runtime discovery, managed venv, and tool preparation.

Directory layout under `<app_data_dir>/runtime/`:
```text
runtime/
  venv/          # Independent Python venv for skills & scripts
  npm-tools/     # Local npm prefix (e.g. @mermaid-js/mermaid-cli)
  cache/         # Runtime configs (e.g. puppeteer-config.json) & caches
  mcp-procs/     # Existing MCP subprocess registry (preserved untouched)
```

Key design invariant:
NEVER treat PyInstaller's frozen `sys.executable` or the backend Sidecar's own
`.venv` as the skills Python interpreter. Always probe standalone host runtimes
deterministically (without depending on Finder's minimal GUI PATH) and prepare
an isolated venv under `<app_data_dir>/runtime/venv/`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

SAFE_INHERITED_ENV_KEYS: tuple[str, ...] = (
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
)

FORBIDDEN_ENV_PREFIXES: tuple[str, ...] = (
    "OPENAI_",
    "DEEPSEEK_",
    "SILICONFLOW_",
    "LANGSMITH_",
    "LANGCHAIN_",
    "KNOWLEDGEAGENT_",
    "DYLD_",
    "LD_",
    "PYINSTALLER_",
    "_MEIPASS",
)

FORBIDDEN_ENV_EXACT: frozenset[str] = frozenset({
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
    "__PYVENV_LAUNCHER__",
})

MACOS_CHROME_CANDIDATES: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)

PACKAGE_SPEC_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-\[\],<>=!~ ]{0,127}$")
DEFAULT_PREPARE_TIMEOUT_SECONDS = 300.0
PROBE_TIMEOUT_SECONDS = 4.0


class RuntimeError_(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ToolProbe:
    available: bool
    path: str | None = None
    version: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    runtime_dir: str
    venv_path: str
    venv_ready: bool
    venv_python: str | None
    installed_python_packages: dict[str, str]
    puppeteer_config_path: str | None
    uv: ToolProbe
    python: ToolProbe
    node: ToolProbe
    npm: ToolProbe
    npx: ToolProbe
    git: ToolProbe
    chrome: ToolProbe
    mermaid_cli: ToolProbe
    shell: ToolProbe
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrepareStepResult:
    step: str
    ok: bool
    message: str
    duration_ms: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class PrepareResult:
    ok: bool
    steps: tuple[PrepareStepResult, ...]
    status: RuntimeStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "steps": [asdict(step) for step in self.steps],
            "status": self.status.to_dict(),
        }


def _semver_sort_key(name: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", name)
    return tuple(int(n) for n in nums) if nums else (0,)


def _is_executable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _is_forbidden_sidecar_python(candidate: Path) -> bool:
    """Reject PyInstaller frozen executable and the backend Sidecar's own .venv binary."""
    try:
        resolved = candidate.resolve()
    except OSError:
        return False

    if getattr(sys, "frozen", False):
        with suppress(OSError):
            if resolved == Path(sys.executable).resolve():
                return True
        return False

    # In development mode, avoid picking the backend Sidecar's own .venv/bin/python*,
    # while still allowing the underlying standalone interpreter it symlinks to.
    backend_venv_bin = Path(sys.prefix) / "bin"
    with suppress(OSError):
        if candidate.parent.resolve() == backend_venv_bin.resolve():
            return True
    return False


def _has_xcode_clt() -> bool:
    """Check whether macOS Command Line Tools are installed so /usr/bin/python3 won't trigger a GUI prompt."""
    if sys.platform != "darwin":
        return True
    return Path("/Library/Developer/CommandLineTools/usr/bin/python3").exists() or Path(
        "/Applications/Xcode.app/Contents/Developer/usr/bin/python3"
    ).exists()


def _quick_version(executable: Path, args: Sequence[str] = ("--version",), *, env: Mapping[str, str]) -> str | None:
    try:
        proc = subprocess.run(
            [str(executable), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            env=dict(env),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    combined = (proc.stdout or proc.stderr or "").strip()
    if not combined:
        return None
    return combined.splitlines()[0].strip()[:120]


class RuntimeManager:
    """Probes host tooling, maintains independent venv/npm-tools under `<app_data_dir>/runtime/`."""

    def __init__(self, app_data_dir: Path) -> None:
        self.app_data_dir = app_data_dir
        self.runtime_dir = app_data_dir / "runtime"
        self.venv_dir = self.runtime_dir / "venv"
        self.npm_tools_dir = self.runtime_dir / "npm-tools"
        self.cache_dir = self.runtime_dir / "cache"
        self._prepare_lock = asyncio.Lock()

    @property
    def venv_bin_dir(self) -> Path:
        return self.venv_dir / ("Scripts" if os.name == "nt" else "bin")

    @property
    def venv_python_path(self) -> Path:
        return self.venv_bin_dir / ("python.exe" if os.name == "nt" else "python")

    @property
    def npm_bin_dir(self) -> Path:
        return self.npm_tools_dir / "node_modules" / ".bin"

    @property
    def puppeteer_config_file(self) -> Path:
        return self.cache_dir / "puppeteer-config.json"

    def ensure_runtime_dirs(self) -> None:
        for directory in (self.runtime_dir, self.npm_tools_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Deterministic search paths & clean child environment
    # ------------------------------------------------------------------

    def search_dirs(self, *, include_runtime_bins: bool = True, current_path: str | None = None) -> list[Path]:
        dirs: list[Path] = []
        seen: set[str] = set()

        def _add(path: Path | str) -> None:
            text = str(path).strip()
            if not text or text in seen:
                return
            seen.add(text)
            dirs.append(Path(text))

        if include_runtime_bins:
            if self.venv_bin_dir.is_dir():
                _add(self.venv_bin_dir)
            if self.npm_bin_dir.is_dir():
                _add(self.npm_bin_dir)

        home = Path(os.environ.get("HOME", str(Path.home())))

        # Standard macOS Homebrew & user tool locations first (Finder GUI apps lack these in PATH).
        for fixed in ("/opt/homebrew/bin", "/usr/local/bin", home / ".local" / "bin", home / ".cargo" / "bin"):
            _add(fixed)

        # Node version managers (nvm / volta / fnm) if present.
        volta_bin = home / ".volta" / "bin"
        if volta_bin.is_dir():
            _add(volta_bin)
        nvm_versions = home / ".nvm" / "versions" / "node"
        if nvm_versions.is_dir():
            with suppress(OSError):
                for entry in sorted(nvm_versions.iterdir(), key=lambda p: _semver_sort_key(p.name), reverse=True):
                    bin_dir = entry / "bin"
                    if bin_dir.is_dir():
                        _add(bin_dir)

        # python.org macOS Framework installations (sorted newest first).
        framework_versions = Path("/Library/Frameworks/Python.framework/Versions")
        if framework_versions.is_dir():
            with suppress(OSError):
                for entry in sorted(
                    (p for p in framework_versions.iterdir() if p.is_dir() and p.name != "Current"),
                    key=lambda p: _semver_sort_key(p.name),
                    reverse=True,
                ):
                    bin_dir = entry / "bin"
                    if bin_dir.is_dir():
                        _add(bin_dir)

        # Include inherited PATH entries (excluding backend .venv/bin in dev mode).
        raw_path = os.environ.get("PATH") if current_path is None else current_path
        backend_venv_bin = None
        if not getattr(sys, "frozen", False):
            with suppress(OSError):
                backend_venv_bin = (Path(sys.prefix) / "bin").resolve()
        if raw_path:
            for item in raw_path.split(os.pathsep):
                item = item.strip()
                if not item:
                    continue
                if backend_venv_bin is not None:
                    with suppress(OSError):
                        if Path(item).resolve() == backend_venv_bin:
                            continue
                _add(item)

        for sys_dir in ("/usr/bin", "/bin", "/usr/sbin", "/sbin"):
            _add(sys_dir)
        return dirs

    def build_clean_env(
        self,
        *,
        env_overrides: Mapping[str, str] | None = None,
        include_venv: bool = True,
        host_environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Build a clean subprocess environment free of PyInstaller vars and app secrets."""
        source = os.environ if host_environ is None else host_environ
        env: dict[str, str] = {}
        for key in SAFE_INHERITED_ENV_KEYS:
            value = source.get(key)
            if value is None or "\x00" in value or value.startswith("()"):
                continue
            env[key] = value

        env.setdefault("LANG", "en_US.UTF-8")
        env.setdefault("LC_ALL", "en_US.UTF-8")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        path_dirs = self.search_dirs(include_runtime_bins=include_venv, current_path=source.get("PATH"))
        env["PATH"] = os.pathsep.join(str(d) for d in path_dirs)
        if include_venv and self.venv_python_path.exists():
            env["VIRTUAL_ENV"] = str(self.venv_dir)

        if env_overrides:
            for key, value in env_overrides.items():
                upper = key.upper()
                if upper in FORBIDDEN_ENV_EXACT or upper.startswith(FORBIDDEN_ENV_PREFIXES):
                    continue
                if "\x00" in value:
                    continue
                env[key] = value

        # Defense-in-depth scrub.
        for key in tuple(env):
            upper = key.upper()
            if upper in FORBIDDEN_ENV_EXACT or upper.startswith(FORBIDDEN_ENV_PREFIXES):
                del env[key]
        return env

    # ------------------------------------------------------------------
    # Host tool probing
    # ------------------------------------------------------------------

    def _find_binary(
        self,
        names: Sequence[str],
        *,
        include_runtime_bins: bool = True,
        skip_predicate=None,
    ) -> Path | None:
        for directory in self.search_dirs(include_runtime_bins=include_runtime_bins):
            for name in names:
                candidate = directory / name
                if not _is_executable(candidate):
                    continue
                if skip_predicate is not None and skip_predicate(candidate):
                    continue
                return candidate
        return None

    def _probe_binary(
        self,
        names: Sequence[str],
        *,
        version_args: Sequence[str] = ("--version",),
        include_runtime_bins: bool = True,
        env: Mapping[str, str],
        skip_predicate=None,
        source_label: str | None = None,
    ) -> ToolProbe:
        found = self._find_binary(names, include_runtime_bins=include_runtime_bins, skip_predicate=skip_predicate)
        if found is None:
            return ToolProbe(available=False)
        version = _quick_version(found, version_args, env=env)
        src = source_label
        if src is None:
            if found.is_relative_to(self.runtime_dir):
                src = "runtime"
            elif str(found).startswith("/opt/homebrew"):
                src = "homebrew"
            else:
                src = "system"
        return ToolProbe(available=True, path=str(found), version=version, source=src)

    def _probe_uv_python(self, uv_path: Path, env: Mapping[str, str]) -> ToolProbe | None:
        for spec in ("3.12", ">=3.11,<3.14", ">=3.10"):
            try:
                proc = subprocess.run(
                    [str(uv_path), "python", "find", "--no-python-downloads", spec],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=PROBE_TIMEOUT_SECONDS,
                    env=dict(env),
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            if proc.returncode == 0:
                found_str = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
                if found_str:
                    candidate = Path(found_str)
                    if _is_executable(candidate) and not _is_forbidden_sidecar_python(candidate):
                        ver = _quick_version(candidate, env=env)
                        return ToolProbe(available=True, path=str(candidate), version=ver, source="uv")
        return None

    def _probe_host_python(self, uv_probe: ToolProbe, env: Mapping[str, str]) -> ToolProbe:
        if uv_probe.available and uv_probe.path:
            uv_py = self._probe_uv_python(Path(uv_probe.path), env)
            if uv_py is not None:
                return uv_py

        clt_ready = _has_xcode_clt()

        def _skip_python(candidate: Path) -> bool:
            if _is_forbidden_sidecar_python(candidate):
                return True
            if sys.platform == "darwin" and str(candidate) == "/usr/bin/python3" and not clt_ready:
                return True
            return False

        return self._probe_binary(
            ("python3.12", "python3.13", "python3.11", "python3"),
            include_runtime_bins=False,
            env=env,
            skip_predicate=_skip_python,
        )

    def _probe_chrome(self, env: Mapping[str, str]) -> ToolProbe:
        home = Path(os.environ.get("HOME", str(Path.home())))
        candidates = [
            *(Path(p) for p in MACOS_CHROME_CANDIDATES),
            home / "Applications" / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome",
        ]
        for candidate in candidates:
            if _is_executable(candidate):
                ver = _quick_version(candidate, ("--version",), env=env)
                return ToolProbe(available=True, path=str(candidate), version=ver, source="application")
        return ToolProbe(available=False)

    def _probe_venv_packages(self, env: Mapping[str, str]) -> dict[str, str]:
        if not _is_executable(self.venv_python_path):
            return {}
        script = (
            "import importlib.metadata as m, json; "
            "print(json.dumps({d.metadata['Name'].lower(): d.version "
            "for d in m.distributions() if d.metadata and d.metadata.get('Name')}))"
        )
        try:
            proc = subprocess.run(
                [str(self.venv_python_path), "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
                env=dict(env),
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                parsed = json.loads(proc.stdout.strip())
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in sorted(parsed.items())}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return {}

    def ensure_puppeteer_config(self, chrome_path: str | None) -> Path | None:
        if not chrome_path:
            return self.puppeteer_config_file if self.puppeteer_config_file.is_file() else None
        self.ensure_runtime_dirs()
        payload = {"executablePath": chrome_path}
        self.puppeteer_config_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.puppeteer_config_file

    def probe_status(self) -> RuntimeStatus:
        """Synchronously probe available host tools and managed runtime state (safe to run via asyncio.to_thread)."""
        env = self.build_clean_env(include_venv=False)
        uv_probe = self._probe_binary(("uv",), include_runtime_bins=False, env=env)
        python_probe = self._probe_host_python(uv_probe, env)
        node_probe = self._probe_binary(("node",), include_runtime_bins=False, env=env)
        npm_probe = self._probe_binary(("npm",), include_runtime_bins=False, env=env)
        npx_probe = self._probe_binary(("npx",), include_runtime_bins=False, env=env)
        git_probe = self._probe_binary(("git",), include_runtime_bins=False, env=env)
        chrome_probe = self._probe_chrome(env)
        mermaid_probe = self._probe_binary(("mmdc",), include_runtime_bins=True, env=self.build_clean_env(include_venv=True))
        shell_probe = self._probe_binary(("zsh", "bash", "sh"), include_runtime_bins=False, env=env)

        venv_ready = _is_executable(self.venv_python_path) and not _is_forbidden_sidecar_python(self.venv_python_path)
        venv_python = str(self.venv_python_path) if venv_ready else None
        packages = self._probe_venv_packages(self.build_clean_env(include_venv=True)) if venv_ready else {}

        puppeteer_cfg: str | None = None
        if self.puppeteer_config_file.is_file():
            puppeteer_cfg = str(self.puppeteer_config_file)

        warnings: list[str] = []
        if not venv_ready:
            warnings.append("独立 Skills Python 虚拟环境尚未初始化，请执行运行环境准备。")
        if not uv_probe.available and not python_probe.available:
            warnings.append("未检测到 uv 或独立的 Python 3 解释器，请先安装 uv（推荐）或 Python 3.12。")
        if not node_probe.available:
            warnings.append("未检测到 Node.js，Node 脚本与 Mermaid CLI 渲染将不可用。")
        if not mermaid_probe.available:
            warnings.append("尚未安装 Mermaid CLI（mmdc），思维导图 SVG/PNG 渲染需先准备。")

        return RuntimeStatus(
            runtime_dir=str(self.runtime_dir),
            venv_path=str(self.venv_dir),
            venv_ready=venv_ready,
            venv_python=venv_python,
            installed_python_packages=packages,
            puppeteer_config_path=puppeteer_cfg,
            uv=uv_probe,
            python=python_probe,
            node=node_probe,
            npm=npm_probe,
            npx=npx_probe,
            git=git_probe,
            chrome=chrome_probe,
            mermaid_cli=mermaid_probe,
            shell=shell_probe,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Command resolution for RuntimeExecutor
    # ------------------------------------------------------------------

    def resolve_command(self, command: str) -> Path:
        """Resolve a command name or absolute path deterministically without invoking a shell."""
        if not command or not isinstance(command, str) or "\x00" in command or "\n" in command or "\r" in command:
            raise RuntimeError_("RUNTIME_INVALID_COMMAND", "命令名无效。")
        cmd = command.strip()
        if not cmd:
            raise RuntimeError_("RUNTIME_INVALID_COMMAND", "命令名不能为空。")

        # Map `python` / `python3` to managed independent venv first, then probed standalone Python.
        if cmd in ("python", "python3"):
            if _is_executable(self.venv_python_path) and not _is_forbidden_sidecar_python(self.venv_python_path):
                return self.venv_python_path
            status = self.probe_status()
            if status.python.available and status.python.path:
                return Path(status.python.path)
            raise RuntimeError_(
                "RUNTIME_PYTHON_NOT_READY",
                "未找到可用的独立 Python 运行环境，请先准备独立 Python 环境或安装 uv / Python 3.12。",
            )

        if "/" in cmd or "\\" in cmd:
            if "\\" in cmd:
                raise RuntimeError_("RUNTIME_INVALID_COMMAND", "命令路径不支持反斜杠。")
            candidate = Path(cmd)
            if not candidate.is_absolute():
                raise RuntimeError_("RUNTIME_INVALID_COMMAND", "带路径分隔符的命令必须使用绝对路径。")
            if not _is_executable(candidate):
                raise RuntimeError_("RUNTIME_COMMAND_NOT_FOUND", f"未找到可执行文件：{cmd}")
            if _is_forbidden_sidecar_python(candidate):
                raise RuntimeError_("RUNTIME_FORBIDDEN_INTERPRETER", "禁止将 Sidecar 自身作为技能 Python 解释器执行。")
            return candidate

        found = self._find_binary((cmd,), include_runtime_bins=True)
        if found is None:
            raise RuntimeError_("RUNTIME_COMMAND_NOT_FOUND", f"未在运行环境中找到命令：{cmd}")
        if cmd.startswith("python") and _is_forbidden_sidecar_python(found):
            raise RuntimeError_("RUNTIME_FORBIDDEN_INTERPRETER", "禁止将 Sidecar 自身作为技能 Python 解释器执行。")
        return found

    # ------------------------------------------------------------------
    # On-demand preparation (venv + Python packages + Mermaid CLI)
    # ------------------------------------------------------------------

    async def _run_prepare_cmd(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> tuple[int, str, str, int]:
        started = monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=dict(env),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            with suppress(OSError):
                os.killpg(proc.pid, 9)
            with suppress(Exception):
                await proc.wait()
            elapsed = max(0, round((monotonic() - started) * 1000))
            return -1, "", f"准备步骤超时（>{timeout_seconds:.0f}s）。", elapsed
        except asyncio.CancelledError:
            with suppress(OSError):
                os.killpg(proc.pid, 9)
            with suppress(Exception):
                await proc.wait()
            raise

        elapsed = max(0, round((monotonic() - started) * 1000))
        stdout = stdout_b.decode("utf-8", errors="replace")[-8000:]
        stderr = stderr_b.decode("utf-8", errors="replace")[-8000:]
        return int(proc.returncode or 0), stdout, stderr, elapsed

    async def prepare(
        self,
        *,
        python_packages: Sequence[str] = (),
        install_mermaid_cli: bool = False,
        ensure_python_venv: bool = True,
        timeout_seconds: float = DEFAULT_PREPARE_TIMEOUT_SECONDS,
    ) -> PrepareResult:
        """Prepare independent Python venv, Python packages, and/or Mermaid CLI."""
        cleaned_packages: list[str] = []
        for pkg in python_packages:
            spec = pkg.strip()
            if not spec:
                continue
            if not PACKAGE_SPEC_PATTERN.fullmatch(spec):
                raise RuntimeError_("RUNTIME_INVALID_PACKAGE", f"无效的 Python 依赖包声明：{pkg}")
            cleaned_packages.append(spec)

        async with self._prepare_lock:
            self.ensure_runtime_dirs()
            status = await asyncio.to_thread(self.probe_status)
            steps: list[PrepareStepResult] = []
            all_ok = True

            # Step 1: Ensure independent Python venv under `<app_data_dir>/runtime/venv`
            if (ensure_python_venv or cleaned_packages) and not status.venv_ready:
                env = self.build_clean_env(include_venv=False)
                if status.uv.available and status.uv.path:
                    argv = [status.uv.path, "venv", "--python", "3.12", str(self.venv_dir)]
                    rc, out, err, ms = await self._run_prepare_cmd(
                        argv, cwd=self.runtime_dir, env=env, timeout_seconds=timeout_seconds,
                    )
                    if rc != 0:
                        # Fallback to any uv-discoverable Python if 3.12 isn't available offline.
                        argv = [status.uv.path, "venv", str(self.venv_dir)]
                        rc, out, err, ms = await self._run_prepare_cmd(
                            argv, cwd=self.runtime_dir, env=env, timeout_seconds=timeout_seconds,
                        )
                    ok = rc == 0 and _is_executable(self.venv_python_path)
                    steps.append(PrepareStepResult(
                        step="create_python_venv",
                        ok=ok,
                        message="已通过 uv 创建独立 Python 虚拟环境。" if ok else "使用 uv 创建独立 Python 虚拟环境失败。",
                        duration_ms=ms,
                        stdout=out,
                        stderr=err,
                    ))
                    if not ok:
                        all_ok = False
                elif status.python.available and status.python.path:
                    argv = [status.python.path, "-m", "venv", str(self.venv_dir)]
                    rc, out, err, ms = await self._run_prepare_cmd(
                        argv, cwd=self.runtime_dir, env=env, timeout_seconds=timeout_seconds,
                    )
                    ok = rc == 0 and _is_executable(self.venv_python_path)
                    steps.append(PrepareStepResult(
                        step="create_python_venv",
                        ok=ok,
                        message="已通过本机 Python 创建独立虚拟环境。" if ok else "创建独立 Python 虚拟环境失败。",
                        duration_ms=ms,
                        stdout=out,
                        stderr=err,
                    ))
                    if not ok:
                        all_ok = False
                else:
                    all_ok = False
                    steps.append(PrepareStepResult(
                        step="create_python_venv",
                        ok=False,
                        message="未找到 uv 或独立 Python 3 解释器，无法创建独立虚拟环境。",
                        duration_ms=0,
                    ))

            # Step 2: Install requested Python packages into the independent venv
            if cleaned_packages and all_ok and _is_executable(self.venv_python_path):
                env = self.build_clean_env(include_venv=True)
                if status.uv.available and status.uv.path:
                    argv = [status.uv.path, "pip", "install", "--python", str(self.venv_python_path), *cleaned_packages]
                else:
                    argv = [str(self.venv_python_path), "-m", "pip", "install", *cleaned_packages]
                rc, out, err, ms = await self._run_prepare_cmd(
                    argv, cwd=self.runtime_dir, env=env, timeout_seconds=timeout_seconds,
                )
                ok = rc == 0
                steps.append(PrepareStepResult(
                    step="install_python_packages",
                    ok=ok,
                    message=(
                        f"已安装 Python 依赖：{', '.join(cleaned_packages)}。"
                        if ok
                        else f"安装 Python 依赖失败：{', '.join(cleaned_packages)}。"
                    ),
                    duration_ms=ms,
                    stdout=out,
                    stderr=err,
                ))
                if not ok:
                    all_ok = False

            # Step 3: Install @mermaid-js/mermaid-cli into `<app_data_dir>/runtime/npm-tools`
            if install_mermaid_cli:
                if status.chrome.available and status.chrome.path:
                    self.ensure_puppeteer_config(status.chrome.path)
                if not status.npm.available or not status.npm.path:
                    all_ok = False
                    steps.append(PrepareStepResult(
                        step="install_mermaid_cli",
                        ok=False,
                        message="未检测到 npm，无法安装 @mermaid-js/mermaid-cli。",
                        duration_ms=0,
                    ))
                else:
                    pkg_json = self.npm_tools_dir / "package.json"
                    if not pkg_json.exists():
                        pkg_json.write_text(
                            json.dumps({"name": "knowledgeagent-runtime-tools", "private": True}, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    overrides: dict[str, str] = {}
                    if status.chrome.available and status.chrome.path:
                        overrides["PUPPETEER_SKIP_DOWNLOAD"] = "1"
                        overrides["PUPPETEER_SKIP_CHROMIUM_DOWNLOAD"] = "1"
                        overrides["PUPPETEER_EXECUTABLE_PATH"] = status.chrome.path
                    env = self.build_clean_env(include_venv=True, env_overrides=overrides)
                    argv = [
                        status.npm.path,
                        "install",
                        "--prefix",
                        str(self.npm_tools_dir),
                        "--no-audit",
                        "--no-fund",
                        "@mermaid-js/mermaid-cli",
                    ]
                    rc, out, err, ms = await self._run_prepare_cmd(
                        argv, cwd=self.npm_tools_dir, env=env, timeout_seconds=timeout_seconds,
                    )
                    mmdc_path = self.npm_bin_dir / "mmdc"
                    ok = rc == 0 and _is_executable(mmdc_path)
                    steps.append(PrepareStepResult(
                        step="install_mermaid_cli",
                        ok=ok,
                        message=(
                            "已安装 @mermaid-js/mermaid-cli（复用本机 Chrome）。"
                            if ok and status.chrome.available
                            else "已安装 @mermaid-js/mermaid-cli。" if ok
                            else "安装 @mermaid-js/mermaid-cli 失败。"
                        ),
                        duration_ms=ms,
                        stdout=out,
                        stderr=err,
                    ))
                    if not ok:
                        all_ok = False

            updated = await asyncio.to_thread(self.probe_status)
            return PrepareResult(ok=all_ok, steps=tuple(steps), status=updated)
