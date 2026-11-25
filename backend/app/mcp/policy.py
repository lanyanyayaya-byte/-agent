"""Application-level MCP configuration, path, environment, network, and hashing policy.

Security boundary note:
This module enforces **application-level** validation, environment allowlisting,
shell-free executable resolution, URL/IP controls, and redaction. It is NOT an
operating-system sandbox: a local stdio MCP server process still runs with the
current user's OS privileges.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.mcp.models import McpError, McpServerConfig

SERVER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]{1,128}$")
BEARER_PATTERN = re.compile(r"(?i)\b(bearer|token|api[_-]?key|secret)\s*[:=]?\s*[A-Za-z0-9._~+/=-]{8,}")
SK_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")

MAX_NAME_LENGTH = 100
MAX_ARGS_COUNT = 64
MAX_ARG_LENGTH = 4096
MAX_ENV_COUNT = 64
MAX_HEADER_COUNT = 32
MAX_SECRET_VALUE_BYTES = 4096
MAX_DIAGNOSTIC_MESSAGE_CHARS = 300

# Strictly allowlisted base environment variables inherited from the host process.
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
)

# Forbidden prefixes that must NEVER leak into or be injected into MCP child processes.
FORBIDDEN_ENV_PREFIXES: tuple[str, ...] = (
    "OPENAI_",
    "DEEPSEEK_",
    "SILICONFLOW_",
    "LANGSMITH_",
    "LANGCHAIN_",
    "KNOWLEDGEAGENT_",
    "DYLD_",
    "LD_",
)

FORBIDDEN_ENV_EXACT: frozenset[str] = frozenset({
    "PYTHONSTARTUP",
    "PYTHONPATH",
})

# Fixed, deterministic, shell-free macOS executable search directories.
MACOS_STANDARD_BIN_DIRS: tuple[str, ...] = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
)

HIGH_RISK_EXECUTABLES: frozenset[str] = frozenset({
    "npx",
    "uvx",
    "bunx",
    "pipx",
    "docker",
    "sh",
    "bash",
    "zsh",
    "osascript",
})

LOOPBACK_HOSTNAMES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

# Reserved HTTP headers managed by the MCP transport / HTTP layer.
FORBIDDEN_CUSTOM_HEADERS: frozenset[str] = frozenset({
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "mcp-session-id",
    "mcp-protocol-version",
})


@dataclass(frozen=True, slots=True)
class ValidatedHttpTarget:
    url: str
    scheme: str
    hostname: str
    port: int
    allow_private_network: bool


def _has_control_chars(value: str, *, allow_tabs: bool = False) -> bool:
    for ch in value:
        code = ord(ch)
        if code == 0 or code in (10, 13):
            return True
        if code < 32 and not (allow_tabs and ch == "\t"):
            return True
        if code == 127:
            return True
    return False


def _validate_env_mapping(mapping: Mapping[str, str], *, is_secret: bool) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str) or not ENV_KEY_PATTERN.fullmatch(key):
            raise McpError("MCP_INVALID_CONFIG", "环境变量名格式无效。")
        upper = key.upper()
        if upper in FORBIDDEN_ENV_EXACT or upper.startswith(FORBIDDEN_ENV_PREFIXES):
            raise McpError("MCP_INVALID_CONFIG", f"不允许向 MCP 子进程传入受保护的环境变量: {key}")
        if not isinstance(value, str) or "\x00" in value:
            raise McpError("MCP_INVALID_CONFIG", f"环境变量 {key} 的值无效。")
        if is_secret:
            if not value or "\r" in value or "\n" in value or len(value.encode("utf-8")) > MAX_SECRET_VALUE_BYTES:
                raise McpError("MCP_INVALID_CONFIG", f"秘钥环境变量 {key} 超出长度或包含非法换行符。")


def _validate_header_mapping(mapping: Mapping[str, str], *, is_secret: bool) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str) or not HEADER_NAME_PATTERN.fullmatch(key):
            raise McpError("MCP_INVALID_CONFIG", "HTTP Header 名称格式无效。")
        if key.lower() in FORBIDDEN_CUSTOM_HEADERS:
            raise McpError("MCP_INVALID_CONFIG", f"不允许覆盖受保护的 HTTP Header: {key}")
        if not isinstance(value, str) or _has_control_chars(value, allow_tabs=True):
            raise McpError("MCP_INVALID_CONFIG", f"HTTP Header {key} 包含非法控制字符。")
        if is_secret:
            if not value or len(value.encode("utf-8")) > MAX_SECRET_VALUE_BYTES:
                raise McpError("MCP_INVALID_CONFIG", f"秘钥 Header {key} 不能为空或超过 4096 字节。")


def validate_server_config(config: McpServerConfig) -> None:
    """Validate structural, transport, command/URL, and header/env constraints."""
    if not isinstance(config.server_id, str) or not SERVER_ID_PATTERN.fullmatch(config.server_id):
        raise McpError("MCP_INVALID_CONFIG", "server_id 格式无效。")
    if not isinstance(config.name, str) or not 1 <= len(config.name.strip()) <= MAX_NAME_LENGTH:
        raise McpError("MCP_INVALID_CONFIG", "服务名称不能为空且不超过 100 字符。")
    if _has_control_chars(config.name):
        raise McpError("MCP_INVALID_CONFIG", "服务名称不能包含控制字符。")
    if config.config_revision < 1:
        raise McpError("MCP_INVALID_CONFIG", "config_revision 必须大于等于 1。")

    if config.transport == "stdio":
        if not config.command or not isinstance(config.command, str) or not config.command.strip():
            raise McpError("MCP_INVALID_CONFIG", "stdio 服务必须指定 command。")
        if _has_control_chars(config.command):
            raise McpError("MCP_INVALID_CONFIG", "stdio command 包含非法控制字符。")
        if config.url is not None or config.headers or config.secret_headers:
            raise McpError("MCP_INVALID_CONFIG", "stdio 服务不能配置 url 或 HTTP headers。")
        if len(config.args) > MAX_ARGS_COUNT:
            raise McpError("MCP_INVALID_CONFIG", f"stdio 参数数量不能超过 {MAX_ARGS_COUNT}。")
        for arg in config.args:
            if not isinstance(arg, str) or "\x00" in arg or len(arg) > MAX_ARG_LENGTH:
                raise McpError("MCP_INVALID_CONFIG", "stdio 参数包含非法字符或超出长度上限。")
        if config.cwd is not None:
            if not isinstance(config.cwd, str) or _has_control_chars(config.cwd):
                raise McpError("MCP_INVALID_CONFIG", "cwd 包含非法字符。")
            cwd_path = Path(config.cwd)
            if not cwd_path.is_absolute() or ".." in cwd_path.parts:
                raise McpError("MCP_INVALID_CONFIG", "cwd 必须为不含相对跳转的绝对路径。")
            if not cwd_path.is_dir():
                raise McpError("MCP_INVALID_CONFIG", "指定的 cwd 目录不存在。")
        total_env = len(config.env) + len(config.secret_env)
        if total_env > MAX_ENV_COUNT:
            raise McpError("MCP_INVALID_CONFIG", f"环境变量总数不能超过 {MAX_ENV_COUNT}。")
        overlap = set(config.env) & set(config.secret_env)
        if overlap:
            raise McpError("MCP_INVALID_CONFIG", "明文环境变量与秘钥环境变量存在同名冲突。")
        _validate_env_mapping(config.env, is_secret=False)
        _validate_env_mapping(config.secret_env, is_secret=True)
        return

    if config.transport == "streamable_http":
        if not config.url or not isinstance(config.url, str):
            raise McpError("MCP_INVALID_CONFIG", "Streamable HTTP 服务必须指定 url。")
        if config.command is not None or config.args or config.cwd is not None or config.env or config.secret_env:
            raise McpError("MCP_INVALID_CONFIG", "Streamable HTTP 服务不能配置 stdio 相关字段。")
        validate_http_url(config.url, allow_private_network=config.allow_private_network)
        total_headers = len(config.headers) + len(config.secret_headers)
        if total_headers > MAX_HEADER_COUNT:
            raise McpError("MCP_INVALID_CONFIG", f"HTTP Header 总数不能超过 {MAX_HEADER_COUNT}。")
        lower_plain = {k.lower() for k in config.headers}
        lower_secret = {k.lower() for k in config.secret_headers}
        if lower_plain & lower_secret:
            raise McpError("MCP_INVALID_CONFIG", "明文 Header 与秘钥 Header 存在同名冲突。")
        _validate_header_mapping(config.headers, is_secret=False)
        _validate_header_mapping(config.secret_headers, is_secret=True)
        return

    raise McpError("MCP_INVALID_CONFIG", f"不支持的传输协议: {config.transport}")


def _deterministic_search_dirs(current_path: str | None = None) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        raw = raw.strip()
        if not raw or raw in seen:
            return
        seen.add(raw)
        dirs.append(Path(raw))

    if current_path:
        for entry in current_path.split(os.pathsep):
            _add(entry)
    for entry in MACOS_STANDARD_BIN_DIRS[:2]:  # /opt/homebrew/bin, /usr/local/bin
        _add(entry)
    home = os.environ.get("HOME")
    if home:
        _add(str(Path(home) / ".local" / "bin"))
    for entry in MACOS_STANDARD_BIN_DIRS[2:]:  # /usr/bin, /bin
        _add(entry)
    return dirs


def resolve_stdio_executable(command: str, *, path_env: str | None = None) -> Path:
    """Resolve an executable path deterministically without invoking a shell."""
    if not command or _has_control_chars(command):
        raise McpError("MCP_INVALID_CONFIG", "stdio command 无效。")

    if "/" in command or "\\" in command:
        if "\\" in command:
            raise McpError("MCP_INVALID_CONFIG", "stdio command 不支持反斜杠路径。")
        candidate = Path(command)
        if not candidate.is_absolute():
            raise McpError("MCP_INVALID_CONFIG", "包含路径分隔符的 command 必须使用绝对路径。")
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise McpError("MCP_EXECUTABLE_NOT_FOUND", f"找不到可执行文件或缺少执行权限: {command}")
        return candidate.resolve()

    search_dirs = _deterministic_search_dirs(os.environ.get("PATH") if path_env is None else path_env)
    for directory in search_dirs:
        candidate = directory / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise McpError("MCP_EXECUTABLE_NOT_FOUND", f"在受控路径中未找到可执行程序: {command}")


def is_high_risk_command(resolved_executable: Path, args: tuple[str, ...]) -> bool:
    """Identify dynamic package runners or shells that merit prominent trust warnings."""
    name = resolved_executable.name.lower()
    if name in HIGH_RISK_EXECUTABLES:
        return True
    if name.startswith("python") and any(arg == "-c" for arg in args):
        return True
    return False


def build_stdio_environment(
    config: McpServerConfig,
    *,
    host_environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct a minimal, isolated environment for a stdio MCP subprocess.

    Model API keys, internal Sidecar tokens, SHELL, and dynamic linker variables
    are NEVER inherited from the host process.
    """
    source = os.environ if host_environ is None else host_environ
    result: dict[str, str] = {}

    for key in SAFE_INHERITED_ENV_KEYS:
        value = source.get(key)
        if value is None or value.startswith("()") or "\x00" in value:
            continue
        result[key] = value

    # Provide a deterministic PATH so packaged macOS GUI apps can still locate
    # sibling runtimes (e.g. node/python in Homebrew) without inheriting secrets.
    path_entries = [str(d) for d in _deterministic_search_dirs(source.get("PATH"))]
    if path_entries:
        result["PATH"] = os.pathsep.join(path_entries)

    for mapping in (config.env, config.secret_env):
        for key, value in mapping.items():
            upper = key.upper()
            if upper in FORBIDDEN_ENV_EXACT or upper.startswith(FORBIDDEN_ENV_PREFIXES):
                raise McpError("MCP_INVALID_CONFIG", f"禁止注入受保护的环境变量: {key}")
            result[key] = value

    # Defense-in-depth scrub on the final environment dictionary.
    for key in tuple(result):
        upper = key.upper()
        if upper in FORBIDDEN_ENV_EXACT or upper.startswith(FORBIDDEN_ENV_PREFIXES):
            del result[key]

    return result


def validate_http_url(url: str, *, allow_private_network: bool = False) -> ValidatedHttpTarget:
    """Validate Streamable HTTP endpoint URL against transport & network policy."""
    if not url or _has_control_chars(url):
        raise McpError("MCP_INVALID_CONFIG", "HTTP URL 无效。")

    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("https", "http"):
        raise McpError("MCP_NETWORK_POLICY_VIOLATION", "仅支持 https:// 或本机回环 http:// 端点。")
    if parsed.username is not None or parsed.password is not None:
        raise McpError("MCP_NETWORK_POLICY_VIOLATION", "URL 中禁止携带明文凭据（user:pass@）。")
    if parsed.fragment:
        raise McpError("MCP_INVALID_CONFIG", "HTTP URL 不能包含片段标识（#）。")
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise McpError("MCP_INVALID_CONFIG", "HTTP URL 缺少有效主机名。")

    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise McpError("MCP_INVALID_CONFIG", "HTTP URL 端口号无效。") from exc
    if not 1 <= port <= 65535:
        raise McpError("MCP_INVALID_CONFIG", "HTTP URL 端口号超出范围。")

    if scheme == "http" and hostname not in LOOPBACK_HOSTNAMES:
        raise McpError(
            "MCP_NETWORK_POLICY_VIOLATION",
            "非本机回环地址禁止使用明文 HTTP，请改用 HTTPS。",
        )

    # If hostname is already a literal IP address, validate it immediately too.
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        validate_resolved_ip(
            hostname,
            scheme=scheme,
            hostname=hostname,
            allow_private_network=allow_private_network,
        )

    return ValidatedHttpTarget(
        url=url,
        scheme=scheme,
        hostname=hostname,
        port=port,
        allow_private_network=allow_private_network,
    )


def validate_resolved_ip(
    ip_text: str,
    *,
    scheme: str,
    hostname: str,
    allow_private_network: bool = False,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Validate an actual resolved IP right before TCP connect (anti-DNS-rebinding)."""
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError as exc:
        raise McpError("MCP_NETWORK_POLICY_VIOLATION", "无法识别解析后的目标 IP 地址。") from exc

    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    if ip.is_unspecified or ip.is_multicast or ip.is_link_local or ip.is_reserved:
        raise McpError(
            "MCP_NETWORK_POLICY_VIOLATION",
            "目标地址解析至受禁网段（链路本地、云元数据或保留地址），已拒绝连接。",
        )

    if scheme == "http" and not ip.is_loopback:
        raise McpError(
            "MCP_NETWORK_POLICY_VIOLATION",
            f"明文 HTTP 主机 {hostname} 解析到非回环地址，已拒绝连接。",
        )

    if ip.is_loopback:
        return ip

    if ip.is_private and not allow_private_network:
        raise McpError(
            "MCP_NETWORK_POLICY_VIOLATION",
            "目标地址解析至私有内网网段，未显式允许内网访问，已拒绝连接。",
        )

    return ip


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_tool_definition_hash(
    *,
    name: str,
    title: str | None,
    description: str,
    input_schema: Mapping[str, Any],
    output_schema: Mapping[str, Any] | None,
    annotations: Mapping[str, Any],
) -> str:
    """Compute a stable SHA-256 digest over a tool's security-relevant definition."""
    payload = {
        "name": name,
        "title": title,
        "description": description,
        "input_schema": dict(input_schema),
        "output_schema": dict(output_schema) if output_schema is not None else None,
        "annotations": dict(annotations),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def compute_config_trust_fingerprint(
    config: McpServerConfig,
    *,
    resolved_command: Path | None = None,
) -> str:
    """Compute a SHA-256 fingerprint over security-relevant server configuration.

    Display `name` is intentionally excluded so renaming a server does not
    alter its trust fingerprint or tool identity. Secret values are excluded
    (only secret key names are bound) so secret rotation does not embed secrets.
    """
    payload = {
        "transport": config.transport,
        "command": str(resolved_command) if resolved_command is not None else config.command,
        "args": list(config.args),
        "cwd": config.cwd,
        "url": config.url,
        "env": sorted(config.env.items()),
        "secret_env_keys": sorted(config.secret_env.keys()),
        "headers": sorted((k.lower(), v) for k, v in config.headers.items()),
        "secret_header_keys": sorted(k.lower() for k in config.secret_headers.keys()),
        "allow_private_network": config.allow_private_network,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def redact_text(
    text: str,
    *,
    secrets: Iterable[str] = (),
    max_chars: int = MAX_DIAGNOSTIC_MESSAGE_CHARS,
) -> str:
    """Redact known secret values and common token shapes from diagnostics/errors."""
    sanitized = text.replace("\r", " ").replace("\n", " ").strip()
    for secret in secrets:
        if secret and len(secret) >= 4:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = BEARER_PATTERN.sub(lambda m: f"{m.group(1)} [REDACTED]", sanitized)
    sanitized = SK_TOKEN_PATTERN.sub("[REDACTED]", sanitized)
    if len(sanitized) > max_chars:
        sanitized = sanitized[: max_chars - 1] + "…"
    return sanitized
