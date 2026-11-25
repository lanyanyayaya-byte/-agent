from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

TransportKind = Literal["stdio", "streamable_http"]
ServerRuntimeState = Literal["disconnected", "connecting", "connected", "disconnecting", "error"]

McpErrorCode = Literal[
    "MCP_INVALID_CONFIG",
    "MCP_NETWORK_POLICY_VIOLATION",
    "MCP_EXECUTABLE_NOT_FOUND",
    "MCP_CONNECT_TIMEOUT",
    "MCP_CONNECT_FAILED",
    "MCP_SERVER_UNAVAILABLE",
    "MCP_SERVER_BUSY",
    "MCP_TOOL_TIMEOUT",
    "MCP_TOOL_RETURNED_ERROR",
    "MCP_TOOL_CANCELLED",
    "MCP_RESPONSE_TOO_LARGE",
    "MCP_MALFORMED_RESPONSE",
    "MCP_PROTOCOL_ERROR",
    "MCP_DISCOVERY_FAILED",
    "MCP_INTERNAL_ERROR",
]

MCP_ERROR_MESSAGES: dict[McpErrorCode, str] = {
    "MCP_INVALID_CONFIG": "MCP 服务配置无效。",
    "MCP_NETWORK_POLICY_VIOLATION": "MCP HTTP 目标地址违反安全网络策略。",
    "MCP_EXECUTABLE_NOT_FOUND": "未找到可执行的 MCP 本地程序。",
    "MCP_CONNECT_TIMEOUT": "连接 MCP 服务超时。",
    "MCP_CONNECT_FAILED": "无法连接到 MCP 服务。",
    "MCP_SERVER_UNAVAILABLE": "MCP 服务当前未连接或不可用。",
    "MCP_SERVER_BUSY": "该 MCP 服务当前请求排队已满，请稍后重试。",
    "MCP_TOOL_TIMEOUT": "MCP 工具调用超时，外部执行结果未知。",
    "MCP_TOOL_RETURNED_ERROR": "MCP 工具返回了错误结果。",
    "MCP_TOOL_CANCELLED": "MCP 工具调用已取消。",
    "MCP_RESPONSE_TOO_LARGE": "MCP 响应体积超过安全上限，已中止读取。",
    "MCP_MALFORMED_RESPONSE": "MCP 服务返回了畸形或无法解析的数据。",
    "MCP_PROTOCOL_ERROR": "MCP 协议通信失败或连接中断。",
    "MCP_DISCOVERY_FAILED": "获取 MCP 工具列表失败。",
    "MCP_INTERNAL_ERROR": "MCP 运行时内部错误。",
}


class McpError(Exception):
    """Stable, redacted MCP runtime error."""

    def __init__(
        self,
        code: McpErrorCode,
        message: str | None = None,
        *,
        uncertain_outcome: bool = False,
    ) -> None:
        resolved = message or MCP_ERROR_MESSAGES[code]
        super().__init__(resolved)
        self.code: McpErrorCode = code
        self.message: str = resolved
        self.uncertain_outcome: bool = uncertain_outcome


@dataclass(frozen=True, slots=True)
class McpRuntimeLimits:
    """Configurable safety & lifecycle limits for MCP Client and Manager."""

    connect_timeout_seconds: float = 15.0
    discovery_timeout_seconds: float = 15.0
    call_timeout_seconds: float = 60.0
    max_discovery_pages: int = 10
    max_discovered_tools: int = 200
    max_transport_bytes: int = 2 * 1024 * 1024  # 2 MiB pre-parse cap
    max_stderr_bytes: int = 8_192
    max_stderr_lines: int = 50
    max_queued_calls_per_server: int = 4
    shutdown_total_timeout_seconds: float = 2.5
    stdio_eof_wait_seconds: float = 0.8
    stdio_sigterm_wait_seconds: float = 0.8
    max_connect_concurrency: int = 3
    max_servers: int = 16
    max_active_servers: int = 8

    def __post_init__(self) -> None:
        for field_name in (
            "connect_timeout_seconds",
            "discovery_timeout_seconds",
            "call_timeout_seconds",
            "shutdown_total_timeout_seconds",
            "stdio_eof_wait_seconds",
            "stdio_sigterm_wait_seconds",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "max_discovery_pages",
            "max_discovered_tools",
            "max_transport_bytes",
            "max_stderr_bytes",
            "max_stderr_lines",
            "max_queued_calls_per_server",
            "max_connect_concurrency",
            "max_servers",
            "max_active_servers",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """In-memory MCP Server configuration (independent of P6.3 persistence)."""

    server_id: str
    name: str
    transport: TransportKind
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: str | None = None
    url: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    secret_env: Mapping[str, str] = field(default_factory=dict, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict)
    secret_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    enabled: bool = True
    config_revision: int = 1
    allow_private_network: bool = False

    def all_secret_values(self) -> tuple[str, ...]:
        values: list[str] = []
        for mapping in (self.secret_env, self.secret_headers):
            for value in mapping.values():
                if value:
                    values.append(value)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class McpToolDefinition:
    """Normalized MCP tool definition discovered from a server."""

    server_id: str
    original_tool_name: str
    title: str | None
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any]
    definition_hash: str


@dataclass(frozen=True, slots=True)
class McpContentBlock:
    """Normalized content block returned by tools/call."""

    kind: Literal["text", "omitted_non_text"]
    text: str | None = None
    original_type: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class McpToolCallResult:
    """Normalized outcome of an MCP tools/call invocation."""

    server_id: str
    original_tool_name: str
    generation: int
    is_error: bool
    content_blocks: tuple[McpContentBlock, ...]
    text: str
    structured_content: dict[str, Any] | None
    omitted_non_text_types: tuple[str, ...]
    duration_ms: int


@dataclass(frozen=True, slots=True)
class McpDiagnostics:
    """Redacted runtime snapshot for one MCP Server."""

    server_id: str
    name: str
    transport: TransportKind
    state: ServerRuntimeState
    generation: int
    protocol_version: str | None = None
    server_name_reported: str | None = None
    server_version_reported: str | None = None
    resolved_command: str | None = None
    child_pid: int | None = None
    tool_count: int = 0
    catalog_version: int = 0
    tools_truncated: bool = False
    last_error_code: McpErrorCode | None = None
    last_error_message: str | None = None
    stderr_tail: tuple[str, ...] = ()
    connected_at: str | None = None
