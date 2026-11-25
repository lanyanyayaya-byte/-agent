import json
from dataclasses import dataclass
from typing import Literal

ToolEvent = Literal["tool_started", "tool_completed", "tool_failed"]
ToolSource = Literal["native", "mcp"]

ERROR_MESSAGES = {
    "VALIDATION_ERROR": "工具参数无效。",
    "TOOL_NOT_FOUND": "该工具未注册。",
    "FILE_PERMISSION_DENIED": "文件或目录未获有效读取授权，或目标已变化。",
    "FILE_TOO_LARGE": "文件超过 1 MiB 读取上限。",
    "UNSUPPORTED_FILE_TYPE": "只支持普通 TXT 或 Markdown 文件。",
    "INVALID_TEXT_FILE": "文件包含二进制数据或无法按 UTF-8、GBK 解码。",
    "DIRECTORY_LIMIT_EXCEEDED": "目录超出单次扫描上限，请指定更小的子目录。",
    "RATE_LIMITED": "本轮工具调用次数或返回内容已达到上限。",
    "RUN_CANCELLED": "工具调用已取消。",
    "RUN_TIMEOUT": "本轮对话执行总时长已超过上限。",
    "INTERNAL_ERROR": "无法完成工具调用。",
    "MCP_SERVER_UNAVAILABLE": "MCP 服务当前未连接或不可用。",
    "MCP_PERMISSION_DENIED": "MCP 工具未获得有效授权，或服务配置/工具定义已发生变化。",
    "MCP_TOOL_TIMEOUT": "MCP 工具调用超时，外部执行结果未知。",
    "MCP_TOOL_RETURNED_ERROR": "MCP 工具返回了错误结果。",
    "MCP_PROTOCOL_ERROR": "MCP 协议通信失败或连接中断。",
    "MCP_RESPONSE_TOO_LARGE": "MCP 响应体积超过安全上限，已中止读取。",
    "MCP_MALFORMED_RESPONSE": "MCP 服务返回了畸形或无法解析的数据。",
    "MCP_SERVER_BUSY": "该 MCP 服务当前请求排队已满，请稍后重试。",
    "MCP_TOOL_LIMIT_EXCEEDED": "本轮暴露给模型的工具数量或 Schema 总大小超出安全上限，请精简已授权工具。",
    "MCP_CALL_ABORTED": "同轮前置工具调用已失败，已中止后续外部 MCP 调用。",
    "MCP_EGRESS_AUTHORIZATION_REQUIRED": "本轮存在受保护的本地上下文，默认禁止同轮调用外部 MCP；请关闭其中一项或显式授权允许外部数据外发。",
}


class ToolError(Exception):
    def __init__(self, code: str, message: str | None = None):
        resolved = message or ERROR_MESSAGES.get(code, ERROR_MESSAGES["INTERNAL_ERROR"])
        super().__init__(resolved)
        self.code = code


@dataclass(frozen=True)
class ToolCall:
    tool_call_id: str
    tool_name: str
    arguments: object
    session_id: str
    run_id: str
    source: ToolSource = "native"
    server_id: str | None = None
    server_name: str | None = None
    original_tool_name: str | None = None
    definition_hash: str | None = None


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    ok: bool
    content: str | dict | None
    display_summary: str
    truncated: bool = False
    error_code: str | None = None
    next_offset_line: int | None = None
    source: ToolSource = "native"
    server_id: str | None = None
    server_name: str | None = None
    original_tool_name: str | None = None
    duration_ms: int | None = None

    @classmethod
    def failure(
        cls,
        call_id: str,
        code: str,
        *,
        display_summary: str | None = None,
        source: ToolSource = "native",
        server_id: str | None = None,
        server_name: str | None = None,
        original_tool_name: str | None = None,
        duration_ms: int | None = None,
    ) -> "ToolResult":
        message = display_summary or ERROR_MESSAGES.get(code, ERROR_MESSAGES["INTERNAL_ERROR"])
        return cls(
            call_id,
            False,
            None,
            message,
            error_code=code,
            source=source,
            server_id=server_id,
            server_name=server_name,
            original_tool_name=original_tool_name,
            duration_ms=duration_ms,
        )

    def for_model(self) -> str:
        return encode_json({
            "tool_call_id": self.tool_call_id, "ok": self.ok, "content": self.content,
            "display_summary": self.display_summary, "truncated": self.truncated,
            "error_code": self.error_code, "next_offset_line": self.next_offset_line,
        })


def encode_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
