import json
from dataclasses import dataclass
from typing import Literal

ToolEvent = Literal["tool_started", "tool_completed", "tool_failed"]

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
    "INTERNAL_ERROR": "无法完成工具调用。",
}


class ToolError(Exception):
    def __init__(self, code: str):
        super().__init__(ERROR_MESSAGES[code])
        self.code = code


@dataclass(frozen=True)
class ToolCall:
    tool_call_id: str
    tool_name: str
    arguments: object
    session_id: str
    run_id: str


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    ok: bool
    content: str | dict | None
    display_summary: str
    truncated: bool = False
    error_code: str | None = None
    next_offset_line: int | None = None

    @classmethod
    def failure(cls, call_id: str, code: str) -> "ToolResult":
        return cls(call_id, False, None, ERROR_MESSAGES[code], error_code=code)

    def for_model(self) -> str:
        return encode_json({
            "tool_call_id": self.tool_call_id, "ok": self.ok, "content": self.content,
            "display_summary": self.display_summary, "truncated": self.truncated,
            "error_code": self.error_code, "next_offset_line": self.next_offset_line,
        })


def encode_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
