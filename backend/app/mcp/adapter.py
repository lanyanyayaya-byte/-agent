"""MCP Tool Adapter bridging P6.2/P6.3 MCP tools to DeepSeek Tool Calling and RunTools.

Design guarantees:
- Model tool alias is anchored strictly to stable `(server_id, original_tool_name)` so
  renaming a server's display `name` NEVER alters the model alias or tool identity, and
  same-named tools on different servers NEVER collide.
- Schema compatibility reuses P6.3 `check_tool_schema_compatibility()`; incompatible schemas
  are rejected explicitly without silently stripping security constraints.
- Pre-execution argument validation runs `jsonschema` against the ORIGINAL unmodified
  `input_schema` with remote `$ref` retrieval strictly forbidden and a 32 KiB JSON cap.
- Result adaptation reuses P6.2 `McpToolCallResult` (no duplicate wire parsing), supports
  text + `structured_content`, keeps non-text blocks as safe placeholders (never fetching
  resource URIs), enforces 16,000 chars / 24,000 UTF-8 bytes truncation with explicit marker,
  and strictly separates redacted `display_summary` from model-visible content.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jsonschema
from referencing import Registry
from referencing.exceptions import NoSuchResource

from app.mcp.models import McpError, McpToolCallResult
from app.mcp.policy import (
    MAX_TOOL_SCHEMA_BYTES,
    MAX_TOOL_SCHEMA_DEPTH,
    canonical_json,
    check_tool_schema_compatibility,
)
from app.tools.protocol import ToolError, ToolResult

DEEPSEEK_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
DEEPSEEK_MAX_TOOL_NAME_LEN = 128
MAX_MODEL_DESCRIPTION_CHARS = 1024
MAX_TOOL_ARGUMENTS_BYTES = 32 * 1024
MAX_MCP_RESULT_CHARS = 16_000
MAX_MCP_RESULT_BYTES = 24_000
MAX_DISPLAY_SUMMARY_CHARS = 200

_PURE_METADATA_SCHEMA_KEYS = frozenset({"$schema", "$id", "$comment", "examples", "$defs", "definitions"})


def _reject_remote_ref(uri: str):  # pragma: no cover - defensive retrieval blocker
    raise NoSuchResource(ref=uri)


# Offline-only referencing registry: forbids any network/file remote $ref retrieval.
_OFFLINE_SCHEMA_REGISTRY: Registry[Any] = Registry(retrieve=_reject_remote_ref)


def _slugify_ascii(value: str, *, max_len: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not cleaned:
        cleaned = "item"
    return cleaned[:max_len]


def make_mcp_tool_alias(
    server_id: str,
    original_tool_name: str,
    *,
    existing_aliases: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Generate a deterministic, DeepSeek-compliant alias anchored ONLY to `(server_id, original_tool_name)`.

    Changing `server.name` NEVER alters this alias.
    Different servers with the same `original_tool_name` produce distinct aliases.
    Always prefixed with `mcp__` so it can never collide with native tools (`list_files`, `read_text`).
    """
    server_slug = _slugify_ascii(server_id, max_len=16)
    server_hash = hashlib.sha256(server_id.encode("utf-8")).hexdigest()[:10]
    server_key = f"{server_slug}_{server_hash}"

    prefix = f"mcp__{server_key}__"
    max_tool_part = DEEPSEEK_MAX_TOOL_NAME_LEN - len(prefix)
    tool_hash = hashlib.sha256(original_tool_name.encode("utf-8")).hexdigest()[:10]

    is_plain_ascii = re.fullmatch(r"[A-Za-z0-9_-]+", original_tool_name) is not None
    if is_plain_ascii and len(original_tool_name) <= max_tool_part:
        candidate = f"{prefix}{original_tool_name}"
        if candidate not in existing_aliases and DEEPSEEK_TOOL_NAME_PATTERN.fullmatch(candidate):
            return candidate

    tool_slug_budget = max(1, max_tool_part - len(tool_hash) - 1)
    tool_slug = _slugify_ascii(original_tool_name, max_len=tool_slug_budget)
    candidate = f"{prefix}{tool_slug}_{tool_hash}"
    if candidate not in existing_aliases and DEEPSEEK_TOOL_NAME_PATTERN.fullmatch(candidate):
        return candidate

    # Extreme fallback (should be unreachable in practice): full deterministic hash suffix
    pair_hash = hashlib.sha256(f"{server_id}\0{original_tool_name}".encode("utf-8")).hexdigest()[:16]
    return f"mcp__{server_hash}__{pair_hash}"


def normalize_mcp_input_schema(raw_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an MCP tool `inputSchema` into a DeepSeek-compatible JSON Schema object.

    Refuses incompatible schemas via `McpError("MCP_SCHEMA_INCOMPATIBLE")` rather than
    silently stripping validation constraints.
    """
    compatible, reason = check_tool_schema_compatibility(raw_schema)
    if not compatible:
        raise McpError("MCP_SCHEMA_INCOMPATIBLE", reason)

    defs: dict[str, Any] = {}
    for defs_key in ("$defs", "definitions"):
        raw_defs = raw_schema.get(defs_key)
        if isinstance(raw_defs, Mapping):
            for key, value in raw_defs.items():
                defs[f"#/{defs_key}/{key}"] = value

    def _expand(node: Any, depth: int, active_refs: frozenset[str]) -> Any:
        if depth > MAX_TOOL_SCHEMA_DEPTH:
            raise McpError("MCP_SCHEMA_INCOMPATIBLE", f"工具参数 Schema 嵌套深度超过 {MAX_TOOL_SCHEMA_DEPTH} 层上限。")
        if isinstance(node, list):
            return [_expand(item, depth + 1, active_refs) for item in node]
        if not isinstance(node, Mapping):
            return copy.deepcopy(node)

        ref = node.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or not ref.startswith("#/") or ref in active_refs or ref not in defs:
                raise McpError("MCP_SCHEMA_INCOMPATIBLE", "工具参数 Schema 包含不支持或无法解析的 $ref 引用。")
            resolved = _expand(defs[ref], depth + 1, active_refs | {ref})
            if not isinstance(resolved, dict):
                return resolved
            merged = dict(resolved)
            for k, v in node.items():
                if k == "$ref" or k in _PURE_METADATA_SCHEMA_KEYS:
                    continue
                merged[k] = _expand(v, depth + 1, active_refs | {ref})
            return merged

        result: dict[str, Any] = {}
        for k, v in node.items():
            if k in _PURE_METADATA_SCHEMA_KEYS:
                continue
            if k == "properties" and isinstance(v, Mapping):
                result["properties"] = {
                    str(prop_name): _expand(prop_schema, depth + 1, active_refs)
                    for prop_name, prop_schema in v.items()
                }
            elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                result[k] = [_expand(branch, depth + 1, active_refs) for branch in v]
            elif k in ("items", "additionalProperties") and isinstance(v, (Mapping, list)):
                result[k] = _expand(v, depth + 1, active_refs)
            else:
                result[k] = copy.deepcopy(v)
        return result

    normalized = _expand(dict(raw_schema), 0, frozenset())
    if not isinstance(normalized, dict):
        raise McpError("MCP_SCHEMA_INCOMPATIBLE", "工具参数 Schema 顶层必须为 JSON 对象。")
    normalized.setdefault("type", "object")
    if normalized["type"] != "object":
        raise McpError("MCP_SCHEMA_INCOMPATIBLE", f"工具参数 Schema 顶层 type 必须为 object（当前为 {normalized['type']}）。")
    if "properties" not in normalized or not isinstance(normalized["properties"], dict):
        normalized["properties"] = {}

    serialized = canonical_json(normalized).encode("utf-8")
    if len(serialized) > MAX_TOOL_SCHEMA_BYTES:
        raise McpError("MCP_SCHEMA_INCOMPATIBLE", f"规范化后的工具参数 Schema 超过 {MAX_TOOL_SCHEMA_BYTES // 1024} KiB 上限。")
    return normalized


def build_deepseek_tool_schema(
    *,
    alias: str,
    server_name: str,
    original_tool_name: str,
    description: str,
    normalized_input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Build the OpenAI/DeepSeek function tool schema dict passed to `ChatDeepSeek.bind_tools()`."""
    if not DEEPSEEK_TOOL_NAME_PATTERN.fullmatch(alias):
        raise McpError("MCP_INVALID_CONFIG", f"生成的模型工具别名无效: {alias}")

    prefix = f"[MCP:{server_name}/{original_tool_name}] "
    cleaned_desc = (description or "").strip()
    combined = f"{prefix}{cleaned_desc}" if cleaned_desc else f"{prefix}外部只读 MCP 工具。"
    if len(combined) > MAX_MODEL_DESCRIPTION_CHARS:
        combined = combined[: MAX_MODEL_DESCRIPTION_CHARS - 1] + "…"

    return {
        "type": "function",
        "function": {
            "name": alias,
            "description": combined,
            "parameters": normalized_input_schema,
        },
    }


def _select_validator_cls(raw_schema: Mapping[str, Any]):
    schema_uri = str(raw_schema.get("$schema") or "").lower()
    if "draft-07" in schema_uri or "draft7" in schema_uri:
        return jsonschema.Draft7Validator
    if "2019-09" in schema_uri:
        return jsonschema.Draft201909Validator
    return jsonschema.Draft202012Validator


def validate_mcp_tool_arguments(raw_input_schema: Mapping[str, Any], arguments: object) -> dict[str, Any]:
    """Validate model-generated arguments against the ORIGINAL unmodified `input_schema`.

    Enforces:
    - Top-level JSON object (`dict`)
    - Serialized JSON UTF-8 size <= 32 KiB
    - Strict `jsonschema` validation with remote `$ref` retrieval disabled
    """
    if not isinstance(arguments, dict):
        raise ToolError("VALIDATION_ERROR")
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        raise ToolError("VALIDATION_ERROR") from None
    if len(encoded) > MAX_TOOL_ARGUMENTS_BYTES:
        raise ToolError("VALIDATION_ERROR")

    compatible, _ = check_tool_schema_compatibility(raw_input_schema)
    if not compatible:
        raise ToolError("VALIDATION_ERROR")

    validator_cls = _select_validator_cls(raw_input_schema)
    schema_dict = dict(raw_input_schema)
    try:
        validator = validator_cls(schema_dict, registry=_OFFLINE_SCHEMA_REGISTRY)
        validator.validate(arguments)
    except (jsonschema.ValidationError, jsonschema.SchemaError, Exception):
        raise ToolError("VALIDATION_ERROR") from None
    return dict(arguments)


def _truncate_text_by_chars_and_bytes(
    text: str,
    *,
    max_chars: int = MAX_MCP_RESULT_CHARS,
    max_bytes: int = MAX_MCP_RESULT_BYTES,
) -> tuple[str, bool]:
    """Truncate text cleanly on Unicode character boundaries respecting both char and UTF-8 byte caps."""
    truncated = False
    candidate = text
    if len(candidate) > max_chars:
        candidate = candidate[:max_chars]
        truncated = True

    encoded = candidate.encode("utf-8")
    if len(encoded) <= max_bytes:
        return candidate, truncated

    truncated = True
    # Trim cleanly on UTF-8 boundary without splitting multi-byte characters.
    trimmed = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return trimmed, truncated


def adapt_mcp_call_result(
    *,
    call_id: str,
    server_id: str,
    server_name: str,
    original_tool_name: str,
    raw_result: McpToolCallResult,
) -> ToolResult:
    """Adapt a P6.2-normalized `McpToolCallResult` into a `ToolResult`.

    Strictly separates:
    - `content`: model-visible bounded body (text + structured_content + non-text placeholders)
    - `display_summary`: redacted metadata-only summary (<= 200 chars, safe for DB/checkpoints/UI)
    """
    duration_ms = max(0, int(raw_result.duration_ms))

    if raw_result.is_error:
        summary = f"MCP 工具 {original_tool_name} 返回错误结果。"
        return ToolResult(
            tool_call_id=call_id,
            ok=False,
            content=None,
            display_summary=summary[:MAX_DISPLAY_SUMMARY_CHARS],
            truncated=False,
            error_code="MCP_TOOL_RETURNED_ERROR",
            source="mcp",
            server_id=server_id,
            server_name=server_name,
            original_tool_name=original_tool_name,
            duration_ms=duration_ms,
        )

    parts: list[str] = []
    for block in raw_result.content_blocks:
        if block.kind == "text":
            if block.text is not None:
                parts.append(block.text)
        else:
            block_type = block.original_type or "unknown"
            parts.append(f"[已忽略不受支持的 MCP 非文本内容: {block_type}]")

    if raw_result.structured_content is not None:
        structured_text = json.dumps(raw_result.structured_content, ensure_ascii=False, separators=(",", ":"))
        if not parts:
            parts.append(structured_text)
        else:
            parts.append(f"[structured_content]\n{structured_text}")

    raw_body = "\n".join(parts)
    bounded_body, truncated = _truncate_text_by_chars_and_bytes(raw_body)
    if truncated:
        marker = "\n[输出已按安全上限截断]"
        marker_bytes = len(marker.encode("utf-8"))
        prefix_body, _ = _truncate_text_by_chars_and_bytes(
            bounded_body,
            max_chars=max(0, MAX_MCP_RESULT_CHARS - len(marker)),
            max_bytes=max(0, MAX_MCP_RESULT_BYTES - marker_bytes),
        )
        bounded_body = prefix_body + marker

    # Redacted summary for tool_calls.summary and checkpoints — NEVER embeds raw external content!
    char_count = len(bounded_body)
    summary_parts = [f"MCP 工具返回 {char_count} 字符"]
    if raw_result.structured_content is not None:
        summary_parts.append("含结构化数据")
    if raw_result.omitted_non_text_types:
        summary_parts.append(f"已忽略 {len(raw_result.omitted_non_text_types)} 项非文本内容")
    if truncated:
        summary_parts.append("已截断")
    display_summary = "（".join([summary_parts[0], "，".join(summary_parts[1:]) + "）"]) if len(summary_parts) > 1 else f"{summary_parts[0]}。"
    if len(display_summary) > MAX_DISPLAY_SUMMARY_CHARS:
        display_summary = display_summary[: MAX_DISPLAY_SUMMARY_CHARS - 1] + "…"

    model_payload: dict[str, Any] = {
        "source": "mcp",
        "server_name": server_name,
        "tool_name": original_tool_name,
        "text": bounded_body,
        "truncated": truncated,
    }
    if raw_result.omitted_non_text_types:
        model_payload["omitted_non_text_types"] = list(raw_result.omitted_non_text_types)
    if raw_result.structured_content is not None and not truncated:
        # Include structured_content directly only when total body fits within the single-call cap.
        structured_probe = json.dumps(model_payload, ensure_ascii=False, separators=(",", ":"))
        extra_probe = json.dumps(raw_result.structured_content, ensure_ascii=False, separators=(",", ":"))
        if (
            len(structured_probe) + len(extra_probe) <= MAX_MCP_RESULT_CHARS
            and len(structured_probe.encode("utf-8")) + len(extra_probe.encode("utf-8")) <= MAX_MCP_RESULT_BYTES
        ):
            model_payload["structured_content"] = raw_result.structured_content

    return ToolResult(
        tool_call_id=call_id,
        ok=True,
        content=model_payload,
        display_summary=display_summary,
        truncated=truncated,
        error_code=None,
        source="mcp",
        server_id=server_id,
        server_name=server_name,
        original_tool_name=original_tool_name,
        duration_ms=duration_ms,
    )


@dataclass(frozen=True, slots=True)
class McpAuthorizedToolSnapshot:
    """Immutable per-run snapshot of one effectively authorized MCP tool."""

    alias: str
    tool_id: str
    server_id: str
    server_name_snapshot: str
    original_tool_name: str
    description: str
    raw_input_schema: dict[str, Any]
    normalized_input_schema: dict[str, Any]
    model_schema: dict[str, Any]
    definition_hash: str
    config_revision: int
    generation: int
    catalog_version: int
