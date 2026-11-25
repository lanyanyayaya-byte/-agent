"""SQLite persistence layer for MCP servers and discovered tools (Migration V6).

Reuses `BusinessStore`'s SQLite connection and `BEGIN IMMEDIATE` transaction
semantics inside `knowledgeagent.sqlite3`. Plaintext secret values are NEVER
persisted in SQLite — only typed `{kind: "plain", value}` or
`{kind: "secret", secret_key}` descriptors are stored.

Security note:
Persisted SQLite authorization rows (`permission_state = 'allowed'`) are NEVER
treated as sufficient authorization on their own. Effective runtime authorization
must always combine persisted state with live `McpManager` connection state,
fingerprint verification, credential completeness, non-stale catalog state, and
live `definition_hash` equality.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from app.mcp.models import McpError, McpServerConfig, McpToolDefinition, TransportKind
from app.mcp.policy import (
    MAX_ARGS_COUNT,
    MAX_ENV_COUNT,
    MAX_HEADER_COUNT,
    canonical_json,
    check_secret_bypass,
    check_tool_schema_compatibility,
    compute_config_trust_fingerprint,
    compute_tool_definition_hash,
    resolve_stdio_executable,
    validate_server_config,
)
from app.storage.business import BusinessStore, utc_now

TrustState = Literal["untrusted", "trusted"]
ToolPermissionState = Literal["denied", "allowed"]
ConfigValueKind = Literal["plain", "secret"]

MAX_CONFIGURED_SERVERS = 16


@dataclass(frozen=True, slots=True)
class ConfigValueSpec:
    """Typed non-secret descriptor persisted in env_config_json / header_config_json."""

    kind: ConfigValueKind
    value: str | None = None
    secret_key: str | None = None

    def to_json_dict(self) -> dict[str, str]:
        if self.kind == "plain":
            return {"kind": "plain", "value": self.value or ""}
        return {"kind": "secret", "secret_key": self.secret_key or ""}

    @classmethod
    def from_json_dict(cls, key: str, raw: Mapping[str, Any]) -> ConfigValueSpec:
        kind = raw.get("kind")
        if kind == "plain":
            val = raw.get("value")
            if not isinstance(val, str):
                raise McpError("MCP_INVALID_CONFIG", f"配置项 {key} 的明文值无效。")
            return cls(kind="plain", value=val)
        if kind == "secret":
            secret_key = raw.get("secret_key") or key
            if not isinstance(secret_key, str) or not secret_key.strip():
                raise McpError("MCP_INVALID_CONFIG", f"配置项 {key} 的秘钥引用键无效。")
            if secret_key != key:
                raise McpError("MCP_INVALID_CONFIG", f"配置项 {key} 的 secret_key 必须与键名一致。")
            return cls(kind="secret", secret_key=secret_key)
        raise McpError("MCP_INVALID_CONFIG", f"配置项 {key} 包含不支持的类型: {kind}")


def _parse_config_map(raw_json: str) -> dict[str, ConfigValueSpec]:
    loaded = json.loads(raw_json)
    if not isinstance(loaded, dict):
        raise McpError("MCP_INVALID_CONFIG", "持久化配置映射格式损坏。")
    result: dict[str, ConfigValueSpec] = {}
    for key, entry in loaded.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            raise McpError("MCP_INVALID_CONFIG", "持久化配置项格式损坏。")
        result[key] = ConfigValueSpec.from_json_dict(key, entry)
    return result


def _serialize_config_map(mapping: Mapping[str, ConfigValueSpec]) -> str:
    payload = {key: spec.to_json_dict() for key, spec in sorted(mapping.items())}
    return canonical_json(payload)


def try_resolve_executable(command: str | None) -> Path | None:
    if not command:
        return None
    try:
        return resolve_stdio_executable(command)
    except McpError:
        return None


@dataclass(frozen=True, slots=True)
class StoredMcpServer:
    server_id: str
    name: str
    transport: TransportKind
    command: str | None
    args: tuple[str, ...]
    cwd: str | None
    url: str | None
    allow_private_network: bool
    env_config: Mapping[str, ConfigValueSpec]
    header_config: Mapping[str, ConfigValueSpec]
    credential_ref: str | None
    enabled: bool
    trust_state: TrustState
    config_revision: int
    trusted_revision: int | None
    trusted_fingerprint: str | None
    created_at: str
    updated_at: str

    def plain_env(self) -> dict[str, str]:
        return {k: (v.value or "") for k, v in self.env_config.items() if v.kind == "plain"}

    def secret_env_keys(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.env_config.items() if v.kind == "secret"))

    def plain_headers(self) -> dict[str, str]:
        return {k: (v.value or "") for k, v in self.header_config.items() if v.kind == "plain"}

    def secret_header_keys(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.header_config.items() if v.kind == "secret"))

    def all_secret_keys(self) -> tuple[str, ...]:
        if self.transport == "stdio":
            return self.secret_env_keys()
        return self.secret_header_keys()

    def to_runtime_config(self, secrets: Mapping[str, str] = {}) -> McpServerConfig:
        """Build an in-memory McpServerConfig by combining SQLite descriptors with in-memory secrets."""
        secret_env: dict[str, str] = {}
        secret_headers: dict[str, str] = {}
        if self.transport == "stdio":
            for key in self.secret_env_keys():
                if key in secrets:
                    secret_env[key] = secrets[key]
        else:
            for key in self.secret_header_keys():
                if key in secrets:
                    secret_headers[key] = secrets[key]
        return McpServerConfig(
            server_id=self.server_id,
            name=self.name,
            transport=self.transport,
            command=self.command,
            args=self.args,
            cwd=self.cwd,
            url=self.url,
            env=self.plain_env(),
            secret_env=secret_env,
            headers=self.plain_headers(),
            secret_headers=secret_headers,
            enabled=self.enabled,
            config_revision=self.config_revision,
            allow_private_network=self.allow_private_network,
        )

    def compute_current_fingerprint(self, *, resolved_command: Path | None = None) -> str:
        """Compute the authoritative security-relevant SHA-256 fingerprint for this server.

        Secret values are never included — only secret key names and non-secret values.
        Display `name` is excluded so renaming does not invalidate trust or tool identity.
        """
        dummy_secret_env = {k: "placeholder" for k in self.secret_env_keys()}
        dummy_secret_headers = {k: "placeholder" for k in self.secret_header_keys()}
        cfg = McpServerConfig(
            server_id=self.server_id,
            name=self.name,
            transport=self.transport,
            command=self.command,
            args=self.args,
            cwd=self.cwd,
            url=self.url,
            env=self.plain_env(),
            secret_env=dummy_secret_env,
            headers=self.plain_headers(),
            secret_headers=dummy_secret_headers,
            enabled=self.enabled,
            config_revision=self.config_revision,
            allow_private_network=self.allow_private_network,
        )
        effective_resolved = (
            resolved_command
            if resolved_command is not None
            else (try_resolve_executable(self.command) if self.transport == "stdio" else None)
        )
        return compute_config_trust_fingerprint(cfg, resolved_command=effective_resolved)


@dataclass(frozen=True, slots=True)
class StoredMcpTool:
    tool_id: str
    server_id: str
    original_tool_name: str
    title: str | None
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any]
    definition_hash: str
    schema_compatible: bool
    incompatibility_reason: str | None
    permission_state: ToolPermissionState
    approved_definition_hash: str | None
    approved_config_revision: int | None
    present_in_last_discovery: bool
    discovered_at: str
    updated_at: str

    @property
    def definition_changed(self) -> bool:
        return (
            self.approved_definition_hash is not None
            and self.approved_definition_hash != self.definition_hash
        )

    @property
    def is_destructive(self) -> bool:
        return self.annotations.get("destructiveHint") is True


@dataclass(frozen=True, slots=True)
class CredentialRefRecord:
    server_id: str
    credential_ref: str
    secret_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ServerUpdateResult:
    server: StoredMcpServer
    security_changed: bool
    name_changed: bool
    enabled_changed: bool
    removed_secret_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SyncCatalogResult:
    synced: bool
    reason: str | None
    tools: tuple[StoredMcpTool, ...]
    invalidated_tool_ids: tuple[str, ...]


def _security_signature(
    *,
    transport: TransportKind,
    command: str | None,
    args: tuple[str, ...],
    cwd: str | None,
    url: str | None,
    allow_private_network: bool,
    env_config: Mapping[str, ConfigValueSpec],
    header_config: Mapping[str, ConfigValueSpec],
) -> str:
    payload = {
        "transport": transport,
        "command": command,
        "args": list(args),
        "cwd": cwd,
        "url": url,
        "allow_private_network": allow_private_network,
        "env": sorted((k, v.kind, v.value if v.kind == "plain" else v.secret_key) for k, v in env_config.items()),
        "headers": sorted(
            (k.lower(), v.kind, v.value if v.kind == "plain" else (v.secret_key or "").lower())
            for k, v in header_config.items()
        ),
    }
    return canonical_json(payload)


def _validate_persisted_fields(
    *,
    server_id: str,
    name: str,
    transport: TransportKind,
    command: str | None,
    args: tuple[str, ...],
    cwd: str | None,
    url: str | None,
    allow_private_network: bool,
    env_config: Mapping[str, ConfigValueSpec],
    header_config: Mapping[str, ConfigValueSpec],
) -> None:
    if len(args) > MAX_ARGS_COUNT:
        raise McpError("MCP_INVALID_CONFIG", f"stdio 参数数量不能超过 {MAX_ARGS_COUNT}。")
    if len(env_config) > MAX_ENV_COUNT:
        raise McpError("MCP_INVALID_CONFIG", f"环境变量总数不能超过 {MAX_ENV_COUNT}。")
    if len(header_config) > MAX_HEADER_COUNT:
        raise McpError("MCP_INVALID_CONFIG", f"HTTP Header 总数不能超过 {MAX_HEADER_COUNT}。")

    if transport == "stdio":
        if header_config:
            raise McpError("MCP_INVALID_CONFIG", "stdio 服务不能配置 HTTP headers。")
        if allow_private_network:
            raise McpError("MCP_INVALID_CONFIG", "stdio 服务不能设置 allow_private_network。")
    else:
        if env_config:
            raise McpError("MCP_INVALID_CONFIG", "Streamable HTTP 服务不能配置 stdio 环境变量。")

    plain_env = {k: (v.value or "") for k, v in env_config.items() if v.kind == "plain"}
    dummy_secret_env = {k: "placeholder-secret" for k, v in env_config.items() if v.kind == "secret"}
    plain_headers = {k: (v.value or "") for k, v in header_config.items() if v.kind == "plain"}
    dummy_secret_headers = {k: "placeholder-secret" for k, v in header_config.items() if v.kind == "secret"}

    check_secret_bypass(
        transport=transport,
        args=args,
        url=url,
        plain_env=plain_env,
        plain_headers=plain_headers,
    )
    validate_server_config(
        McpServerConfig(
            server_id=server_id,
            name=name,
            transport=transport,
            command=command,
            args=args,
            cwd=cwd,
            url=url,
            env=plain_env,
            secret_env=dummy_secret_env,
            headers=plain_headers,
            secret_headers=dummy_secret_headers,
            enabled=False,
            allow_private_network=allow_private_network,
        )
    )


class McpStore:
    """Persistent SQLite storage for MCP server configurations, trust state, and tool permissions."""

    def __init__(self, business: BusinessStore, *, max_servers: int = MAX_CONFIGURED_SERVERS) -> None:
        self._business = business
        self._max_servers = max_servers
        # Per-process monotonic catalog sync fence: server_id -> (config_revision, generation, catalog_version).
        self._fence_lock = threading.Lock()
        self._sync_fences: dict[str, tuple[int, int, int]] = {}

    def _row_to_server(self, row: tuple[Any, ...]) -> StoredMcpServer:
        (
            server_id,
            name,
            transport,
            command,
            args_json,
            cwd,
            url,
            allow_private_network,
            env_config_json,
            header_config_json,
            credential_ref,
            enabled,
            trust_state,
            config_revision,
            trusted_revision,
            trusted_fingerprint,
            created_at,
            updated_at,
        ) = row
        args_list = json.loads(args_json)
        if not isinstance(args_list, list) or not all(isinstance(a, str) for a in args_list):
            raise McpError("MCP_INVALID_CONFIG", "持久化 args_json 格式损坏。")
        return StoredMcpServer(
            server_id=server_id,
            name=name,
            transport=transport,
            command=command,
            args=tuple(args_list),
            cwd=cwd,
            url=url,
            allow_private_network=bool(allow_private_network),
            env_config=_parse_config_map(env_config_json),
            header_config=_parse_config_map(header_config_json),
            credential_ref=credential_ref,
            enabled=bool(enabled),
            trust_state=trust_state,
            config_revision=int(config_revision),
            trusted_revision=int(trusted_revision) if trusted_revision is not None else None,
            trusted_fingerprint=trusted_fingerprint,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _row_to_tool(self, row: tuple[Any, ...]) -> StoredMcpTool:
        (
            tool_id,
            server_id,
            original_tool_name,
            title,
            description,
            input_schema_json,
            output_schema_json,
            annotations_json,
            definition_hash,
            schema_compatible,
            incompatibility_reason,
            permission_state,
            approved_definition_hash,
            approved_config_revision,
            present_in_last_discovery,
            discovered_at,
            updated_at,
        ) = row
        return StoredMcpTool(
            tool_id=tool_id,
            server_id=server_id,
            original_tool_name=original_tool_name,
            title=title,
            description=description,
            input_schema=json.loads(input_schema_json),
            output_schema=json.loads(output_schema_json) if output_schema_json is not None else None,
            annotations=json.loads(annotations_json),
            definition_hash=definition_hash,
            schema_compatible=bool(schema_compatible),
            incompatibility_reason=incompatibility_reason,
            permission_state=permission_state,
            approved_definition_hash=approved_definition_hash,
            approved_config_revision=int(approved_config_revision) if approved_config_revision is not None else None,
            present_in_last_discovery=bool(present_in_last_discovery),
            discovered_at=discovered_at,
            updated_at=updated_at,
        )

    _SERVER_COLUMNS = (
        "server_id, name, transport, command, args_json, cwd, url, allow_private_network, "
        "env_config_json, header_config_json, credential_ref, enabled, trust_state, "
        "config_revision, trusted_revision, trusted_fingerprint, created_at, updated_at"
    )
    _TOOL_COLUMNS = (
        "tool_id, server_id, original_tool_name, title, description, input_schema_json, "
        "output_schema_json, annotations_json, definition_hash, schema_compatible, "
        "incompatibility_reason, permission_state, approved_definition_hash, "
        "approved_config_revision, present_in_last_discovery, discovered_at, updated_at"
    )

    def _select_server(self, connection: sqlite3.Connection, server_id: str) -> StoredMcpServer:
        row = connection.execute(
            f"SELECT {self._SERVER_COLUMNS} FROM mcp_servers WHERE server_id = ?",
            (server_id,),
        ).fetchone()
        if row is None:
            raise McpError("MCP_SERVER_NOT_FOUND")
        return self._row_to_server(row)

    def create_server(
        self,
        *,
        name: str,
        transport: TransportKind,
        command: str | None = None,
        args: Sequence[str] = (),
        cwd: str | None = None,
        url: str | None = None,
        allow_private_network: bool = False,
        env_config: Mapping[str, ConfigValueSpec] = {},
        header_config: Mapping[str, ConfigValueSpec] = {},
        server_id: str | None = None,
    ) -> StoredMcpServer:
        """Create a new MCP server configuration. Always starts disabled and untrusted."""
        sid = server_id or str(uuid4())
        clean_name = name.strip()
        args_tuple = tuple(args)
        clean_command = command.strip() if isinstance(command, str) else None
        clean_url = url.strip() if isinstance(url, str) else None
        clean_cwd = cwd.strip() if isinstance(cwd, str) and cwd.strip() else None

        _validate_persisted_fields(
            server_id=sid,
            name=clean_name,
            transport=transport,
            command=clean_command,
            args=args_tuple,
            cwd=clean_cwd,
            url=clean_url,
            allow_private_network=allow_private_network if transport == "streamable_http" else False,
            env_config=env_config,
            header_config=header_config,
        )

        has_secrets = any(v.kind == "secret" for v in env_config.values()) or any(
            v.kind == "secret" for v in header_config.values()
        )
        credential_ref = f"mcp-server:{sid}" if has_secrets else None
        now = utc_now()

        with closing(self._business._connect()) as connection, self._business._write(connection):
            count = connection.execute("SELECT count(*) FROM mcp_servers").fetchone()[0]
            if count >= self._max_servers:
                raise McpError("MCP_INVALID_CONFIG", f"最多允许配置 {self._max_servers} 个 MCP 服务。")
            try:
                connection.execute(
                    """INSERT INTO mcp_servers (
                        server_id, name, transport, command, args_json, cwd, url,
                        allow_private_network, env_config_json, header_config_json,
                        credential_ref, enabled, trust_state, config_revision,
                        trusted_revision, trusted_fingerprint, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'untrusted', 1, NULL, NULL, ?, ?)""",
                    (
                        sid,
                        clean_name,
                        transport,
                        clean_command,
                        canonical_json(list(args_tuple)),
                        clean_cwd,
                        clean_url,
                        1 if (transport == "streamable_http" and allow_private_network) else 0,
                        _serialize_config_map(env_config),
                        _serialize_config_map(header_config),
                        credential_ref,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                msg = str(exc).lower()
                if "mcp_servers.name" in msg or "mcp_servers_name_unique" in msg or "unique" in msg:
                    raise McpError("MCP_SERVER_NAME_CONFLICT") from None
                raise McpError("MCP_INVALID_CONFIG", "MCP 服务配置违反数据库约束。") from None
            return self._select_server(connection, sid)

    def get_server(self, server_id: str) -> StoredMcpServer:
        with closing(self._business._connect()) as connection:
            return self._select_server(connection, server_id)

    def list_servers(self) -> list[StoredMcpServer]:
        with closing(self._business._connect()) as connection:
            rows = connection.execute(
                f"SELECT {self._SERVER_COLUMNS} FROM mcp_servers ORDER BY created_at ASC, server_id ASC"
            ).fetchall()
            return [self._row_to_server(row) for row in rows]

    def list_credential_refs(self) -> list[CredentialRefRecord]:
        refs: list[CredentialRefRecord] = []
        for server in self.list_servers():
            keys = server.all_secret_keys()
            if server.credential_ref and keys:
                refs.append(
                    CredentialRefRecord(
                        server_id=server.server_id,
                        credential_ref=server.credential_ref,
                        secret_keys=keys,
                    )
                )
        return refs

    def update_server(
        self,
        server_id: str,
        *,
        name: str | None = None,
        transport: TransportKind | None = None,
        command: str | None = None,
        args: Sequence[str] | None = None,
        cwd: str | None = None,
        clear_cwd: bool = False,
        url: str | None = None,
        allow_private_network: bool | None = None,
        env_config: Mapping[str, ConfigValueSpec] | None = None,
        header_config: Mapping[str, ConfigValueSpec] | None = None,
        enabled: bool | None = None,
    ) -> ServerUpdateResult:
        """Update server configuration or enabled flag.

        Security-critical changes increment `config_revision`, reset `trust_state`
        to `'untrusted'`, clear `trusted_revision`/`trusted_fingerprint`, and
        immediately revoke all stored tool approvals for this server.
        Display `name`-only changes preserve trust and internal tool identities.
        """
        with closing(self._business._connect()) as connection, self._business._write(connection):
            current = self._select_server(connection, server_id)

            new_name = current.name if name is None else name.strip()
            new_transport: TransportKind = current.transport if transport is None else transport
            if new_transport != current.transport:
                # Transport switched: do not carry over incompatible fields from old transport.
                new_command = command.strip() if isinstance(command, str) else None
                new_args = tuple(args) if args is not None else ()
                new_cwd = None if clear_cwd else (cwd.strip() if isinstance(cwd, str) and cwd.strip() else None)
                new_url = url.strip() if isinstance(url, str) else None
                new_allow_private = bool(allow_private_network) if new_transport == "streamable_http" else False
                new_env = dict(env_config) if env_config is not None else {}
                new_headers = dict(header_config) if header_config is not None else {}
            else:
                new_command = current.command if command is None else command.strip()
                new_args = current.args if args is None else tuple(args)
                if clear_cwd:
                    new_cwd = None
                elif cwd is not None:
                    new_cwd = cwd.strip() or None
                else:
                    new_cwd = current.cwd
                new_url = current.url if url is None else url.strip()
                new_allow_private = (
                    current.allow_private_network
                    if allow_private_network is None
                    else bool(allow_private_network)
                )
                if new_transport == "stdio":
                    new_allow_private = False
                new_env = dict(current.env_config) if env_config is None else dict(env_config)
                new_headers = dict(current.header_config) if header_config is None else dict(header_config)

            _validate_persisted_fields(
                server_id=server_id,
                name=new_name,
                transport=new_transport,
                command=new_command,
                args=new_args,
                cwd=new_cwd,
                url=new_url,
                allow_private_network=new_allow_private,
                env_config=new_env,
                header_config=new_headers,
            )

            old_sig = _security_signature(
                transport=current.transport,
                command=current.command,
                args=current.args,
                cwd=current.cwd,
                url=current.url,
                allow_private_network=current.allow_private_network,
                env_config=current.env_config,
                header_config=current.header_config,
            )
            new_sig = _security_signature(
                transport=new_transport,
                command=new_command,
                args=new_args,
                cwd=new_cwd,
                url=new_url,
                allow_private_network=new_allow_private,
                env_config=new_env,
                header_config=new_headers,
            )
            security_changed = old_sig != new_sig
            name_changed = new_name != current.name

            new_config_revision = current.config_revision + 1 if security_changed else current.config_revision
            new_trust_state: TrustState = "untrusted" if security_changed else current.trust_state
            new_trusted_revision = None if security_changed else current.trusted_revision
            new_trusted_fingerprint = None if security_changed else current.trusted_fingerprint

            new_enabled = current.enabled if enabled is None else bool(enabled)
            if security_changed and enabled is None:
                # Force disabled on security-critical change until user re-trusts.
                new_enabled = False
            if new_enabled and new_trust_state != "trusted":
                raise McpError("MCP_UNTRUSTED_SERVER", "未信任的 MCP 服务无法启用，请先完成原生信任确认。")
            enabled_changed = new_enabled != current.enabled

            old_secret_keys = set(current.all_secret_keys())
            new_secret_keys = set(
                (k for k, v in new_env.items() if v.kind == "secret")
                if new_transport == "stdio"
                else (k for k, v in new_headers.items() if v.kind == "secret")
            )
            removed_secret_keys = tuple(sorted(old_secret_keys - new_secret_keys))
            credential_ref = f"mcp-server:{server_id}" if new_secret_keys else None
            now = utc_now()

            try:
                connection.execute(
                    """UPDATE mcp_servers SET
                        name = ?, transport = ?, command = ?, args_json = ?, cwd = ?, url = ?,
                        allow_private_network = ?, env_config_json = ?, header_config_json = ?,
                        credential_ref = ?, enabled = ?, trust_state = ?, config_revision = ?,
                        trusted_revision = ?, trusted_fingerprint = ?, updated_at = ?
                       WHERE server_id = ?""",
                    (
                        new_name,
                        new_transport,
                        new_command,
                        canonical_json(list(new_args)),
                        new_cwd,
                        new_url,
                        1 if new_allow_private else 0,
                        _serialize_config_map(new_env),
                        _serialize_config_map(new_headers),
                        credential_ref,
                        1 if new_enabled else 0,
                        new_trust_state,
                        new_config_revision,
                        new_trusted_revision,
                        new_trusted_fingerprint,
                        now,
                        server_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                msg = str(exc).lower()
                if "mcp_servers.name" in msg or "mcp_servers_name_unique" in msg or "unique" in msg:
                    raise McpError("MCP_SERVER_NAME_CONFLICT") from None
                raise McpError("MCP_INVALID_CONFIG", "更新 MCP 服务配置违反数据库约束。") from None

            if security_changed:
                connection.execute(
                    """UPDATE mcp_tools SET
                        permission_state = 'denied',
                        approved_definition_hash = NULL,
                        approved_config_revision = NULL,
                        updated_at = ?
                       WHERE server_id = ?""",
                    (now, server_id),
                )
                with self._fence_lock:
                    self._sync_fences.pop(server_id, None)

            updated = self._select_server(connection, server_id)
            return ServerUpdateResult(
                server=updated,
                security_changed=security_changed,
                name_changed=name_changed,
                enabled_changed=enabled_changed,
                removed_secret_keys=removed_secret_keys,
            )

    def trust_server(
        self,
        server_id: str,
        *,
        expected_config_revision: int,
        expected_fingerprint: str,
        enable: bool = False,
    ) -> StoredMcpServer:
        """Bind server trust atomically to (config_revision, current_fingerprint)."""
        with closing(self._business._connect()) as connection, self._business._write(connection):
            server = self._select_server(connection, server_id)
            resolved_exe: Path | None = None
            if server.transport == "stdio":
                assert server.command is not None
                resolved_exe = resolve_stdio_executable(server.command)
            current_fp = server.compute_current_fingerprint(resolved_command=resolved_exe)
            if server.config_revision != expected_config_revision or current_fp != expected_fingerprint:
                raise McpError("MCP_TRUST_STALE")

            now = utc_now()
            new_enabled = 1 if (enable or server.enabled) else 0
            connection.execute(
                """UPDATE mcp_servers SET
                    trust_state = 'trusted',
                    trusted_revision = ?,
                    trusted_fingerprint = ?,
                    enabled = ?,
                    updated_at = ?
                   WHERE server_id = ?""",
                (server.config_revision, current_fp, new_enabled, now, server_id),
            )
            return self._select_server(connection, server_id)

    def revoke_trust(self, server_id: str) -> StoredMcpServer:
        """Explicitly revoke server trust, disable the server, and revoke all its tool approvals."""
        now = utc_now()
        with closing(self._business._connect()) as connection, self._business._write(connection):
            self._select_server(connection, server_id)
            connection.execute(
                """UPDATE mcp_servers SET
                    trust_state = 'untrusted',
                    trusted_revision = NULL,
                    trusted_fingerprint = NULL,
                    enabled = 0,
                    updated_at = ?
                   WHERE server_id = ?""",
                (now, server_id),
            )
            connection.execute(
                """UPDATE mcp_tools SET
                    permission_state = 'denied',
                    approved_definition_hash = NULL,
                    approved_config_revision = NULL,
                    updated_at = ?
                   WHERE server_id = ?""",
                (now, server_id),
            )
            return self._select_server(connection, server_id)

    def revoke_all_tool_permissions(self, server_id: str) -> int:
        """Revoke all tool-level authorizations for a server without writing any secret data."""
        now = utc_now()
        with closing(self._business._connect()) as connection, self._business._write(connection):
            self._select_server(connection, server_id)
            changed = connection.execute(
                """UPDATE mcp_tools SET
                    permission_state = 'denied',
                    approved_definition_hash = NULL,
                    approved_config_revision = NULL,
                    updated_at = ?
                   WHERE server_id = ? AND (
                       permission_state = 'allowed'
                       OR approved_definition_hash IS NOT NULL
                       OR approved_config_revision IS NOT NULL
                   )""",
                (now, server_id),
            ).rowcount
            return int(changed)

    def delete_server(self, server_id: str) -> StoredMcpServer:
        """Delete a server and its discovered mcp_tools while preserving historical tool_calls."""
        with closing(self._business._connect()) as connection, self._business._write(connection):
            server = self._select_server(connection, server_id)
            connection.execute("DELETE FROM mcp_servers WHERE server_id = ?", (server_id,))
        with self._fence_lock:
            self._sync_fences.pop(server_id, None)
        return server

    def sync_discovered_tools(
        self,
        server_id: str,
        tools: Sequence[McpToolDefinition],
        *,
        generation: int,
        catalog_version: int,
        expected_config_revision: int,
    ) -> SyncCatalogResult:
        """Synchronize discovered tools into SQLite with strict generation & revision fencing.

        - New tools always default to `permission_state = 'denied'`.
        - Definition hash is recomputed over the FULL untruncated definition.
        - If a tool's definition_hash changed, or schema became incompatible, or
          annotations mark destructiveHint=True, existing authorization is revoked
          immediately (`permission_state = 'denied'`).
        - Tools absent from this discovery are marked `present_in_last_discovery = 0`
          and `permission_state = 'denied'`.
        """
        fence_key = (expected_config_revision, generation, catalog_version)
        with self._fence_lock:
            prev = self._sync_fences.get(server_id)
            if prev is not None and fence_key < prev:
                return SyncCatalogResult(
                    synced=False,
                    reason="stale_generation",
                    tools=(),
                    invalidated_tool_ids=(),
                )

            now = utc_now()
            invalidated: list[str] = []
            with closing(self._business._connect()) as connection, self._business._write(connection):
                row = connection.execute(
                    f"SELECT {self._SERVER_COLUMNS} FROM mcp_servers WHERE server_id = ?",
                    (server_id,),
                ).fetchone()
                if row is None:
                    return SyncCatalogResult(
                        synced=False,
                        reason="server_deleted",
                        tools=(),
                        invalidated_tool_ids=(),
                    )
                server = self._row_to_server(row)
                if server.config_revision != expected_config_revision:
                    return SyncCatalogResult(
                        synced=False,
                        reason="stale_config_revision",
                        tools=(),
                        invalidated_tool_ids=(),
                    )

                existing_rows = connection.execute(
                    f"SELECT {self._TOOL_COLUMNS} FROM mcp_tools WHERE server_id = ?",
                    (server_id,),
                ).fetchall()
                existing_by_name = {t.original_tool_name: t for t in (self._row_to_tool(r) for r in existing_rows)}

                seen_names: set[str] = set()
                for item in tools:
                    if item.server_id != server_id or item.original_tool_name in seen_names:
                        continue
                    seen_names.add(item.original_tool_name)

                    # Recompute over full untruncated definition so truncated display text can never hide drift.
                    full_hash = compute_tool_definition_hash(
                        name=item.original_tool_name,
                        title=item.title,
                        description=item.description,
                        input_schema=item.input_schema,
                        output_schema=item.output_schema,
                        annotations=item.annotations,
                    )
                    compatible, incompat_reason = check_tool_schema_compatibility(item.input_schema)
                    is_destructive = item.annotations.get("destructiveHint") is True
                    input_schema_json = canonical_json(item.input_schema)
                    output_schema_json = canonical_json(item.output_schema) if item.output_schema is not None else None
                    annotations_json = canonical_json(item.annotations)

                    prev_tool = existing_by_name.get(item.original_tool_name)
                    if prev_tool is None:
                        connection.execute(
                            """INSERT INTO mcp_tools (
                                tool_id, server_id, original_tool_name, title, description,
                                input_schema_json, output_schema_json, annotations_json,
                                definition_hash, schema_compatible, incompatibility_reason,
                                permission_state, approved_definition_hash, approved_config_revision,
                                present_in_last_discovery, discovered_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'denied', NULL, NULL, 1, ?, ?)""",
                            (
                                str(uuid4()),
                                server_id,
                                item.original_tool_name,
                                item.title,
                                item.description,
                                input_schema_json,
                                output_schema_json,
                                annotations_json,
                                full_hash,
                                1 if compatible else 0,
                                incompat_reason,
                                now,
                                now,
                            ),
                        )
                    else:
                        still_valid = (
                            prev_tool.permission_state == "allowed"
                            and prev_tool.approved_definition_hash == full_hash
                            and prev_tool.approved_config_revision == server.config_revision
                            and compatible
                            and not is_destructive
                        )
                        new_perm: ToolPermissionState = "allowed" if still_valid else "denied"
                        # Preserve the previous approved_definition_hash when a previously allowed tool
                        # drifted so callers/UI can display "definition changed, re-approval required".
                        if still_valid:
                            new_approved_hash = prev_tool.approved_definition_hash
                            new_approved_rev = prev_tool.approved_config_revision
                        elif prev_tool.permission_state == "allowed" and prev_tool.approved_definition_hash != full_hash:
                            new_approved_hash = prev_tool.approved_definition_hash
                            new_approved_rev = None
                            invalidated.append(prev_tool.tool_id)
                        elif prev_tool.permission_state == "allowed":
                            new_approved_hash = None
                            new_approved_rev = None
                            invalidated.append(prev_tool.tool_id)
                        else:
                            # Keep existing drift marker if definition still differs from prior approval.
                            if prev_tool.approved_definition_hash is not None and prev_tool.approved_definition_hash != full_hash:
                                new_approved_hash = prev_tool.approved_definition_hash
                            else:
                                new_approved_hash = None
                            new_approved_rev = None

                        connection.execute(
                            """UPDATE mcp_tools SET
                                title = ?, description = ?, input_schema_json = ?,
                                output_schema_json = ?, annotations_json = ?,
                                definition_hash = ?, schema_compatible = ?, incompatibility_reason = ?,
                                permission_state = ?, approved_definition_hash = ?,
                                approved_config_revision = ?, present_in_last_discovery = 1,
                                updated_at = ?
                               WHERE tool_id = ?""",
                            (
                                item.title,
                                item.description,
                                input_schema_json,
                                output_schema_json,
                                annotations_json,
                                full_hash,
                                1 if compatible else 0,
                                incompat_reason,
                                new_perm,
                                new_approved_hash,
                                new_approved_rev,
                                now,
                                prev_tool.tool_id,
                            ),
                        )

                # Tools removed from the server must immediately lose authorization and visibility.
                for name, prev_tool in existing_by_name.items():
                    if name not in seen_names:
                        if prev_tool.permission_state == "allowed":
                            invalidated.append(prev_tool.tool_id)
                        connection.execute(
                            """UPDATE mcp_tools SET
                                present_in_last_discovery = 0,
                                permission_state = 'denied',
                                approved_definition_hash = NULL,
                                approved_config_revision = NULL,
                                updated_at = ?
                               WHERE tool_id = ?""",
                            (now, prev_tool.tool_id),
                        )

                synced_rows = connection.execute(
                    f"""SELECT {self._TOOL_COLUMNS} FROM mcp_tools
                        WHERE server_id = ?
                        ORDER BY present_in_last_discovery DESC, original_tool_name ASC""",
                    (server_id,),
                ).fetchall()
                synced_tools = tuple(self._row_to_tool(r) for r in synced_rows)

            self._sync_fences[server_id] = fence_key
            return SyncCatalogResult(
                synced=True,
                reason=None,
                tools=synced_tools,
                invalidated_tool_ids=tuple(invalidated),
            )

    def list_tools(self, server_id: str, *, include_missing: bool = True) -> list[StoredMcpTool]:
        with closing(self._business._connect()) as connection:
            self._select_server(connection, server_id)
            predicate = "" if include_missing else "AND present_in_last_discovery = 1"
            rows = connection.execute(
                f"""SELECT {self._TOOL_COLUMNS} FROM mcp_tools
                    WHERE server_id = ? {predicate}
                    ORDER BY present_in_last_discovery DESC, original_tool_name ASC""",
                (server_id,),
            ).fetchall()
            return [self._row_to_tool(row) for row in rows]

    def get_tool(self, server_id: str, tool_id: str) -> StoredMcpTool:
        with closing(self._business._connect()) as connection:
            self._select_server(connection, server_id)
            row = connection.execute(
                f"SELECT {self._TOOL_COLUMNS} FROM mcp_tools WHERE server_id = ? AND tool_id = ?",
                (server_id, tool_id),
            ).fetchone()
            if row is None:
                raise McpError("MCP_TOOL_NOT_FOUND")
            return self._row_to_tool(row)

    def set_tool_permission(
        self,
        server_id: str,
        tool_id: str,
        *,
        permission_state: ToolPermissionState,
        expected_definition_hash: str | None = None,
        expected_config_revision: int | None = None,
    ) -> StoredMcpTool:
        """Grant or revoke per-tool permission bound to (definition_hash, config_revision)."""
        now = utc_now()
        with closing(self._business._connect()) as connection, self._business._write(connection):
            server = self._select_server(connection, server_id)
            row = connection.execute(
                f"SELECT {self._TOOL_COLUMNS} FROM mcp_tools WHERE server_id = ? AND tool_id = ?",
                (server_id, tool_id),
            ).fetchone()
            if row is None:
                raise McpError("MCP_TOOL_NOT_FOUND")
            tool = self._row_to_tool(row)

            if permission_state == "denied":
                connection.execute(
                    """UPDATE mcp_tools SET
                        permission_state = 'denied',
                        approved_definition_hash = NULL,
                        approved_config_revision = NULL,
                        updated_at = ?
                       WHERE tool_id = ?""",
                    (now, tool_id),
                )
            else:
                if not self.is_server_trust_valid(server):
                    raise McpError("MCP_UNTRUSTED_SERVER", "服务尚未完成有效信任确认，无法授权工具。")
                if expected_config_revision is None or expected_config_revision != server.config_revision:
                    raise McpError("MCP_TRUST_STALE", "服务配置版本已变更，请刷新后再授权工具。")
                if not tool.present_in_last_discovery:
                    raise McpError("MCP_TOOL_NOT_FOUND", "该工具已从服务端移除，无法授权。")
                if not tool.schema_compatible:
                    raise McpError("MCP_SCHEMA_INCOMPATIBLE", tool.incompatibility_reason or None)
                if tool.is_destructive:
                    raise McpError("MCP_DESTRUCTIVE_TOOL_BLOCKED")
                if expected_definition_hash is None or expected_definition_hash != tool.definition_hash:
                    raise McpError("MCP_TRUST_STALE", "工具定义已发生变化，请重新查看后再授权。")

                connection.execute(
                    """UPDATE mcp_tools SET
                        permission_state = 'allowed',
                        approved_definition_hash = ?,
                        approved_config_revision = ?,
                        updated_at = ?
                       WHERE tool_id = ?""",
                    (tool.definition_hash, server.config_revision, now, tool_id),
                )

            updated_row = connection.execute(
                f"SELECT {self._TOOL_COLUMNS} FROM mcp_tools WHERE tool_id = ?",
                (tool_id,),
            ).fetchone()
            assert updated_row is not None
            return self._row_to_tool(updated_row)

    @staticmethod
    def is_server_trust_valid(
        server: StoredMcpServer,
        *,
        resolved_command: Path | None = None,
    ) -> bool:
        if server.trust_state != "trusted":
            return False
        if server.trusted_revision is None or server.trusted_revision != server.config_revision:
            return False
        if not server.trusted_fingerprint:
            return False
        if server.transport == "stdio":
            exe = resolved_command if resolved_command is not None else try_resolve_executable(server.command)
            if exe is None:
                return False
            current_fp = server.compute_current_fingerprint(resolved_command=exe)
        else:
            current_fp = server.compute_current_fingerprint()
        return current_fp == server.trusted_fingerprint

    @classmethod
    def is_tool_effectively_authorized(
        cls,
        server: StoredMcpServer,
        tool: StoredMcpTool,
        *,
        live_connected: bool,
        live_catalog_stale: bool,
        live_tool: McpToolDefinition | None,
        credentials_complete: bool,
        resolved_command: Path | None = None,
    ) -> bool:
        """Evaluate whether a tool is truly callable right now.

        Persisted SQLite state alone is NEVER sufficient: live runtime connection,
        non-stale catalog, live definition_hash, credential completeness, and
        re-verified trust fingerprint must ALL hold simultaneously.
        """
        if not live_connected or live_catalog_stale:
            return False
        if not server.enabled or not credentials_complete:
            return False
        if not cls.is_server_trust_valid(server, resolved_command=resolved_command):
            return False
        if not tool.present_in_last_discovery or not tool.schema_compatible or tool.is_destructive:
            return False
        if tool.permission_state != "allowed":
            return False
        if tool.approved_config_revision != server.config_revision:
            return False
        if tool.approved_definition_hash != tool.definition_hash:
            return False
        if live_tool is None:
            return False
        if live_tool.server_id != server.server_id or live_tool.original_tool_name != tool.original_tool_name:
            return False
        return live_tool.definition_hash == tool.definition_hash
