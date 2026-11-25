"""MCP orchestration service bridging McpStore (SQLite), McpManager (runtime), and memory credentials.

Security invariants:
- Plaintext secrets live ONLY in `self._secrets` (process memory) and inside the
  single server's `McpServerConfig` in `McpManager`; they are NEVER written to
  SQLite, NEVER logged, NEVER returned in API responses, and NEVER shared across
  different `server_id`s.
- Untrusted, disabled, fingerprint-mismatched, or credential-incomplete servers
  can NEVER be connected manually or during background startup recovery.
- Background recovery is strictly non-blocking, bounded by timeouts and
  concurrency caps, and cancels cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

from app.diagnostics import record
from app.mcp.adapter import (
    McpAuthorizedToolSnapshot,
    build_deepseek_tool_schema,
    make_mcp_tool_alias,
    normalize_mcp_input_schema,
)
from app.mcp.manager import McpManager
from app.mcp.models import McpDiagnostics, McpError, McpToolCallResult, McpToolDefinition, TransportKind
from app.mcp.policy import (
    inspect_high_risk_command,
    is_private_network_url,
    redact_args,
    redact_text,
    redact_url,
    validate_secret_value,
)
from app.storage.mcp import (
    ConfigValueSpec,
    CredentialRefRecord,
    McpStore,
    StoredMcpServer,
    StoredMcpTool,
    ToolPermissionState,
    try_resolve_executable,
)

MAX_TOOLS_PER_RUN = 32
MAX_DISPLAY_DESCRIPTION_CHARS = 1024
STARTUP_RECOVERY_PER_SERVER_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    server_id: str
    configured_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    credentials_complete: bool


class McpService:
    """High-level coordinator for MCP persistence, memory credentials, and live connections."""

    def __init__(self, store: McpStore, manager: McpManager) -> None:
        self._store = store
        self._manager = manager
        self._secrets: dict[str, dict[str, str]] = {}
        self._recovery_task: asyncio.Task[None] | None = None
        self._shutting_down = False
        self._restore_completed = False

        # Hook catalog discovery/refresh events from McpManager into McpStore with generation fencing.
        self._manager._on_catalog_event = self._handle_catalog_event  # noqa: SLF001

    @property
    def store(self) -> McpStore:
        return self._store

    @property
    def manager(self) -> McpManager:
        return self._manager

    def _handle_catalog_event(
        self,
        server_id: str,
        tools: tuple[McpToolDefinition, ...],
        generation: int,
        catalog_version: int,
        config_revision: int,
        stale: bool,
    ) -> None:
        if stale:
            record(
                "mcp_catalog_stale",
                server_id=server_id,
                generation=generation,
                config_revision=config_revision,
            )
            return
        result = self._store.sync_discovered_tools(
            server_id,
            tools,
            generation=generation,
            catalog_version=catalog_version,
            expected_config_revision=config_revision,
        )
        record(
            "mcp_catalog_synced",
            server_id=server_id,
            generation=generation,
            config_revision=config_revision,
            synced=result.synced,
            reason=result.reason,
            tool_count=len(tools),
            invalidated_count=len(result.invalidated_tool_ids),
        )

    def _server_secrets(self, server: StoredMcpServer) -> dict[str, str]:
        raw = self._secrets.get(server.server_id, {})
        allowed = set(server.all_secret_keys())
        return {k: v for k, v in raw.items() if k in allowed}

    def credential_status(self, server: StoredMcpServer) -> CredentialStatus:
        declared = server.all_secret_keys()
        loaded = self._server_secrets(server)
        configured = tuple(sorted(k for k in declared if k in loaded))
        missing = tuple(sorted(k for k in declared if k not in loaded))
        return CredentialStatus(
            server_id=server.server_id,
            configured_keys=configured,
            missing_keys=missing,
            credentials_complete=len(missing) == 0,
        )

    async def sync_server_to_manager(self, server: StoredMcpServer) -> None:
        """Synchronize one persisted server's config + in-memory secrets into McpManager."""
        if self._shutting_down:
            return
        secrets = self._server_secrets(server)
        runtime_cfg = server.to_runtime_config(secrets)
        await self._manager.register_server(runtime_cfg)

    async def load_persisted_servers(self) -> None:
        """Load persisted server definitions into McpManager on startup (without connecting)."""
        for server in self._store.list_servers():
            with suppress(McpError):
                await self.sync_server_to_manager(server)

    async def create_server(
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
    ) -> dict[str, Any]:
        server = await asyncio.to_thread(
            self._store.create_server,
            name=name,
            transport=transport,
            command=command,
            args=args,
            cwd=cwd,
            url=url,
            allow_private_network=allow_private_network,
            env_config=env_config,
            header_config=header_config,
        )
        await self.sync_server_to_manager(server)
        record(
            "mcp_server_created",
            server_id=server.server_id,
            transport=server.transport,
            config_revision=server.config_revision,
        )
        return self.build_server_view(server)

    async def update_server(
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
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            self._store.update_server,
            server_id,
            name=name,
            transport=transport,
            command=command,
            args=args,
            cwd=cwd,
            clear_cwd=clear_cwd,
            url=url,
            allow_private_network=allow_private_network,
            env_config=env_config,
            header_config=header_config,
            enabled=enabled,
        )
        server = result.server
        if result.removed_secret_keys and server_id in self._secrets:
            for removed_key in result.removed_secret_keys:
                self._secrets[server_id].pop(removed_key, None)
            if not self._secrets[server_id]:
                self._secrets.pop(server_id, None)

        if result.security_changed or (result.enabled_changed and not server.enabled):
            self._manager.cancel_server_calls(server_id)
            with suppress(McpError):
                await self._manager.disconnect(server_id)

        await self.sync_server_to_manager(server)
        record(
            "mcp_server_updated",
            server_id=server_id,
            security_changed=result.security_changed,
            enabled=server.enabled,
            trust_state=server.trust_state,
            config_revision=server.config_revision,
        )
        return self.build_server_view(server)

    async def trust_server(
        self,
        server_id: str,
        *,
        expected_config_revision: int,
        expected_fingerprint: str,
        enable: bool = False,
    ) -> dict[str, Any]:
        server = await asyncio.to_thread(
            self._store.trust_server,
            server_id,
            expected_config_revision=expected_config_revision,
            expected_fingerprint=expected_fingerprint,
            enable=enable,
        )
        await self.sync_server_to_manager(server)
        record(
            "mcp_server_trusted",
            server_id=server_id,
            config_revision=server.config_revision,
            enabled=server.enabled,
        )
        return self.build_server_view(server)

    async def revoke_trust(self, server_id: str) -> dict[str, Any]:
        self._manager.cancel_server_calls(server_id)
        with suppress(McpError):
            await self._manager.disconnect(server_id)
        server = await asyncio.to_thread(self._store.revoke_trust, server_id)
        await self.sync_server_to_manager(server)
        record("mcp_server_trust_revoked", server_id=server_id)
        return self.build_server_view(server)

    async def delete_server(self, server_id: str) -> None:
        # Ensure server exists before mutating runtime state.
        await asyncio.to_thread(self._store.get_server, server_id)
        self._manager.cancel_server_calls(server_id)
        await self._manager.remove_server(server_id)
        self._secrets.pop(server_id, None)
        await asyncio.to_thread(self._store.delete_server, server_id)
        record("mcp_server_deleted", server_id=server_id)

    async def set_credentials(
        self,
        server_id: str,
        secrets: Mapping[str, str],
        *,
        mode: Literal["restore", "update"] = "update",
    ) -> CredentialStatus:
        """Store validated secrets in Sidecar process memory only (never SQLite or logs)."""
        server = await asyncio.to_thread(self._store.get_server, server_id)
        declared = set(server.all_secret_keys())
        for key, value in secrets.items():
            if key not in declared:
                raise McpError("MCP_INVALID_CONFIG", f"秘钥键 {key} 未在服务配置中声明。")
            validate_secret_value(key, value)

        previous = dict(self._secrets.get(server_id, {}))
        cleaned = {k: secrets[k] for k in sorted(secrets) if k in declared}
        rotated = mode == "update" and previous != cleaned

        if cleaned:
            self._secrets[server_id] = cleaned
        else:
            self._secrets.pop(server_id, None)

        revoked_tool_count = 0
        if rotated:
            # Credential rotation must immediately disconnect any active session and revoke
            # tool approvals so an account/scope change with identical tool definitions
            # cannot silently inherit prior tool permissions. Normal startup `mode="restore"`
            # never enters this branch.
            self._manager.cancel_server_calls(server_id)
            with suppress(McpError):
                await self._manager.disconnect(server_id)
            if previous or any(
                t.permission_state == "allowed"
                for t in await asyncio.to_thread(self._store.list_tools, server_id, include_missing=False)
            ):
                revoked_tool_count = await asyncio.to_thread(
                    self._store.revoke_all_tool_permissions, server_id
                )

        await self.sync_server_to_manager(server)
        status = self.credential_status(server)
        record(
            "mcp_credentials_updated",
            server_id=server_id,
            mode=mode,
            rotated=rotated,
            revoked_tool_count=revoked_tool_count,
            configured_count=len(status.configured_keys),
            missing_count=len(status.missing_keys),
        )
        return status

    async def clear_credentials(self, server_id: str) -> None:
        server = await asyncio.to_thread(self._store.get_server, server_id)
        self._secrets.pop(server_id, None)
        self._manager.cancel_server_calls(server_id)
        with suppress(McpError):
            await self._manager.disconnect(server_id)
        await self.sync_server_to_manager(server)
        record("mcp_credentials_cleared", server_id=server_id)

    def _verify_connectable(self, server: StoredMcpServer) -> None:
        if not server.enabled:
            raise McpError("MCP_SERVER_UNAVAILABLE", "该 MCP 服务已停用，请先启用。")
        if server.transport == "stdio":
            resolved = try_resolve_executable(server.command)
            if resolved is None:
                raise McpError("MCP_EXECUTABLE_NOT_FOUND")
            trust_ok = self._store.is_server_trust_valid(server, resolved_command=resolved)
        else:
            trust_ok = self._store.is_server_trust_valid(server)
        if not trust_ok:
            raise McpError("MCP_UNTRUSTED_SERVER")
        cred_status = self.credential_status(server)
        if not cred_status.credentials_complete:
            raise McpError("MCP_CREDENTIALS_MISSING")

    async def connect_server(self, server_id: str) -> dict[str, Any]:
        server = await asyncio.to_thread(self._store.get_server, server_id)
        self._verify_connectable(server)
        await self.sync_server_to_manager(server)
        await self._manager.connect(server_id)
        # Re-read authoritative server record (may have been updated concurrently).
        latest = await asyncio.to_thread(self._store.get_server, server_id)
        return self.build_server_view(latest)

    async def disconnect_server(self, server_id: str) -> dict[str, Any]:
        server = await asyncio.to_thread(self._store.get_server, server_id)
        self._manager.cancel_server_calls(server_id)
        with suppress(McpError):
            await self._manager.disconnect(server_id)
        return self.build_server_view(server)

    async def refresh_tools(self, server_id: str) -> list[dict[str, Any]]:
        server = await asyncio.to_thread(self._store.get_server, server_id)
        self._verify_connectable(server)
        await self._manager.refresh_tools(server_id)
        return self.list_tool_views(server_id)

    async def set_tool_permission(
        self,
        server_id: str,
        tool_id: str,
        *,
        permission_state: ToolPermissionState,
        expected_definition_hash: str | None = None,
        expected_config_revision: int | None = None,
    ) -> dict[str, Any]:
        tool = await asyncio.to_thread(
            self._store.set_tool_permission,
            server_id,
            tool_id,
            permission_state=permission_state,
            expected_definition_hash=expected_definition_hash,
            expected_config_revision=expected_config_revision,
        )
        if permission_state == "denied":
            self._manager.cancel_server_calls(server_id)
        record(
            "mcp_tool_permission_updated",
            server_id=server_id,
            tool_id=tool_id,
            original_tool_name=tool.original_tool_name,
            permission_state=permission_state,
        )
        server = await asyncio.to_thread(self._store.get_server, server_id)
        return self._build_tool_view(server, tool)

    def list_credential_refs(self) -> list[CredentialRefRecord]:
        return self._store.list_credential_refs()

    def notify_restore_complete(self) -> None:
        """Trigger non-blocking background auto-connect for eligible servers."""
        if self._shutting_down:
            return
        self._restore_completed = True
        if self._recovery_task is not None and not self._recovery_task.done():
            return
        self._recovery_task = asyncio.create_task(
            self._run_auto_connect(),
            name="mcp-startup-recovery",
        )

    async def _run_auto_connect(self) -> None:
        servers = await asyncio.to_thread(self._store.list_servers)
        eligible: list[StoredMcpServer] = []
        for server in servers:
            if self._shutting_down:
                return
            if not server.enabled:
                continue
            resolved = try_resolve_executable(server.command) if server.transport == "stdio" else None
            if not self._store.is_server_trust_valid(server, resolved_command=resolved):
                continue
            if not self.credential_status(server).credentials_complete:
                record("mcp_auto_connect_skipped", server_id=server.server_id, reason="credentials_missing")
                continue
            eligible.append(server)

        if not eligible:
            return

        async def _connect_one(target: StoredMcpServer) -> None:
            if self._shutting_down:
                return
            try:
                # Re-verify latest persisted state right before connecting (guards against concurrent edit/delete).
                fresh = await asyncio.to_thread(self._store.get_server, target.server_id)
                if fresh.config_revision != target.config_revision:
                    return
                self._verify_connectable(fresh)
                await self.sync_server_to_manager(fresh)
                async with asyncio.timeout(STARTUP_RECOVERY_PER_SERVER_TIMEOUT_SECONDS):
                    await self._manager.connect(fresh.server_id)
                record("mcp_auto_connect_succeeded", server_id=fresh.server_id)
            except asyncio.CancelledError:
                raise
            except (McpError, TimeoutError) as exc:
                code = exc.code if isinstance(exc, McpError) else "MCP_CONNECT_TIMEOUT"
                record("mcp_auto_connect_failed", server_id=target.server_id, error_code=code)
            except Exception as exc:
                record(
                    "mcp_auto_connect_failed",
                    server_id=target.server_id,
                    error_code="MCP_INTERNAL_ERROR",
                    exception_type=type(exc).__name__,
                )

        tasks = [asyncio.create_task(_connect_one(server)) for server in eligible]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def shutdown(self) -> None:
        self._shutting_down = True
        if self._recovery_task is not None and not self._recovery_task.done():
            self._recovery_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._recovery_task, timeout=1.0)
        self._recovery_task = None
        await self._manager.shutdown()
        self._secrets.clear()

    def _safe_diagnostics(self, server: StoredMcpServer) -> McpDiagnostics:
        try:
            return self._manager.get_diagnostics(server.server_id)
        except McpError:
            return McpDiagnostics(
                server_id=server.server_id,
                name=server.name,
                transport=server.transport,
                state="disconnected",
                generation=0,
            )

    def _live_tools_by_name(self, server_id: str) -> dict[str, McpToolDefinition]:
        try:
            tools = self._manager.list_tools(server_id)
        except McpError:
            return {}
        return {tool.original_tool_name: tool for tool in tools}

    def _build_tool_view(self, server: StoredMcpServer, tool: StoredMcpTool) -> dict[str, Any]:
        diag = self._safe_diagnostics(server)
        live_map = self._live_tools_by_name(server.server_id)
        cred = self.credential_status(server)
        resolved = try_resolve_executable(server.command) if server.transport == "stdio" else None
        effective = self._store.is_tool_effectively_authorized(
            server,
            tool,
            live_connected=(diag.state == "connected"),
            live_catalog_stale=diag.catalog_stale,
            live_tool=live_map.get(tool.original_tool_name),
            credentials_complete=cred.credentials_complete,
            resolved_command=resolved,
        )
        display_desc = tool.description
        if len(display_desc) > MAX_DISPLAY_DESCRIPTION_CHARS:
            display_desc = display_desc[: MAX_DISPLAY_DESCRIPTION_CHARS - 1] + "…"
        return {
            "tool_id": tool.tool_id,
            "server_id": tool.server_id,
            "original_tool_name": tool.original_tool_name,
            "title": tool.title,
            "description": display_desc,
            "input_schema": tool.input_schema,
            "annotations": tool.annotations,
            "definition_hash": tool.definition_hash,
            "schema_compatible": tool.schema_compatible,
            "incompatibility_reason": tool.incompatibility_reason,
            "permission_state": tool.permission_state,
            "approved_definition_hash": tool.approved_definition_hash,
            "approved_config_revision": tool.approved_config_revision,
            "definition_changed": tool.definition_changed,
            "is_destructive": tool.is_destructive,
            "present_in_last_discovery": tool.present_in_last_discovery,
            "effective_allowed": effective,
            "discovered_at": tool.discovered_at,
            "updated_at": tool.updated_at,
        }

    def build_server_view(self, server: StoredMcpServer) -> dict[str, Any]:
        """Return a strictly redacted server view safe for UI and Host consumption."""
        diag = self._safe_diagnostics(server)
        cred = self.credential_status(server)
        loaded_secrets = self._server_secrets(server)
        secret_values = tuple(loaded_secrets.values())

        resolved_exe = try_resolve_executable(server.command) if server.transport == "stdio" else None
        current_fp = server.compute_current_fingerprint(resolved_command=resolved_exe)
        trust_valid = self._store.is_server_trust_valid(server, resolved_command=resolved_exe)

        high_risk = False
        high_risk_reasons: tuple[str, ...] = ()
        if server.transport == "stdio" and server.command:
            high_risk, high_risk_reasons = inspect_high_risk_command(
                server.command,
                server.args,
                resolved_executable=resolved_exe,
            )
        private_net_target = bool(server.url and is_private_network_url(server.url))

        env_view: dict[str, dict[str, Any]] = {}
        for k, spec in sorted(server.env_config.items()):
            if spec.kind == "plain":
                env_view[k] = {"kind": "plain", "value": spec.value or ""}
            else:
                env_view[k] = {"kind": "secret", "secret_key": k, "configured": k in loaded_secrets}

        header_view: dict[str, dict[str, Any]] = {}
        for k, spec in sorted(server.header_config.items()):
            if spec.kind == "plain":
                header_view[k] = {"kind": "plain", "value": spec.value or ""}
            else:
                header_view[k] = {"kind": "secret", "secret_key": k, "configured": k in loaded_secrets}

        tools = self._store.list_tools(server.server_id, include_missing=False)
        live_map = self._live_tools_by_name(server.server_id)
        allowed_count = sum(1 for t in tools if t.permission_state == "allowed")
        effective_count = sum(
            1
            for t in tools
            if self._store.is_tool_effectively_authorized(
                server,
                t,
                live_connected=(diag.state == "connected"),
                live_catalog_stale=diag.catalog_stale,
                live_tool=live_map.get(t.original_tool_name),
                credentials_complete=cred.credentials_complete,
                resolved_command=resolved_exe,
            )
        )

        safe_error_msg = (
            redact_text(diag.last_error_message, secrets=secret_values)
            if diag.last_error_message
            else None
        )
        safe_stderr = tuple(redact_text(line, secrets=secret_values) for line in diag.stderr_tail)

        return {
            "server_id": server.server_id,
            "name": server.name,
            "transport": server.transport,
            "command": server.command,
            "resolved_command": str(resolved_exe) if resolved_exe is not None else None,
            "executable_exists": resolved_exe is not None if server.transport == "stdio" else None,
            "args": list(redact_args(server.args, secrets=secret_values)),
            "cwd": server.cwd,
            "url": redact_url(server.url, secrets=secret_values) if server.url else None,
            "allow_private_network": server.allow_private_network,
            "private_network_target": private_net_target,
            "high_risk_command": high_risk,
            "high_risk_reasons": list(high_risk_reasons),
            "env_config": env_view,
            "header_config": header_view,
            "credential_ref": server.credential_ref,
            "credentials_complete": cred.credentials_complete,
            "missing_secret_keys": list(cred.missing_keys),
            "enabled": server.enabled,
            "trust_state": server.trust_state,
            "trust_valid": trust_valid,
            "config_revision": server.config_revision,
            "trusted_revision": server.trusted_revision,
            "trusted_fingerprint": server.trusted_fingerprint,
            "current_fingerprint": current_fp,
            "discovered_tool_count": len(tools),
            "allowed_tool_count": allowed_count,
            "effective_allowed_tool_count": effective_count,
            "runtime": {
                "state": diag.state,
                "generation": diag.generation,
                "protocol_version": diag.protocol_version,
                "server_name_reported": diag.server_name_reported,
                "server_version_reported": diag.server_version_reported,
                "tool_count": diag.tool_count,
                "catalog_version": diag.catalog_version,
                "catalog_stale": diag.catalog_stale,
                "tools_truncated": diag.tools_truncated,
                "last_error_code": diag.last_error_code,
                "last_error_message": safe_error_msg,
                "stderr_tail": list(safe_stderr),
                "connected_at": diag.connected_at,
            },
            "created_at": server.created_at,
            "updated_at": server.updated_at,
        }

    def list_server_views(self) -> list[dict[str, Any]]:
        return [self.build_server_view(server) for server in self._store.list_servers()]

    def get_server_view(self, server_id: str) -> dict[str, Any]:
        return self.build_server_view(self._store.get_server(server_id))

    def list_tool_views(self, server_id: str) -> list[dict[str, Any]]:
        server = self._store.get_server(server_id)
        tools = self._store.list_tools(server_id, include_missing=True)
        return [self._build_tool_view(server, tool) for tool in tools]

    def summary(self) -> dict[str, Any]:
        views = self.list_server_views()
        connected = sum(1 for v in views if v["runtime"]["state"] == "connected")
        errored = sum(1 for v in views if v["runtime"]["state"] == "error")
        enabled = sum(1 for v in views if v["enabled"])
        effective_tools = sum(v["effective_allowed_tool_count"] for v in views)
        return {
            "total_servers": len(views),
            "enabled_servers": enabled,
            "connected_servers": connected,
            "error_servers": errored,
            "effective_allowed_tools": effective_tools,
            "max_tools_per_run": MAX_TOOLS_PER_RUN,
            "exceeds_tool_limit": effective_tools > MAX_TOOLS_PER_RUN,
        }

    # ------------------------------------------------------------------
    # Authoritative Agent Runtime Entrypoint (used exclusively by RunTools)
    # ------------------------------------------------------------------

    def snapshot_authorized_tools(self) -> tuple[McpAuthorizedToolSnapshot, ...]:
        """Build an immutable snapshot of all currently connected, trusted, authorized MCP tools."""
        snapshots: list[McpAuthorizedToolSnapshot] = []
        used_aliases: set[str] = set()

        for server in self._store.list_servers():
            if not server.enabled:
                continue
            cred = self.credential_status(server)
            if not cred.credentials_complete:
                continue
            resolved = try_resolve_executable(server.command) if server.transport == "stdio" else None
            if not self._store.is_server_trust_valid(server, resolved_command=resolved):
                continue

            diag = self._safe_diagnostics(server)
            if diag.state != "connected" or diag.catalog_stale or diag.generation <= 0:
                continue

            live_map = self._live_tools_by_name(server.server_id)
            if not live_map:
                continue

            for tool in self._store.list_tools(server.server_id, include_missing=False):
                live_tool = live_map.get(tool.original_tool_name)
                if not self._store.is_tool_effectively_authorized(
                    server,
                    tool,
                    live_connected=True,
                    live_catalog_stale=False,
                    live_tool=live_tool,
                    credentials_complete=True,
                    resolved_command=resolved,
                ):
                    continue
                try:
                    normalized_schema = normalize_mcp_input_schema(tool.input_schema)
                    alias = make_mcp_tool_alias(
                        server.server_id,
                        tool.original_tool_name,
                        existing_aliases=used_aliases,
                    )
                    model_schema = build_deepseek_tool_schema(
                        alias=alias,
                        server_name=server.name,
                        original_tool_name=tool.original_tool_name,
                        description=tool.description,
                        normalized_input_schema=normalized_schema,
                    )
                except McpError:
                    continue

                used_aliases.add(alias)
                snapshots.append(
                    McpAuthorizedToolSnapshot(
                        alias=alias,
                        tool_id=tool.tool_id,
                        server_id=server.server_id,
                        server_name_snapshot=server.name,
                        original_tool_name=tool.original_tool_name,
                        description=tool.description,
                        raw_input_schema=dict(tool.input_schema),
                        normalized_input_schema=normalized_schema,
                        model_schema=model_schema,
                        definition_hash=tool.definition_hash,
                        config_revision=server.config_revision,
                        generation=diag.generation,
                        catalog_version=diag.catalog_version,
                    )
                )

        return tuple(snapshots)

    def verify_tool_snapshot_authorized(self, snapshot: McpAuthorizedToolSnapshot) -> None:
        """Re-verify live server, generation, catalog, trust, and tool authorization against a snapshot."""
        try:
            server = self._store.get_server(snapshot.server_id)
        except McpError:
            raise McpError("MCP_PERMISSION_DENIED", "该 MCP 服务已被删除。") from None

        if server.config_revision != snapshot.config_revision:
            raise McpError("MCP_PERMISSION_DENIED", "MCP 服务配置已变更，原快照已失效。")

        if not server.enabled:
            raise McpError("MCP_SERVER_UNAVAILABLE", "MCP 服务已停用。")

        cred = self.credential_status(server)
        if not cred.credentials_complete:
            raise McpError("MCP_PERMISSION_DENIED", "MCP 服务缺少有效凭据。")

        resolved = try_resolve_executable(server.command) if server.transport == "stdio" else None
        if not self._store.is_server_trust_valid(server, resolved_command=resolved):
            raise McpError("MCP_PERMISSION_DENIED", "MCP 服务信任状态已失效。")

        diag = self._safe_diagnostics(server)
        if diag.state != "connected":
            raise McpError("MCP_SERVER_UNAVAILABLE", "MCP 服务当前未连接。")
        if diag.catalog_stale:
            raise McpError("MCP_SERVER_UNAVAILABLE", "MCP 工具目录正在刷新或已失效。")
        if diag.generation != snapshot.generation:
            raise McpError("MCP_PERMISSION_DENIED", "MCP 服务连接代际已变化，原快照已失效。")

        try:
            tool = self._store.get_tool(snapshot.server_id, snapshot.tool_id)
        except McpError:
            raise McpError("MCP_PERMISSION_DENIED", "该 MCP 工具已不存在。") from None

        live_tool = self._live_tools_by_name(snapshot.server_id).get(snapshot.original_tool_name)
        if (
            live_tool is None
            or tool.definition_hash != snapshot.definition_hash
            or live_tool.definition_hash != snapshot.definition_hash
            or not self._store.is_tool_effectively_authorized(
                server,
                tool,
                live_connected=True,
                live_catalog_stale=False,
                live_tool=live_tool,
                credentials_complete=True,
                resolved_command=resolved,
            )
        ):
            raise McpError("MCP_PERMISSION_DENIED", "该 MCP 工具授权已被撤销或定义已发生变化。")

    async def execute_authorized_tool(
        self,
        snapshot: McpAuthorizedToolSnapshot,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolCallResult:
        """Execute an authorized MCP tool through McpManager with pre-queue, pre-dispatch, and post-result checks."""
        # 1. Pre-enqueue verification
        self.verify_tool_snapshot_authorized(snapshot)

        # 2. Enqueue with pre-dispatch callback invoked inside _ServerActor right before conn.call_tool()
        result = await self._manager.call_tool(
            snapshot.server_id,
            snapshot.original_tool_name,
            arguments,
            timeout_seconds=timeout_seconds,
            before_dispatch=lambda: self.verify_tool_snapshot_authorized(snapshot),
        )

        # 3. Linearization check immediately after result arrives (discard if revoked in-flight)
        if result.generation != snapshot.generation:
            raise McpError("MCP_PERMISSION_DENIED", "MCP 服务连接代际已变化，结果已丢弃。")
        self.verify_tool_snapshot_authorized(snapshot)
        return result
