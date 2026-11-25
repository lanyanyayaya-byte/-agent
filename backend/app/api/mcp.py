"""FastAPI routers for MCP read-only UI queries and Host-token privileged management.

Channel separation:
- UI Token (`/api/v1/mcp/...`): strictly read-only GET queries returning redacted views.
- Host Token (`/api/v1/_host/mcp/...`): protected by `authenticate_host_namespace` middleware;
  handles all mutations, trust confirmations, connect/disconnect/refresh, tool permissions,
  in-memory credential injection/clearing, and startup recovery notifications.
- No arbitrary tool execution endpoint is exposed on either channel.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints

from app.mcp.models import McpError, TransportKind
from app.security import ApiError
from app.services.mcp import McpService
from app.storage.mcp import ConfigValueSpec, ToolPermissionState

_ERROR_HTTP_STATUS: dict[str, int] = {
    "MCP_INVALID_CONFIG": 422,
    "MCP_NETWORK_POLICY_VIOLATION": 422,
    "MCP_EXECUTABLE_NOT_FOUND": 422,
    "MCP_SCHEMA_INCOMPATIBLE": 422,
    "MCP_SERVER_NOT_FOUND": 404,
    "MCP_TOOL_NOT_FOUND": 404,
    "MCP_SERVER_NAME_CONFLICT": 409,
    "MCP_TRUST_STALE": 409,
    "MCP_CREDENTIALS_MISSING": 409,
    "MCP_ACTIVE_SERVER_LIMIT": 409,
    "MCP_SERVER_BUSY": 429,
    "MCP_UNTRUSTED_SERVER": 403,
    "MCP_PERMISSION_DENIED": 403,
    "MCP_DESTRUCTIVE_TOOL_BLOCKED": 403,
    "MCP_SERVER_UNAVAILABLE": 409,
    "MCP_CONNECT_TIMEOUT": 504,
    "MCP_TOOL_TIMEOUT": 504,
    "MCP_CONNECT_FAILED": 502,
    "MCP_DISCOVERY_FAILED": 502,
    "MCP_PROTOCOL_ERROR": 502,
    "MCP_MALFORMED_RESPONSE": 502,
    "MCP_RESPONSE_TOO_LARGE": 502,
}


def _raise_api_error(exc: McpError) -> None:
    status_code = _ERROR_HTTP_STATUS.get(exc.code, 500)
    raise ApiError(status_code, exc.code, exc.message) from None


def _require_mcp_service(request: Request) -> McpService:
    service: McpService | None = getattr(request.app.state, "mcp_service", None)
    if service is None:
        raise ApiError(503, "MCP_UNAVAILABLE", "MCP 服务尚未初始化。")
    return service


class PlainValueInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["plain"]
    value: str


class SecretRefInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["secret"]
    secret_key: Annotated[str, StringConstraints(min_length=1, max_length=128)]


ConfigEntryInput = PlainValueInput | SecretRefInput


def _to_specs(raw: dict[str, ConfigEntryInput] | None) -> dict[str, ConfigValueSpec] | None:
    if raw is None:
        return None
    specs: dict[str, ConfigValueSpec] = {}
    for key, item in raw.items():
        if isinstance(item, PlainValueInput):
            specs[key] = ConfigValueSpec(kind="plain", value=item.value)
        else:
            if item.secret_key != key:
                raise ApiError(422, "MCP_INVALID_CONFIG", f"秘钥引用键 {item.secret_key} 必须与配置键 {key} 一致。")
            specs[key] = ConfigValueSpec(kind="secret", secret_key=item.secret_key)
    return specs


class CreateMcpServerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    transport: TransportKind
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    url: str | None = None
    allow_private_network: bool = False
    env_config: dict[str, ConfigEntryInput] = Field(default_factory=dict)
    header_config: dict[str, ConfigEntryInput] = Field(default_factory=dict)


class UpdateMcpServerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Annotated[str, StringConstraints(min_length=1, max_length=100)] | None = None
    transport: TransportKind | None = None
    command: str | None = None
    args: list[str] | None = None
    cwd: str | None = None
    clear_cwd: bool = False
    url: str | None = None
    allow_private_network: bool | None = None
    env_config: dict[str, ConfigEntryInput] | None = None
    header_config: dict[str, ConfigEntryInput] | None = None
    enabled: bool | None = None


class TrustMcpServerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_config_revision: Annotated[int, Field(ge=1)]
    expected_fingerprint: Annotated[str, StringConstraints(min_length=64, max_length=64)]
    enable: bool = False


class SetToolPermissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    permission_state: ToolPermissionState
    expected_definition_hash: Annotated[str, StringConstraints(min_length=64, max_length=64)] | None = None
    expected_config_revision: Annotated[int, Field(ge=1)] | None = None


class SetMcpCredentialsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    secrets: dict[str, SecretStr]
    mode: Literal["restore", "update"] = "update"


def mcp_router(host_namespace: str, require_ui, require_ready) -> APIRouter:
    router = APIRouter()

    # ------------------------------------------------------------------
    # UI Token: strictly read-only redacted queries
    # ------------------------------------------------------------------

    @router.get("/api/v1/mcp/servers", dependencies=[Depends(require_ui), Depends(require_ready)])
    def list_mcp_servers(request: Request, response: Response) -> list[dict[str, Any]]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        return service.list_server_views()

    @router.get(
        "/api/v1/mcp/servers/{server_id}/tools",
        dependencies=[Depends(require_ui), Depends(require_ready)],
    )
    def list_mcp_server_tools(server_id: str, request: Request, response: Response) -> list[dict[str, Any]]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return service.list_tool_views(server_id)
        except McpError as exc:
            _raise_api_error(exc)

    @router.get("/api/v1/mcp/summary", dependencies=[Depends(require_ui), Depends(require_ready)])
    def get_mcp_summary(request: Request, response: Response) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        return service.summary()

    # ------------------------------------------------------------------
    # Host Token: privileged management routes (authenticated by middleware)
    # ------------------------------------------------------------------

    @router.get(f"{host_namespace}/mcp/servers/{{server_id}}", dependencies=[Depends(require_ready)])
    def host_get_mcp_server(server_id: str, request: Request, response: Response) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return service.get_server_view(server_id)
        except McpError as exc:
            _raise_api_error(exc)

    @router.post(
        f"{host_namespace}/mcp/servers",
        status_code=201,
        dependencies=[Depends(require_ready)],
    )
    async def host_create_mcp_server(
        body: CreateMcpServerRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.create_server(
                name=body.name,
                transport=body.transport,
                command=body.command,
                args=body.args,
                cwd=body.cwd,
                url=body.url,
                allow_private_network=body.allow_private_network,
                env_config=_to_specs(body.env_config) or {},
                header_config=_to_specs(body.header_config) or {},
            )
        except McpError as exc:
            _raise_api_error(exc)

    @router.patch(
        f"{host_namespace}/mcp/servers/{{server_id}}",
        dependencies=[Depends(require_ready)],
    )
    async def host_update_mcp_server(
        server_id: str,
        body: UpdateMcpServerRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.update_server(
                server_id,
                name=body.name,
                transport=body.transport,
                command=body.command,
                args=body.args,
                cwd=body.cwd,
                clear_cwd=body.clear_cwd,
                url=body.url,
                allow_private_network=body.allow_private_network,
                env_config=_to_specs(body.env_config),
                header_config=_to_specs(body.header_config),
                enabled=body.enabled,
            )
        except McpError as exc:
            _raise_api_error(exc)

    @router.post(
        f"{host_namespace}/mcp/servers/{{server_id}}/trust",
        dependencies=[Depends(require_ready)],
    )
    async def host_trust_mcp_server(
        server_id: str,
        body: TrustMcpServerRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.trust_server(
                server_id,
                expected_config_revision=body.expected_config_revision,
                expected_fingerprint=body.expected_fingerprint,
                enable=body.enable,
            )
        except McpError as exc:
            _raise_api_error(exc)

    @router.post(
        f"{host_namespace}/mcp/servers/{{server_id}}/revoke-trust",
        dependencies=[Depends(require_ready)],
    )
    async def host_revoke_mcp_server_trust(
        server_id: str,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.revoke_trust(server_id)
        except McpError as exc:
            _raise_api_error(exc)

    @router.delete(
        f"{host_namespace}/mcp/servers/{{server_id}}",
        status_code=204,
        dependencies=[Depends(require_ready)],
    )
    async def host_delete_mcp_server(server_id: str, request: Request) -> Response:
        service = _require_mcp_service(request)
        try:
            await service.delete_server(server_id)
        except McpError as exc:
            _raise_api_error(exc)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @router.post(
        f"{host_namespace}/mcp/servers/{{server_id}}/connect",
        dependencies=[Depends(require_ready)],
    )
    async def host_connect_mcp_server(
        server_id: str,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.connect_server(server_id)
        except McpError as exc:
            _raise_api_error(exc)

    @router.post(
        f"{host_namespace}/mcp/servers/{{server_id}}/disconnect",
        dependencies=[Depends(require_ready)],
    )
    async def host_disconnect_mcp_server(
        server_id: str,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.disconnect_server(server_id)
        except McpError as exc:
            _raise_api_error(exc)

    @router.post(
        f"{host_namespace}/mcp/servers/{{server_id}}/refresh",
        dependencies=[Depends(require_ready)],
    )
    async def host_refresh_mcp_tools(
        server_id: str,
        request: Request,
        response: Response,
    ) -> list[dict[str, Any]]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.refresh_tools(server_id)
        except McpError as exc:
            _raise_api_error(exc)

    @router.patch(
        f"{host_namespace}/mcp/servers/{{server_id}}/tools/{{tool_id}}",
        dependencies=[Depends(require_ready)],
    )
    async def host_set_mcp_tool_permission(
        server_id: str,
        tool_id: str,
        body: SetToolPermissionRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        try:
            return await service.set_tool_permission(
                server_id,
                tool_id,
                permission_state=body.permission_state,
                expected_definition_hash=body.expected_definition_hash,
                expected_config_revision=body.expected_config_revision,
            )
        except McpError as exc:
            _raise_api_error(exc)

    @router.get(f"{host_namespace}/mcp/credential-refs", dependencies=[Depends(require_ready)])
    def host_list_mcp_credential_refs(request: Request, response: Response) -> list[dict[str, Any]]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        return [
            {
                "server_id": ref.server_id,
                "credential_ref": ref.credential_ref,
                "secret_keys": list(ref.secret_keys),
            }
            for ref in service.list_credential_refs()
        ]

    @router.put(
        f"{host_namespace}/mcp/servers/{{server_id}}/credentials",
        dependencies=[Depends(require_ready)],
    )
    async def host_put_mcp_credentials(
        server_id: str,
        body: SetMcpCredentialsRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        service = _require_mcp_service(request)
        response.headers["Cache-Control"] = "no-store"
        plain_secrets = {key: value.get_secret_value() for key, value in body.secrets.items()}
        try:
            status = await service.set_credentials(server_id, plain_secrets, mode=body.mode)
        except McpError as exc:
            _raise_api_error(exc)
        return {
            "server_id": status.server_id,
            "configured_keys": list(status.configured_keys),
            "missing_keys": list(status.missing_keys),
            "credentials_complete": status.credentials_complete,
        }

    @router.delete(
        f"{host_namespace}/mcp/servers/{{server_id}}/credentials",
        status_code=204,
        dependencies=[Depends(require_ready)],
    )
    async def host_delete_mcp_credentials(server_id: str, request: Request) -> Response:
        service = _require_mcp_service(request)
        try:
            await service.clear_credentials(server_id)
        except McpError as exc:
            _raise_api_error(exc)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @router.post(
        f"{host_namespace}/mcp/restore-complete",
        status_code=204,
        dependencies=[Depends(require_ready)],
    )
    async def host_mcp_restore_complete(request: Request) -> Response:
        service = _require_mcp_service(request)
        service.notify_restore_complete()
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return router
