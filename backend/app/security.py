import re
from secrets import compare_digest

from fastapi import Request

BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]+=*")


class ApiError(Exception):
    def __init__(
        self, status_code: int, code: str, message: str,
        headers: dict[str, str] | None = None, *, run_id: str | None = None,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers
        self.run_id = run_id


def validate_tokens(ui_token: str, host_token: str) -> tuple[bytes, bytes]:
    if not all(
        isinstance(token, str) and BEARER_TOKEN.fullmatch(token)
        for token in (ui_token, host_token)
    ) or ui_token == host_token:
        raise ValueError("需要分别配置有效且不同的 UI 与宿主 Token。")
    return ui_token.encode("ascii"), host_token.encode("ascii")


def require_bearer(request: Request, expected: bytes, other_channel: bytes) -> None:
    values = request.headers.getlist("authorization")
    if len(values) == 1:
        scheme, _, token = values[0].partition(" ")
        if scheme.lower() == "bearer":
            supplied = token.lstrip(" ").encode("utf-8")
            if compare_digest(supplied, expected):
                return
            if compare_digest(supplied, other_channel):
                raise ApiError(403, "FORBIDDEN", "此凭证无权访问该接口。")

    raise ApiError(401, "AUTH_REQUIRED", "需要有效的 Bearer Token。", {"WWW-Authenticate": "Bearer"})
