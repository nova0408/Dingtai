"""RecordReplay HTTP/JSON 服务。"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.parse import parse_qs, urlsplit

from loguru import logger

from .application import RecordReplayApplication, RecordReplayApplicationError
from .config_store import RuntimeParameterValue
from ..contracts import ReplayErrorCode
from ..device_status import DeviceStatusResponse
from .protocol import (
    PriorUploadResponse,
    RecordReplayHealthResponse,
    RecordReplayErrorResponse,
    RecordReplayPlanResponse,
    RecordReplayResponse,
)

_ERROR_NOT_FOUND: Final[ReplayErrorCode] = "not_found"
_ERROR_INTERNAL: Final[ReplayErrorCode] = "internal_error"
_ERROR_INVALID_STATE: Final[ReplayErrorCode] = "invalid_state"
_ERROR_METHOD_NOT_ALLOWED: Final[ReplayErrorCode] = "method_not_allowed"
_GET_PATHS: Final = frozenset({"/health", "/status", "/plan", "/config", "/device-status"})
_POST_PATHS: Final = frozenset(
    {"/start", "/stop", "/reset", "/config", "/prior/ball-pose", "/prior/charuco"}
)

ApiResponse = (
    RecordReplayResponse
    | RecordReplayPlanResponse
    | RecordReplayErrorResponse
    | DeviceStatusResponse
    | PriorUploadResponse
    | RecordReplayHealthResponse
)


class ApiRequestError(ValueError):
    """带有稳定错误码的 HTTP 请求校验异常。"""

    def __init__(self, error_code: ReplayErrorCode, message: str) -> None:
        super().__init__(message)
        self.error_code: ReplayErrorCode = error_code


class RecordReplayServer:
    """通过标准 HTTP API 提供状态、配置和人工启动入口。"""

    def __init__(self, host: str, port: int, application: RecordReplayApplication) -> None:
        self._application = application
        self._server = ThreadingHTTPServer((host, port), self._build_handler())

    def serve(self) -> None:
        """持续处理 HTTP 请求。"""

        self._server.serve_forever(poll_interval=0.5)

    def close(self) -> None:
        """在 `serve_forever()` 已退出后关闭监听 socket。

        `HTTPServer.shutdown()` 只能由运行 `serve_forever()` 之外的线程调用。服务入口
        通过主线程信号异常退出请求循环，因此这里不得再次调用 `shutdown()`，否则主
        线程会等待自身结束并形成停机死锁。
        """

        self._server.server_close()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        application = self._application

        class Handler(BaseHTTPRequestHandler):
            """将固定 HTTP 路径映射到 application。"""

            request_id = ""

            def do_GET(self) -> None:
                self._begin_request()
                try:
                    path = urlsplit(self.path).path
                    if path == "/health":
                        self._send(HTTPStatus.OK, application.health())
                        return
                    if path == "/status":
                        self._send(HTTPStatus.OK, application.status())
                        return
                    if path == "/plan":
                        plan_options = self._read_plan_options()
                        response = application.get_plan(*plan_options)
                        if not response.accepted:
                            self._send_error(
                                HTTPStatus.BAD_REQUEST,
                                response.error_text or "回放计划未被接受",
                                response.error_code or _ERROR_INVALID_STATE,
                            )
                            return
                        self._send(HTTPStatus.OK, response)
                        return
                    if path == "/config":
                        self._send(HTTPStatus.OK, application.get_parameters())
                        return
                    if path == "/device-status":
                        self._send(HTTPStatus.OK, application.get_device_status())
                        return
                    if path in _POST_PATHS:
                        self._send_method_not_allowed("POST")
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "请求路径不存在", _ERROR_NOT_FOUND)
                except Exception as error:
                    self._handle_exception(error)

            def do_POST(self) -> None:
                self._begin_request()
                try:
                    path = urlsplit(self.path).path
                    if path == "/start":
                        response = application.start(*self._read_start_options())
                        if not response.accepted:
                            self._send_error(
                                HTTPStatus.BAD_REQUEST,
                                response.error_text or "回放启动未被接受",
                                response.error_code or _ERROR_INVALID_STATE,
                            )
                            return
                        self._send(HTTPStatus.ACCEPTED, response)
                        return
                    if path == "/stop":
                        response = application.stop()
                        if not response.accepted:
                            self._send_error(
                                HTTPStatus.BAD_REQUEST,
                                response.error_text or "停止请求未被接受",
                                response.error_code or _ERROR_INVALID_STATE,
                            )
                            return
                        self._send(HTTPStatus.OK, response)
                        return
                    if path == "/reset":
                        response = application.reset()
                        if not response.accepted:
                            self._send_error(
                                HTTPStatus.BAD_REQUEST,
                                response.error_text or "复位请求未被接受",
                                response.error_code or _ERROR_INVALID_STATE,
                            )
                            return
                        self._send(HTTPStatus.OK, response)
                        return
                    if path == "/config":
                        self._send(HTTPStatus.OK, application.update_parameters(self._read_changes()))
                        return
                    if path == "/prior/ball-pose":
                        self._send(
                            HTTPStatus.OK,
                            application.replace_ball_pose_prior(self._read_json_object()),
                        )
                        return
                    if path == "/prior/charuco":
                        self._send(
                            HTTPStatus.OK,
                            application.replace_charuco_prior(self._read_json_object()),
                        )
                        return
                    if path in _GET_PATHS:
                        self._send_method_not_allowed("GET")
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "请求路径不存在", _ERROR_NOT_FOUND)
                except Exception as error:
                    self._handle_exception(error)

            def do_PUT(self) -> None:
                self._reject_unsupported_method()

            def do_PATCH(self) -> None:
                self._reject_unsupported_method()

            def do_DELETE(self) -> None:
                self._reject_unsupported_method()

            def do_OPTIONS(self) -> None:
                self._reject_unsupported_method()

            def do_HEAD(self) -> None:
                self._reject_unsupported_method()

            def do_CONNECT(self) -> None:
                self._reject_unsupported_method()

            def do_TRACE(self) -> None:
                self._reject_unsupported_method()

            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _begin_request(self) -> None:
                """为一次 HTTP 请求生成可跨客户端与服务日志检索的标识。"""

                self.request_id = uuid.uuid4().hex[:12]
                logger.info(
                    "RecordReplay HTTP 请求开始 request_id={} method={} path={}",
                    self.request_id,
                    self.command,
                    self.path,
                )

            def _reject_unsupported_method(self) -> None:
                """对服务未实现的 HTTP 方法返回结构化 405。"""

                self._begin_request()
                path = urlsplit(self.path).path
                if path in _GET_PATHS:
                    self._send_method_not_allowed("GET")
                    return
                if path in _POST_PATHS:
                    self._send_method_not_allowed("POST")
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "请求路径不存在", _ERROR_NOT_FOUND)

            def _send_method_not_allowed(self, allowed_method: str) -> None:
                """返回已知资源支持的方法，避免 BaseHTTPRequestHandler 的 HTML 错误页。"""

                self._send_error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    f"请求方法 {self.command} 不适用于 {urlsplit(self.path).path}；"
                    f"请使用 {allowed_method}",
                    _ERROR_METHOD_NOT_ALLOWED,
                    extra_headers={"Allow": allowed_method},
                )

            def _handle_exception(self, error: Exception) -> None:
                """把已知拒绝和未知异常统一转换为稳定 JSON 错误。"""

                if isinstance(error, ApiRequestError | RecordReplayApplicationError):
                    self._send_error(HTTPStatus.BAD_REQUEST, str(error), error.error_code)
                    return
                logger.exception(
                    "RecordReplay HTTP 未处理异常 request_id={} method={} path={} type={}",
                    self.request_id,
                    self.command,
                    self.path,
                    type(error).__name__,
                )
                self._send_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "服务处理发生未预期异常；"
                    f"request_id={self.request_id} type={type(error).__name__} "
                    f"detail={_safe_error_detail(error)}",
                    _ERROR_INTERNAL,
                )

            def _read_changes(self) -> dict[str, RuntimeParameterValue]:
                payload = self._read_json_object()
                changes: dict[str, RuntimeParameterValue] = {}
                for key, value in payload.items():
                    if isinstance(value, bool):
                        raise ApiRequestError("invalid_request", "配置值不能是 bool")
                    if isinstance(value, int | float):
                        changes[key] = value
                        continue
                    raise ApiRequestError(
                        "invalid_request",
                        "运行参数值必须是数字；动作 speed/zone 请修改 action_sequence.json",
                    )
                return changes

            def _read_start_options(self) -> tuple[int, int, int, int, bool, str]:
                payload = self._read_json_object()
                required = {
                    "old_tray_current_index",
                    "old_tray_put_index",
                    "new_tray_current_index",
                    "new_tray_put_index",
                    "enable_agv_navigation",
                    "agv_target",
                }
                if set(payload) != required:
                    raise ApiRequestError(
                        "invalid_request",
                        "start 请求必须且只能包含 old_tray_current_index、old_tray_put_index、"
                        "new_tray_current_index、new_tray_put_index、"
                        "enable_agv_navigation、agv_target"
                    )
                old_tray_current_index = _read_positive_json_int(
                    payload, "old_tray_current_index"
                )
                old_tray_put_index = _read_positive_json_int(payload, "old_tray_put_index")
                new_tray_current_index = _read_positive_json_int(
                    payload, "new_tray_current_index"
                )
                new_tray_put_index = _read_positive_json_int(payload, "new_tray_put_index")
                enable_agv_navigation = payload["enable_agv_navigation"]
                if not isinstance(enable_agv_navigation, bool):
                    raise ApiRequestError("invalid_request", "enable_agv_navigation 必须是 bool")
                agv_target = payload["agv_target"]
                if not isinstance(agv_target, str) or not agv_target.strip():
                    raise ApiRequestError("invalid_request", "agv_target 必须是非空字符串")
                return (
                    old_tray_current_index,
                    old_tray_put_index,
                    new_tray_current_index,
                    new_tray_put_index,
                    enable_agv_navigation,
                    agv_target.strip(),
                )

            def _read_plan_options(self) -> tuple[int, int, int, int]:
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                required = {
                    "old_tray_current_index",
                    "old_tray_put_index",
                    "new_tray_current_index",
                    "new_tray_put_index",
                }
                if set(query) != required:
                    raise ApiRequestError(
                        "invalid_request",
                        "plan 查询参数必须且只能包含 old_tray_current_index、old_tray_put_index、"
                        "new_tray_current_index、new_tray_put_index"
                    )
                return (
                    _parse_positive_query_int(query, "old_tray_current_index"),
                    _parse_positive_query_int(query, "old_tray_put_index"),
                    _parse_positive_query_int(query, "new_tray_current_index"),
                    _parse_positive_query_int(query, "new_tray_put_index"),
                )

            def _read_json_object(self) -> dict[str, object]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as error:
                    raise ApiRequestError(
                        "invalid_request",
                        "Content-Length 必须是非负整数",
                    ) from error
                if length < 0:
                    raise ApiRequestError("invalid_request", "Content-Length 必须是非负整数")
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ApiRequestError(
                        "invalid_request",
                        "请求 body 必须是合法 UTF-8 JSON object",
                    ) from error
                if not isinstance(payload, dict):
                    raise ApiRequestError("invalid_request", "请求 body 必须是 JSON object")
                result: dict[str, object] = {}
                for key, value in payload.items():
                    if not isinstance(key, str):
                        raise ApiRequestError("invalid_request", "JSON object 字段名必须是字符串")
                    result[key] = value
                return result

            def _send(
                self,
                status: HTTPStatus,
                response: ApiResponse,
                *,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                payload = json.dumps(asdict(response), ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Request-ID", self.request_id)
                if extra_headers is not None:
                    for name, value in extra_headers.items():
                        self.send_header(name, value)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
                logger.info(
                    "RecordReplay HTTP 请求结束 request_id={} method={} path={} status={}",
                    self.request_id,
                    self.command,
                    self.path,
                    status.value,
                )

            def _send_error(
                self,
                status: HTTPStatus,
                message: str,
                error_code: ReplayErrorCode,
                *,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                logger.warning(
                    "RecordReplay HTTP 请求拒绝 request_id={} method={} path={} status={} "
                    "error_code={} error_text={}",
                    self.request_id,
                    self.command,
                    self.path,
                    status.value,
                    error_code,
                    message,
                )
                self._send(
                    status,
                    RecordReplayErrorResponse(error_code=error_code, error_text=message),
                    extra_headers=extra_headers,
                )

        return Handler


def _safe_error_detail(error: Exception) -> str:
    """把异常摘要压成单行并限长；完整堆栈只写入服务日志。"""

    detail = " ".join(str(error).splitlines()).strip()
    if not detail:
        detail = "异常未提供文本说明"
    return detail[:1000]


def _parse_positive_query_int(query: dict[str, list[str]], key: str) -> int:
    """读取一个单值正整数查询参数。"""

    values = query.get(key)
    if values is None or len(values) != 1:
        raise ApiRequestError("invalid_index", f"{key} 必须是单个正整数")
    try:
        value = int(values[0])
    except ValueError as error:
        raise ApiRequestError("invalid_index", f"{key} 必须是单个正整数") from error
    if value <= 0:
        raise ApiRequestError("invalid_index", f"{key} 必须是大于 0 的整数")
    return value


def _read_positive_json_int(payload: dict[str, object], key: str) -> int:
    """读取一个 JSON object 中的单值正整数。"""

    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ApiRequestError("invalid_index", f"{key} 必须是大于 0 的整数")
    return value

