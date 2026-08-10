"""RecordReplay HTTP/JSON 服务。"""

from __future__ import annotations

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.parse import parse_qs, urlsplit

from .application import RecordReplayApplication, RecordReplayApplicationError
from .config_store import RuntimeParameterValue
from ..contracts import ReplayErrorCode
from ..device_status import DeviceStatusResponse
from .protocol import (
    PriorUploadResponse,
    RecordReplayErrorResponse,
    RecordReplayPlanResponse,
    RecordReplayResponse,
)

_ERROR_NOT_FOUND: Final[ReplayErrorCode] = "not_found"
_ERROR_INTERNAL: Final[ReplayErrorCode] = "internal_error"
_ERROR_INVALID_STATE: Final[ReplayErrorCode] = "invalid_state"

ApiResponse = (
    RecordReplayResponse
    | RecordReplayPlanResponse
    | RecordReplayErrorResponse
    | DeviceStatusResponse
    | PriorUploadResponse
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

            def do_GET(self) -> None:
                try:
                    path = urlsplit(self.path).path
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
                    self._send_error(HTTPStatus.NOT_FOUND, "请求路径不存在", _ERROR_NOT_FOUND)
                except ApiRequestError as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, str(exc), exc.error_code)
                except RecordReplayApplicationError as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, str(exc), exc.error_code)
                except ValueError as exc:
                    self._send_error(
                        HTTPStatus.BAD_REQUEST,
                        f"请求参数无效：{exc}",
                        "invalid_request",
                    )
                except Exception:
                    self._send_error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "服务内部处理失败",
                        _ERROR_INTERNAL,
                    )

            def do_POST(self) -> None:
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
                    self._send_error(HTTPStatus.NOT_FOUND, "请求路径不存在", _ERROR_NOT_FOUND)
                except ApiRequestError as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, str(exc), exc.error_code)
                except RecordReplayApplicationError as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, str(exc), exc.error_code)
                except ValueError as exc:
                    self._send_error(
                        HTTPStatus.BAD_REQUEST,
                        f"请求参数无效：{exc}",
                        "invalid_request",
                    )
                except Exception:
                    self._send_error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "服务内部处理失败",
                        _ERROR_INTERNAL,
                    )

            def log_message(self, format: str, *args: object) -> None:
                del format, args

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
                length = int(self.headers.get("Content-Length", "0"))
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

            def _send(self, status: HTTPStatus, response: ApiResponse) -> None:
                payload = json.dumps(asdict(response), ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_error(
                self,
                status: HTTPStatus,
                message: str,
                error_code: ReplayErrorCode,
            ) -> None:
                self._send(
                    status,
                    RecordReplayErrorResponse(error_code=error_code, error_text=message),
                )

        return Handler


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

