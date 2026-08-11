"""手眼标定与先验记录 HTTP 服务器。"""

from __future__ import annotations

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import urlsplit

from .application import CalibrationApplication, CalibrationKind
from .protocol import CalibrationResponse


class CalibrationServer:
    """把固定 API 路径映射到标定应用。"""

    def __init__(self, host: str, port: int, application: CalibrationApplication) -> None:
        self._server = ThreadingHTTPServer((host, port), self._build_handler(application))
        self._server.daemon_threads = True

    def serve(self) -> None:
        """持续处理 HTTP 请求。"""

        self._server.serve_forever(poll_interval=0.5)

    def close(self) -> None:
        """关闭 HTTP 监听，不触发任何设备操作。"""

        self._server.shutdown()
        self._server.server_close()

    @staticmethod
    def _build_handler(application: CalibrationApplication) -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            """标定服务固定路由处理器。"""

            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/api/v1/health":
                    self._send(HTTPStatus.OK, application.health())
                    return
                if path == "/api/v1/status":
                    self._send(HTTPStatus.OK, application.status())
                    return
                if path == "/api/v1/hand-eye/config":
                    self._send(HTTPStatus.OK, application.get_hand_eye_config())
                    return
                if path == "/api/v1/results/hand-eye":
                    self._send(HTTPStatus.OK, application.get_result("hand_eye"))
                    return
                if path == "/api/v1/results/head-eye":
                    self._send(HTTPStatus.OK, application.get_result("head_eye", "left"))
                    return
                if path.startswith("/api/v1/results/head-eye/"):
                    self._send(HTTPStatus.OK, application.get_result("head_eye", path.rsplit("/", 1)[-1]))
                    return
                if path == "/api/v1/results/prior/head":
                    self._send(HTTPStatus.OK, application.get_result("head_prior"))
                    return
                if path == "/api/v1/results/prior/hand":
                    self._send(HTTPStatus.OK, application.get_result("hand_prior"))
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "unsupported path")

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                try:
                    payload = self._read_json_object()
                    if path == "/api/v1/start":
                        response = application.start_calibration(
                            _read_calibration_kind(payload),
                            _read_arm_side(payload),
                        )
                    elif path == "/api/v1/end":
                        response = application.end_calibration(
                            _read_calibration_kind(payload),
                            _read_arm_side(payload),
                        )
                    elif path == "/api/v1/cancel":
                        response = application.cancel()
                    elif path == "/api/v1/replacements/confirm":
                        response = application.confirm_replacement(
                            _read_string(payload, "replacement_id"),
                            _read_bool(payload, "confirmed"),
                        )
                    elif path == "/api/v1/prior/head":
                        response = application.record_head_prior()
                    elif path == "/api/v1/prior/hand":
                        response = application.record_hand_prior(_read_arm_side(payload))
                    elif path == "/api/v1/hand-eye/sample":
                        response = application.capture_hand_eye_sample(_read_arm_side(payload))
                    elif path == "/api/v1/head-eye/sample":
                        response = application.capture_head_eye_sample(_read_arm_side(payload))
                    elif path == "/api/v1/hand-eye/solve":
                        response = application.solve_hand_eye(payload)
                    elif path == "/api/v1/head-eye/solve":
                        response = application.solve_head_eye(_read_arm_side(payload), payload)
                    else:
                        self._send_error(HTTPStatus.NOT_FOUND, "unsupported path")
                        return
                    status = HTTPStatus.OK if response.accepted else HTTPStatus.CONFLICT
                    self._send(status, response)
                except Exception as error:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"{type(error).__name__}: {error}")

            def do_PATCH(self) -> None:
                path = urlsplit(self.path).path
                if path != "/api/v1/hand-eye/config":
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported path")
                    return
                try:
                    response = application.update_hand_eye_config(self._read_json_object())
                    status = HTTPStatus.OK if response.accepted else HTTPStatus.CONFLICT
                    self._send(status, response)
                except Exception as error:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"{type(error).__name__}: {error}")

            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _read_json_object(self) -> dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("请求 body 必须是 JSON object")
                return value

            def _send(self, status: HTTPStatus, response: CalibrationResponse) -> None:
                body = json.dumps(asdict(response), ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_error(self, status: HTTPStatus, message: str) -> None:
                self._send(status, CalibrationResponse(accepted=False, error=message))

        return Handler


def _read_arm_side(payload: dict[str, object]) -> str:
    value = payload.get("arm_side", "left")
    if not isinstance(value, str):
        raise ValueError("arm_side 必须是字符串")
    return value


def _read_calibration_kind(payload: dict[str, object]) -> CalibrationKind:
    value = payload.get("calibration_kind")
    if value not in {"left_eye_in_hand", "head_eye_to_hand"}:
        raise ValueError(
            "calibration_kind 必须是 left_eye_in_hand 或 head_eye_to_hand"
        )
    return cast(CalibrationKind, value)


def _read_string(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value


def _read_bool(payload: dict[str, object], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是 boolean")
    return value
