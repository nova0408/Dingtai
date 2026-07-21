"""RecordReplay HTTP/JSON 服务。"""

from __future__ import annotations

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .application import RecordReplayApplication
from .protocol import RecordReplayResponse


class RecordReplayServer:
    """通过标准 HTTP API 提供状态、配置和人工启动入口。"""

    def __init__(self, host: str, port: int, application: RecordReplayApplication) -> None:
        self._application = application
        self._server = ThreadingHTTPServer((host, port), self._build_handler())

    def serve(self) -> None:
        """持续处理 HTTP 请求。"""

        self._server.serve_forever(poll_interval=0.5)

    def close(self) -> None:
        """停止请求循环并关闭监听 socket。"""

        self._server.shutdown()
        self._server.server_close()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        application = self._application

        class Handler(BaseHTTPRequestHandler):
            """将固定 HTTP 路径映射到 application。"""

            def do_GET(self) -> None:
                try:
                    if self.path == "/status":
                        self._send(HTTPStatus.OK, application.status())
                        return
                    if self.path == "/config":
                        self._send(HTTPStatus.OK, application.get_parameters())
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported path")
                except Exception as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"{type(exc).__name__}: {exc}")

            def do_POST(self) -> None:
                try:
                    if self.path == "/start":
                        self._require_empty_body()
                        self._send(HTTPStatus.ACCEPTED, application.start())
                        return
                    if self.path == "/config":
                        self._send(HTTPStatus.OK, application.update_parameters(self._read_changes()))
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported path")
                except Exception as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"{type(exc).__name__}: {exc}")

            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _read_changes(self) -> dict[str, float | int]:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("配置请求必须是 JSON object")
                changes: dict[str, float | int] = {}
                for key, value in payload.items():
                    if not isinstance(key, str) or not isinstance(value, int | float):
                        raise ValueError("配置字段名必须是字符串，值必须是数字")
                    changes[key] = value
                return changes

            def _require_empty_body(self) -> None:
                if int(self.headers.get("Content-Length", "0")) != 0:
                    raise ValueError("start 请求不接受 body")

            def _send(self, status: HTTPStatus, response: RecordReplayResponse) -> None:
                payload = json.dumps(asdict(response), ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_error(self, status: HTTPStatus, message: str) -> None:
                response = application.status(accepted=False)
                payload = RecordReplayResponse(
                    state=response.state,
                    accepted=False,
                    left_csv_state=response.left_csv_state,
                    plan_index=response.plan_index,
                    error_text=message,
                )
                self._send(status, payload)

        return Handler

