"""RecordReplay HTTP/JSON 服务。"""

from __future__ import annotations

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .application import RecordReplayApplication
from .config_store import RuntimeParameterValue
from ..device_status import DeviceStatusResponse
from .protocol import PriorUploadResponse, RecordReplayResponse

ApiResponse = RecordReplayResponse | DeviceStatusResponse | PriorUploadResponse


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
                    if self.path == "/status":
                        self._send(HTTPStatus.OK, application.status())
                        return
                    if self.path == "/config":
                        self._send(HTTPStatus.OK, application.get_parameters())
                        return
                    if self.path == "/device-status":
                        self._send(HTTPStatus.OK, application.get_device_status())
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported path")
                except Exception as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"{type(exc).__name__}: {exc}")

            def do_POST(self) -> None:
                try:
                    if self.path == "/start":
                        enable_agv_navigation = self._read_start_options()
                        response = application.start(enable_agv_navigation)
                        response_status = (
                            HTTPStatus.BAD_REQUEST
                            if not response.accepted and response.error_text is not None
                            else HTTPStatus.ACCEPTED
                        )
                        self._send(response_status, response)
                        return
                    if self.path == "/stop":
                        self._send(HTTPStatus.OK, application.stop())
                        return
                    if self.path == "/reset":
                        self._send(HTTPStatus.OK, application.reset())
                        return
                    if self.path == "/config":
                        self._send(HTTPStatus.OK, application.update_parameters(self._read_changes()))
                        return
                    if self.path == "/prior/ball-pose":
                        self._send(
                            HTTPStatus.OK,
                            application.replace_ball_pose_prior(self._read_json_object()),
                        )
                        return
                    if self.path == "/prior/charuco":
                        self._send(
                            HTTPStatus.OK,
                            application.replace_charuco_prior(self._read_json_object()),
                        )
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported path")
                except Exception as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, f"{type(exc).__name__}: {exc}")

            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _read_changes(self) -> dict[str, RuntimeParameterValue]:
                payload = self._read_json_object()
                changes: dict[str, RuntimeParameterValue] = {}
                for key, value in payload.items():
                    if isinstance(value, bool):
                        raise ValueError("配置值不能是 bool")
                    if isinstance(value, int | float):
                        changes[key] = value
                        continue
                    raise ValueError("运行参数值必须是数字；动作 speed/zone 请修改 action_sequence.json")
                return changes

            def _read_start_options(self) -> bool:
                payload = self._read_json_object()
                if set(payload) != {"enable_agv_navigation"}:
                    raise ValueError("start 请求必须且只能包含 enable_agv_navigation")
                enable_agv_navigation = payload["enable_agv_navigation"]
                if not isinstance(enable_agv_navigation, bool):
                    raise ValueError("enable_agv_navigation 必须是 bool")
                return enable_agv_navigation

            def _read_json_object(self) -> dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("请求 body 必须是 JSON object")
                result: dict[str, object] = {}
                for key, value in payload.items():
                    if not isinstance(key, str):
                        raise ValueError("JSON object 字段名必须是字符串")
                    result[key] = value
                return result

            def _send(self, status: HTTPStatus, response: ApiResponse) -> None:
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
                    action_sequence_sha256=response.action_sequence_sha256,
                    left_csv_state=response.left_csv_state,
                    plan_index=response.plan_index,
                    error_text=message,
                    left_csv_files=response.left_csv_files,
                    right_csv_files=response.right_csv_files,
                    execution_tasks=response.execution_tasks,
                    current_task_sequence=response.current_task_sequence,
                    current_task_active=response.current_task_active,
                    total_execution_count=response.total_execution_count,
                    current_left_csv=response.current_left_csv,
                    current_left_action_name=response.current_left_action_name,
                    current_left_action_index=response.current_left_action_index,
                    current_right_csv=response.current_right_csv,
                    current_right_action_name=response.current_right_action_name,
                    current_right_action_index=response.current_right_action_index,
                    current_left_row=response.current_left_row,
                    current_right_row=response.current_right_row,
                    current_left_total_rows=response.current_left_total_rows,
                    current_right_total_rows=response.current_right_total_rows,
                    offset_statuses=response.offset_statuses,
                )
                self._send(status, payload)

        return Handler

