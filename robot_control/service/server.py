"""RobotControl HTTP/JSON 服务。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from typing import Any

from .application import RobotControlApplication


class RobotControlServer:
    """把固定的 GET/POST 路径映射到 RobotControlApplication。"""

    def __init__(
        self,
        host: str,
        port: int,
        application: RobotControlApplication,
        status_stream_interval_s: float = 0.2,
    ) -> None:
        """创建 HTTP 监听器。"""

        self._stop_event = threading.Event()
        self._server = ThreadingHTTPServer(
            (host, port),
            self._build_handler(
                application, self._stop_event, status_stream_interval_s
            ),
        )
        self._server.daemon_threads = True

    def serve(self) -> None:
        """持续处理 HTTP 请求。"""

        self._server.serve_forever(poll_interval=0.5)

    def close(self) -> None:
        """关闭 HTTP socket；不发送任何硬件控制请求。"""

        self._stop_event.set()
        self._server.shutdown()
        self._server.server_close()

    @staticmethod
    def _build_handler(
        application: RobotControlApplication,
        stop_event: threading.Event,
        status_stream_interval_s: float,
    ) -> type[BaseHTTPRequestHandler]:
        """构造绑定应用门面的请求处理器。"""

        class Handler(BaseHTTPRequestHandler):
            """RobotControl 固定路由处理器。"""

            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                """处理健康检查与只读状态请求。"""

                path = urlsplit(self.path).path
                try:
                    if path == "/api/v1/health":
                        self._send(HTTPStatus.OK, application.health())
                        return
                    if path in {"/api/v1/status", "/api/v1/devices"}:
                        self._send(HTTPStatus.OK, asdict(application.status()))
                        return
                    if path == "/api/v1/qmlinker/agv/targets":
                        self._send(
                            HTTPStatus.OK, application.qmlinker_get_agv_targets()
                        )
                        return
                    if path == "/api/v1/status/stream":
                        self._stream_status()
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported GET path")
                except Exception as exc:
                    self._send_error(
                        HTTPStatus.SERVICE_UNAVAILABLE, f"{type(exc).__name__}: {exc}"
                    )

            def do_POST(self) -> None:
                """处理人工发起的控制请求。

                Codex 不调用本方法对应的接口；本机验证只允许 GET。
                """

                path = urlsplit(self.path).path
                try:
                    payload = self._read_json_object()
                    response = self._dispatch_control(path, payload)
                    self._send(HTTPStatus.ACCEPTED, asdict(response))
                except ValueError as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                except Exception as exc:
                    self._send_error(
                        HTTPStatus.SERVICE_UNAVAILABLE, f"{type(exc).__name__}: {exc}"
                    )

            def log_message(self, format: str, *args: object) -> None:
                """交给上层服务日志，避免标准错误输出重复。"""

                del format, args

            def _stream_status(self) -> None:
                """以 SSE 推送完整设备状态快照。"""

                interval_s = _status_stream_interval(
                    urlsplit(self.path).query,
                    status_stream_interval_s,
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                sequence = 0
                while not stop_event.is_set():
                    payload = asdict(application.status())
                    event = (
                        "event: robot_status\n"
                        f"id: {sequence}\n"
                        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    )
                    try:
                        self.wfile.write(event.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    sequence += 1
                    if stop_event.wait(interval_s):
                        return

            def _dispatch_control(self, path: str, payload: dict[str, object]) -> Any:
                """按显式路径分发控制请求，不使用动态方法名调用。"""

                parts = [part for part in path.split("/") if part]
                if parts[:3] == ["api", "v1", "qmlinker"]:
                    return self._dispatch_qmlinker(parts[3:], payload)
                if parts[:3] == ["api", "v1", "ar5"]:
                    return self._dispatch_ar5(parts[3:], payload)
                raise ValueError("unsupported POST path")

            def _dispatch_qmlinker(
                self, parts: list[str], payload: dict[str, object]
            ) -> Any:
                """分发 qmlinker 控制路径。"""

                if parts == ["head"]:
                    return application.qmlinker_set_head(
                        _optional_bool(payload, "enable"),
                        _optional_number(payload, "yaw_deg"),
                        _optional_number(payload, "pitch_deg"),
                    )
                if parts == ["lift"]:
                    return application.qmlinker_set_lift(
                        _optional_bool(payload, "enable"),
                        _optional_number(payload, "height_mm"),
                    )
                if parts == ["gripper"]:
                    return application.qmlinker_set_gripper_position(
                        _int_field(payload, "position")
                    )
                if parts == ["gripper", "enable"]:
                    return application.qmlinker_set_gripper_enabled(
                        _bool_field(payload, "enabled")
                    )
                if parts == ["gripper", "calibrate"]:
                    return application.qmlinker_calibrate_gripper()
                if parts == ["right-hand"]:
                    return application.qmlinker_set_right_hand(
                        _number_sequence(payload, "positions")
                    )
                if parts == ["right-hand", "enable"]:
                    return application.qmlinker_set_right_hand_enabled(
                        _bool_field(payload, "enabled")
                    )
                if parts == ["agv", "navigate"]:
                    return application.qmlinker_navigate_to(
                        _string_field(payload, "target")
                    )
                if parts == ["agv", "enable"]:
                    return application.qmlinker_set_agv_enabled(
                        _bool_field(payload, "enabled")
                    )
                if parts == ["agv", "translate"]:
                    return application.qmlinker_translate_agv(
                        _number_field(payload, "speed_mps"),
                        _number_field(payload, "direction_deg"),
                    )
                if parts == ["agv", "stop"]:
                    return application.qmlinker_stop_agv()
                raise ValueError("unsupported qmlinker control path")

            def _dispatch_ar5(
                self, parts: list[str], payload: dict[str, object]
            ) -> Any:
                """分发 AR5 控制路径。"""

                if len(parts) != 2:
                    raise ValueError("AR5 path must contain side and action")
                side, action = parts
                if action == "power":
                    return application.ar5_set_power(
                        side, _bool_field(payload, "enabled")
                    )
                if action == "mode":
                    return application.ar5_set_operate_mode(
                        side, _bool_field(payload, "automatic")
                    )
                if action == "recover-estop":
                    return application.ar5_recover_estop(side)
                if action == "drag":
                    return application.ar5_set_drag_enabled(
                        side, _bool_field(payload, "enabled")
                    )
                if action == "jog":
                    return application.ar5_start_jog(
                        side,
                        _string_field(payload, "space"),
                        _int_field(payload, "axis_index"),
                        _bool_field(payload, "direction_positive"),
                        _number_field(payload, "rate"),
                        _number_field(payload, "step"),
                    )
                if action == "stop":
                    return application.ar5_stop(side)
                if action == "move-joints":
                    return application.ar5_move_joints(
                        side,
                        _number_sequence(payload, "joint_deg"),
                        _number_field(payload, "speed_mm_s"),
                        _number_field(payload, "zone_mm"),
                    )
                if action == "move-cartesian":
                    return application.ar5_move_cartesian(
                        side,
                        _number_sequence(payload, "xyz_mm"),
                        _number_sequence(payload, "rpy_deg"),
                        _number_field(payload, "elbow_deg"),
                        _number_field(payload, "speed_mm_s"),
                        _number_field(payload, "zone_mm"),
                    )
                if action == "move-elbow":
                    return application.ar5_move_elbow(
                        side,
                        _number_field(payload, "elbow_deg"),
                        _number_field(payload, "speed_mm_s"),
                        _number_field(payload, "zone_mm"),
                    )
                raise ValueError("unsupported AR5 control action")

            def _read_json_object(self) -> dict[str, object]:
                """读取并校验 JSON object。"""

                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                value = json.loads(raw or "{}")
                if not isinstance(value, dict):
                    raise ValueError("request body must be a JSON object")
                return {str(key): item for key, item in value.items()}

            def _send(self, status: HTTPStatus, payload: object) -> None:
                """发送 JSON 响应。"""

                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_error(self, status: HTTPStatus, message: str) -> None:
                """发送统一错误响应。"""

                self._send(status, {"ok": False, "error": message})

        return Handler


def _status_stream_interval(query: str, default_interval_s: float) -> float:
    """读取并校验 SSE 推送间隔。"""

    values = parse_qs(query, keep_blank_values=True).get("interval_s")
    if not values or not values[0]:
        return default_interval_s
    try:
        interval_s = float(values[0])
    except ValueError as exc:
        raise ValueError("interval_s must be a number") from exc
    if not 0.05 <= interval_s <= 5.0:
        raise ValueError("interval_s must be between 0.05 and 5.0")
    return interval_s


def _number_field(payload: dict[str, object], name: str) -> float:
    """读取有限数值字段。"""

    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_number(payload: dict[str, object], name: str) -> float | None:
    """读取可选数值字段。"""

    if name not in payload:
        return None
    return _number_field(payload, name)


def _number_sequence(payload: dict[str, object], name: str) -> tuple[float, ...]:
    """读取数值序列。"""

    value = payload.get(name)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be an array")
    result = tuple(_number_field({"value": item}, "value") for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _bool_field(payload: dict[str, object], name: str) -> bool:
    """读取布尔字段。"""

    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _optional_bool(payload: dict[str, object], name: str) -> bool | None:
    """读取可选布尔字段。"""

    if name not in payload:
        return None
    return _bool_field(payload, name)


def _int_field(payload: dict[str, object], name: str) -> int:
    """读取整数域。"""

    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _string_field(payload: dict[str, object], name: str) -> str:
    """读取非空字符串域。"""

    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
