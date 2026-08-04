"""CameraPipeline 对外 HTTP/JSON 适配层。

该模块只负责把 HTTP JSON 转换为现有协议对象，再调用
``CameraPipelineApplication``。算法、相机缓存和内部 ZMQ RPC 不在这里重复实现。
"""

from __future__ import annotations

import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from loguru import logger

from ..ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPosePriorInfo,
)
from ..protocol import CameraName
from .application import CameraPipelineApplication
from .external_codec import JsonObject, JsonValue, to_json_value
from .protocol import (
    PROTOCOL_VERSION,
    SERVICE_VERSION,
    CameraPipelineServiceRequest,
    CharucoDetectionRequest,
)


class CameraPipelineHttpServer:
    """在独立 HTTP 端口提供 CameraPipeline 外部控制接口。"""

    def __init__(
        self,
        host: str,
        port: int,
        application: CameraPipelineApplication,
    ) -> None:
        self._application = application
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """后台启动 HTTP 服务。"""

        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="camera-pipeline-http",
            daemon=True,
        )
        self._thread.start()
        logger.info("camera pipeline HTTP API started address={}", self._server.server_address)

    def close(self) -> None:
        """停止 HTTP 服务并释放监听 socket。"""

        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._server.server_close()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        application = self._application

        class Handler(BaseHTTPRequestHandler):
            """把稳定的外部路径映射为内部协议请求。"""

            def do_GET(self) -> None:
                try:
                    path, query = self._path_and_query()
                    if path == "/api/v1/health":
                        self._send_ok(
                            {
                                "service_version": SERVICE_VERSION,
                                "zmq_protocol_version": PROTOCOL_VERSION,
                            }
                        )
                        return
                    parts = self._parts(path)
                    if parts == ("api", "v1", "cameras"):
                        self._send_payload(application.get_camera_inventory())
                        return
                    if len(parts) == 5 and parts[:3] == ("api", "v1", "cameras"):
                        camera_name = self._camera_name(parts[3])
                        timeout_s = self._query_timeout(query)
                        if parts[4] == "status":
                            response = application.handle(
                                CameraPipelineServiceRequest(
                                    operation="camera_status",
                                    camera_name=camera_name,
                                    timeout_s=timeout_s,
                                )
                            )
                            self._send_payload(response.camera_status)
                            return
                        if parts[4] == "intrinsics":
                            response = application.handle(
                                CameraPipelineServiceRequest(
                                    operation="camera_intrinsics",
                                    camera_name=camera_name,
                                    timeout_s=timeout_s,
                                )
                            )
                            self._send_payload(response.camera_intrinsics)
                            return
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported path", "not_found")
                except Exception as exc:  # noqa: BLE001
                    self._handle_exception(exc)

            def do_POST(self) -> None:
                try:
                    path, _query = self._path_and_query()
                    parts = self._parts(path)
                    body = self._read_json_object()
                    if len(parts) == 5 and parts[:3] == ("api", "v1", "cameras"):
                        camera_name = self._camera_name(parts[3])
                        if parts[4] == "stable-frame":
                            timeout_s = _optional_float(body, "timeout_s", 10.0)
                            response = application.handle(
                                CameraPipelineServiceRequest(
                                    operation="stable_frame",
                                    camera_name=camera_name,
                                    timeout_s=timeout_s,
                                )
                            )
                            self._send_payload(response.stable_frame)
                            return
                    if parts == ("api", "v1", "detections", "ball"):
                        request = _parse_ball_request(body)
                        response = application.handle(
                            CameraPipelineServiceRequest(
                                operation="detect_ball",
                                camera_name=request.camera_name,
                                detect_ball=request,
                            )
                        )
                        self._send_payload(response.detect_ball)
                        return
                    if parts == ("api", "v1", "detections", "charuco"):
                        request = _parse_charuco_request(body)
                        response = application.handle(
                            CameraPipelineServiceRequest(
                                operation="detect_charuco",
                                camera_name=request.camera_name,
                                detect_charuco=request,
                            )
                        )
                        self._send_payload(response.detect_charuco)
                        return
                    self._send_error(HTTPStatus.NOT_FOUND, "unsupported path", "not_found")
                except Exception as exc:  # noqa: BLE001
                    self._handle_exception(exc)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _path_and_query(self) -> tuple[str, dict[str, list[str]]]:
                parsed = urlsplit(self.path)
                return parsed.path, parse_qs(parsed.query, keep_blank_values=True)

            @staticmethod
            def _parts(path: str) -> tuple[str, ...]:
                return tuple(part for part in path.split("/") if part)

            @staticmethod
            def _camera_name(value: str) -> CameraName:
                try:
                    return CameraName(unquote(value))
                except ValueError as exc:
                    raise ValueError(f"unsupported camera_name: {value}") from exc

            def _query_timeout(self, query: dict[str, list[str]]) -> float:
                values = query.get("timeout_s")
                timeout_s = 10.0 if not values else _finite_float(values[0], "timeout_s")
                if timeout_s <= 0.0:
                    raise ValueError("timeout_s must be greater than zero")
                return timeout_s

            def _read_json_object(self) -> JsonObject:
                length_text = self.headers.get("Content-Length", "0")
                try:
                    length = int(length_text)
                except ValueError as exc:
                    raise ValueError("Content-Length must be an integer") from exc
                if length < 0 or length > 8 * 1024 * 1024:
                    raise ValueError("request body is too large")
                if length == 0:
                    return {}
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                return {
                    str(key): _json_value(value)
                    for key, value in payload.items()
                }

            def _send_payload(self, payload: object) -> None:
                if payload is None:
                    raise RuntimeError("response payload missing")
                self._send_ok(to_json_value(payload))

            def _send_ok(self, data: JsonValue) -> None:
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "service_version": SERVICE_VERSION,
                        "zmq_protocol_version": PROTOCOL_VERSION,
                        "data": data,
                        "error": None,
                    },
                )

            def _send_error(
                self,
                status: HTTPStatus,
                message: str,
                code: str,
            ) -> None:
                self._send_json(
                    status,
                    {
                        "ok": False,
                        "service_version": SERVICE_VERSION,
                        "zmq_protocol_version": PROTOCOL_VERSION,
                        "data": None,
                        "error": {"code": code, "message": message},
                    },
                )

            def _handle_exception(self, exc: Exception) -> None:
                if isinstance(exc, (ValueError, TypeError, KeyError)):
                    status = HTTPStatus.BAD_REQUEST
                    code = "invalid_request"
                elif isinstance(exc, TimeoutError):
                    status = HTTPStatus.GATEWAY_TIMEOUT
                    code = "timeout"
                else:
                    status = HTTPStatus.SERVICE_UNAVAILABLE
                    code = "service_error"
                logger.warning("camera pipeline HTTP request failed code={} error={}", code, exc)
                self._send_error(status, f"{type(exc).__name__}: {exc}", code)

            def _send_json(self, status: HTTPStatus, payload: JsonObject) -> None:
                body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        return Handler


def _json_value(value: object) -> JsonValue:
    """把 json.loads 的边界值收窄到本模块允许的 JSON 类型。"""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _parse_ball_request(payload: JsonObject) -> BallPoseDetectionRequest:
    priors_payload = _required_list(payload, "priors")
    priors = tuple(_parse_ball_prior(item) for item in priors_payload)
    return BallPoseDetectionRequest(
        request_id=_required_int(payload, "request_id"),
        camera_name=_parse_camera(payload, "camera_name"),
        frame_id=_optional_int(payload, "frame_id", -1),
        enable_debug=_optional_bool(payload, "enable_debug", False),
        priors=priors,
    )


def _parse_ball_prior(payload: JsonValue) -> BallPosePriorInfo:
    item = _as_object(payload, "prior")
    hsv_payload = item.get("hsv_ranges", [])
    hsv_ranges: list[tuple[int, int, int, int, int, int]] = []
    for hsv in _as_list(hsv_payload, "hsv_ranges"):
        values = _as_list(hsv, "hsv_range")
        if len(values) != 6:
            raise ValueError("each hsv_range must contain six integers")
        integers = tuple(_as_int(value, "hsv_range value") for value in values)
        hsv_ranges.append(integers)  # type: ignore[arg-type]
    center = _as_list(item.get("model_center_mm"), "model_center_mm")
    if len(center) != 3:
        raise ValueError("model_center_mm must contain three numbers")
    center_values = tuple(_as_float(value, "model_center_mm value") for value in center)
    return BallPosePriorInfo(
        color_hex=_as_string(item.get("color_hex"), "color_hex"),
        diameter_mm=_as_float(item.get("diameter_mm"), "diameter_mm"),
        model_center_mm=(center_values[0], center_values[1], center_values[2]),
        hsv_ranges=tuple(hsv_ranges),
    )


def _parse_charuco_request(payload: JsonObject) -> CharucoDetectionRequest:
    return CharucoDetectionRequest(
        camera_name=_parse_camera(payload, "camera_name"),
        dictionary_name=_required_string(payload, "dictionary_name"),
        squares_x=_required_int(payload, "squares_x"),
        squares_y=_required_int(payload, "squares_y"),
        square_length_mm=_required_float(payload, "square_length_mm"),
        marker_length_mm=_required_float(payload, "marker_length_mm"),
        min_charuco_corners=_required_int(payload, "min_charuco_corners"),
        max_frames=_required_int(payload, "max_frames"),
        stable_timeout_s=_required_float(payload, "stable_timeout_s"),
        enable_debug=_optional_bool(payload, "enable_debug", False),
    )


def _parse_camera(payload: JsonObject, key: str) -> CameraName:
    try:
        return CameraName(_required_string(payload, key))
    except ValueError as exc:
        raise ValueError(f"unsupported {key}") from exc


def _required_string(payload: JsonObject, key: str) -> str:
    return _as_string(payload.get(key), key)


def _required_int(payload: JsonObject, key: str) -> int:
    return _as_int(payload.get(key), key)


def _required_float(payload: JsonObject, key: str) -> float:
    return _as_float(payload.get(key), key)


def _optional_int(payload: JsonObject, key: str, default: int) -> int:
    value = payload.get(key)
    return default if value is None else _as_int(value, key)


def _optional_bool(payload: JsonObject, key: str, default: bool) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be bool")
    return value


def _optional_float(payload: JsonObject, key: str, default: float) -> float:
    value = payload.get(key)
    number = default if value is None else _as_float(value, key)
    if number <= 0.0:
        raise ValueError(f"{key} must be greater than zero")
    return number


def _finite_float(value: str, key: str) -> float:
    return _as_float(value, key)


def _as_string(value: JsonValue | None, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _as_int(value: JsonValue | None, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be int")
    return value


def _as_float(value: JsonValue | None, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _as_list(value: JsonValue | None, key: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be array")
    return value


def _as_object(value: JsonValue | None, key: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be object")
    return value


def _required_list(payload: JsonObject, key: str) -> list[JsonValue]:
    return _as_list(payload.get(key), key)
