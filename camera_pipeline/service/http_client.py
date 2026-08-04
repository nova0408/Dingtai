"""CameraPipeline 对外 HTTP 与 WebSocket 客户端。

该客户端面向 GUI 和其它项目使用，不依赖内部 ZMQ transport。HTTP 负责控制和
检测请求，WebSocket 负责 CPWS1 最新帧订阅；返回值仍转换为 CameraPipeline
现有协议 dataclass，业务层不需要解析外部 JSON 细节。
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import socket
import ssl
import struct
from collections.abc import Iterator
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np

from ..ball_pose_detection.protocol import (
    BallDetectionInfo,
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPosePriorInfo,
)
from ..protocol import (
    CameraColorFramePacket,
    CameraDepthFramePacket,
    CameraFramePacket,
    CameraName,
)
from .protocol import (
    SERVICE_VERSION,
    CameraIntrinsicsResponse,
    CameraStatusResponse,
    CharucoDetectionRequest,
    CharucoDetectionResponse,
    StableFrameResponse,
)

_STREAM_MAGIC = b"CPWS1"
_STREAM_HEADER = struct.Struct("!5sI")
_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_DIRECT_OPENER = build_opener(ProxyHandler({}))


class CameraPipelineHttpClient:
    """通过 CameraPipeline 外部 HTTP/WebSocket API 访问服务。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6400",
        websocket_url: str | None = None,
        timeout_s: float = 30.0,
        stream_timeout_s: float = 5.0,
        *,
        api_prefix: str = "",
        websocket_prefix: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        default_websocket_url = self._base_url.replace("http://", "ws://", 1).replace(
            "https://", "wss://", 1
        )
        self._websocket_url = (websocket_url or default_websocket_url).rstrip("/")
        self._api_prefix = _normalize_prefix(api_prefix)
        self._websocket_prefix = _normalize_prefix(websocket_prefix)
        self._timeout_s = float(timeout_s)
        self._stream_timeout_s = float(stream_timeout_s)

    def close(self) -> None:
        """关闭客户端；已返回的 WebSocket 迭代器由自身释放 socket。"""

    @property
    def expected_service_version(self) -> str:
        """返回当前客户端代码要求的 CameraPipeline 功能版本。"""

        return SERVICE_VERSION

    def get_health(self) -> dict[str, object]:
        """读取外部 HTTP 健康检查结果。"""

        return self._get("/api/v1/health", self._timeout_s)

    def get_camera_inventory(
        self, timeout_s: float = 10.0
    ) -> tuple[CameraStatusResponse, ...]:
        """读取当前已配置、已连接且已有最新帧的相机清单。"""

        data = self._get("/api/v1/cameras", timeout_s)
        return tuple(
            _parse_camera_status(_as_object(item, "camera"))
            for item in _as_list(data, "cameras")
        )

    def get_camera_status(
        self, camera_name: CameraName, timeout_s: float = 10.0
    ) -> CameraStatusResponse:
        data = self._get(f"/api/v1/cameras/{quote(camera_name.value)}/status", timeout_s)
        return _parse_camera_status(data)

    def get_camera_intrinsics(
        self, camera_name: CameraName, timeout_s: float = 10.0
    ) -> CameraIntrinsicsResponse:
        data = self._get(
            f"/api/v1/cameras/{quote(camera_name.value)}/intrinsics", timeout_s
        )
        return CameraIntrinsicsResponse(
            camera_name=_as_string(data, "camera_name"),
            fx=_as_float(data, "fx"),
            fy=_as_float(data, "fy"),
            cx=_as_float(data, "cx"),
            cy=_as_float(data, "cy"),
            distortion=tuple(_as_float_value(item, "distortion") for item in _as_list(data, "distortion")),
            width=_as_int(data, "width"),
            height=_as_int(data, "height"),
            error=None,
        )

    def get_stable_frame(
        self, camera_name: CameraName, timeout_s: float = 10.0
    ) -> StableFrameResponse:
        data = self._post(
            f"/api/v1/cameras/{quote(camera_name.value)}/stable-frame",
            {"timeout_s": timeout_s},
            timeout_s + 1.0,
        )
        return StableFrameResponse(
            frame_id=_as_int(data, "frame_id"),
            camera_name=_as_string(data, "camera_name"),
            timestamp_ms=_as_float(data, "timestamp_ms"),
            error=None,
        )

    def detect_ball(self, request: BallPoseDetectionRequest) -> BallPoseDetectionResponse:
        payload = {
            "request_id": request.request_id,
            "camera_name": request.camera_name.value,
            "frame_id": request.frame_id,
            "enable_debug": request.enable_debug,
            "priors": [_ball_prior_payload(prior) for prior in request.priors],
        }
        data = self._post("/api/v1/detections/ball", payload, self._timeout_s)
        return _parse_ball_response(data)

    def detect_charuco(
        self, request: CharucoDetectionRequest
    ) -> CharucoDetectionResponse:
        payload = {
            "camera_name": request.camera_name.value,
            "dictionary_name": request.dictionary_name,
            "squares_x": request.squares_x,
            "squares_y": request.squares_y,
            "square_length_mm": request.square_length_mm,
            "marker_length_mm": request.marker_length_mm,
            "min_charuco_corners": request.min_charuco_corners,
            "max_frames": request.max_frames,
            "stable_timeout_s": request.stable_timeout_s,
            "enable_debug": request.enable_debug,
        }
        data = self._post("/api/v1/detections/charuco", payload, self._timeout_s)
        overlay = _decode_json_value(data.get("overlay_bgr"))
        return CharucoDetectionResponse(
            status=_as_string(data, "status"),
            camera_name=CameraName(_as_string(data, "camera_name")),
            t_cam_board_mm=_matrix4(data.get("t_cam_board_mm")),
            error_px=_as_float(data, "error_px"),
            marker_num=_as_int(data, "marker_num"),
            charuco_num=_as_int(data, "charuco_num"),
            overlay_bgr=_as_array(overlay, "overlay_bgr"),
        )

    def subscribe_camera_color_frames(
        self, camera_name: CameraName
    ) -> Iterator[CameraColorFramePacket]:
        for fields in self._subscribe(camera_name, "color"):
            yield CameraColorFramePacket(
                frame_id=_as_int(fields, "frame_id"),
                camera_name=_as_string(fields, "camera_name"),
                timestamp_ms=_as_float(fields, "timestamp_ms"),
                color_bgr=_as_array(fields.get("color_bgr"), "color_bgr"),
                fx=_as_float(fields, "fx"),
                fy=_as_float(fields, "fy"),
                cx=_as_float(fields, "cx"),
                cy=_as_float(fields, "cy"),
                distortion=tuple(
                    _as_float_value(item, "distortion")
                    for item in _as_list(fields, "distortion")
                ),
            )

    def subscribe_camera_depth_frames(
        self, camera_name: CameraName
    ) -> Iterator[CameraDepthFramePacket]:
        for fields in self._subscribe(camera_name, "depth"):
            yield CameraDepthFramePacket(
                frame_id=_as_int(fields, "frame_id"),
                camera_name=_as_string(fields, "camera_name"),
                timestamp_ms=_as_float(fields, "timestamp_ms"),
                depth_mm=_as_array(fields.get("depth_mm"), "depth_mm"),
                fx=_as_float(fields, "fx"),
                fy=_as_float(fields, "fy"),
                cx=_as_float(fields, "cx"),
                cy=_as_float(fields, "cy"),
                distortion=tuple(
                    _as_float_value(item, "distortion")
                    for item in _as_list(fields, "distortion")
                ),
            )

    def subscribe_camera_frames(self, camera_name: CameraName) -> Iterator[CameraFramePacket]:
        for fields in self._subscribe(camera_name, "rgbd"):
            yield CameraFramePacket(
                frame_id=_as_int(fields, "frame_id"),
                camera_name=_as_string(fields, "camera_name"),
                timestamp_ms=_as_float(fields, "timestamp_ms"),
                color_bgr=_as_array(fields.get("color_bgr"), "color_bgr"),
                depth_mm=_as_array(fields.get("depth_mm"), "depth_mm"),
                fx=_as_float(fields, "fx"),
                fy=_as_float(fields, "fy"),
                cx=_as_float(fields, "cx"),
                cy=_as_float(fields, "cy"),
                distortion=tuple(
                    _as_float_value(item, "distortion")
                    for item in _as_list(fields, "distortion")
                ),
            )

    def _get(self, path: str, timeout_s: float) -> dict[str, object]:
        return self._request("GET", path, None, timeout_s)

    def _post(
        self, path: str, payload: dict[str, object], timeout_s: float
    ) -> dict[str, object]:
        return self._request("POST", path, payload, timeout_s)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        timeout_s: float,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self._build_http_url(path),
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with _DIRECT_OPENER.open(
                request,
                timeout=max(0.1, float(timeout_s)),
            ) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            try:
                error_envelope = json.loads(response_body)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"CameraPipeline HTTP {exc.code}: "
                    f"{response_body or exc.reason}"
                ) from exc
            if isinstance(error_envelope, dict):
                error = error_envelope.get("error")
                if isinstance(error, dict):
                    code = str(error.get("code", "http_error"))
                    message = str(error.get("message", exc.reason))
                    raise RuntimeError(
                        f"CameraPipeline HTTP {exc.code} [{code}]: {message}"
                    ) from exc
            raise RuntimeError(
                f"CameraPipeline HTTP {exc.code}: "
                f"{response_body or exc.reason}"
            ) from exc
        if not isinstance(envelope, dict) or envelope.get("ok") is not True:
            raise RuntimeError(f"CameraPipeline HTTP response failed: {envelope!r}")
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("CameraPipeline HTTP response data is not an object")
        return {str(key): _decode_json_value(value) for key, value in data.items()}

    def _subscribe(self, camera_name: CameraName, frame_kind: str) -> Iterator[dict[str, object]]:
        host, port, secure = _websocket_target(self._websocket_url)
        path = (
            self._build_prefixed_path(
                self._websocket_prefix,
                f"/api/v1/ws/cameras/{quote(camera_name.value)}/{frame_kind}",
                internal_prefix="/api/v1/ws",
            )
        )
        raw_socket = socket.create_connection(
            (host, port),
            timeout=self._stream_timeout_s,
        )
        socket_obj: socket.socket = raw_socket
        try:
            if secure:
                socket_obj = ssl.create_default_context().wrap_socket(
                    raw_socket,
                    server_hostname=host,
                )
            _websocket_handshake(socket_obj, host, port, path)
            while True:
                yield _read_stream_packet(socket_obj)
        finally:
            socket_obj.close()
            if socket_obj is not raw_socket:
                raw_socket.close()

    def _build_http_url(self, path: str) -> str:
        """拼接直连或 Gateway HTTP 前缀，避免重复拼接 `/api/v1`。"""

        return self._base_url + self._build_prefixed_path(self._api_prefix, path)

    @staticmethod
    def _build_prefixed_path(
        prefix: str,
        path: str,
        *,
        internal_prefix: str = "/api/v1",
    ) -> str:
        """把完整服务前缀与客户端内部 API 路径拼接为一个路径。"""

        if prefix and path.startswith(internal_prefix + "/"):
            path = path.removeprefix(internal_prefix)
        return prefix + path


def _normalize_prefix(value: str) -> str:
    """规范统一入口的服务 URL 前缀。"""

    normalized = value.strip().strip("/")
    return f"/{normalized}" if normalized else ""


def _parse_camera_status(data: dict[str, object]) -> CameraStatusResponse:
    return CameraStatusResponse(
        service_version=_as_string(data, "service_version"),
        camera_name=_as_string(data, "camera_name"),
        camera_id=_as_string(data, "camera_id"),
        camera_model=_as_string(data, "camera_model"),
        width=_as_int(data, "width"),
        height=_as_int(data, "height"),
        color_enabled=_as_bool(data, "color_enabled"),
        depth_enabled=_as_bool(data, "depth_enabled"),
        online=_as_bool(data, "online"),
        error=None,
    )


def _ball_prior_payload(prior: BallPosePriorInfo) -> dict[str, object]:
    return {
        "color_hex": prior.color_hex,
        "diameter_mm": prior.diameter_mm,
        "model_center_mm": list(prior.model_center_mm),
        "hsv_ranges": [list(item) for item in prior.hsv_ranges],
    }


def _parse_ball_response(data: dict[str, object]) -> BallPoseDetectionResponse:
    artifacts = tuple(
        _parse_ball_debug_artifact(item)
        for item in _as_list_value(data.get("debug_artifacts"), "debug_artifacts")
    )
    detections = tuple(
        _parse_ball_detection(item)
        for item in _as_list_value(data.get("detections"), "detections")
    )
    return BallPoseDetectionResponse(
        request_id=_as_int(data, "request_id"),
        frame_id=_as_int(data, "frame_id"),
        camera_name=_as_string(data, "camera_name"),
        timestamp_ms=_as_float(data, "timestamp_ms"),
        elapsed_ms=_as_float(data, "elapsed_ms"),
        matched_count=_as_int(data, "matched_count"),
        detections=detections,
        debug_artifacts=artifacts,
    )


def _parse_ball_detection(value: object) -> BallDetectionInfo:
    data = _as_object(value, "detection")
    return BallDetectionInfo(
        color_hex=_as_string(data, "color_hex"),
        detected=_as_bool(data, "detected"),
        center_px=_number_tuple(data, "center_px"),
        center_mm=_number_tuple(data, "center_mm"),
        diameter_mm=_as_float(data, "diameter_mm"),
        radius_px=_as_float(data, "radius_px"),
        center_norm=_number_tuple(data, "center_norm"),
        radius_norm=_as_float(data, "radius_norm"),
        point_count=_as_int(data, "point_count"),
        status=_as_string(data, "status"),
        observed_hsv=_number_tuple(data, "observed_hsv"),
    )


def _parse_ball_debug_artifact(value: object) -> BallPoseDetectionDebugArtifacts:
    data = _as_object(value, "debug_artifact")
    detections = tuple(
        _parse_ball_detection(item)
        for item in _as_list_value(data.get("detections"), "detections")
    )
    return BallPoseDetectionDebugArtifacts(
        color_bgr=_as_array(data.get("color_bgr"), "color_bgr"),
        depth_mm=_as_array(data.get("depth_mm"), "depth_mm"),
        camera_intrinsics=_intrinsics_tuple(data),
        overlay_bgr=_as_array(data.get("overlay_bgr"), "overlay_bgr"),
        detection_overlay_bgr=_as_array(
            data.get("detection_overlay_bgr"), "detection_overlay_bgr"
        ),
        detections=detections,
    )


def _decode_json_value(value: object) -> object:
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if isinstance(value, dict):
        if value.get("encoding") == "base64":
            raw = base64.b64decode(_as_string(value, "data"))
            array = np.frombuffer(raw, dtype=np.dtype(_as_string(value, "dtype")))
            return array.reshape(tuple(_as_int_value(item, "shape") for item in _as_list(value, "shape"))).copy()
        return {str(key): _decode_json_value(item) for key, item in value.items()}
    return value


def _websocket_target(value: str) -> tuple[str, int, bool]:
    parsed = urlsplit(value)
    if parsed.scheme not in {"ws", "wss"} or parsed.hostname is None:
        raise ValueError(f"invalid WebSocket URL: {value}")
    return (
        parsed.hostname,
        parsed.port or (443 if parsed.scheme == "wss" else 80),
        parsed.scheme == "wss",
    )


def _websocket_handshake(socket_obj: socket.socket, host: str, port: int, path: str) -> None:
    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    expected = base64.b64encode(
        hashlib.sha1((key + _WEBSOCKET_GUID).encode("ascii")).digest()
    ).decode("ascii")
    socket_obj.sendall(
        (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
    )
    response = _recv_until(socket_obj, b"\r\n\r\n").decode("iso-8859-1")
    lines = response.split("\r\n")
    if not lines or not lines[0].startswith("HTTP/1.1 101"):
        raise RuntimeError(f"WebSocket handshake failed: {lines[0] if lines else response}")
    headers = {
        line.split(":", 1)[0].lower(): line.split(":", 1)[1].strip()
        for line in lines[1:]
        if ":" in line
    }
    if headers.get("sec-websocket-accept") != expected:
        raise RuntimeError("WebSocket handshake accept mismatch")


def _read_stream_packet(socket_obj: socket.socket) -> dict[str, object]:
    first_two = _recv_exact(socket_obj, 2)
    opcode = first_two[0] & 0x0F
    length = first_two[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(socket_obj, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(socket_obj, 8))[0]
    payload = _recv_exact(socket_obj, length)
    if opcode == 0x8:
        raise ConnectionError("WebSocket server closed the stream")
    if opcode != 0x2:
        return _read_stream_packet(socket_obj)
    if len(payload) < _STREAM_HEADER.size:
        raise RuntimeError("CPWS1 packet header is incomplete")
    magic, metadata_length = _STREAM_HEADER.unpack_from(payload)
    if magic != _STREAM_MAGIC:
        raise RuntimeError(f"unexpected WebSocket packet magic: {magic!r}")
    metadata_start = _STREAM_HEADER.size
    metadata_end = metadata_start + metadata_length
    if metadata_end > len(payload):
        raise RuntimeError("CPWS1 metadata is incomplete")
    metadata = json.loads(payload[metadata_start:metadata_end].decode("utf-8"))
    if not isinstance(metadata, dict) or metadata.get("protocol_version") != 1:
        raise RuntimeError("unsupported CPWS1 metadata")
    raw = payload[metadata_end:]
    fields = metadata.get("fields")
    if not isinstance(fields, dict):
        raise RuntimeError("CPWS1 fields are not an object")
    decoded: dict[str, object] = {}
    for key, value in fields.items():
        if isinstance(value, dict) and value.get("encoding") == "raw":
            offset = _as_int(value, "offset")
            nbytes = _as_int(value, "nbytes")
            array = np.frombuffer(
                raw[offset : offset + nbytes],
                dtype=np.dtype(_as_string(value, "dtype")),
            )
            decoded[str(key)] = array.reshape(
                tuple(_as_int_value(item, "shape") for item in _as_list(value, "shape"))
            ).copy()
        else:
            decoded[str(key)] = value
    return decoded


def _recv_until(socket_obj: socket.socket, marker: bytes) -> bytes:
    data = bytearray()
    while marker not in data:
        # 握手响应后端可能立即发送首帧；逐字节读取避免预读并丢弃首帧数据。
        chunk = socket_obj.recv(1)
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)


def _recv_exact(socket_obj: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = socket_obj.recv(length - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)


def _matrix4(value: object) -> tuple[tuple[float, float, float, float], ...]:
    rows = _as_list_value(value, "t_cam_board_mm")
    if len(rows) != 4:
        raise ValueError("t_cam_board_mm must contain four rows")
    parsed_rows: list[tuple[float, float, float, float]] = []
    for row in rows:
        values = _as_list_value(row, "matrix row")
        if len(values) != 4:
            raise ValueError("each t_cam_board_mm row must contain four values")
        parsed_rows.append(
            tuple(_as_float_value(item, "matrix value") for item in values)  # type: ignore[arg-type]
        )
    return tuple(parsed_rows)


def _number_tuple(data: dict[str, object], key: str) -> tuple[float, ...]:
    return tuple(_as_float_value(item, key) for item in _as_list(data, key))


def _intrinsics_tuple(data: dict[str, object]) -> tuple[float, float, float, float]:
    values = _as_list(data, "camera_intrinsics")
    if len(values) != 4:
        raise ValueError("camera_intrinsics must contain four values")
    return tuple(_as_float_value(item, "camera_intrinsics") for item in values)  # type: ignore[return-value]


def _as_array(value: object, key: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{key} is not an array")
    return value


def _as_object(value: object, key: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} is not an object")
    return {str(name): item for name, item in value.items()}


def _as_list(data: dict[str, object], key: str) -> list[object]:
    return _as_list_value(data.get(key), key)


def _as_list_value(value: object, key: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{key} is not an array")
    return value


def _as_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} is not a string")
    return value


def _as_bool(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} is not a bool")
    return value


def _as_int(data: dict[str, object], key: str) -> int:
    return _as_int_value(data.get(key), key)


def _as_int_value(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} is not an int")
    return value


def _as_float(data: dict[str, object], key: str) -> float:
    return _as_float_value(data.get(key), key)


def _as_float_value(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} is not a number")
    return float(value)
