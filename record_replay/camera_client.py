"""RecordReplay 使用的 CameraPipeline HTTP 协议 DTO 与客户端。"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from enum import Enum
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np


class CameraName(str, Enum):
    """RecordReplay 使用的逻辑相机名称。"""

    HEAD = "head_camera"
    LEFT_ARM = "left_hand_camera"
    RIGHT_ARM = "right_hand_camera"


HsvRange = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class BallPosePriorInfo:
    """三球检测请求中的单球先验。"""

    color_hex: str
    diameter_mm: float
    model_center_mm: tuple[float, float, float]
    hsv_ranges: tuple[HsvRange, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BallPoseDetectionRequest:
    """三球检测请求协议。"""

    request_id: int
    camera_name: CameraName
    frame_id: int = -1
    enable_debug: bool = False
    priors: tuple[BallPosePriorInfo, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BallDetectionInfo:
    """单球检测结果协议。"""

    color_hex: str
    detected: bool
    center_mm: tuple[float, ...]
    status: str


@dataclass(frozen=True, slots=True)
class BallPoseDetectionResponse:
    """三球检测响应协议。"""

    request_id: int
    frame_id: int
    camera_name: str
    matched_count: int
    detections: tuple[BallDetectionInfo, ...]


@dataclass(frozen=True, slots=True)
class CharucoDetectionRequest:
    """ChArUco 检测请求协议。"""

    camera_name: CameraName
    dictionary_name: str
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    min_charuco_corners: int
    max_frames: int
    stable_timeout_s: float


@dataclass(frozen=True, slots=True)
class CharucoDetectionResponse:
    """ChArUco 检测响应协议。"""

    status: str
    t_cam_board_mm: tuple[tuple[float, float, float, float], ...]
    marker_num: int
    charuco_num: int


class CameraPipelineHttpClient:
    """只通过 CameraPipeline HTTP API 请求检测，不导入其 Python 包。"""

    def __init__(self, base_url: str = "http://127.0.0.1:6400", timeout_s: float = 55.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._opener = build_opener(ProxyHandler({}))

    def close(self) -> None:
        """HTTP 客户端无持久连接，保留统一生命周期接口。"""

    def detect_ball(self, request: BallPoseDetectionRequest) -> BallPoseDetectionResponse:
        """请求一次三球检测。"""

        data = self._post(
            "/api/v1/detections/ball",
            {
                "request_id": request.request_id,
                "camera_name": request.camera_name.value,
                "frame_id": request.frame_id,
                "enable_debug": request.enable_debug,
                "priors": [
                    {
                        "color_hex": prior.color_hex,
                        "diameter_mm": prior.diameter_mm,
                        "model_center_mm": list(prior.model_center_mm),
                        "hsv_ranges": [list(item) for item in prior.hsv_ranges],
                    }
                    for prior in request.priors
                ],
            },
        )
        detections = tuple(
            _parse_detection(item)
            for item in _list(data.get("detections"), "detections")
        )
        return BallPoseDetectionResponse(
            request_id=_integer(data, "request_id"),
            frame_id=_integer(data, "frame_id"),
            camera_name=_string(data, "camera_name"),
            matched_count=_integer(data, "matched_count"),
            detections=detections,
        )

    def detect_charuco(self, request: CharucoDetectionRequest) -> CharucoDetectionResponse:
        """请求一次 ChArUco 检测。"""

        data = self._post(
            "/api/v1/detections/charuco",
            {
                "camera_name": request.camera_name.value,
                "dictionary_name": request.dictionary_name,
                "squares_x": request.squares_x,
                "squares_y": request.squares_y,
                "square_length_mm": request.square_length_mm,
                "marker_length_mm": request.marker_length_mm,
                "min_charuco_corners": request.min_charuco_corners,
                "max_frames": request.max_frames,
                "stable_timeout_s": request.stable_timeout_s,
                "enable_debug": False,
            },
        )
        return CharucoDetectionResponse(
            status=_string(data, "status"),
            t_cam_board_mm=_matrix4(data.get("t_cam_board_mm")),
            marker_num=_integer(data, "marker_num"),
            charuco_num=_integer(data, "charuco_num"),
        )

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=max(0.1, self._timeout_s)) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CameraPipeline HTTP {error.code}: {detail}") from error
        if not isinstance(envelope, dict) or envelope.get("ok") is not True:
            raise RuntimeError(f"CameraPipeline HTTP response failed: {envelope!r}")
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("CameraPipeline HTTP response data is not an object")
        return {str(key): _decode_json_value(value) for key, value in data.items()}


def _parse_detection(value: object) -> BallDetectionInfo:
    data = _object(value, "detection")
    return BallDetectionInfo(
        color_hex=_string(data, "color_hex"),
        detected=_boolean(data, "detected"),
        center_mm=_numbers(data, "center_mm"),
        status=_string(data, "status"),
    )


def _decode_json_value(value: object) -> object:
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if isinstance(value, dict):
        if value.get("encoding") == "base64":
            raw = base64.b64decode(_string(value, "data"))
            array = np.frombuffer(raw, dtype=np.dtype(_string(value, "dtype")))
            shape = tuple(
                _integer_value(item, "shape")
                for item in _list(value.get("shape"), "shape")
            )
            return array.reshape(shape).copy()
        return {str(key): _decode_json_value(item) for key, item in value.items()}
    return value


def _object(value: object, key: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} 必须是 object")
    return {str(name): item for name, item in value.items()}


def _list(value: object, key: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{key} 必须是 array")
    return value


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} 必须是 string")
    return value


def _boolean(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} 必须是 bool")
    return value


def _integer(data: dict[str, object], key: str) -> int:
    return _integer_value(data.get(key), key)


def _integer_value(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} 必须是 int")
    return value


def _numbers(data: dict[str, object], key: str) -> tuple[float, ...]:
    value = data.get(key)
    items = _list(value, key)
    values = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{key} 必须是 number array")
        values.append(float(item))
    return tuple(values)


def _matrix4(value: object) -> tuple[tuple[float, float, float, float], ...]:
    rows = _list(value, "t_cam_board_mm")
    if len(rows) != 4:
        raise ValueError("t_cam_board_mm 必须包含四行")
    parsed = []
    for row in rows:
        values = _list(row, "matrix row")
        if len(values) != 4:
            raise ValueError("矩阵每行必须包含四个数值")
        parsed.append(
            tuple(
                float(item)
                if isinstance(item, int | float) and not isinstance(item, bool)
                else (_raise_matrix_value())
                for item in values
            )
        )
    return tuple(parsed)  # type: ignore[return-value]


def _raise_matrix_value() -> float:
    raise ValueError("矩阵必须只包含数值")
