from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
from ..protocol import CameraName
PROTOCOL_VERSION = 10
"CameraPipeline 线协议版本。"

SERVICE_VERSION = "1.10.0"
"CameraPipeline 服务功能版本，必须与 CHANGELOG 当前版本一致。"

CAMERA_STREAM_TOPIC_SEPARATOR = b"\x00"
"相机名与帧协议载荷之间的 topic 分隔字节。"

CameraPipelineOperation = Literal[
    "camera_summary",
    "camera_intrinsics",
    "camera_status",
    "stable_frame",
    "camera_frame_subscribe",
    "camera_color_frame_subscribe",
    "camera_depth_frame_subscribe",
    "detect_ball",
    "detect_charuco",
]


@dataclass(frozen=True, slots=True)
class CharucoDetectionRequest:
    """调用方完整提供板型和检测边界的 ChArUco 请求。"""

    camera_name: CameraName
    "逻辑相机名称。"
    dictionary_name: str
    "ArUco 字典名称；当前支持 DICT_APRILTAG_16H5。"
    squares_x: int
    "Board 横向方格数量。"
    squares_y: int
    "Board 纵向方格数量。"
    square_length_mm: float
    "单个方格边长，单位 mm。"
    marker_length_mm: float
    "单个 marker 边长，单位 mm。"
    min_charuco_corners: int
    "进入 PnP 的最少 ChArUco 角点数量。"
    max_frames: int
    "允许尝试的稳定帧数量。"
    stable_timeout_s: float
    "每次等待稳定帧的超时时间，单位 s。"
    enable_debug: bool = False
    "是否返回最终检测帧的 marker、角点和坐标轴叠加图。"


@dataclass(frozen=True, slots=True)
class CharucoDetectionResponse:
    """CameraPipeline 返回的 Board 到相机坐标系位姿。"""

    status: str
    camera_name: CameraName
    t_cam_board_mm: tuple[tuple[float, float, float, float], ...]
    error_px: float
    marker_num: int
    charuco_num: int
    overlay_bgr: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    "最终检测帧的 BGR 叠加图；未启用 debug 时形状为 `(0, 0, 3)`。"


# region 相机查询协议


@dataclass(frozen=True, slots=True)
class CameraSummaryResponse:
    """相机首帧摘要响应。"""

    frame_id: int
    camera_name: str
    timestamp_ms: float
    color_shape: tuple[int, int, int]
    depth_shape: tuple[int, int]
    fx: float
    fy: float
    cx: float
    cy: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CameraIntrinsicsResponse:
    """相机内参响应。"""

    camera_name: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    width: int
    height: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CameraStatusResponse:
    """相机在线状态响应。"""

    service_version: str
    "远端 CameraPipeline 功能版本。"
    camera_name: str
    camera_id: str
    camera_model: str
    width: int
    height: int
    color_enabled: bool
    depth_enabled: bool
    online: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CameraInventoryResponse:
    """当前可用相机清单响应。"""

    cameras: tuple[CameraStatusResponse, ...]
    """只包含已配置、已连接且已有最新帧的相机。"""


@dataclass(frozen=True, slots=True)
class StableFrameResponse:
    """稳定时间窗中点帧响应。"""

    frame_id: int
    camera_name: str
    timestamp_ms: float
    error: str | None = None


# endregion


# region 帧订阅协议


@dataclass(frozen=True, slots=True)
class CameraFrameSubscribeResponse:
    """完整 RGBD 帧订阅响应。"""

    stream_addr: str
    camera_name: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CameraColorFrameSubscribeResponse:
    """彩色帧订阅响应。"""

    stream_addr: str
    camera_name: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CameraDepthFrameSubscribeResponse:
    """深度帧订阅响应。"""

    stream_addr: str
    camera_name: str
    error: str | None = None


# endregion


# region 帧 topic


def build_camera_stream_topic(camera_name: str) -> bytes:
    """构造帧发布与订阅共用的相机 topic 前缀。"""

    if not camera_name:
        raise ValueError("camera_name must not be empty")
    return camera_name.encode("utf-8") + CAMERA_STREAM_TOPIC_SEPARATOR


# endregion


# region 统一服务信封


@dataclass(frozen=True, slots=True)
class CameraPipelineServiceRequest:
    """统一 CameraPipeline 服务请求信封。"""

    operation: CameraPipelineOperation
    "当前请求操作名；只有对应 operation 的 payload 应该被设置。"
    camera_name: CameraName
    "请求使用的逻辑相机名称。"
    protocol_version: int = PROTOCOL_VERSION
    "统一协议版本号。"
    timeout_s: float = 10.0
    "查询和稳定帧等待超时，单位 s。"
    detect_ball: BallPoseDetectionRequest | None = None
    "三球检测协议；operation 不是 `detect_ball` 时必须为空。"
    detect_charuco: CharucoDetectionRequest | None = None
    "Board 检测请求；采集、稳定帧和检测全部由 CameraPipeline 处理。"


@dataclass(frozen=True, slots=True)
class CameraPipelineServiceResponse:
    """统一 CameraPipeline 服务响应信封。"""

    operation: CameraPipelineOperation
    "与请求一致的操作名。"
    protocol_version: int = PROTOCOL_VERSION
    "统一协议版本号。"
    camera_summary: CameraSummaryResponse | None = None
    camera_intrinsics: CameraIntrinsicsResponse | None = None
    camera_status: CameraStatusResponse | None = None
    stable_frame: StableFrameResponse | None = None
    camera_frame_subscribe: CameraFrameSubscribeResponse | None = None
    camera_color_frame_subscribe: CameraColorFrameSubscribeResponse | None = None
    camera_depth_frame_subscribe: CameraDepthFrameSubscribeResponse | None = None
    detect_ball: BallPoseDetectionResponse | None = None
    "球位姿成功响应 payload；失败时为空并由 error 表达失败。"
    detect_charuco: CharucoDetectionResponse | None = None
    "Board 检测成功或空结果响应。"
    error: str | None = None
    "服务级错误文本；成功响应为空，客户端会将非空错误转换为 RuntimeError。"


# endregion
