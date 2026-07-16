from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
PROTOCOL_VERSION = 4

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
    "ball_pose_detection",
]


# region 相机查询协议


@dataclass(frozen=True, slots=True)
class CameraSummaryRequest:
    """相机首帧摘要请求。"""

    timeout_s: float = 10.0


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
class CameraIntrinsicsRequest:
    """指定安装位的相机内参请求。"""

    camera_name: str = "left_hand_camera"
    timeout_s: float = 10.0


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
class CameraStatusRequest:
    """相机在线状态请求。"""

    timeout_s: float = 10.0


@dataclass(frozen=True, slots=True)
class CameraStatusResponse:
    """相机在线状态响应。"""

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
class StableFrameRequest:
    """指定安装位的稳定帧等待请求。"""

    camera_name: str = "left_hand_camera"
    timeout_s: float = 10.0


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
class CameraFrameSubscribeRequest:
    """完整 RGBD 帧订阅请求。"""

    camera_name: str = "left_hand_camera"


@dataclass(frozen=True, slots=True)
class CameraFrameSubscribeResponse:
    """完整 RGBD 帧订阅响应。"""

    stream_addr: str
    camera_name: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CameraColorFrameSubscribeRequest:
    """彩色帧订阅请求。"""

    camera_name: str = "left_hand_camera"


@dataclass(frozen=True, slots=True)
class CameraColorFrameSubscribeResponse:
    """彩色帧订阅响应。"""

    stream_addr: str
    camera_name: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CameraDepthFrameSubscribeRequest:
    """深度帧订阅请求。"""

    camera_name: str = "left_hand_camera"


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
    protocol_version: int = PROTOCOL_VERSION
    "统一协议版本号。"
    camera_summary: CameraSummaryRequest | None = None
    camera_intrinsics: CameraIntrinsicsRequest | None = None
    camera_status: CameraStatusRequest | None = None
    stable_frame: StableFrameRequest | None = None
    camera_frame_subscribe: CameraFrameSubscribeRequest | None = None
    camera_color_frame_subscribe: CameraColorFrameSubscribeRequest | None = None
    camera_depth_frame_subscribe: CameraDepthFrameSubscribeRequest | None = None
    ball_pose_detection: BallPoseDetectionRequest | None = None
    "球位姿请求 payload；operation 不是 `ball_pose_detection` 时必须为空。"


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
    ball_pose_detection: BallPoseDetectionResponse | None = None
    "球位姿成功响应 payload；失败时为空并由 error 表达失败。"
    error: str | None = None
    "服务级错误文本；成功响应为空，客户端会将非空错误转换为 RuntimeError。"


# endregion
