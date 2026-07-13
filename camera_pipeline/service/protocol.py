from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
from ..opening_detection.protocol import (
    OpeningDetectionPipelineRequest,
    OpeningDetectionPipelineResponse,
)
from ..tray_detection.protocol import (
    OrinTrayDetectionRequest,
    OrinTrayDetectionResponse,
)

PROTOCOL_VERSION = 1

CameraPipelineOperation = Literal[
    "camera_summary",
    "camera_intrinsics",
    "camera_status",
    "stable_frame",
    "camera_frame_subscribe",
    "camera_color_frame_subscribe",
    "camera_depth_frame_subscribe",
    "tray_detection",
    "opening_detection",
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
    """相机内参请求。"""

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
    """显式等待稳定帧请求。"""

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


# region 统一服务信封


@dataclass(frozen=True, slots=True)
class CameraPipelineServiceRequest:
    """统一 CameraPipeline 服务请求信封。"""

    operation: CameraPipelineOperation
    protocol_version: int = PROTOCOL_VERSION
    camera_summary: CameraSummaryRequest | None = None
    camera_intrinsics: CameraIntrinsicsRequest | None = None
    camera_status: CameraStatusRequest | None = None
    stable_frame: StableFrameRequest | None = None
    camera_frame_subscribe: CameraFrameSubscribeRequest | None = None
    camera_color_frame_subscribe: CameraColorFrameSubscribeRequest | None = None
    camera_depth_frame_subscribe: CameraDepthFrameSubscribeRequest | None = None
    tray_detection: OrinTrayDetectionRequest | None = None
    opening_detection: OpeningDetectionPipelineRequest | None = None
    ball_pose_detection: BallPoseDetectionRequest | None = None


@dataclass(frozen=True, slots=True)
class CameraPipelineServiceResponse:
    """统一 CameraPipeline 服务响应信封。"""

    operation: CameraPipelineOperation
    protocol_version: int = PROTOCOL_VERSION
    camera_summary: CameraSummaryResponse | None = None
    camera_intrinsics: CameraIntrinsicsResponse | None = None
    camera_status: CameraStatusResponse | None = None
    stable_frame: StableFrameResponse | None = None
    camera_frame_subscribe: CameraFrameSubscribeResponse | None = None
    camera_color_frame_subscribe: CameraColorFrameSubscribeResponse | None = None
    camera_depth_frame_subscribe: CameraDepthFrameSubscribeResponse | None = None
    tray_detection: OrinTrayDetectionResponse | None = None
    opening_detection: OpeningDetectionPipelineResponse | None = None
    ball_pose_detection: BallPoseDetectionResponse | None = None
    error: str | None = None


# endregion
