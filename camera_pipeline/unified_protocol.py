from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .ball_pose_detection.protocol import BallPoseDetectionRequest, BallPoseDetectionResponse
from .camera_stream import CameraFramePacket
from .opening_detection.protocol import OpeningDetectionPipelineRequest, OpeningDetectionPipelineResponse
from .tray_detection.protocol import OrinTrayDetectionRequest, OrinTrayDetectionResponse


@dataclass(frozen=True)
class CameraSummaryRequest:
    """统一相机服务首帧摘要请求。"""

    timeout_s: float = 10.0


@dataclass(frozen=True)
class CameraSummaryResponse:
    """统一相机服务首帧摘要响应。"""

    frame_id: int
    camera_name: str
    timestamp_ms: float
    color_shape: tuple[int, int, int]
    depth_shape: tuple[int, int]
    fx: float
    fy: float
    cx: float
    cy: float
    source_meta: dict[str, str]
    error: Optional[str] = None


@dataclass(frozen=True)
class CameraIntrinsicsRequest:
    """单独获取相机内参请求。"""

    timeout_s: float = 10.0


@dataclass(frozen=True)
class CameraIntrinsicsResponse:
    """单独获取相机内参响应。"""

    camera_name: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    width: int
    height: int
    error: Optional[str] = None


@dataclass(frozen=True)
class CameraStatusRequest:
    """相机状态查询请求。"""

    timeout_s: float = 10.0


@dataclass(frozen=True)
class CameraStatusResponse:
    """相机状态查询响应。"""

    camera_name: str
    camera_id: str
    camera_model: str
    width: int
    height: int
    color_enabled: bool
    depth_enabled: bool
    online: bool
    source_meta: dict[str, str]
    error: Optional[str] = None


@dataclass(frozen=True)
class CameraFrameSubscribeRequest:
    """订阅原始帧请求。"""

    camera_name: str = "left_hand_camera"


@dataclass(frozen=True)
class CameraFrameSubscribeResponse:
    """订阅原始帧响应。"""

    stream_addr: str
    camera_name: str
    error: Optional[str] = None


@dataclass(frozen=True)
class CameraPipelineServiceRequest:
    """统一远端 camera pipeline 服务请求。"""

    operation: Literal["camera_summary", "camera_intrinsics", "camera_status", "camera_frame_subscribe", "tray_detection", "opening_detection", "ball_pose_detection"]
    camera_summary: CameraSummaryRequest | None = None
    camera_intrinsics: CameraIntrinsicsRequest | None = None
    camera_status: CameraStatusRequest | None = None
    camera_frame_subscribe: CameraFrameSubscribeRequest | None = None
    tray_detection: OrinTrayDetectionRequest | None = None
    opening_detection: OpeningDetectionPipelineRequest | None = None
    ball_pose_detection: BallPoseDetectionRequest | None = None


@dataclass(frozen=True)
class CameraPipelineServiceResponse:
    """统一远端 camera pipeline 服务响应。"""

    operation: Literal["camera_summary", "camera_intrinsics", "camera_status", "camera_frame_subscribe", "tray_detection", "opening_detection", "ball_pose_detection"]
    camera_summary: CameraSummaryResponse | None = None
    camera_intrinsics: CameraIntrinsicsResponse | None = None
    camera_status: CameraStatusResponse | None = None
    camera_frame_subscribe: CameraFrameSubscribeResponse | None = None
    tray_detection: OrinTrayDetectionResponse | None = None
    opening_detection: OpeningDetectionPipelineResponse | None = None
    ball_pose_detection: BallPoseDetectionResponse | None = None
    error: Optional[str] = None
