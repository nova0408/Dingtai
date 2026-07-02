from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .ball_pose_detection.protocol import BallPoseDetectionRequest, BallPoseDetectionResponse
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
class CameraPipelineServiceRequest:
    """统一远端 camera pipeline 服务请求。"""

    operation: Literal["camera_summary", "tray_detection", "opening_detection", "ball_pose_detection"]
    camera_summary: CameraSummaryRequest | None = None
    tray_detection: OrinTrayDetectionRequest | None = None
    opening_detection: OpeningDetectionPipelineRequest | None = None
    ball_pose_detection: BallPoseDetectionRequest | None = None


@dataclass(frozen=True)
class CameraPipelineServiceResponse:
    """统一远端 camera pipeline 服务响应。"""

    operation: Literal["camera_summary", "tray_detection", "opening_detection", "ball_pose_detection"]
    camera_summary: CameraSummaryResponse | None = None
    tray_detection: OrinTrayDetectionResponse | None = None
    opening_detection: OpeningDetectionPipelineResponse | None = None
    ball_pose_detection: BallPoseDetectionResponse | None = None
    error: Optional[str] = None
