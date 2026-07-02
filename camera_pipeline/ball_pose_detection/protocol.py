from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..ports import BALL_POSE_DETECTION_BIND_ADDR


@dataclass(frozen=True)
class BallPoseDetectionServiceEndpointConfig:
    """球位姿检测服务端点配置。"""

    request_bind_addr: str = BALL_POSE_DETECTION_BIND_ADDR


@dataclass(frozen=True)
class BallPosePriorInfo:
    """单个小球的先验信息。"""

    color_hex: str
    radius_mm: float
    model_center_mm: Tuple[float, float, float]


@dataclass(frozen=True)
class BallPoseDetectionRequest:
    """球位姿检测请求。"""

    request_id: int = 0
    camera_name: str = "left_hand_camera"
    frame_id: int = -1
    enable_debug: bool = True
    priors: Tuple[BallPosePriorInfo, ...] = field(default_factory=tuple)
    reference_relative_transform_mm: Optional[Tuple[Tuple[float, float, float, float], ...]] = None


@dataclass(frozen=True)
class BallPoseDetectionDebugArtifacts:
    """球位姿检测调试信息。"""

    color_bgr: Optional[np.ndarray] = None
    depth_mm: Optional[np.ndarray] = None
    camera_intrinsics: Optional[Tuple[float, float, float, float]] = None
    overlay_bgr: Optional[np.ndarray] = None
    detection_overlay_bgr: Optional[np.ndarray] = None
    detections: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BallPoseDetectionResponse:
    """球位姿检测响应。"""

    request_id: int
    frame_id: int
    camera_name: str
    timestamp_ms: float
    source_meta: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    pose_transform: Optional[Any] = None
    pose_translation_mm: Optional[Any] = None
    pose_rotation: Optional[Any] = None
    residual_mm: Optional[float] = None
    matched_count: int = 0
    detections: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    debug: Optional[BallPoseDetectionDebugArtifacts] = None
    error: Optional[str] = None
