from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, slots=True)
class BallPosePriorInfo:
    """单个小球的先验信息。"""

    color_hex: str
    radius_mm: float
    model_center_mm: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BallPoseDetectionRequest:
    """球位姿检测请求。"""

    request_id: int = 0
    camera_name: str = "left_hand_camera"
    frame_id: int = -1
    "请求帧号。正数表示精确缓存帧；非正数表示等待并使用稳定帧。"
    enable_debug: bool = False
    priors: tuple[BallPosePriorInfo, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BallDetectionInfo:
    """单个球的明确检测结果；未检测到时坐标元组为空。"""

    color_hex: str
    detected: bool
    center_px: tuple[float, ...]
    center_mm: tuple[float, ...]
    radius_mm: float
    radius_px: float
    center_norm: tuple[float, ...]
    radius_norm: float
    point_count: int
    status: str


@dataclass(frozen=True, slots=True)
class BallPoseDetectionDebugArtifacts:
    """球位姿检测调试信息。"""

    color_bgr: np.ndarray
    depth_mm: np.ndarray
    camera_intrinsics: tuple[float, float, float, float]
    overlay_bgr: np.ndarray
    detection_overlay_bgr: np.ndarray
    detections: tuple[BallDetectionInfo, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BallPoseDetectionResponse:
    """球位姿检测响应。"""

    request_id: int
    frame_id: int
    camera_name: str
    timestamp_ms: float
    elapsed_ms: float = 0.0
    matched_count: int = 0
    detections: tuple[BallDetectionInfo, ...] = field(default_factory=tuple)
    debug_artifacts: tuple[BallPoseDetectionDebugArtifacts, ...] = field(
        default_factory=tuple
    )
