from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..tray_detection.protocol import OrinTrayDetectionInfo
from ..ports import OPENING_DETECTION_BIND_ADDR


@dataclass(frozen=True)
class GraspPoseRequest:
    """单托盘抓取位姿请求。"""

    request_id: int
    frame_id: int
    camera_name: str
    timestamp_ms: float
    source_meta: Dict[str, Any] = field(default_factory=dict)
    target_tray_index: int = 0
    enable_debug: bool = True
    tray_mask: Any = None


@dataclass(frozen=True)
class TrayPoseInfo:
    """单托盘开口与位姿结果。"""

    tray_index: int
    tray_bbox_xywh: Any
    tray_center_uv: Any
    opening_center_uv: Optional[Any] = None
    opening_quad_uv: Optional[Any] = None
    top_quad_uv: Optional[Any] = None
    pose: Optional["GraspPoseInfo"] = None


@dataclass(frozen=True)
class GraspPoseInfo:
    """抓取位姿结果。"""

    grasp_point_mm: Any
    pre_grasp_point_mm: Any
    rotation: Optional[Any] = None
    rpy_deg: Optional[Any] = None


@dataclass(frozen=True)
class DebugArtifacts:
    """开口检测与位姿计算调试信息。"""

    color_bgr: Any = None
    depth_mm: Any = None
    camera_intrinsics: Optional[Tuple[float, float, float, float]] = None
    overlay_bgr: Any = None
    contrast_bgr: Any = None
    tray_instance_masks: Tuple[Any, ...] = field(default_factory=tuple)
    selected_tray_mask: Any = None
    near_plane_mask: Any = None
    no_hole_mask: Any = None
    opening_center_uv: Optional[Any] = None
    opening_quad_uv: Optional[Any] = None
    opening_bbox_xywh: Optional[Any] = None
    opening_score: Optional[float] = None
    top_quad_uv: Optional[Any] = None
    grasp_point_mm: Optional[Any] = None
    pre_grasp_point_mm: Optional[Any] = None
    rotation: Optional[Any] = None
    rpy_deg: Optional[Any] = None


@dataclass(frozen=True)
class GraspPoseResponse:
    """单托盘抓取位姿响应。"""

    request_id: int
    frame_id: int
    camera_name: str
    timestamp_ms: float
    source_meta: Dict[str, Any] = field(default_factory=dict)
    target_tray_index: int = 0
    selected_result: Optional[TrayPoseInfo] = None
    debug: Optional[DebugArtifacts] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class OpeningDetectionPipelineServiceEndpointConfig:
    """抓取位姿主服务端点配置。"""

    request_bind_addr: str = OPENING_DETECTION_BIND_ADDR


@dataclass(frozen=True)
class OpeningDetectionPipelineRequest:
    """抓取位姿主服务请求。"""

    request_id: int = 0
    camera_name: str = "left_hand_camera"
    frame_id: int = -1
    target_tray_index: int = 0
    enable_debug: bool = True


@dataclass(frozen=True)
class OpeningDetectionPipelineResponse:
    """抓取位姿主服务响应。"""

    request_id: int
    frame_id: int
    camera_name: str
    timestamp_ms: float
    source_meta: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    tray_count: int = 0
    tray_results: Tuple[OrinTrayDetectionInfo, ...] = field(default_factory=tuple)
    selected_tray_index: int = 0
    selected_result: Optional[TrayPoseInfo] = None
    all_tray_results: Tuple[TrayPoseInfo, ...] = field(default_factory=tuple)
    debug: Optional[DebugArtifacts] = None
    error: Optional[str] = None
