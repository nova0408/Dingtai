from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..tray_detection.protocol import OrinTrayDetectionInfo

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Quad2 = tuple[Point2, Point2, Point2, Point2]
Matrix3 = tuple[Point3, Point3, Point3]


@dataclass(frozen=True, slots=True)
class GraspPoseInfo:
    """相机坐标系中的完整抓取位姿。"""

    grasp_point_mm: Point3
    pre_grasp_point_mm: Point3
    rotation: Matrix3


@dataclass(frozen=True, slots=True)
class TrayPoseInfo:
    """单托盘开口与位姿结果。"""

    tray_index: int
    tray_bbox_xywh: tuple[int, int, int, int]
    tray_center_uv: Point2
    opening_center_uv: Point2
    opening_quad_uv: Quad2
    pose: GraspPoseInfo


@dataclass(frozen=True, slots=True)
class DebugArtifacts:
    """启用 debug 时一次性构造的完整开口检测调试产物。"""

    color_bgr: np.ndarray
    depth_mm: np.ndarray
    camera_intrinsics: tuple[float, float, float, float]
    overlay_bgr: np.ndarray
    contrast_bgr: np.ndarray
    tray_instance_masks: tuple[np.ndarray, ...]
    selected_tray_mask: np.ndarray
    near_plane_masks: tuple[np.ndarray, ...]
    no_hole_masks: tuple[np.ndarray, ...]
    opening_center_uv: Point2
    opening_quad_uv: Quad2
    opening_bbox_xywh: tuple[int, int, int, int]
    opening_score: float
    grasp_point_mm: Point3
    pre_grasp_point_mm: Point3
    rotation: Matrix3


@dataclass(frozen=True, slots=True)
class OpeningDetectionPipelineRequest:
    """抓取位姿主服务请求。"""

    request_id: int = 0
    camera_name: str = "left_hand_camera"
    frame_id: int = -1
    "请求帧号。正数表示精确缓存帧；非正数表示等待并使用稳定帧。"
    target_tray_index: int = 0
    enable_debug: bool = True


@dataclass(frozen=True, slots=True)
class OpeningDetectionPipelineResponse:
    """成功完成开口检测后的完整响应。"""

    request_id: int
    frame_id: int
    camera_name: str
    timestamp_ms: float
    elapsed_ms: float
    tray_count: int
    tray_results: tuple[OrinTrayDetectionInfo, ...]
    selected_tray_index: int
    selected_result: TrayPoseInfo
    all_tray_results: tuple[TrayPoseInfo, ...]
    debug_artifacts: tuple[DebugArtifacts, ...] = field(default_factory=tuple)
