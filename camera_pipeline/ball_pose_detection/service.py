from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
from .detector import BallPoseDetector
from .types import BallPoseDetectionConfig
from .protocol import (
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
from .priors import BallPosePrior


class BallPoseDetectionService:
    """球位姿检测纯计算执行器包装。

    职责边界：
    - 只接收单帧 RGBD 和球位姿请求。
    - 不负责相机流、PipelineContext、RPC 监听或请求轮询。
    - 只负责球检测与位姿求解。

    设计思想：
    - 保持算法对象与 IO 编排分离。
    - 让上层决定 frame 与 request 的来源，子模块只处理输入。

    生命周期：
    - 不持有硬件资源。
    - 可跨线程复用，但默认仅作为单次请求处理器使用。

    继承关系：
    - 不继承业务基类。
    """

    def __init__(self, config: Optional[BallPoseDetectionConfig] = None) -> None:
        self._detector = BallPoseDetector(config=config)

    def compute(self, frame: Any, request: BallPoseDetectionRequest) -> BallPoseDetectionResponse:
        """基于输入帧和请求计算球位姿结果。"""

        priors = [
            BallPosePrior(
                color_hex=str(prior.color_hex),
                radius_mm=float(prior.radius_mm),
                model_center_mm=np.asarray(prior.model_center_mm, dtype=np.float64),
            )
            for prior in request.priors
        ]
        result = self._detector.detect(frame, priors)
        pose_transform = None
        pose_rotation = None
        pose_translation_mm = None
        if result.pose_transform is not None:
            pose_transform_matrix = np.asarray(result.pose_transform, dtype=np.float64)
            pose_transform_matrix = _apply_reference_relative_transform(
                pose_transform_matrix,
                request.reference_relative_transform_mm,
            )
            pose_transform = _matrix4_to_tuple(pose_transform_matrix)
            pose_rotation = _matrix3_to_tuple(pose_transform_matrix[:3, :3])
            pose_translation_mm = _vector3_to_tuple(pose_transform_matrix[:3, 3])
        debug = BallPoseDetectionDebugArtifacts(
            color_bgr=np.asarray(frame.color_bgr, dtype=np.uint8),
            depth_mm=np.asarray(frame.depth_mm, dtype=np.float64),
            camera_intrinsics=(
                float(frame.fx),
                float(frame.fy),
                float(frame.cx),
                float(frame.cy),
            ),
            overlay_bgr=_build_overlay(frame, result, pose_transform),
            detection_overlay_bgr=_build_detection_overlay(frame, result),
            detections=tuple(
                {
                    "color_hex": item.color_hex,
                    "detected": bool(item.detected),
                    "center_px": None if item.center_px is None else [float(v) for v in np.asarray(item.center_px, dtype=np.float64)],
                    "center_mm": None if item.center_mm is None else [float(v) for v in np.asarray(item.center_mm, dtype=np.float64)],
                    "radius_mm": float(item.radius_mm),
                    "radius_px": float(item.radius_px),
                    "center_norm": None if item.center_norm is None else [float(v) for v in np.asarray(item.center_norm, dtype=np.float64)],
                    "radius_norm": float(item.radius_norm),
                    "point_count": int(item.point_count),
                    "status": item.status,
                }
                for item in result.detections
            ),
        )
        return BallPoseDetectionResponse(
            request_id=int(request.request_id),
            frame_id=int(frame.frame_id),
            camera_name=str(request.camera_name),
            timestamp_ms=float(frame.timestamp_ms),
            source_meta=dict(frame.source_meta),
            elapsed_ms=float(result.timings_ms.get("detect_balls", 0.0) + result.timings_ms.get("estimate_pose", 0.0)),
            pose_transform=pose_transform,
            pose_translation_mm=pose_translation_mm,
            pose_rotation=pose_rotation,
            residual_mm=result.residual_mm,
            matched_count=int(result.matched_count),
            detections=tuple(
                {
                    "color_hex": item.color_hex,
                    "detected": bool(item.detected),
                    "center_px": None if item.center_px is None else [float(v) for v in np.asarray(item.center_px, dtype=np.float64)],
                    "center_mm": None if item.center_mm is None else [float(v) for v in np.asarray(item.center_mm, dtype=np.float64)],
                    "radius_mm": float(item.radius_mm),
                    "radius_px": float(item.radius_px),
                    "center_norm": None if item.center_norm is None else [float(v) for v in np.asarray(item.center_norm, dtype=np.float64)],
                    "radius_norm": float(item.radius_norm),
                    "point_count": int(item.point_count),
                    "status": item.status,
                }
                for item in result.detections
            ),
            debug=debug,
            error=None,
        )


def _apply_reference_relative_transform(
    pose_transform: np.ndarray,
    reference_relative_transform_mm: tuple[tuple[float, float, float, float], ...] | None,
) -> np.ndarray:
    if reference_relative_transform_mm is None:
        return pose_transform
    relative = np.asarray(reference_relative_transform_mm, dtype=np.float64)
    if relative.shape != (4, 4):
        raise ValueError("invalid reference_relative_transform_mm shape")
    return pose_transform @ relative


def _vector3_to_tuple(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _matrix3_to_tuple(values: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return (
        (float(values[0, 0]), float(values[0, 1]), float(values[0, 2])),
        (float(values[1, 0]), float(values[1, 1]), float(values[1, 2])),
        (float(values[2, 0]), float(values[2, 1]), float(values[2, 2])),
    )


def _matrix4_to_tuple(values: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (float(values[0, 0]), float(values[0, 1]), float(values[0, 2]), float(values[0, 3])),
        (float(values[1, 0]), float(values[1, 1]), float(values[1, 2]), float(values[1, 3])),
        (float(values[2, 0]), float(values[2, 1]), float(values[2, 2]), float(values[2, 3])),
        (float(values[3, 0]), float(values[3, 1]), float(values[3, 2]), float(values[3, 3])),
    )


def _build_detection_overlay(frame, result) -> np.ndarray:
    overlay = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
    for item in result.detections:
        if item.contour is not None:
            base_color = np.asarray(item.debug_bgr, dtype=np.uint8)
            contour_color = tuple(int(value) for value in base_color.tolist())
            fitted_color = tuple(int(value) for value in np.clip(base_color.astype(np.int16) * 0.65, 0, 255).tolist())
            cv2.drawContours(overlay, [np.asarray(item.contour, dtype=np.int32)], -1, contour_color, 2)
            if item.center_px is not None:
                center = tuple(int(round(value)) for value in np.asarray(item.center_px, dtype=np.float64).tolist())
                cv2.circle(overlay, center, max(4, int(round(float(item.radius_px)))), fitted_color, 2)
                cv2.putText(
                    overlay,
                    f"{item.color_hex}:{item.status}",
                    (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    contour_color,
                    2,
                    cv2.LINE_AA,
                )
    return overlay


def _build_overlay(frame, result, pose_transform) -> np.ndarray:
    overlay = _build_detection_overlay(frame, result)
    if pose_transform is None:
        return overlay
    pose = np.asarray(pose_transform, dtype=np.float64)
    origin = pose[:3, 3]
    axes = pose[:3, :3]
    points = [origin, origin + axes[:, 0] * 60.0, origin + axes[:, 1] * 60.0, origin + axes[:, 2] * 60.0]
    projected = []
    for point in points:
        z = float(point[2])
        if abs(z) <= 1e-6:
            return overlay
        u = int(round(point[0] * float(frame.fx) / z + float(frame.cx)))
        v = int(round(point[1] * float(frame.fy) / z + float(frame.cy)))
        projected.append((u, v))
    if len(projected) != 4:
        return overlay
    cv2.circle(overlay, projected[0], 7, (255, 255, 255), 2)
    cv2.line(overlay, projected[0], projected[1], (0, 0, 255), 2)
    cv2.line(overlay, projected[0], projected[2], (0, 255, 0), 2)
    cv2.line(overlay, projected[0], projected[3], (255, 0, 0), 2)
    cv2.putText(
        overlay,
        "pose",
        (projected[0][0] + 8, projected[0][1] + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay
