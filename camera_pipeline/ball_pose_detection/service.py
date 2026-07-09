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
        debug = None
        if request.enable_debug:
            debug = BallPoseDetectionDebugArtifacts(
                color_bgr=np.asarray(frame.color_bgr, dtype=np.uint8),
                depth_mm=np.asarray(frame.depth_mm, dtype=np.float64),
                camera_intrinsics=(
                    float(frame.fx),
                    float(frame.fy),
                    float(frame.cx),
                    float(frame.cy),
                ),
                overlay_bgr=_build_detection_overlay(frame, result),
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
            matched_count=int(result.matched_count),
            detections=tuple(
                {
                    "color_hex": item.color_hex,
                    "detected": bool(item.detected),
                    "center_px": None if item.center_px is None else [float(v) for v in np.asarray(item.center_px, dtype=np.float64)],
                    "center_mm": None if item.center_mm is None else [float(v) for v in np.asarray(item.center_mm, dtype=np.float64)],
                    "radius_mm": float(_estimate_radius_mm(frame, item)),
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


def _estimate_radius_mm(frame: Any, item: Any) -> float:
    if item.center_mm is None:
        return float(item.radius_mm)
    center_mm = np.asarray(item.center_mm, dtype=np.float64)
    if center_mm.shape != (3,) or not np.all(np.isfinite(center_mm)):
        return float(item.radius_mm)
    if not np.isfinite(item.radius_px) or float(item.radius_px) <= 1e-6:
        return float(item.radius_mm)
    focal = 0.5 * (float(frame.fx) + float(frame.fy))
    if focal <= 1e-6:
        return float(item.radius_mm)
    return float(item.radius_px) * float(center_mm[2]) / focal
