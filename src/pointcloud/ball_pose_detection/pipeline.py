from __future__ import annotations

import cv2
import numpy as np

from .detector import BallPoseDetector
from .priors import BallPosePrior
from .types import BallPoseDetectionConfig, BallPoseDetectionResult, BallPoseFrame


class BallPoseDetectionPipeline:
    """多球检测服务编排层。"""

    def __init__(self, config: BallPoseDetectionConfig | None = None) -> None:
        self._detector = BallPoseDetector(config=config)

    def detect(self, frame: BallPoseFrame, priors: list[BallPosePrior]) -> BallPoseDetectionResult:
        t0 = cv2.getTickCount()
        detections = self._detector.detect(frame, priors)
        detect_cost = self._elapsed_ms(t0)
        t1 = cv2.getTickCount()
        fit = self._detector.estimate_pose(detections, priors)
        pose_cost = self._elapsed_ms(t1)
        pose_transform = None
        pose_translation_mm = None
        pose_rotation = None
        residual_mm = None
        status = "pose_unavailable"
        if fit is not None:
            pose_rotation = fit.rotation
            pose_translation_mm = fit.translation_mm
            pose_transform = np.eye(4, dtype=np.float64)
            pose_transform[:3, :3] = pose_rotation
            pose_transform[:3, 3] = pose_translation_mm
            residual_mm = fit.residual_mm
            status = "pose_estimated"
        debug_colors = {prior.color_hex: np.asarray([0, 0, 0], dtype=np.uint8).copy() for prior in priors}
        debug_radii = {prior.color_hex: float(prior.radius_mm) for prior in priors}
        debug_positions = {
            item.color_hex: np.asarray(item.center_mm, dtype=np.float64).copy()
            for item in detections
            if item.center_mm is not None
        }
        debug_model_positions = {prior.color_hex: np.asarray(prior.model_center_mm, dtype=np.float64).copy() for prior in priors}
        return BallPoseDetectionResult(
            detections=detections,
            pose_translation_mm=pose_translation_mm,
            pose_rotation=pose_rotation,
            pose_transform=pose_transform,
            residual_mm=residual_mm,
            matched_count=sum(1 for item in detections if item.detected and item.center_mm is not None),
            debug_ball_colors_bgr=debug_colors,
            debug_ball_radii_mm=debug_radii,
            debug_ball_positions_mm=debug_positions,
            debug_ball_model_positions_mm=debug_model_positions,
            status=status,
            timings_ms={"detect_balls": detect_cost, "estimate_pose": pose_cost},
        )

    @staticmethod
    def _elapsed_ms(ticks: int) -> float:
        return float((cv2.getTickCount() - ticks) / cv2.getTickFrequency() * 1000.0)
