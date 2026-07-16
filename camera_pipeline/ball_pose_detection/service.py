from __future__ import annotations

# pyright: reportMissingImports=false

import cv2
import numpy as np
from loguru import logger
import time

from ..protocol import RgbdFrameProtocol
from .detector import BallPoseDetector
from .priors import BallPosePrior
from .protocol import (
    BallDetectionInfo,
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
from .types import BallObservation, BallPoseDetectionConfig, BallPoseDetectionResult


class BallPoseDetectionService:
    """只负责基于单帧 RGBD 执行球检测与位姿求解。"""

    def __init__(
        self, config: BallPoseDetectionConfig = BallPoseDetectionConfig()
    ) -> None:
        self._detector = BallPoseDetector(config=config)

    def compute(
        self, frame: RgbdFrameProtocol, request: BallPoseDetectionRequest
    ) -> BallPoseDetectionResponse:
        """基于输入帧和请求计算球位姿结果。"""

        started_at = time.perf_counter()
        logger.info(
            "ball pose service compute started request_id={} camera_name={} frame_id={} prior_count={} debug_enabled={}",
            request.request_id,
            request.camera_name,
            frame.frame_id,
            len(request.priors),
            request.enable_debug,
        )
        priors = [
            BallPosePrior(
                color_hex=prior.color_hex,
                radius_mm=prior.radius_mm,
                model_center_mm=np.asarray(prior.model_center_mm, dtype=np.float64),
            )
            for prior in request.priors
        ]
        result = self._detector.detect(frame, priors)
        detections = tuple(
            _build_detection_info(frame, item) for item in result.detections
        )
        debug_artifacts: tuple[BallPoseDetectionDebugArtifacts, ...] = ()
        if request.enable_debug:
            overlay = _build_detection_overlay(frame, result)
            debug_artifacts = (
                BallPoseDetectionDebugArtifacts(
                    color_bgr=frame.color_bgr,
                    depth_mm=frame.depth_mm,
                    camera_intrinsics=(frame.fx, frame.fy, frame.cx, frame.cy),
                    overlay_bgr=overlay,
                    detection_overlay_bgr=overlay,
                    detections=detections,
                ),
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        response = BallPoseDetectionResponse(
            request_id=request.request_id,
            frame_id=frame.frame_id,
            camera_name=request.camera_name,
            timestamp_ms=frame.timestamp_ms,
            elapsed_ms=elapsed_ms,
            matched_count=result.matched_count,
            detections=detections,
            debug_artifacts=debug_artifacts,
        )
        log_method = logger.info
        if result.matched_count < len(priors):
            log_method = logger.warning
        log_method(
            "ball pose service compute completed request_id={} camera_name={} frame_id={} status={} matched_count={}/{} elapsed_ms={:.3f}",
            request.request_id,
            request.camera_name,
            frame.frame_id,
            result.status,
            result.matched_count,
            len(priors),
            elapsed_ms,
        )
        return response


def _build_detection_info(
    frame: RgbdFrameProtocol, item: BallObservation
) -> BallDetectionInfo:
    return BallDetectionInfo(
        color_hex=item.color_hex,
        detected=item.detected,
        center_px=(
            () if item.center_px is None else tuple(float(v) for v in item.center_px)
        ),
        center_mm=(
            () if item.center_mm is None else tuple(float(v) for v in item.center_mm)
        ),
        radius_mm=_estimate_radius_mm(frame, item),
        radius_px=item.radius_px,
        center_norm=(
            ()
            if item.center_norm is None
            else tuple(float(v) for v in item.center_norm)
        ),
        radius_norm=item.radius_norm,
        point_count=item.point_count,
        status=item.status,
    )


def _build_detection_overlay(
    frame: RgbdFrameProtocol, result: BallPoseDetectionResult
) -> np.ndarray:
    overlay = frame.color_bgr.copy()
    for item in result.detections:
        if item.contour is None:
            continue
        base_color = item.debug_bgr
        contour_color = tuple(int(value) for value in base_color.tolist())
        fitted_color = tuple(
            int(value)
            for value in np.clip(base_color.astype(np.int16) * 0.65, 0, 255).tolist()
        )
        cv2.drawContours(overlay, [item.contour], -1, contour_color, 2)
        if item.center_px is None:
            continue
        center = tuple(int(round(value)) for value in item.center_px.tolist())
        cv2.circle(overlay, center, max(4, int(round(item.radius_px))), fitted_color, 2)
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


def _estimate_radius_mm(frame: RgbdFrameProtocol, item: BallObservation) -> float:
    if item.center_mm is None:
        return item.radius_mm
    if item.center_mm.shape != (3,) or not np.all(np.isfinite(item.center_mm)):
        return item.radius_mm
    if not np.isfinite(item.radius_px) or item.radius_px <= 1e-6:
        return item.radius_mm
    focal = 0.5 * (frame.fx + frame.fy)
    if focal <= 1e-6:
        return item.radius_mm
    return item.radius_px * float(item.center_mm[2]) / focal
