from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Optional

import cv2
import zmq
import numpy as np

from ..camera_stream import CameraStreamRuntimeConfig
from ..pipeline_context import PipelineContext, PipelineContextConfig
from ..ports import (
    DEFAULT_CAMERA_HOST,
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_NAME,
    DEFAULT_CONTROL_PORT,
    DEFAULT_STREAM_PORT,
    BALL_POSE_DETECTION_BIND_ADDR,
)

from .detector import BallPoseDetector
from .protocol import (
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPoseDetectionServiceEndpointConfig,
)
from .priors import BallPosePrior
from .transport import BallPoseDetectionRpcServer, ZmqSocketOptions


LOGGER = logging.getLogger("..ball_pose_detection.service")


class BallPoseDetectionService:
    """球位姿检测独立服务。"""

    def __init__(
        self,
        endpoint_config: BallPoseDetectionServiceEndpointConfig,
        frame_runtime_config: CameraStreamRuntimeConfig,
        socket_options: Optional[ZmqSocketOptions] = None,
    ) -> None:
        self._context = PipelineContext(PipelineContextConfig(camera_runtime=frame_runtime_config))
        self._context.start()
        self._server = BallPoseDetectionRpcServer(endpoint_config.request_bind_addr, options=socket_options)
        self._detector = BallPoseDetector()
        self._running = True

    def close(self) -> None:
        self._running = False
        self._server.close()
        self._context.close()

    def run_forever(self) -> None:
        LOGGER.info("ball pose detection rpc service started")
        while self._running:
            try:
                request = self._server.recv_request()
            except zmq.error.Again:
                continue
            try:
                response = self._process_request(request)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("ball pose detection service failed: %s", exc)
                response = BallPoseDetectionResponse(
                    request_id=int(request.request_id),
                    frame_id=-1,
                    camera_name=str(request.camera_name),
                    timestamp_ms=0.0,
                    source_meta={},
                    error="{0}: {1}".format(type(exc).__name__, exc),
                )
            self._server.send_response(response)

    def _process_request(self, request: BallPoseDetectionRequest) -> BallPoseDetectionResponse:
        frame = self._context.resolve_frame(request.frame_id)
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
            frame_id=int(getattr(frame, "frame_id", request.frame_id)),
            camera_name=str(request.camera_name),
            timestamp_ms=float(getattr(frame, "timestamp_ms", 0.0)),
            source_meta=dict(getattr(frame, "source_meta", {})),
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ball pose detection RPC service")
    parser.add_argument("--bind-addr", type=str, default=BALL_POSE_DETECTION_BIND_ADDR)
    parser.add_argument("--host", type=str, default=DEFAULT_CAMERA_HOST)
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT)
    parser.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT)
    parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = BallPoseDetectionService(
        endpoint_config=BallPoseDetectionServiceEndpointConfig(request_bind_addr=str(args.bind_addr)),
        frame_runtime_config=CameraStreamRuntimeConfig(
            host=str(args.host),
            control_port=int(args.control_port),
            stream_port=int(args.stream_port),
            camera_id=str(args.camera_id),
            camera_name=str(args.camera_name),
        ),
        socket_options=ZmqSocketOptions(),
    )
    if not service._context.wait_until_ready(timeout_s=8.0):  # noqa: SLF001
        LOGGER.warning("camera stream not ready within timeout")

    def _handle_signal(signum, _frame) -> None:  # noqa: ANN001
        LOGGER.info("received stop signal %s", signum)
        service.close()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        service.run_forever()
    finally:
        service.close()
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
