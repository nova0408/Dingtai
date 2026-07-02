from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .protocol import (
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPosePriorInfo,
)


def encode_request(packet: BallPoseDetectionRequest) -> List[bytes]:
    meta = {
        "request_id": int(packet.request_id),
        "camera_name": str(packet.camera_name),
        "frame_id": int(packet.frame_id),
        "enable_debug": bool(packet.enable_debug),
        "priors": [_encode_prior(item) for item in packet.priors],
        "reference_relative_transform_mm": packet.reference_relative_transform_mm,
    }
    return [b"ball_pose_detection_request", json.dumps(meta, ensure_ascii=False).encode("utf-8")]


def decode_request(parts: List[bytes]) -> BallPoseDetectionRequest:
    if len(parts) != 2:
        raise RuntimeError("invalid ball pose detection request multipart count")
    topic, meta_bytes = parts
    if topic != b"ball_pose_detection_request":
        raise RuntimeError("unexpected ball pose detection request topic")
    meta = json.loads(meta_bytes.decode("utf-8"))
    return BallPoseDetectionRequest(
        request_id=int(meta.get("request_id", 0)),
        camera_name=str(meta.get("camera_name", "left_hand_camera")),
        frame_id=int(meta.get("frame_id", -1)),
        enable_debug=bool(meta.get("enable_debug", False)),
        priors=tuple(_decode_prior(item) for item in meta.get("priors", [])),
        reference_relative_transform_mm=None
        if meta.get("reference_relative_transform_mm") is None
        else tuple(
            (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
            )
            for row in meta["reference_relative_transform_mm"]
        ),
    )


def encode_response(packet: BallPoseDetectionResponse) -> List[bytes]:
    meta: Dict[str, Any] = {
        "request_id": int(packet.request_id),
        "frame_id": int(packet.frame_id),
        "camera_name": str(packet.camera_name),
        "timestamp_ms": float(packet.timestamp_ms),
        "source_meta": dict(packet.source_meta),
        "elapsed_ms": float(packet.elapsed_ms),
        "pose_transform": packet.pose_transform,
        "pose_translation_mm": packet.pose_translation_mm,
        "pose_rotation": packet.pose_rotation,
        "residual_mm": packet.residual_mm,
        "matched_count": int(packet.matched_count),
        "detections": list(packet.detections),
        "error": packet.error,
    }
    debug = packet.debug
    color_bytes = _encode_jpeg(None if debug is None else debug.color_bgr)
    depth_bytes = _encode_png_depth(None if debug is None else debug.depth_mm)
    overlay_bytes = _encode_jpeg(None if debug is None else debug.overlay_bgr)
    detection_overlay_bytes = _encode_jpeg(None if debug is None else debug.detection_overlay_bgr)
    meta["debug_camera_intrinsics"] = None if debug is None or debug.camera_intrinsics is None else list(debug.camera_intrinsics)
    meta["debug_detections"] = [] if debug is None else list(debug.detections)
    return [
        b"ball_pose_detection_response",
        json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        color_bytes,
        depth_bytes,
        overlay_bytes,
        detection_overlay_bytes,
    ]


def decode_response(parts: List[bytes]) -> BallPoseDetectionResponse:
    if len(parts) != 6:
        raise RuntimeError("invalid ball pose detection response multipart count")
    topic, meta_bytes, color_bytes, depth_bytes, overlay_bytes, detection_overlay_bytes = parts
    if topic != b"ball_pose_detection_response":
        raise RuntimeError("unexpected ball pose detection response topic")
    meta = json.loads(meta_bytes.decode("utf-8"))
    debug = None
    if len(color_bytes) > 0 or len(depth_bytes) > 0 or len(overlay_bytes) > 0 or len(detection_overlay_bytes) > 0:
        debug = BallPoseDetectionDebugArtifacts(
            color_bgr=_decode_jpeg(color_bytes),
            depth_mm=_decode_png_depth(depth_bytes),
            camera_intrinsics=None if meta.get("debug_camera_intrinsics") is None else _decode_point4(meta["debug_camera_intrinsics"]),
            overlay_bgr=_decode_jpeg(overlay_bytes),
            detection_overlay_bgr=_decode_jpeg(detection_overlay_bytes),
            detections=tuple(meta.get("debug_detections", [])),
        )
    pose_transform = None if meta.get("pose_transform") is None else tuple(tuple(float(v) for v in row) for row in meta["pose_transform"])
    pose_translation_mm = None if meta.get("pose_translation_mm") is None else _decode_point3(meta["pose_translation_mm"])
    pose_rotation = None if meta.get("pose_rotation") is None else tuple(tuple(float(v) for v in row) for row in meta["pose_rotation"])
    return BallPoseDetectionResponse(
        request_id=int(meta.get("request_id", 0)),
        frame_id=int(meta["frame_id"]),
        camera_name=str(meta["camera_name"]),
        timestamp_ms=float(meta["timestamp_ms"]),
        source_meta=dict(meta.get("source_meta", {})),
        elapsed_ms=float(meta["elapsed_ms"]),
        pose_transform=pose_transform,
        pose_translation_mm=pose_translation_mm,
        pose_rotation=pose_rotation,
        residual_mm=meta.get("residual_mm"),
        matched_count=int(meta.get("matched_count", 0)),
        detections=tuple(meta.get("detections", [])),
        debug=debug,
        error=meta.get("error"),
    )


def _encode_prior(info: BallPosePriorInfo) -> Dict[str, Any]:
    return {
        "color_hex": str(info.color_hex),
        "radius_mm": float(info.radius_mm),
        "model_center_mm": list(info.model_center_mm),
    }


def _decode_prior(raw: Dict[str, Any]) -> BallPosePriorInfo:
    return BallPosePriorInfo(
        color_hex=str(raw["color_hex"]),
        radius_mm=float(raw["radius_mm"]),
        model_center_mm=(float(raw["model_center_mm"][0]), float(raw["model_center_mm"][1]), float(raw["model_center_mm"][2])),
    )


def _encode_jpeg(image_bgr: Any) -> bytes:
    if image_bgr is None:
        return b""
    ok, encoded = cv2.imencode(".jpg", np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode ball pose jpeg")
    return encoded.tobytes()


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    if len(data) == 0:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode ball pose jpeg")
    return np.asarray(image, dtype=np.uint8)


def _encode_png_depth(depth_mm: Any) -> bytes:
    if depth_mm is None:
        return b""
    arr = np.asarray(depth_mm, dtype=np.uint16)
    ok, encoded = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("failed to encode ball pose depth png")
    return bytes(encoded.tobytes())


def _decode_png_depth(data: bytes) -> Optional[np.ndarray]:
    if len(data) == 0:
        return None
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("failed to decode ball pose depth png")
    return np.asarray(arr, dtype=np.uint16)


def _decode_point3(values: Any) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _decode_point4(values: Any) -> tuple[float, float, float, float]:
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])
