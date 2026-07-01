from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..tray_detection.codec import _decode_result, _encode_result, _decode_jpeg, _decode_mask_stack, _encode_jpeg, _encode_mask_stack

from .protocol import DebugArtifacts, GraspPoseInfo, OpeningDetectionPipelineRequest, OpeningDetectionPipelineResponse, TrayPoseInfo


def encode_request(packet: OpeningDetectionPipelineRequest) -> List[bytes]:
    meta = {
        "request_id": int(packet.request_id),
        "camera_name": str(packet.camera_name),
        "frame_id": int(packet.frame_id),
        "target_tray_index": int(packet.target_tray_index),
        "enable_debug": bool(packet.enable_debug),
    }
    return [b"opening_detection_pipeline_request", json.dumps(meta, ensure_ascii=False).encode("utf-8")]


def decode_request(parts: List[bytes]) -> OpeningDetectionPipelineRequest:
    if len(parts) != 2:
        raise RuntimeError("invalid opening detection pipeline request multipart count")
    topic, meta_bytes = parts
    if topic != b"opening_detection_pipeline_request":
        raise RuntimeError("unexpected opening detection pipeline request topic")
    meta = json.loads(meta_bytes.decode("utf-8"))
    return OpeningDetectionPipelineRequest(
        request_id=int(meta.get("request_id", 0)),
        camera_name=str(meta.get("camera_name", "left_hand_camera")),
        frame_id=int(meta.get("frame_id", -1)),
        target_tray_index=int(meta.get("target_tray_index", 0)),
        enable_debug=bool(meta.get("enable_debug", False)),
    )


def encode_response(packet: OpeningDetectionPipelineResponse) -> List[bytes]:
    meta: Dict[str, Any] = {
        "request_id": int(packet.request_id),
        "frame_id": int(packet.frame_id),
        "camera_name": str(packet.camera_name),
        "timestamp_ms": float(packet.timestamp_ms),
        "source_meta": dict(packet.source_meta),
        "elapsed_ms": float(packet.elapsed_ms),
        "tray_count": int(packet.tray_count),
        "tray_results": [_encode_result(item) for item in packet.tray_results],
        "selected_tray_index": int(packet.selected_tray_index),
        "selected_result": None if packet.selected_result is None else _encode_tray_pose_info(packet.selected_result),
        "all_tray_results": [_encode_tray_pose_info(item) for item in packet.all_tray_results],
        "error": packet.error,
    }
    debug = packet.debug
    color_bytes = _encode_jpeg(None if debug is None else debug.color_bgr)
    depth_bytes = _encode_png_depth(None if debug is None else debug.depth_mm)
    intrinsics = None if debug is None else debug.camera_intrinsics
    meta["debug_camera_intrinsics"] = None if intrinsics is None else list(intrinsics)
    meta["debug_opening_center_uv"] = None if debug is None or debug.opening_center_uv is None else list(debug.opening_center_uv)
    meta["debug_opening_quad_uv"] = None if debug is None or debug.opening_quad_uv is None else [list(item) for item in debug.opening_quad_uv]
    meta["debug_opening_bbox_xywh"] = None if debug is None or debug.opening_bbox_xywh is None else list(debug.opening_bbox_xywh)
    meta["debug_opening_score"] = None if debug is None or debug.opening_score is None else float(debug.opening_score)
    meta["debug_top_quad_uv"] = None if debug is None or debug.top_quad_uv is None else [list(item) for item in debug.top_quad_uv]
    meta["debug_grasp_point_mm"] = None if debug is None or debug.grasp_point_mm is None else list(debug.grasp_point_mm)
    meta["debug_pre_grasp_point_mm"] = None if debug is None or debug.pre_grasp_point_mm is None else list(debug.pre_grasp_point_mm)
    meta["debug_rotation"] = None if debug is None or debug.rotation is None else [list(row) for row in debug.rotation]
    meta["debug_rpy_deg"] = None if debug is None or debug.rpy_deg is None else list(debug.rpy_deg)
    overlay_bytes = _encode_jpeg(None if debug is None else debug.overlay_bgr)
    contrast_bytes = _encode_jpeg(None if debug is None else debug.contrast_bgr)
    tray_masks_bytes = _encode_mask_stack(tuple() if debug is None else debug.tray_instance_masks)
    selected_tray_mask_bytes = _encode_png_mask(None if debug is None else debug.selected_tray_mask)
    near_mask_bytes = _encode_png_mask(None if debug is None else debug.near_plane_mask)
    top_mask_bytes = _encode_png_mask(None if debug is None else debug.no_hole_mask)
    return [
        b"opening_detection_pipeline_response",
        json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        color_bytes,
        depth_bytes,
        overlay_bytes,
        contrast_bytes,
        tray_masks_bytes,
        selected_tray_mask_bytes,
        near_mask_bytes,
        top_mask_bytes,
    ]


def decode_response(parts: List[bytes]) -> OpeningDetectionPipelineResponse:
    if len(parts) != 10:
        raise RuntimeError("invalid opening detection pipeline response multipart count")
    topic, meta_bytes, color_bytes, depth_bytes, overlay_bytes, contrast_bytes, tray_masks_bytes, selected_tray_mask_bytes, near_mask_bytes, top_mask_bytes = parts
    if topic != b"opening_detection_pipeline_response":
        raise RuntimeError("unexpected opening detection pipeline response topic")
    meta = json.loads(meta_bytes.decode("utf-8"))
    debug = None
    if len(color_bytes) > 0 or len(depth_bytes) > 0 or len(overlay_bytes) > 0 or len(contrast_bytes) > 0 or len(tray_masks_bytes) > 0:
        raw_intrinsics = meta.get("debug_camera_intrinsics")
        debug = DebugArtifacts(
            color_bgr=_decode_jpeg(color_bytes),
            depth_mm=_decode_png_depth(depth_bytes),
            camera_intrinsics=None if raw_intrinsics is None else _decode_point4(raw_intrinsics),
            overlay_bgr=_decode_jpeg(overlay_bytes),
            contrast_bgr=_decode_jpeg(contrast_bytes),
            tray_instance_masks=_decode_mask_stack(tray_masks_bytes),
            selected_tray_mask=_decode_png_mask(selected_tray_mask_bytes),
            near_plane_mask=_decode_png_mask(near_mask_bytes),
            no_hole_mask=_decode_png_mask(top_mask_bytes),
            opening_center_uv=None if meta.get("debug_opening_center_uv") is None else _decode_point2(meta["debug_opening_center_uv"]),
            opening_quad_uv=None if meta.get("debug_opening_quad_uv") is None else _decode_quad2(meta["debug_opening_quad_uv"]),
            opening_bbox_xywh=None if meta.get("debug_opening_bbox_xywh") is None else _decode_bbox_xywh(meta["debug_opening_bbox_xywh"]),
            opening_score=None if meta.get("debug_opening_score") is None else float(meta["debug_opening_score"]),
            top_quad_uv=None if meta.get("debug_top_quad_uv") is None else _decode_quad2(meta["debug_top_quad_uv"]),
            grasp_point_mm=None if meta.get("debug_grasp_point_mm") is None else _decode_point3(meta["debug_grasp_point_mm"]),
            pre_grasp_point_mm=None if meta.get("debug_pre_grasp_point_mm") is None else _decode_point3(meta["debug_pre_grasp_point_mm"]),
            rotation=None if meta.get("debug_rotation") is None else _decode_rotation3(meta["debug_rotation"]),
            rpy_deg=None if meta.get("debug_rpy_deg") is None else _decode_point3(meta["debug_rpy_deg"]),
        )
    return OpeningDetectionPipelineResponse(
        request_id=int(meta.get("request_id", 0)),
        frame_id=int(meta["frame_id"]),
        camera_name=str(meta["camera_name"]),
        timestamp_ms=float(meta["timestamp_ms"]),
        source_meta=dict(meta.get("source_meta", {})),
        elapsed_ms=float(meta["elapsed_ms"]),
        tray_count=int(meta["tray_count"]),
        tray_results=tuple(_decode_result(item) for item in meta.get("tray_results", [])),
        selected_tray_index=int(meta.get("selected_tray_index", 0)),
        selected_result=_decode_tray_pose_info(meta.get("selected_result")),
        all_tray_results=tuple(item for item in (_decode_tray_pose_info(raw) for raw in meta.get("all_tray_results", [])) if item is not None),
        debug=debug,
        error=meta.get("error"),
    )


def _encode_tray_pose_info(info: TrayPoseInfo) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tray_index": int(info.tray_index),
        "tray_bbox_xywh": list(info.tray_bbox_xywh),
        "tray_center_uv": list(info.tray_center_uv),
        "opening_center_uv": None if info.opening_center_uv is None else list(info.opening_center_uv),
        "opening_quad_uv": None if info.opening_quad_uv is None else [list(item) for item in info.opening_quad_uv],
        "top_quad_uv": None if info.top_quad_uv is None else [list(item) for item in info.top_quad_uv],
        "pose": None if info.pose is None else _encode_grasp_pose_info(info.pose),
    }
    return payload


def _encode_grasp_pose_info(info: GraspPoseInfo) -> Dict[str, Any]:
    return {
        "grasp_point_mm": list(info.grasp_point_mm),
        "pre_grasp_point_mm": list(info.pre_grasp_point_mm),
        "rotation": None if info.rotation is None else [list(row) for row in info.rotation],
        "rpy_deg": None if info.rpy_deg is None else list(info.rpy_deg),
    }


def _decode_tray_pose_info(raw: Optional[Dict[str, Any]]) -> Optional[TrayPoseInfo]:
    if raw is None:
        return None
    pose_raw = raw.get("pose")
    return TrayPoseInfo(
        tray_index=int(raw["tray_index"]),
        tray_bbox_xywh=tuple(int(v) for v in raw["tray_bbox_xywh"]),
        tray_center_uv=tuple(float(v) for v in raw["tray_center_uv"]),
        opening_center_uv=None if raw.get("opening_center_uv") is None else tuple(float(v) for v in raw["opening_center_uv"]),
        opening_quad_uv=None if raw.get("opening_quad_uv") is None else tuple(tuple(float(v) for v in point) for point in raw["opening_quad_uv"]),
        top_quad_uv=None if raw.get("top_quad_uv") is None else tuple(tuple(float(v) for v in point) for point in raw["top_quad_uv"]),
        pose=None if pose_raw is None else _decode_grasp_pose_info(pose_raw),
    )


def _decode_grasp_pose_info(raw: Dict[str, Any]) -> GraspPoseInfo:
    return GraspPoseInfo(
        grasp_point_mm=tuple(float(v) for v in raw["grasp_point_mm"]),
        pre_grasp_point_mm=tuple(float(v) for v in raw["pre_grasp_point_mm"]),
        rotation=None if raw.get("rotation") is None else tuple(tuple(float(v) for v in row) for row in raw["rotation"]),
        rpy_deg=None if raw.get("rpy_deg") is None else tuple(float(v) for v in raw["rpy_deg"]),
    )


def _decode_point2(values: Any) -> Tuple[float, float]:
    return float(values[0]), float(values[1])


def _decode_point3(values: Any) -> Tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _decode_point4(values: Any) -> Tuple[float, float, float, float]:
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _decode_quad2(values: Any) -> Tuple[Tuple[float, float], ...]:
    return tuple(_decode_point2(item) for item in values)


def _decode_rotation3(values: Any) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(tuple(float(v) for v in row) for row in values)


def _decode_bbox_xywh(values: Any) -> Tuple[int, int, int, int]:
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def _encode_png_depth(depth_mm: Any) -> bytes:
    if depth_mm is None:
        return b""
    arr = np.asarray(depth_mm, dtype=np.uint16)
    ok, encoded = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("depth png encode failed")
    return bytes(encoded.tobytes())


def _encode_png_mask(mask: Any) -> bytes:
    if mask is None:
        return b""
    arr = np.asarray(mask, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("mask png encode failed")
    return bytes(encoded.tobytes())


def _decode_png_depth(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("depth png decode failed")
    return np.asarray(arr, dtype=np.uint16)


def _decode_png_mask(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("mask png decode failed")
    return np.asarray(arr, dtype=np.uint8)
