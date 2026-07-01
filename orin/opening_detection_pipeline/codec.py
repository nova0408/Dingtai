from __future__ import annotations

import json
from typing import List

from ..opening_detection.codec import _decode_point4, _decode_tray_pose_info, _encode_jpeg, _encode_mask_stack, _encode_png_depth, _encode_png_mask, _decode_jpeg, _decode_mask_stack, _decode_png_depth, _decode_png_mask, _encode_tray_pose_info
from ..opening_detection.protocol import DebugArtifacts
from ..tray_detection.codec import _decode_result, _encode_result

from .protocol import OpeningDetectionPipelineRequest, OpeningDetectionPipelineResponse


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
    meta = {
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
