from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .protocol import (
    OrinTrayDetectionDebugArtifacts,
    OrinTrayDetectionInfo,
    OrinTrayDetectionRequest,
    OrinTrayDetectionResponse,
)


def encode_request(packet: OrinTrayDetectionRequest) -> List[bytes]:
    meta = {
        "request_id": int(packet.request_id),
        "camera_name": str(packet.camera_name),
        "frame_id": int(packet.frame_id),
        "enable_debug": bool(packet.enable_debug),
    }
    return [b"orin_tray_detection_request", json.dumps(meta, ensure_ascii=False).encode("utf-8")]


def decode_request(parts: List[bytes]) -> OrinTrayDetectionRequest:
    if len(parts) != 2:
        raise RuntimeError("invalid tray detection request multipart count")
    topic, meta_bytes = parts
    if topic != b"orin_tray_detection_request":
        raise RuntimeError("unexpected tray detection request topic")
    meta = json.loads(meta_bytes.decode("utf-8"))
    return OrinTrayDetectionRequest(
        request_id=int(meta.get("request_id", 0)),
        camera_name=str(meta.get("camera_name", "left_hand_camera")),
        frame_id=int(meta.get("frame_id", -1)),
        enable_debug=bool(meta.get("enable_debug", False)),
    )


def encode_response(packet: OrinTrayDetectionResponse) -> List[bytes]:
    meta = {
        "request_id": int(packet.request_id),
        "frame_id": int(packet.frame_id),
        "camera_name": str(packet.camera_name),
        "timestamp_ms": float(packet.timestamp_ms),
        "source_meta": dict(packet.source_meta),
        "elapsed_ms": float(packet.elapsed_ms),
        "tray_count": int(packet.tray_count),
        "tray_results": [_encode_result(item) for item in packet.tray_results],
        "error": packet.error,
    }
    debug = packet.debug
    overlay_bytes = _encode_jpeg(None if debug is None else debug.overlay_bgr)
    mask_preview_bytes = _encode_jpeg(None if debug is None else debug.mask_bgr)
    tray_masks_bytes = _encode_mask_stack(tuple() if debug is None else debug.tray_masks)
    return [
        b"orin_tray_detection_response",
        json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        overlay_bytes,
        mask_preview_bytes,
        tray_masks_bytes,
    ]


def decode_response(parts: List[bytes]) -> OrinTrayDetectionResponse:
    if len(parts) != 5:
        raise RuntimeError("invalid tray detection response multipart count")
    topic, meta_bytes, overlay_bytes, mask_preview_bytes, tray_masks_bytes = parts
    if topic != b"orin_tray_detection_response":
        raise RuntimeError("unexpected tray detection response topic")
    meta = json.loads(meta_bytes.decode("utf-8"))
    debug = None
    if len(overlay_bytes) > 0 or len(mask_preview_bytes) > 0 or len(tray_masks_bytes) > 0:
        debug = OrinTrayDetectionDebugArtifacts(
            overlay_bgr=_decode_jpeg(overlay_bytes),
            mask_bgr=_decode_jpeg(mask_preview_bytes),
            tray_masks=_decode_mask_stack(tray_masks_bytes),
        )
    return OrinTrayDetectionResponse(
        request_id=int(meta.get("request_id", 0)),
        frame_id=int(meta["frame_id"]),
        camera_name=str(meta["camera_name"]),
        timestamp_ms=float(meta["timestamp_ms"]),
        source_meta=dict(meta.get("source_meta", {})),
        elapsed_ms=float(meta["elapsed_ms"]),
        tray_count=int(meta["tray_count"]),
        tray_results=tuple(_decode_result(item) for item in meta.get("tray_results", [])),
        debug=debug,
        error=meta.get("error"),
    )


def _encode_result(info: OrinTrayDetectionInfo) -> Dict[str, Any]:
    return {
        "tray_id": int(info.tray_id),
        "label_text": str(info.label_text),
        "confidence_2d": float(info.confidence_2d),
        "bbox_xywh": list(info.bbox_xywh),
        "center_uv": list(info.center_uv),
        "mask_area_px": int(info.mask_area_px),
        "source": str(info.source),
    }


def _decode_result(raw: Dict[str, Any]) -> OrinTrayDetectionInfo:
    bbox_xywh = raw["bbox_xywh"]
    center_uv = raw["center_uv"]
    return OrinTrayDetectionInfo(
        tray_id=int(raw["tray_id"]),
        label_text=str(raw["label_text"]),
        confidence_2d=float(raw["confidence_2d"]),
        bbox_xywh=(int(bbox_xywh[0]), int(bbox_xywh[1]), int(bbox_xywh[2]), int(bbox_xywh[3])),
        center_uv=(float(center_uv[0]), float(center_uv[1])),
        mask_area_px=int(raw["mask_area_px"]),
        source=str(raw["source"]),
    )


def _encode_jpeg(image_bgr: Optional[np.ndarray]) -> bytes:
    if image_bgr is None:
        return b""
    ok, encoded = cv2.imencode(".jpg", np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode tray detection jpeg")
    return encoded.tobytes()


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    if len(data) == 0:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode tray detection jpeg")
    return np.asarray(image, dtype=np.uint8)


def _encode_mask_stack(masks: tuple[np.ndarray, ...]) -> bytes:
    if len(masks) == 0:
        return b""
    buffer = io.BytesIO()
    np.save(buffer, np.stack([np.asarray(item, dtype=np.uint8) for item in masks], axis=0), allow_pickle=False)
    return buffer.getvalue()


def _decode_mask_stack(data: bytes) -> tuple[np.ndarray, ...]:
    if len(data) == 0:
        return tuple()
    array = np.load(io.BytesIO(data), allow_pickle=False)
    array = np.asarray(array, dtype=np.uint8)
    if array.ndim == 2:
        return (array,)
    return tuple(np.asarray(array[idx], dtype=np.uint8) for idx in range(array.shape[0]))
