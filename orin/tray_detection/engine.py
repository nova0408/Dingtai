from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from orin.camera_stream import CameraFramePacket, CameraStreamRuntime

from .pipeline import TrayDetectionPipeline
from .protocol import (
    OrinTrayDetectionDebugArtifacts,
    OrinTrayDetectionInfo,
    OrinTrayDetectionRequest,
    OrinTrayDetectionResponse,
)
from .types import TrayDetectionConfig, TrayRuntimeState


@dataclass(frozen=True)
class OrinTrayDetectionExecutorConfig:
    detector_config: TrayDetectionConfig = field(
        default_factory=lambda: TrayDetectionConfig(
            prompt="black tray,black pallet,rectangular black tray",
            target_keywords="rectangular black tray,black tray,black pallet",
            strict_target_filter=True,
            max_targets=3,
            use_sam=False,
            min_confidence=0.22,
            topk_objects=8,
            sam_max_boxes=1,
            sam_primary_only=True,
            combine_prompts_forward=False,
            detect_max_side=640,
            box_threshold=0.12,
            text_threshold=0.05,
        )
    )


class OrinTrayDetectionExecutor:
    """独立托盘检测执行器。"""

    def __init__(self, frame_runtime: CameraStreamRuntime, config: Optional[OrinTrayDetectionExecutorConfig] = None) -> None:
        self._frame_runtime = frame_runtime
        self._config = OrinTrayDetectionExecutorConfig() if config is None else config
        detector = TrayDetectionPipeline.build_detector(self._config.detector_config)
        self._pipeline = TrayDetectionPipeline(tray_detector=detector)
        self._state = TrayRuntimeState()

    def process_request(self, request: OrinTrayDetectionRequest) -> OrinTrayDetectionResponse:
        t0 = time.perf_counter()
        frame = self._resolve_frame(request)
        if frame is None:
            return OrinTrayDetectionResponse(
                request_id=int(request.request_id),
                frame_id=-1,
                camera_name=str(request.camera_name),
                timestamp_ms=0.0,
                source_meta={},
                elapsed_ms=float((time.perf_counter() - t0) * 1000.0),
                tray_count=0,
                tray_results=tuple(),
                debug=None,
                error="camera frame not ready",
            )
        detections, from_detector = self._pipeline.segment_trays(np.asarray(frame.color_bgr, dtype=np.uint8), self._state)
        filtered_detections = self._prune_container_detections(detections)
        ordered_detections = self._sort_detections_left_to_right(filtered_detections)
        tray_results = tuple(self._build_result(index, det) for index, det in enumerate(ordered_detections))
        return OrinTrayDetectionResponse(
            request_id=int(request.request_id),
            frame_id=int(frame.frame_id),
            camera_name=str(frame.camera_name),
            timestamp_ms=float(frame.timestamp_ms),
            source_meta=dict(frame.source_meta),
            elapsed_ms=float((time.perf_counter() - t0) * 1000.0),
            tray_count=len(tray_results),
            tray_results=tray_results,
            debug=self._build_debug_artifacts(frame, ordered_detections, request.enable_debug, from_detector),
            error=None,
        )

    def _resolve_frame(self, request: OrinTrayDetectionRequest) -> Optional[CameraFramePacket]:
        if int(request.frame_id) > 0:
            frame = self._frame_runtime.get_frame_by_id(int(request.frame_id))
            if frame is not None:
                return frame
        return self._frame_runtime.get_latest_frame()

    def _build_result(self, tray_id: int, det) -> OrinTrayDetectionInfo:
        mask = np.asarray(det.mask, dtype=np.uint8)
        bbox_xywh = _mask_bbox_xywh(mask)
        center_uv = (float(bbox_xywh[0] + 0.5 * bbox_xywh[2]), float(bbox_xywh[1] + 0.5 * bbox_xywh[3]))
        return OrinTrayDetectionInfo(
            tray_id=int(tray_id),
            label_text=str(det.label_text),
            confidence_2d=float(det.confidence_2d),
            bbox_xywh=bbox_xywh,
            center_uv=center_uv,
            mask_area_px=int(np.count_nonzero(mask)),
            source=str(getattr(det, "label_text", "fast")),
        )

    def _build_debug_artifacts(
        self,
        frame: CameraFramePacket,
        detections: list,
        enable_debug: bool,
        from_detector: bool,
    ) -> Optional[OrinTrayDetectionDebugArtifacts]:
        if not bool(enable_debug):
            return None
        overlay = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
        mask_preview = np.zeros_like(overlay)
        tray_masks: list[np.ndarray] = []
        for tray_id, det in enumerate(detections):
            mask = np.asarray(det.mask, dtype=np.uint8)
            tray_masks.append(mask)
            color = _debug_color_bgr(tray_id)
            overlay = _blend_mask_overlay(overlay, mask, color, 0.28)
            _draw_mask_outline(overlay, mask, color)
            _draw_mask_outline(mask_preview, mask, color)
            bbox_xywh = _mask_bbox_xywh(mask)
            cv2.rectangle(
                overlay,
                (bbox_xywh[0], bbox_xywh[1]),
                (bbox_xywh[0] + bbox_xywh[2] - 1, bbox_xywh[1] + bbox_xywh[3] - 1),
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"tray_{tray_id} conf {float(det.confidence_2d):.2f}",
                (bbox_xywh[0], max(14, bbox_xywh[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(overlay, f"frame {frame.frame_id} source {'detector' if from_detector else 'fast'}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(mask_preview, "Tray mask preview", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return OrinTrayDetectionDebugArtifacts(
            overlay_bgr=overlay,
            mask_bgr=mask_preview,
            tray_masks=tuple(tray_masks),
        )

    def _sort_detections_left_to_right(self, detections: list) -> list:
        return sorted(
            list(detections),
            key=lambda det: _mask_center_uv(np.asarray(det.mask, dtype=np.uint8))[0],
        )

    def _prune_container_detections(self, detections: list) -> list:
        if len(detections) <= 2:
            return list(detections)
        det_infos = [_DetectionBoxInfo.from_detection(det) for det in detections]
        keep_flags = [True for _ in det_infos]
        for idx, info in enumerate(det_infos):
            contained_indices = [
                other_idx
                for other_idx, other in enumerate(det_infos)
                if other_idx != idx and _bbox_contains_center(info.bbox_xywh, other.center_uv)
            ]
            if len(contained_indices) < 2:
                continue
            other_widths = [det_infos[item].bbox_xywh[2] for item in contained_indices]
            other_areas = [det_infos[item].mask_area_px for item in contained_indices]
            width_is_large = info.bbox_xywh[2] >= max(other_widths) * 1.45
            area_is_large = info.mask_area_px >= max(other_areas) * 1.55
            if width_is_large and area_is_large:
                keep_flags[idx] = False
        filtered = [detections[idx] for idx, keep in enumerate(keep_flags) if keep]
        if len(filtered) == 0:
            return list(detections)
        return filtered


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size == 0:
        height, width = mask.shape[:2]
        return 0, 0, int(width), int(height)
    x1 = int(np.min(xs))
    x2 = int(np.max(xs))
    y1 = int(np.min(ys))
    y2 = int(np.max(ys))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def _mask_center_uv(mask: np.ndarray) -> tuple[float, float]:
    bbox_xywh = _mask_bbox_xywh(mask)
    return (
        float(bbox_xywh[0] + 0.5 * bbox_xywh[2]),
        float(bbox_xywh[1] + 0.5 * bbox_xywh[3]),
    )


def _bbox_contains_center(bbox_xywh: tuple[int, int, int, int], center_uv: tuple[float, float]) -> bool:
    x, y, w, h = bbox_xywh
    cx, cy = center_uv
    return float(x) <= float(cx) <= float(x + w) and float(y) <= float(cy) <= float(y + h)


@dataclass(frozen=True)
class _DetectionBoxInfo:
    bbox_xywh: tuple[int, int, int, int]
    center_uv: tuple[float, float]
    mask_area_px: int

    @staticmethod
    def from_detection(det) -> "_DetectionBoxInfo":
        mask = np.asarray(det.mask, dtype=np.uint8)
        bbox_xywh = _mask_bbox_xywh(mask)
        return _DetectionBoxInfo(
            bbox_xywh=bbox_xywh,
            center_uv=_mask_center_uv(mask),
            mask_area_px=int(np.count_nonzero(mask)),
        )


def _debug_color_bgr(index: int) -> tuple[int, int, int]:
    palette = (
        (0, 220, 255),
        (80, 200, 120),
        (255, 170, 0),
        (255, 110, 180),
        (140, 180, 255),
        (200, 120, 255),
    )
    return palette[int(index) % len(palette)]


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _blend_mask_overlay(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or base_bgr.shape[:2] != mask_u8.shape[:2]:
        return base_bgr
    mask_bool = mask_u8 > 0
    if not np.any(mask_bool):
        return base_bgr
    result = np.asarray(base_bgr, dtype=np.float32).copy()
    color = np.asarray(color_bgr, dtype=np.float32)
    result[mask_bool] = result[mask_bool] * (1.0 - float(alpha)) + color * float(alpha)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)
