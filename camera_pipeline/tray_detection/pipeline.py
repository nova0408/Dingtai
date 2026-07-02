from __future__ import annotations

import threading
from typing import Any, cast

import cv2
import numpy as np

from .detector import TrayPointExcluder
from .motion_shift import estimate_phase_shift, prepare_tracking_gray, warp_mask
from .types import TrayDetection, TrayDetectionConfig, TrayPipelineConfig, TrayRuntimeState


class TrayDetectionPipeline:
    def __init__(
        self, tray_detector: TrayPointExcluder | None, config: TrayPipelineConfig | None = None
    ) -> None:
        self._tray_detector = tray_detector
        self._config = config if config is not None else TrayPipelineConfig()

    @staticmethod
    def build_detector(config: TrayDetectionConfig | None = None) -> TrayPointExcluder | None:
        try:
            cfg = TrayDetectionConfig() if config is None else config
            return TrayPointExcluder(cfg)
        except Exception:
            return None

    @staticmethod
    def build_default_detector() -> TrayPointExcluder | None:
        return TrayDetectionPipeline.build_detector(
            TrayDetectionConfig(
                prompt="black tray,black pallet,rectangular black tray",
                target_keywords="rectangular black tray,black tray,black pallet",
                strict_target_filter=True,
                max_targets=3,
                use_sam=False,
                min_confidence=0.35,
                topk_objects=4,
                sam_max_boxes=1,
                sam_primary_only=True,
                combine_prompts_forward=False,
                detect_max_side=512,
            )
        )

    def segment_trays(
        self, rgb_bgr: np.ndarray, state: TrayRuntimeState
    ) -> tuple[list[TrayDetection], bool]:
        motion_dx, motion_dy = self._estimate_motion(rgb_bgr, state)
        state.compute_count += 1
        fast_detections = self._segment_trays_fast(rgb_bgr)
        should_refresh = self._tray_detector is not None and (
            len(state.cached_detections) == 0
            or (state.compute_count % max(1, int(self._config.detect_every_n)) == 1)
            or (not state.cached_ok)
        )
        if should_refresh:
            self._start_async_refine(rgb_bgr, state)
        with state.lock:
            cached_detections = [
                TrayDetection(
                    label_text=item.label_text,
                    confidence_2d=float(item.confidence_2d),
                    contour=np.asarray(item.contour, dtype=np.int32).copy(),
                    mask=np.asarray(item.mask, dtype=np.uint8).copy(),
                    excluded_points=int(item.excluded_points),
                )
                for item in state.cached_detections
            ]
            cached_ok = bool(state.cached_ok)
        if len(cached_detections) == 0:
            return fast_detections, False
        return self._warp_tray_detections(cached_detections, motion_dx, motion_dy), cached_ok

    def _start_async_refine(self, rgb_bgr: np.ndarray, state: TrayRuntimeState) -> None:
        if self._tray_detector is None:
            return
        detector = self._tray_detector
        with state.lock:
            if state.detect_inflight:
                return
            state.detect_inflight = True
        frame = rgb_bgr.copy()

        def _task() -> None:
            try:
                dets = detector.detect(frame)
                if len(dets) > 0:
                    refined_detections: list[TrayDetection] = []
                    for det in dets:
                        mask = cv2.morphologyEx(
                            np.asarray(det.mask, dtype=np.uint8),
                            cv2.MORPH_CLOSE,
                            np.ones((9, 9), dtype=np.uint8),
                            iterations=2,
                        )
                        contour = self._mask_to_contour(mask)
                        if contour.shape[0] < 3:
                            continue
                        refined_detections.append(
                            TrayDetection(
                                label_text=str(det.label_text),
                                confidence_2d=float(det.confidence_2d),
                                contour=contour,
                                mask=mask,
                                excluded_points=int(det.excluded_points),
                            )
                        )
                    with state.lock:
                        state.cached_detections = refined_detections
                        state.cached_ok = len(refined_detections) > 0
            finally:
                with state.lock:
                    state.detect_inflight = False

        threading.Thread(target=_task, name="tray_refine_async", daemon=True).start()

    def _segment_trays_fast(self, rgb_bgr: np.ndarray) -> list[TrayDetection]:
        h, w = rgb_bgr.shape[:2]
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        gray_f = np.asarray(gray, dtype=np.float64)
        base = (gray_f <= np.percentile(cast(Any, gray_f), float(self._config.fast_gray_percentile))).astype(
            np.uint8
        ) * 255
        base = cv2.morphologyEx(
            base, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=1
        )
        base[: int(float(self._config.fast_top_crop_ratio) * h), :] = 0
        num, cc, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
        if num <= 1:
            return []
        candidates: list[tuple[float, TrayDetection]] = []
        tgt = np.array([0.5 * w, 0.78 * h], dtype=np.float64)
        for idx in range(1, num):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < 400.0:
                continue
            cx = float(stats[idx, cv2.CC_STAT_LEFT] + 0.5 * stats[idx, cv2.CC_STAT_WIDTH])
            cy = float(stats[idx, cv2.CC_STAT_TOP] + 0.5 * stats[idx, cv2.CC_STAT_HEIGHT])
            dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float64) - tgt))
            score = area - 1.2 * dist
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[cc == idx] = 255
            contour = self._mask_to_contour(mask)
            if contour.shape[0] < 3:
                continue
            candidates.append(
                (
                    score,
                    TrayDetection(
                        label_text=f"fast_tray_{idx}",
                        confidence_2d=float(np.clip(area / max(1.0, 0.25 * h * w), 0.0, 1.0)),
                        contour=contour,
                        mask=mask,
                        excluded_points=0,
                    ),
                )
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        max_targets = max(
            1,
            int(
                getattr(
                    self._tray_detector.config
                    if self._tray_detector is not None
                    else TrayDetectionConfig(),
                    "max_targets",
                    1,
                )
            ),
        )
        return [item[1] for item in candidates[:max_targets]]

    def _estimate_motion(self, rgb_bgr: np.ndarray, state: TrayRuntimeState) -> tuple[float, float]:
        max_side = int(
            round(max(rgb_bgr.shape[0], rgb_bgr.shape[1]) * float(self._config.motion_downsample))
        )
        small, scale = prepare_tracking_gray(rgb_bgr, max(32, max_side))
        small_u8 = np.asarray(np.clip(small, 0.0, 255.0), dtype=np.uint8)
        with state.lock:
            prev = (
                None
                if state.prev_motion_gray_small is None
                else state.prev_motion_gray_small.copy()
            )
            state.prev_motion_gray_small = small_u8
            dx_s = float(state.motion_dx_smooth)
            dy_s = float(state.motion_dy_smooth)
        if prev is None or prev.shape != small_u8.shape:
            return dx_s, dy_s
        shift = estimate_phase_shift(
            ref_gray=np.asarray(prev, dtype=np.float32),
            cur_gray=np.asarray(small_u8, dtype=np.float32),
            scale=float(scale),
            min_response=0.02,
            max_shift_px=float(self._config.motion_max_shift_px),
        )
        if not shift.valid:
            return dx_s, dy_s
        dx_raw = float(shift.dx_px)
        dy_raw = float(shift.dy_px)
        a = float(np.clip(float(self._config.motion_smooth_alpha), 0.05, 0.98))
        dx_new = a * dx_s + (1.0 - a) * dx_raw
        dy_new = a * dy_s + (1.0 - a) * dy_raw
        with state.lock:
            state.motion_dx_smooth = dx_new
            state.motion_dy_smooth = dy_new
        return dx_new, dy_new

    @staticmethod
    def _warp_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
        return warp_mask(np.asarray(mask, dtype=np.uint8), dx_px=float(dx), dy_px=float(dy))

    def _warp_tray_detections(
        self, detections: list[TrayDetection], dx: float, dy: float
    ) -> list[TrayDetection]:
        warped: list[TrayDetection] = []
        for item in detections:
            warped_mask = self._warp_mask(np.asarray(item.mask, dtype=np.uint8), dx, dy)
            contour = self._mask_to_contour(warped_mask)
            if contour.shape[0] < 3:
                continue
            warped.append(
                TrayDetection(
                    label_text=str(item.label_text),
                    confidence_2d=float(item.confidence_2d),
                    contour=contour,
                    mask=warped_mask,
                    excluded_points=int(item.excluded_points),
                )
            )
        return warped

    @staticmethod
    def _mask_to_contour(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return np.empty((0, 2), dtype=np.int32)
        contour = max(contours, key=cv2.contourArea)
        return np.asarray(contour.reshape(-1, 2), dtype=np.int32)
