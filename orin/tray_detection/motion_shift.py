from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PhaseShiftEstimate:
    dx_px: float
    dy_px: float
    response: float
    valid: bool


def prepare_tracking_gray(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    long_side = max(1, h, w)
    scale = min(1.0, float(max_side) / float(long_side))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image_bgr
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, scale


def estimate_phase_shift(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    scale: float,
    min_response: float = 0.02,
    max_shift_px: float | None = None,
) -> PhaseShiftEstimate:
    if ref_gray.shape != cur_gray.shape:
        return PhaseShiftEstimate(dx_px=0.0, dy_px=0.0, response=0.0, valid=False)
    try:
        (dx_small, dy_small), response = cv2.phaseCorrelate(np.asarray(ref_gray, dtype=np.float32), np.asarray(cur_gray, dtype=np.float32))
    except cv2.error:
        return PhaseShiftEstimate(dx_px=0.0, dy_px=0.0, response=0.0, valid=False)
    if not np.isfinite(dx_small) or not np.isfinite(dy_small):
        return PhaseShiftEstimate(dx_px=0.0, dy_px=0.0, response=float(response) if np.isfinite(response) else 0.0, valid=False)
    response_value = float(response) if np.isfinite(response) else 0.0
    if response_value < float(min_response):
        return PhaseShiftEstimate(dx_px=0.0, dy_px=0.0, response=response_value, valid=False)
    scale_safe = max(1e-6, float(scale))
    dx_px = float(dx_small) / scale_safe
    dy_px = float(dy_small) / scale_safe
    if max_shift_px is not None:
        limit = abs(float(max_shift_px))
        dx_px = float(np.clip(dx_px, -limit, limit))
        dy_px = float(np.clip(dy_px, -limit, limit))
    return PhaseShiftEstimate(dx_px=dx_px, dy_px=dy_px, response=response_value, valid=True)


def warp_mask(mask: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    h, w = mask.shape[:2]
    mat = np.asarray([[1.0, 0.0, float(dx_px)], [0.0, 1.0, float(dy_px)]], dtype=np.float32)
    return cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8),
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
