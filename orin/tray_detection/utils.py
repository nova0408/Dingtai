from __future__ import annotations

import cv2
import numpy as np


def parse_keywords(text: str) -> list[str]:
    raw = str(text).replace("，", ",").split(",")
    return [s.strip().lower() for s in raw if len(s.strip()) > 0]


def build_combined_prompt(terms: list[str]) -> str:
    clean = [str(t).strip() for t in terms if len(str(t).strip()) > 0]
    if len(clean) == 0:
        return "object"
    return ". ".join(clean)


def normalize_label_list(labels, expect_len: int) -> list[str]:
    if labels is None:
        return ["" for _ in range(expect_len)]
    if isinstance(labels, np.ndarray):
        vals = [str(x) for x in labels.tolist()]
    elif isinstance(labels, (list, tuple)):
        vals = [str(x) for x in labels]
    else:
        try:
            vals = [str(x) for x in labels]
        except Exception:
            vals = []
    if len(vals) < expect_len:
        vals.extend([""] * (expect_len - len(vals)))
    return vals[:expect_len]


def merge_label_text(raw_label: str, prompt_term: str) -> str:
    r = str(raw_label).strip().lower()
    p = str(prompt_term).strip()
    if len(r) == 0:
        return p
    if r in {"object", "objects", "thing", "item"}:
        return p if len(p) > 0 else str(raw_label).strip()
    return str(raw_label).strip()


def resize_for_detection(frame_bgr: np.ndarray, detect_max_side: int) -> tuple[np.ndarray, float]:
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= int(detect_max_side):
        return frame_bgr, 1.0
    scale = float(detect_max_side) / float(m)
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, 1.0 / scale


def scale_box_xyxy(box_xyxy: np.ndarray, scale: float, w: int, h: int) -> np.ndarray:
    b = np.asarray(box_xyxy, dtype=np.float32).copy()
    b *= float(scale)
    b[0] = np.clip(b[0], 0, max(0, w - 1))
    b[2] = np.clip(b[2], 0, max(0, w - 1))
    b[1] = np.clip(b[1], 0, max(0, h - 1))
    b[3] = np.clip(b[3], 0, max(0, h - 1))
    if b[2] < b[0]:
        b[0], b[2] = b[2], b[0]
    if b[3] < b[1]:
        b[1], b[3] = b[3], b[1]
    return b


def build_rect_mask(box_xyxy: np.ndarray, h: int, w: int) -> np.ndarray:
    x1 = int(np.floor(float(box_xyxy[0])))
    y1 = int(np.floor(float(box_xyxy[1])))
    x2 = int(np.ceil(float(box_xyxy[2])))
    y2 = int(np.ceil(float(box_xyxy[3])))
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    x2 = int(np.clip(x2, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    y2 = int(np.clip(y2, 0, max(0, h - 1)))
    out = np.zeros((h, w), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return out
    out[y1 : y2 + 1, x1 : x2 + 1] = 255
    return out


def mask_to_contour(mask: np.ndarray, min_mask_pixels: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.empty((0, 2), dtype=np.int32)
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < float(min_mask_pixels):
        return np.empty((0, 2), dtype=np.int32)
    hull = cv2.convexHull(c).reshape(-1, 2)
    return hull.astype(np.int32)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def suppress_masks_by_iou(
    candidates: list[tuple[np.ndarray, float, str, np.ndarray, int]],
    mask_iou_suppress: float,
    max_count: int,
) -> list[tuple[np.ndarray, float, str, np.ndarray, int]]:
    candidates.sort(key=lambda x: x[1], reverse=True)
    kept: list[tuple[np.ndarray, float, str, np.ndarray, int]] = []
    for cand in candidates:
        keep = True
        for item in kept:
            if mask_iou(cand[0], item[0]) >= float(mask_iou_suppress):
                keep = False
                break
        if keep:
            kept.append(cand)
        if len(kept) >= int(max_count):
            break
    return kept
