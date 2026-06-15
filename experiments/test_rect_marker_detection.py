from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 opencv-python 才能运行矩形标记件检测脚本。") from exc

from pyorbbecsdk import OBFormat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import Gemini305, SessionOptions  # noqa: E402

# region 默认参数（优先在这里直接改）
DEFAULT_PRIOR_ROOT = Path("experiments/rect_marker_prior_sessions")
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_WINDOW_NAME = "矩形标记件实时识别"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 820
DEFAULT_ROI_EXTRA_PX = 6
DEFAULT_LAB_THRESHOLD = 42.0
DEFAULT_HUE_THRESHOLD = 12.0
DEFAULT_MIN_COLOR_SUPPORT = 0.12
DEFAULT_MIN_PRESENT_VISIBLE_RATIO = 0.08
DEFAULT_OCCLUDED_VISIBLE_RATIO = 0.55
DEFAULT_EDGE_BAND_PX = 7
DEFAULT_CANNY_LOW = 45
DEFAULT_CANNY_HIGH = 130
DEFAULT_MIN_EDGE_POINTS_PER_SIDE = 3
DEFAULT_MIN_SCORE = 0.48
# endregion


# region 数据结构
@dataclass(frozen=True)
class CameraCalibration:
    """先验 JSON 中保存的彩色相机参数。"""

    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float]
    distortion_model: str
    coordinate_space: str


@dataclass(frozen=True)
class RectMarkerPrior:
    """单个矩形标记件先验。"""

    marker_id: str
    corners_px: np.ndarray
    rgb_prior: np.ndarray
    expected_area_px: float
    expected_angle_deg: float
    expected_aspect_ratio: float
    max_center_shift_px: float
    max_angle_delta_deg: float
    min_area_ratio: float
    max_area_ratio: float
    roi_expand_px: int
    roi_expand_ratio: float


@dataclass(frozen=True)
class RectMarkerSetPrior:
    """矩形标记件先验集合。"""

    source_path: Path
    camera: CameraCalibration
    markers: list[RectMarkerPrior]


@dataclass(frozen=True)
class DetectionConfig:
    """实验性矩形标记件检测参数。"""

    roi_extra_px: int
    lab_threshold: float
    hue_threshold: float
    min_color_support: float
    min_present_visible_ratio: float
    occluded_visible_ratio: float
    edge_band_px: int
    canny_low: int
    canny_high: int
    min_edge_points_per_side: int
    min_score: float


@dataclass(frozen=True)
class PreparedFrame:
    """一帧图像的共享预处理结果。"""

    bgr: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    gray: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class MarkerDetectionResult:
    """单个标记件检测结果。"""

    marker_id: str
    detected: bool
    status: str
    score: float
    center_px: tuple[float, float] | None
    corners_px: np.ndarray | None
    color_support: float
    visible_area_ratio: float
    edge_support: float
    occluded: bool
    failure_reasons: list[str]
    roi_rect: tuple[int, int, int, int]
    initial_rect: np.ndarray | None
    edge_fit_rect: np.ndarray | None
    color_mask: np.ndarray | None
    clean_mask: np.ndarray | None
    edge_roi: np.ndarray | None


# endregion


# region 主流程
def main(
    prior_path: Path | None,
    timeout_ms: int,
    capture_fps: int,
    config: DetectionConfig,
) -> None:
    resolved_prior_path = (
        prior_path if prior_path is not None else _find_latest_prior(DEFAULT_PRIOR_ROOT)
    )
    prior = _load_prior(resolved_prior_path)
    camera_matrix, dist_coeffs = _build_camera_matrix(prior.camera)

    logger.info(f"使用先验文件：{resolved_prior_path}")
    logger.info(f"标记件数量：{len(prior.markers)}")

    options = SessionOptions(
        timeout=int(timeout_ms),
        preferred_capture_fps=max(1, int(capture_fps)),
    )
    with Gemini305(options=options) as session:
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
        )
        frame_index = 0
        try:
            while True:
                frames = session.wait_for_frames()
                if frames is None:
                    continue

                color_bgr = _decode_color_frame_bgr(frames.get_color_frame())
                if color_bgr is None:
                    continue

                undistorted = cv2.undistort(color_bgr, camera_matrix, dist_coeffs)
                started = time.perf_counter()
                prepared = _prepare_frame(undistorted, config)
                results = [
                    _detect_one_marker(prepared, marker, config)
                    for marker in prior.markers
                ]
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                frame_index += 1
                print(f"frame={frame_index} compute_ms={elapsed_ms:.3f}", flush=True)

                preview = _draw_results(prepared.bgr, results, elapsed_ms)
                cv2.imshow(DEFAULT_WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    logger.warning("收到退出指令，结束检测。")
                    break
                if cv2.getWindowProperty(DEFAULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.warning("预览窗口关闭，结束检测。")
                    break
        finally:
            cv2.destroyWindow(DEFAULT_WINDOW_NAME)


# endregion


# region 检测流程
def _prepare_frame(image_bgr: np.ndarray, config: DetectionConfig) -> PreparedFrame:
    blurred = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(config.canny_low), int(config.canny_high))
    return PreparedFrame(bgr=image_bgr, hsv=hsv, lab=lab, gray=gray, edges=edges)


def _detect_one_marker(
    frame: PreparedFrame,
    marker: RectMarkerPrior,
    config: DetectionConfig,
) -> MarkerDetectionResult:
    roi_rect = _build_roi_rect(frame.bgr.shape, marker, config)
    x0, y0, x1, y1 = roi_rect
    roi_bgr = frame.bgr[y0:y1, x0:x1]
    roi_lab = frame.lab[y0:y1, x0:x1]
    roi_hsv = frame.hsv[y0:y1, x0:x1]
    roi_edges = frame.edges[y0:y1, x0:x1]

    if roi_bgr.size == 0:
        return _failed_result(marker.marker_id, roi_rect, "roi_empty")

    local_prior = marker.corners_px - np.array([x0, y0], dtype=np.float64)
    color_mask, color_support = _compute_color_mask(roi_lab, roi_hsv, marker, config)
    clean_mask = _clean_mask(color_mask)
    visible_area_ratio = _compute_visible_area_ratio(clean_mask, local_prior)
    if visible_area_ratio < config.min_present_visible_ratio:
        return _missing_result(
            marker_id=marker.marker_id,
            roi_rect=roi_rect,
            color_support=color_support,
            visible_area_ratio=visible_area_ratio,
            color_mask=color_mask,
            clean_mask=clean_mask,
            edge_roi=roi_edges,
            reason="marker_absent",
        )

    occluded = visible_area_ratio < config.occluded_visible_ratio
    initial_rect = _build_initial_rect(clean_mask, local_prior)
    if initial_rect is None:
        initial_rect = local_prior.astype(np.float32)

    edge_quad_local: np.ndarray | None = None
    edge_support = 0.0
    if not occluded:
        edge_quad_local, edge_support = _fit_quad_from_edges(
            roi_edges, initial_rect, config
        )

    # 目前实测 initial rect 比后续遮挡匹配更稳定，最终结果以 initial rect 为准。
    # edge fit 和遮挡匹配只保留为诊断阶段，避免调权重时把中心点拉偏。
    quad_local = initial_rect.astype(np.float64)
    quad_global = quad_local + np.array([x0, y0], dtype=np.float64)
    edge_quad_global = (
        None
        if edge_quad_local is None
        else edge_quad_local + np.array([x0, y0], dtype=np.float64)
    )
    score, reasons = _score_quad(
        quad=quad_global,
        marker=marker,
        color_support=color_support,
        edge_support=1.0,
        config=config,
    )
    if occluded:
        reasons.append("occluded_color_area")
    detected = score >= config.min_score and len(reasons) == 0
    status = (
        "detected"
        if detected
        else (
            "occluded"
            if occluded
            else ("uncertain" if score >= config.min_score * 0.65 else "missing")
        )
    )
    center = tuple(float(v) for v in np.mean(quad_global, axis=0))

    return MarkerDetectionResult(
        marker_id=marker.marker_id,
        detected=detected,
        status=status,
        score=float(score),
        center_px=center,
        corners_px=quad_global,
        color_support=float(color_support),
        visible_area_ratio=float(visible_area_ratio),
        edge_support=float(edge_support),
        occluded=occluded,
        failure_reasons=reasons,
        roi_rect=roi_rect,
        initial_rect=initial_rect + np.array([x0, y0], dtype=np.float64),
        edge_fit_rect=edge_quad_global,
        color_mask=color_mask,
        clean_mask=clean_mask,
        edge_roi=roi_edges,
    )


def _compute_color_mask(
    roi_lab: np.ndarray,
    roi_hsv: np.ndarray,
    marker: RectMarkerPrior,
    config: DetectionConfig,
) -> tuple[np.ndarray, float]:
    prior_rgb_u8 = np.asarray(marker.rgb_prior, dtype=np.uint8).reshape(1, 1, 3)
    prior_bgr_u8 = prior_rgb_u8[:, :, ::-1]
    prior_lab = (
        cv2.cvtColor(prior_bgr_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    )
    prior_hsv = (
        cv2.cvtColor(prior_bgr_u8, cv2.COLOR_BGR2HSV).reshape(3).astype(np.float32)
    )

    lab_diff = roi_lab.astype(np.float32) - prior_lab.reshape(1, 1, 3)
    lab_distance = np.linalg.norm(lab_diff, axis=2)

    hue = roi_hsv[:, :, 0].astype(np.float32)
    hue_delta = np.abs(hue - float(prior_hsv[0]))
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
    saturation = roi_hsv[:, :, 1].astype(np.float32)

    mask = (lab_distance <= float(config.lab_threshold)) & (
        (hue_delta <= float(config.hue_threshold)) | (saturation <= 35.0)
    )
    mask_u8 = mask.astype(np.uint8) * 255
    color_support = float(np.count_nonzero(mask_u8)) / float(max(1, mask_u8.size))
    return mask_u8, color_support


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    kernel5 = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel3, iterations=1)
    return opened


def _compute_visible_area_ratio(mask: np.ndarray, local_prior: np.ndarray) -> float:
    prior_area_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(prior_area_mask, np.round(local_prior).astype(np.int32), 255)
    prior_pixels = int(np.count_nonzero(prior_area_mask))
    if prior_pixels <= 0:
        return 0.0
    visible_pixels = int(np.count_nonzero(cv2.bitwise_and(mask, prior_area_mask)))
    return float(visible_pixels) / float(prior_pixels)


def _build_occluded_prior_rect(
    mask: np.ndarray, local_prior: np.ndarray
) -> tuple[np.ndarray, bool]:
    """遮挡时用颜色碎片和完整先验模板做不完整匹配。

    任意油污遮挡会让颜色质心和碎片外接框偏向未遮挡区域，不能直接用于拟合矩形。
    这里只允许颜色证据通过模板匹配给出整体平移，矩形尺寸、角度和长宽比仍来自先验。
    """

    ys, xs = np.nonzero(mask)
    if xs.size < 6:
        return local_prior.astype(np.float64), True

    prior = _order_quad_points_array(local_prior)
    origin = prior[0]
    u_axis = prior[1] - prior[0]
    v_axis = prior[3] - prior[0]
    width = float(np.linalg.norm(u_axis))
    height = float(np.linalg.norm(v_axis))
    if width < 1e-6 or height < 1e-6:
        return prior.astype(np.float64), True

    u_unit = u_axis / width
    v_unit = v_axis / height
    color_points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    rel = color_points - origin
    u_values = rel @ u_unit
    v_values = rel @ v_unit
    u_low, u_high = np.percentile(u_values, [5.0, 95.0])
    v_low, v_high = np.percentile(v_values, [5.0, 95.0])
    u_center = 0.5 * (u_low + u_high)
    v_center = 0.5 * (v_low + v_high)

    u_span_ratio = float((u_high - u_low) / max(1e-6, width))
    v_span_ratio = float((v_high - v_low) / max(1e-6, height))
    u_shifts = [
        (float(u_low), 0.18 if u_span_ratio < 0.82 else 0.04),
        (float(u_high - width), 0.18 if u_span_ratio < 0.82 else 0.04),
        (float(u_center - width * 0.5), 0.0),
    ]
    v_shifts = [
        (float(v_low), 0.18 if v_span_ratio < 0.82 else 0.04),
        (float(v_high - height), 0.18 if v_span_ratio < 0.82 else 0.04),
        (float(v_center - height * 0.5), 0.0),
    ]

    best_rect = prior.astype(np.float64)
    best_score = -1.0
    diagonal = max(1.0, float(np.hypot(width, height)))
    scored_rects: list[tuple[float, np.ndarray]] = []
    for u_shift, u_bonus in u_shifts:
        for v_shift, v_bonus in v_shifts:
            shift = u_unit * u_shift + v_unit * v_shift
            candidate = prior + shift
            color_score = _score_color_points_inside_rect(mask, candidate)
            shift_penalty = 0.10 * float(np.linalg.norm(shift)) / diagonal
            score = color_score + u_bonus + v_bonus - shift_penalty
            scored_rects.append((float(score), candidate))
            if score > best_score:
                best_score = score
                best_rect = candidate

    scored_rects.sort(key=lambda item: item[0], reverse=True)
    if scored_rects:
        top_score, top_rect = scored_rects[0]
        for candidate_score, candidate_rect in scored_rects[1:]:
            if top_score - candidate_score >= 0.10:
                break
            center_delta = float(
                np.linalg.norm(
                    np.mean(top_rect, axis=0) - np.mean(candidate_rect, axis=0)
                )
            )
            if center_delta > diagonal * 0.18:
                return prior.astype(np.float64), True
    return _order_quad_points_array(best_rect), False


def _score_color_points_inside_rect(mask: np.ndarray, rect: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0

    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    contour = np.round(rect).astype(np.float32)
    inside = 0
    for point in points:
        if (
            cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
            >= 0
        ):
            inside += 1
    coverage = float(inside) / float(max(1, points.shape[0]))
    rect_area = max(1.0, abs(_polygon_area(rect)))
    density = float(inside) / rect_area
    edge_bonus = _score_partial_color_edge_contact(points.astype(np.float64), rect)
    return 1.20 * coverage + 0.10 * density + 0.45 * edge_bonus


def _score_partial_color_edge_contact(points: np.ndarray, rect: np.ndarray) -> float:
    ordered = _order_quad_points_array(rect)
    origin = ordered[0]
    u_axis = ordered[1] - ordered[0]
    v_axis = ordered[3] - ordered[0]
    width = float(np.linalg.norm(u_axis))
    height = float(np.linalg.norm(v_axis))
    if width < 1e-6 or height < 1e-6:
        return 0.0

    u_unit = u_axis / width
    v_unit = v_axis / height
    rel = points - origin
    u_values = rel @ u_unit
    v_values = rel @ v_unit
    inside = (
        (u_values >= 0.0)
        & (u_values <= width)
        & (v_values >= 0.0)
        & (v_values <= height)
    )
    if not np.any(inside):
        return 0.0

    u_inside = u_values[inside]
    v_inside = v_values[inside]
    u_low, u_high = np.percentile(u_inside, [5.0, 95.0])
    v_low, v_high = np.percentile(v_inside, [5.0, 95.0])
    u_span_ratio = float((u_high - u_low) / max(1e-6, width))
    v_span_ratio = float((v_high - v_low) / max(1e-6, height))

    u_edge = min(abs(u_low), abs(width - u_high)) / max(1e-6, width)
    v_edge = min(abs(v_low), abs(height - v_high)) / max(1e-6, height)
    u_bonus = max(0.0, 1.0 - u_edge * 8.0) if u_span_ratio < 0.78 else 0.0
    v_bonus = max(0.0, 1.0 - v_edge * 8.0) if v_span_ratio < 0.78 else 0.0
    return max(u_bonus, v_bonus)


def _build_initial_rect(mask: np.ndarray, local_prior: np.ndarray) -> np.ndarray | None:
    prior_gate = _build_prior_gate_mask(mask.shape, local_prior)
    gated_mask = cv2.bitwise_and(mask, prior_gate)

    template_rect = _build_prior_template_rect(gated_mask, local_prior)
    if template_rect is not None:
        return template_rect

    union_rect = _build_component_union_rect(gated_mask, local_prior)
    if union_rect is None:
        return None
    return _refine_rect_from_color_bounds(gated_mask, union_rect)


def _build_prior_template_rect(
    mask: np.ndarray, local_prior: np.ndarray
) -> np.ndarray | None:
    """用先验矩形模板在颜色 mask 中找最佳平移位置。

    该步骤不要求颜色区域连通，适合油污遮挡导致的碎片化颜色支持。
    """

    if mask.size == 0 or np.count_nonzero(mask) < 6:
        return None

    min_xy = np.floor(np.min(local_prior, axis=0)).astype(int)
    max_xy = np.ceil(np.max(local_prior, axis=0)).astype(int)
    x0 = int(np.clip(min_xy[0] - 2, 0, mask.shape[1] - 1))
    y0 = int(np.clip(min_xy[1] - 2, 0, mask.shape[0] - 1))
    x1 = int(np.clip(max_xy[0] + 3, x0 + 2, mask.shape[1]))
    y1 = int(np.clip(max_xy[1] + 3, y0 + 2, mask.shape[0]))

    template = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    template_points = np.round(
        local_prior - np.array([x0, y0], dtype=np.float64)
    ).astype(np.int32)
    cv2.fillConvexPoly(template, template_points, 255)
    if np.count_nonzero(template) < 6:
        return None
    if template.shape[0] > mask.shape[0] or template.shape[1] > mask.shape[1]:
        return None

    source = (mask > 0).astype(np.float32)
    templ = (template > 0).astype(np.float32)
    response = cv2.matchTemplate(source, templ, cv2.TM_CCORR_NORMED)
    if response.size == 0:
        return None

    sx0 = 0
    sy0 = 0
    sx1 = response.shape[1]
    sy1 = response.shape[0]
    if sx1 <= sx0 or sy1 <= sy0:
        return None

    search_response = response[sy0:sy1, sx0:sx1]
    _, max_value, _, local_max_loc = cv2.minMaxLoc(search_response)
    if float(max_value) < 0.08:
        return None

    max_loc = (sx0 + int(local_max_loc[0]), sy0 + int(local_max_loc[1]))
    overlap = source[
        max_loc[1] : max_loc[1] + template.shape[0],
        max_loc[0] : max_loc[0] + template.shape[1],
    ]
    overlap_ratio = float(np.count_nonzero(overlap * templ)) / float(
        max(1, np.count_nonzero(templ))
    )
    if overlap_ratio < 0.06:
        return None

    shift = np.array([float(max_loc[0] - x0), float(max_loc[1] - y0)], dtype=np.float64)
    return local_prior.astype(np.float64) + shift


def _build_component_union_rect(
    mask: np.ndarray, local_prior: np.ndarray
) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    prior_center = np.mean(local_prior, axis=0)
    prior_gate = _build_prior_gate_mask(mask.shape, local_prior)
    selected_points: list[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 12.0:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        center = np.mean(points, axis=0)
        cx = int(np.clip(round(center[0]), 0, mask.shape[1] - 1))
        cy = int(np.clip(round(center[1]), 0, mask.shape[0] - 1))
        distance = float(np.linalg.norm(center - prior_center))
        if prior_gate[cy, cx] == 0 and distance > max(mask.shape) * 0.45:
            continue
        selected_points.append(points)

    if not selected_points:
        return None

    union_points = np.vstack(selected_points).astype(np.float32)
    if union_points.shape[0] < 6:
        return None
    rect = cv2.minAreaRect(union_points)
    box = cv2.boxPoints(rect).astype(np.float64)
    return _order_quad_points_array(box)


def _build_prior_gate_mask(
    image_shape: tuple[int, int], local_prior: np.ndarray
) -> np.ndarray:
    gate = np.zeros(image_shape, dtype=np.uint8)
    points = np.round(local_prior).astype(np.int32)
    cv2.fillConvexPoly(gate, points, 255)
    edge_lengths = np.linalg.norm(
        np.roll(local_prior, -1, axis=0) - local_prior, axis=1
    )
    dilate_px = int(round(max(6.0, float(np.min(edge_lengths)) * 0.65)))
    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=np.uint8)
    return cv2.dilate(gate, kernel, iterations=1)


def _refine_rect_from_color_bounds(mask: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """在颜色支持足够完整时，对矩形边界做小幅修正。

    半遮挡时颜色覆盖跨度不足，本函数会直接返回输入矩形，避免外接框被可见半边拖偏。
    """

    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        return rect

    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    ordered = _order_quad_points_array(rect)
    origin = ordered[0]
    u_axis = ordered[1] - ordered[0]
    v_axis = ordered[3] - ordered[0]
    width = float(np.linalg.norm(u_axis))
    height = float(np.linalg.norm(v_axis))
    if width < 1e-6 or height < 1e-6:
        return ordered

    u_unit = u_axis / width
    v_unit = v_axis / height
    rel = points - origin
    u = rel @ u_unit
    v = rel @ v_unit

    u_low, u_high = np.percentile(u, [5.0, 95.0])
    v_low, v_high = np.percentile(v, [5.0, 95.0])
    u_span = float(u_high - u_low)
    v_span = float(v_high - v_low)
    if u_span < width * 0.68 or v_span < height * 0.68:
        return ordered

    max_adjust = float(np.clip(min(width, height) * 0.14, 1.5, 3.0))
    left = float(np.clip(u_low, -max_adjust, max_adjust))
    right = float(np.clip(u_high, width - max_adjust, width + max_adjust))
    top = float(np.clip(v_low, -max_adjust, max_adjust))
    bottom = float(np.clip(v_high, height - max_adjust, height + max_adjust))
    if right - left < width * 0.72 or bottom - top < height * 0.72:
        return ordered

    refined = np.array(
        [
            origin + u_unit * left + v_unit * top,
            origin + u_unit * right + v_unit * top,
            origin + u_unit * right + v_unit * bottom,
            origin + u_unit * left + v_unit * bottom,
        ],
        dtype=np.float64,
    )
    return _order_quad_points_array(refined)


def _fit_quad_from_edges(
    edges: np.ndarray,
    initial_rect: np.ndarray,
    config: DetectionConfig,
) -> tuple[np.ndarray | None, float]:
    fitted_lines: list[tuple[float, float, float]] = []
    valid_sides = 0
    for side_index in range(4):
        p0 = initial_rect[side_index]
        p1 = initial_rect[(side_index + 1) % 4]
        points = _collect_edge_points_near_side(
            edges, p0, p1, float(config.edge_band_px)
        )
        if points.shape[0] < int(config.min_edge_points_per_side):
            fitted_lines.append(_line_from_two_points(p0, p1))
            continue
        fitted_lines.append(_fit_line(points))
        valid_sides += 1

    intersections: list[tuple[float, float]] = []
    for idx in range(4):
        inter = _intersect_lines(fitted_lines[idx], fitted_lines[(idx + 1) % 4])
        if inter is None:
            return None, float(valid_sides) / 4.0
        intersections.append(inter)

    quad = _order_quad_points_array(np.asarray(intersections, dtype=np.float64))
    if not _quad_inside_image(quad, edges.shape):
        return None, float(valid_sides) / 4.0
    return quad, float(valid_sides) / 4.0


def _collect_edge_points_near_side(
    edges: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    band_px: float,
) -> np.ndarray:
    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return np.empty((0, 2), dtype=np.float64)
    unit = vec / length
    rel = points - p0
    t = rel @ unit
    distance = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
    keep = (t >= -band_px) & (t <= length + band_px) & (distance <= band_px)
    return points[keep]


def _fit_line(points: np.ndarray) -> tuple[float, float, float]:
    line = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy, x0, y0 = [float(v) for v in line.reshape(4)]
    # ax + by + c = 0
    a = -vy
    b = vx
    c = vy * x0 - vx * y0
    norm = max(1e-9, float(np.hypot(a, b)))
    return a / norm, b / norm, c / norm


def _line_from_two_points(p0: np.ndarray, p1: np.ndarray) -> tuple[float, float, float]:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    norm = max(1e-9, float(np.hypot(a, b)))
    return a / norm, b / norm, c / norm


def _intersect_lines(
    line1: tuple[float, float, float],
    line2: tuple[float, float, float],
) -> tuple[float, float] | None:
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return float(x), float(y)


def _score_quad(
    quad: np.ndarray,
    marker: RectMarkerPrior,
    color_support: float,
    edge_support: float,
    config: DetectionConfig,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    area = abs(_polygon_area(quad))
    area_ratio = area / max(1.0, float(marker.expected_area_px))
    center_shift = float(
        np.linalg.norm(np.mean(quad, axis=0) - np.mean(marker.corners_px, axis=0))
    )
    angle = _quad_angle_deg(quad)
    angle_delta = abs(_normalize_angle_delta(angle - marker.expected_angle_deg))
    aspect = _quad_aspect_ratio(quad)
    aspect_delta = abs(
        np.log(max(1e-6, aspect) / max(1e-6, marker.expected_aspect_ratio))
    )

    if color_support < float(config.min_color_support):
        reasons.append("color_support_low")
    if area_ratio < marker.min_area_ratio or area_ratio > marker.max_area_ratio:
        reasons.append("area_ratio_out_of_range")
    if center_shift > marker.max_center_shift_px:
        reasons.append("center_shift_too_large")
    if angle_delta > marker.max_angle_delta_deg:
        reasons.append("angle_delta_too_large")
    if edge_support < 0.5:
        reasons.append("edge_support_low")

    center_score = max(0.0, 1.0 - center_shift / max(1.0, marker.max_center_shift_px))
    area_score = max(0.0, 1.0 - abs(area_ratio - 1.0) / 0.65)
    angle_score = max(0.0, 1.0 - angle_delta / max(1.0, marker.max_angle_delta_deg))
    aspect_score = max(0.0, 1.0 - aspect_delta / 0.6)
    color_score = min(
        1.0, color_support / max(1e-6, float(config.min_color_support) * 2.0)
    )
    edge_score = float(edge_support)

    score = (
        0.20 * center_score
        + 0.16 * area_score
        + 0.14 * angle_score
        + 0.12 * aspect_score
        + 0.20 * color_score
        + 0.18 * edge_score
    )
    return float(score), reasons


# endregion


# region 绘制
def _draw_results(
    image_bgr: np.ndarray,
    results: list[MarkerDetectionResult],
    elapsed_ms: float,
) -> np.ndarray:
    final_panel = _draw_final_panel(image_bgr, results, elapsed_ms)
    color_panel = _draw_color_stage_panel(image_bgr, results)
    initial_panel = _draw_initial_stage_panel(image_bgr, results)
    edge_panel = _draw_edge_stage_panel(image_bgr, results)
    return _compose_stage_grid(
        [
            ("Final", final_panel),
            ("Color Mask", color_panel),
            ("Initial Rect", initial_panel),
            ("Edge Fit", edge_panel),
        ]
    )


def _draw_final_panel(
    image_bgr: np.ndarray,
    results: list[MarkerDetectionResult],
    elapsed_ms: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    for result in results:
        x0, y0, x1, y1 = result.roi_rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (100, 100, 100), 1, cv2.LINE_AA)

        if result.corners_px is not None:
            color = (0, 255, 0) if result.detected else (0, 165, 255)
            pts = np.round(result.corners_px).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [pts],
                isClosed=True,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        if result.center_px is not None:
            _draw_center_cross(canvas, result.center_px)

        text = (
            f"{result.marker_id} {result.status} score={result.score:.2f} "
            f"color={result.color_support:.2f} visible={result.visible_area_ratio:.2f} "
            f"edge={result.edge_support:.2f}"
        )
        anchor = (max(4, x0), max(18, y0 - 8))
        cv2.putText(
            canvas,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"compute {elapsed_ms:.2f} ms",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"compute {elapsed_ms:.2f} ms",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_color_stage_panel(
    image_bgr: np.ndarray, results: list[MarkerDetectionResult]
) -> np.ndarray:
    canvas = np.zeros_like(image_bgr)
    canvas[:] = (25, 25, 25)
    for result in results:
        x0, y0, x1, y1 = result.roi_rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
        if result.clean_mask is not None:
            roi_vis = cv2.cvtColor(result.clean_mask, cv2.COLOR_GRAY2BGR)
            roi_vis[:, :, 1] = np.maximum(roi_vis[:, :, 1], result.clean_mask)
            canvas[y0:y1, x0:x1] = cv2.addWeighted(
                canvas[y0:y1, x0:x1], 0.35, roi_vis, 0.65, 0
            )
        _draw_stage_text(
            canvas,
            f"{result.marker_id} color={result.color_support:.2f}",
            (max(4, x0), max(18, y0 - 8)),
        )
    return canvas


def _draw_initial_stage_panel(
    image_bgr: np.ndarray, results: list[MarkerDetectionResult]
) -> np.ndarray:
    canvas = image_bgr.copy()
    for result in results:
        x0, y0, x1, y1 = result.roi_rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (100, 100, 100), 1, cv2.LINE_AA)
        if result.initial_rect is not None:
            pts = np.round(result.initial_rect).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [pts],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        _draw_stage_text(
            canvas, f"{result.marker_id} minAreaRect", (max(4, x0), max(18, y0 - 8))
        )
    return canvas


def _draw_edge_stage_panel(
    image_bgr: np.ndarray, results: list[MarkerDetectionResult]
) -> np.ndarray:
    canvas = np.zeros_like(image_bgr)
    for result in results:
        x0, y0, x1, y1 = result.roi_rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
        if result.edge_roi is not None:
            edge_vis = cv2.cvtColor(result.edge_roi, cv2.COLOR_GRAY2BGR)
            canvas[y0:y1, x0:x1] = edge_vis
        if result.edge_fit_rect is not None:
            pts = np.round(result.edge_fit_rect).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [pts],
                isClosed=True,
                color=(0, 165, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            edge_center = tuple(float(v) for v in np.mean(result.edge_fit_rect, axis=0))
            _draw_center_cross(canvas, edge_center)
        _draw_stage_text(
            canvas,
            f"{result.marker_id} edge={result.edge_support:.2f}",
            (max(4, x0), max(18, y0 - 8)),
        )
    return canvas


def _compose_stage_grid(stages: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if len(stages) != 4:
        raise ValueError("stage grid requires exactly 4 panels")

    panel_h = 360
    panel_w = 640
    panels: list[np.ndarray] = []
    for title, image in stages:
        resized = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        _draw_panel_title(resized, title)
        panels.append(resized)
    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    return np.vstack([top, bottom])


def _draw_panel_title(canvas: np.ndarray, title: str) -> None:
    cv2.putText(
        canvas,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_stage_text(canvas: np.ndarray, text: str, anchor: tuple[int, int]) -> None:
    cv2.putText(
        canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        text,
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_center_cross(canvas: np.ndarray, center: tuple[float, float]) -> None:
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    radius = 8
    black = (0, 0, 0)
    red = (0, 0, 255)
    cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), black, 3, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), black, 3, cv2.LINE_AA)
    cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), red, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), red, 1, cv2.LINE_AA)


# endregion


# region 几何工具
def _build_roi_rect(
    image_shape: tuple[int, ...],
    marker: RectMarkerPrior,
    config: DetectionConfig,
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    pts = marker.corners_px
    width = float(np.ptp(pts[:, 0]))
    height = float(np.ptp(pts[:, 1]))
    expand_x = int(round(width * 1.50 + config.roi_extra_px))
    expand_y = int(round(height * 1.50 + config.roi_extra_px))
    x0 = max(0, int(np.floor(np.min(pts[:, 0]) - expand_x)))
    y0 = max(0, int(np.floor(np.min(pts[:, 1]) - expand_y)))
    x1 = min(w, int(np.ceil(np.max(pts[:, 0]) + expand_x)))
    y1 = min(h, int(np.ceil(np.max(pts[:, 1]) + expand_y)))
    return x0, y0, x1, y1


def _order_quad_points_array(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    if _polygon_area(ordered) < 0:
        ordered = np.array(
            [ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float64
        )
    return ordered


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _quad_angle_deg(quad: np.ndarray) -> float:
    p0 = quad[0]
    p1 = quad[1]
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    width = float(0.5 * (lengths[0] + lengths[2]))
    height = float(0.5 * (lengths[1] + lengths[3]))
    return width / max(1e-6, height)


def _normalize_angle_delta(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def _quad_inside_image(quad: np.ndarray, image_shape: tuple[int, int]) -> bool:
    h, w = image_shape
    margin = 20.0
    return bool(
        np.all(quad[:, 0] >= -margin)
        and np.all(quad[:, 1] >= -margin)
        and np.all(quad[:, 0] <= w + margin)
        and np.all(quad[:, 1] <= h + margin)
    )


def _failed_result(
    marker_id: str, roi_rect: tuple[int, int, int, int], reason: str
) -> MarkerDetectionResult:
    return MarkerDetectionResult(
        marker_id=marker_id,
        detected=False,
        status="missing",
        score=0.0,
        center_px=None,
        corners_px=None,
        color_support=0.0,
        visible_area_ratio=0.0,
        edge_support=0.0,
        occluded=False,
        failure_reasons=[reason],
        roi_rect=roi_rect,
        initial_rect=None,
        edge_fit_rect=None,
        color_mask=None,
        clean_mask=None,
        edge_roi=None,
    )


def _missing_result(
    marker_id: str,
    roi_rect: tuple[int, int, int, int],
    color_support: float,
    visible_area_ratio: float,
    color_mask: np.ndarray,
    clean_mask: np.ndarray,
    edge_roi: np.ndarray,
    reason: str,
) -> MarkerDetectionResult:
    return MarkerDetectionResult(
        marker_id=marker_id,
        detected=False,
        status="missing",
        score=0.0,
        center_px=None,
        corners_px=None,
        color_support=float(color_support),
        visible_area_ratio=float(visible_area_ratio),
        edge_support=0.0,
        occluded=False,
        failure_reasons=[reason],
        roi_rect=roi_rect,
        initial_rect=None,
        edge_fit_rect=None,
        color_mask=color_mask,
        clean_mask=clean_mask,
        edge_roi=edge_roi,
    )


# endregion


# region 先验 IO
def _find_latest_prior(root: Path) -> Path:
    candidates = sorted(
        Path(root).glob("*/prior.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"未找到先验文件：{root}/*/prior.json")
    return candidates[0]


def _load_prior(path: Path) -> RectMarkerSetPrior:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    camera_data = data["camera"]
    camera = CameraCalibration(
        image_width=int(camera_data["image_width"]),
        image_height=int(camera_data["image_height"]),
        fx=float(camera_data["fx"]),
        fy=float(camera_data["fy"]),
        cx=float(camera_data["cx"]),
        cy=float(camera_data["cy"]),
        distortion=[float(v) for v in camera_data["distortion"]],
        distortion_model=str(camera_data["distortion_model"]),
        coordinate_space=str(camera_data["coordinate_space"]),
    )
    markers: list[RectMarkerPrior] = []
    for item in data["markers"]:
        markers.append(
            RectMarkerPrior(
                marker_id=str(item["marker_id"]),
                corners_px=np.asarray(item["corners_px"], dtype=np.float64).reshape(
                    4, 2
                ),
                rgb_prior=np.asarray(item["rgb_prior"], dtype=np.uint8).reshape(3),
                expected_area_px=float(item["expected_area_px"]),
                expected_angle_deg=float(item["expected_angle_deg"]),
                expected_aspect_ratio=float(item["expected_aspect_ratio"]),
                max_center_shift_px=float(item["max_center_shift_px"]),
                max_angle_delta_deg=float(item["max_angle_delta_deg"]),
                min_area_ratio=float(item["min_area_ratio"]),
                max_area_ratio=float(item["max_area_ratio"]),
                roi_expand_px=int(item["roi_expand_px"]),
                roi_expand_ratio=float(item["roi_expand_ratio"]),
            )
        )
    return RectMarkerSetPrior(source_path=Path(path), camera=camera, markers=markers)


def _build_camera_matrix(camera: CameraCalibration) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        [
            [float(camera.fx), 0.0, float(camera.cx)],
            [0.0, float(camera.fy), float(camera.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.asarray(camera.distortion, dtype=np.float64)
    return matrix, dist


# endregion


# region Orbbec 帧解码
def _decode_color_frame_bgr(color_frame) -> np.ndarray | None:
    if color_frame is None:
        return None

    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if width <= 0 or height <= 0 or data.size == 0:
        return None

    if color_format == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3)).copy()
    if color_format in (OBFormat.YUYV, OBFormat.YUY2):
        yuy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if color_format == OBFormat.I420:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    logger.warning(f"当前 color format 暂未支持直接预览：{color_format}")
    return None


# endregion


# region CLI
def _parse_cli() -> tuple[Path | None, int, int, DetectionConfig]:
    parser = argparse.ArgumentParser(description="矩形标记件实时识别实验脚本")
    parser.add_argument(
        "--prior", type=Path, default=None, help="prior.json 路径；默认读取最新采集结果"
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="wait_for_frames 超时时间（ms）",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=DEFAULT_CAPTURE_FPS,
        help="期望采样帧率（fps）",
    )
    parser.add_argument(
        "--roi-extra-px",
        type=int,
        default=DEFAULT_ROI_EXTRA_PX,
        help="额外 ROI 外扩像素",
    )
    parser.add_argument(
        "--lab-threshold",
        type=float,
        default=DEFAULT_LAB_THRESHOLD,
        help="Lab 颜色距离阈值",
    )
    parser.add_argument(
        "--hue-threshold",
        type=float,
        default=DEFAULT_HUE_THRESHOLD,
        help="HSV Hue 阈值",
    )
    parser.add_argument(
        "--min-color-support",
        type=float,
        default=DEFAULT_MIN_COLOR_SUPPORT,
        help="最小颜色支持率",
    )
    parser.add_argument(
        "--min-present-visible-ratio",
        type=float,
        default=DEFAULT_MIN_PRESENT_VISIBLE_RATIO,
        help="判定标记件存在所需的最低先验区域可见颜色比例",
    )
    parser.add_argument(
        "--occluded-visible-ratio",
        type=float,
        default=DEFAULT_OCCLUDED_VISIBLE_RATIO,
        help="低于该先验区域可见颜色比例时进入遮挡模式",
    )
    parser.add_argument(
        "--edge-band-px",
        type=int,
        default=DEFAULT_EDGE_BAND_PX,
        help="四边边缘搜索带宽",
    )
    parser.add_argument(
        "--canny-low", type=int, default=DEFAULT_CANNY_LOW, help="Canny 低阈值"
    )
    parser.add_argument(
        "--canny-high", type=int, default=DEFAULT_CANNY_HIGH, help="Canny 高阈值"
    )
    parser.add_argument(
        "--min-score", type=float, default=DEFAULT_MIN_SCORE, help="最终通过分数阈值"
    )
    args = parser.parse_args()
    config = DetectionConfig(
        roi_extra_px=int(args.roi_extra_px),
        lab_threshold=float(args.lab_threshold),
        hue_threshold=float(args.hue_threshold),
        min_color_support=float(args.min_color_support),
        min_present_visible_ratio=float(args.min_present_visible_ratio),
        occluded_visible_ratio=float(args.occluded_visible_ratio),
        edge_band_px=int(args.edge_band_px),
        canny_low=int(args.canny_low),
        canny_high=int(args.canny_high),
        min_edge_points_per_side=DEFAULT_MIN_EDGE_POINTS_PER_SIDE,
        min_score=float(args.min_score),
    )
    return args.prior, int(args.timeout_ms), int(args.capture_fps), config


# endregion


if __name__ == "__main__":
    try:
        prior_arg, timeout_arg, fps_arg, config_arg = _parse_cli()
        main(
            prior_path=prior_arg,
            timeout_ms=timeout_arg,
            capture_fps=fps_arg,
            config=config_arg,
        )
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
