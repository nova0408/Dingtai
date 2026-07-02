from __future__ import annotations

"""
脚本流程概览：
1. 通过 Orin 的 `camera_pipeline_service` 订阅 1280x800 彩色流，并执行相机畸变校正。
2. 对同一帧仅生成 CLAHE 与 HoughCompare 两个增强视图。
3. 对候选四边形做透视校正、模板采样和 AprilTag 16h5 位解码，再结合重投影误差计算单帧评分。
4. 将各增强分支的单帧结果做空间去重融合，得到单帧 Fusion。
5. 对最近 1 秒内的 Fusion 做时序投票，只保留至少 3 次一致的稳定实例，得到 TemporalFusion。
6. 非交互式持续运行，直到 TemporalFusion 中稳定识别出先验 tag 3、4、5；随后输出 CSV、截图和最终结果图到 `.archive` 用于离线分析。
详细说明见：test_apriltag_color_space_eval.md
"""

import argparse
import csv
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient  # noqa: E402

# region 默认参数（优先在这里直接改）
DEFAULT_WINDOW_NAME = "AprilTag Color Eval"
DEFAULT_WINDOW_MIN_LONG_SIDE = 1280
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_ORIN_SERVICE_ADDR = "tcp://192.168.1.118:6200"
DEFAULT_COLOR_WIDTH = 1280
DEFAULT_COLOR_HEIGHT = 800
DEFAULT_COLOR_FORMAT_NAME = "BGR"
DEFAULT_TAG_SIZE_MM = 40.0
DEFAULT_DICTIONARY_NAME = "DICT_APRILTAG_16H5"
DEFAULT_OUTPUT_ROOT = Path("test/wuji/.archive/apriltag_detect_runs")
DEFAULT_CLAHE_CLIP_LIMIT = 2.5
DEFAULT_CLAHE_GRID = 8
DEFAULT_TEMPLATE_SIZE_PX = 120
DEFAULT_ALLOWED_TAG_IDS = tuple(range(7))
DEFAULT_TEMPORAL_WINDOW_S = 1.0
DEFAULT_TEMPORAL_MIN_SUPPORT = 3
DEFAULT_TARGET_TAG_IDS = (3, 4, 5)
DEFAULT_TARGET_STABLE_COUNT = 3
DEFAULT_MAX_FRAMES = 100
# endregion


# region 数据结构
@dataclass(frozen=True)
class TagSpec:
    """单个 AprilTag 的物理标签信息。"""

    tag_id: int
    label: str
    foreground: str
    background: str
    border: str
    note: str


@dataclass(frozen=True)
class CameraCalibration:
    """Wuji/ZMQ 彩色相机内参与畸变。"""

    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True)
class TemplateBank:
    """AprilTag 字典模板库。"""

    tag_ids: np.ndarray
    flat_templates: np.ndarray
    marker_size: int
    total_cells: int


@dataclass(frozen=True)
class VariantFrame:
    """单个图像变换视图。"""

    name: str
    detect_images: list[np.ndarray]
    candidate_mode: str
    preview_image: np.ndarray
    edge_image: np.ndarray
    source_bgr: np.ndarray


@dataclass(frozen=True)
class VariantDetections:
    """单个视图下的检测结果和被拒候选。"""

    results: list[DetectionResult]
    rejected_corners: list[np.ndarray]


@dataclass(frozen=True)
class DetectionResult:
    """单个标签在某个色彩空间下的识别与位姿结果。"""

    variant_name: str
    detection_index: int
    tag_id: int
    label: str
    color_signature: str
    detected: bool
    score: float
    template_score: float
    reprojection_error_px: float
    reprojection_score: float
    size_score: float
    perimeter_px: float
    corners_px: np.ndarray | None
    axis_points_px: np.ndarray | None
    rvec: np.ndarray | None
    tvec_mm: np.ndarray | None
    rpy_deg: tuple[float, float, float] | None


@dataclass(frozen=True)
class AppConfig:
    """脚本运行配置。"""

    dictionary_name: str
    tag_size_mm: float
    timeout_ms: int
    capture_fps: int
    output_root: Path
    tag_spec_path: Path | None
    allowed_tag_ids: tuple[int, ...]
    clahe_clip_limit: float
    clahe_grid: int
    max_frames: int


@dataclass
class CaptureRow:
    """一次姿态采样中的单条明细。"""

    pose_index: int
    frame_index: int
    timestamp_s: float
    variant_name: str
    detection_index: int
    tag_id: int
    label: str
    color_signature: str
    detected: bool
    score: float
    template_score: float
    reprojection_error_px: float
    tx_mm: float | None
    ty_mm: float | None
    tz_mm: float | None
    roll_deg: float | None
    pitch_deg: float | None
    yaw_deg: float | None


# endregion


# region 主流程
def main(config: AppConfig) -> None:
    _validate_runtime_requirements()
    session_dir = _create_session_dir(config.output_root)
    tag_specs = _load_tag_specs(config.tag_spec_path)
    dictionary = _get_apriltag_dictionary(config.dictionary_name)
    template_bank = _build_template_bank(
        dictionary,
        tag_specs,
        allowed_tag_ids=config.allowed_tag_ids,
    )
    capture_rows: list[CaptureRow] = []
    temporal_fusion_history: deque[tuple[float, list[DetectionResult]]] = deque()
    final_state: dict[str, object] = {
        "session_dir": str(session_dir),
        "target_tag_ids": list(DEFAULT_TARGET_TAG_IDS),
        "target_stable_count": int(DEFAULT_TARGET_STABLE_COUNT),
        "frames": [],
        "final_result": None,
    }
    final_preview: np.ndarray | None = None

    client = CameraPipelineClient(service_addr=DEFAULT_ORIN_SERVICE_ADDR, timeout_ms=30_000)
    try:
        summary_response = client.get_camera_summary(timeout_s=float(config.timeout_ms) / 1000.0)
        status_response = client.get_camera_status(timeout_s=float(config.timeout_ms) / 1000.0)
        intrinsics_response = client.get_camera_intrinsics(timeout_s=float(config.timeout_ms) / 1000.0)
        calibration = _read_camera_calibration(intrinsics_response)
        if str(status_response.camera_name) != str(DEFAULT_CAMERA_NAME):
            logger.warning(
                "相机状态返回的 camera_name={} 与脚本默认值 {} 不一致，将继续使用订阅相机名 {}",
                status_response.camera_name,
                DEFAULT_CAMERA_NAME,
                DEFAULT_CAMERA_NAME,
            )
        if int(status_response.width) != int(calibration.width) or int(status_response.height) != int(calibration.height):
            logger.warning(
                "相机状态分辨率 {}x{} 与内参分辨率 {}x{} 不一致，以内参为准",
                int(status_response.width),
                int(status_response.height),
                int(calibration.width),
                int(calibration.height),
            )
        logger.info(
            "AprilTag eval Orin stream target: "
            f"{DEFAULT_COLOR_WIDTH}x{DEFAULT_COLOR_HEIGHT} {DEFAULT_COLOR_FORMAT_NAME}, "
            f"actual calibration={calibration.width}x{calibration.height}, "
            f"camera_status={status_response.camera_model}/{status_response.width}x{status_response.height}, "
            f"source_meta={summary_response.source_meta}"
        )
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        frame_index = 0
        for frame in client.subscribe_camera_frames(DEFAULT_CAMERA_NAME):
            color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
            if color_bgr.size == 0:
                logger.warning("Orin camera_frame 返回空图像，跳过本帧")
                continue
            if str(frame.camera_name) != str(DEFAULT_CAMERA_NAME):
                logger.warning(
                    "订阅流返回 camera_name={}，期望 {}，仍继续处理该帧",
                    frame.camera_name,
                    DEFAULT_CAMERA_NAME,
                )

            frame_index += 1
            undistorted_bgr = cv2.undistort(
                color_bgr,
                calibration.camera_matrix,
                calibration.dist_coeffs,
            )
            variant_frames = _build_variant_frames(
                undistorted_bgr=undistorted_bgr,
                clip_limit=config.clahe_clip_limit,
                clahe_grid=config.clahe_grid,
            )
            started = time.perf_counter()
            frame_results = _evaluate_frame(
                variant_frames=variant_frames,
                calibration=calibration,
                dictionary=dictionary,
                template_bank=template_bank,
                tag_specs=tag_specs,
                tag_size_mm=config.tag_size_mm,
            )
            temporal_fusion_history.append(
                (
                    time.perf_counter(),
                    list(frame_results.get("Fusion", VariantDetections([], [])).results),
                )
            )
            _prune_temporal_history(
                temporal_fusion_history=temporal_fusion_history,
                window_s=DEFAULT_TEMPORAL_WINDOW_S,
            )
            frame_results["TemporalFusion"] = _fuse_temporal_detections(
                fusion_history=list(temporal_fusion_history),
                window_s=DEFAULT_TEMPORAL_WINDOW_S,
                min_support=DEFAULT_TEMPORAL_MIN_SUPPORT,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            preview = _compose_preview(
                variant_frames=variant_frames,
                frame_results=frame_results,
                frame_index=frame_index,
                elapsed_ms=elapsed_ms,
                session_dir=session_dir,
            )
            cv2.imshow(DEFAULT_WINDOW_NAME, preview)
            cv2.waitKey(1)
            _save_frame_artifacts(
                session_dir=session_dir,
                frame_index=frame_index,
                preview_image=preview,
                variant_frames=variant_frames,
                frame_results=frame_results,
            )
            _capture_pose(
                pose_index=frame_index,
                frame_index=frame_index,
                frame_results=frame_results,
                rows=capture_rows,
                session_dir=session_dir,
                preview_image=preview,
            )
            final_state["frames"].append(
                {
                    "frame_index": int(frame_index),
                    "elapsed_ms": float(elapsed_ms),
                    "temporal_tag_ids": [int(item.tag_id) for item in frame_results.get("TemporalFusion", VariantDetections([], [])).results],
                }
            )
            if _has_stable_target_tags(frame_results.get("TemporalFusion", VariantDetections([], [])).results, DEFAULT_TARGET_TAG_IDS):
                logger.success("已稳定识别到目标 tag 3,4,5，结束采集。")
                final_state["final_result"] = _build_final_result_payload(frame_index, elapsed_ms, frame_results)
                final_preview = _annotate_final_preview(preview, frame_results["TemporalFusion"].results)
                _write_outputs(session_dir=session_dir, rows=capture_rows, final_state=final_state)
                cv2.imshow(DEFAULT_WINDOW_NAME, final_preview)
                cv2.waitKey(5000)
                break
            if int(config.max_frames) > 0 and frame_index >= int(config.max_frames):
                logger.warning("达到最大帧数 {}，停止采集。", int(config.max_frames))
                final_state["final_result"] = _build_final_result_payload(frame_index, elapsed_ms, frame_results)
                final_preview = _annotate_final_preview(preview, frame_results["TemporalFusion"].results)
                _write_outputs(session_dir=session_dir, rows=capture_rows, final_state=final_state)
                cv2.imshow(DEFAULT_WINDOW_NAME, final_preview)
                cv2.waitKey(5000)
                break
    finally:
        client.close()
    if final_preview is not None:
        cv2.imwrite(str(session_dir / "final_preview.png"), final_preview)
    cv2.destroyAllWindows()
    if final_preview is None:
        _write_outputs(session_dir=session_dir, rows=capture_rows, final_state=final_state)
    logger.success(f"评估结果输出目录：{session_dir}")


# endregion


# region AprilTag 评估
def _evaluate_frame(
    variant_frames: list[VariantFrame],
    calibration: CameraCalibration,
    dictionary,
    template_bank: TemplateBank,
    tag_specs: dict[int, TagSpec],
    tag_size_mm: float,
) -> dict[str, VariantDetections]:
    frame_results: dict[str, VariantDetections] = {}
    for variant_frame in variant_frames:
        variant_results: list[DetectionResult] = []
        corners_list, ids, rejected = _detect_candidates_and_decode_multi_input(
            detect_images=variant_frame.detect_images,
            template_bank=template_bank,
            candidate_mode=variant_frame.candidate_mode,
        )
        primary_image = variant_frame.detect_images[0]
        if ids is None or len(ids) == 0:
            frame_results[variant_frame.name] = VariantDetections(
                results=variant_results,
                rejected_corners=_normalize_rejected_corners(rejected),
            )
            continue
        corner_items = [] if corners_list is None else list(corners_list)
        for detection_index, (corners, tag_id) in enumerate(
            zip(corner_items, ids.flatten(), strict=True),
            start=1,
        ):
            corner_array = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            pose = _estimate_pose(
                corners_px=corner_array,
                calibration=calibration,
                tag_size_mm=tag_size_mm,
            )
            template_score = _compute_template_score(
                detect_image=primary_image,
                corners_px=corner_array,
                dictionary=dictionary,
                tag_id=int(tag_id),
            )
            perimeter_px = float(cv2.arcLength(corner_array.reshape(-1, 1, 2), True))
            size_score = _compute_size_score(
                perimeter_px=perimeter_px,
                image_shape=primary_image.shape[:2],
            )
            reprojection_error_px = pose[2]
            reprojection_score = _compute_reprojection_score(reprojection_error_px)
            score = float(0.55 * template_score + 0.25 * reprojection_score + 0.20 * size_score)
            spec = tag_specs.get(int(tag_id)) or TagSpec(
                tag_id=int(tag_id),
                label=f"tag_{int(tag_id)}",
                foreground="",
                background="",
                border="",
                note="",
            )
            color_signature = _extract_color_signature(
                source_bgr=variant_frame.source_bgr,
                corners_px=corner_array,
            )
            variant_results.append(
                DetectionResult(
                    variant_name=variant_frame.name,
                    detection_index=detection_index,
                    tag_id=int(tag_id),
                    label=spec.label,
                    color_signature=color_signature,
                    detected=True,
                    score=score,
                    template_score=template_score,
                    reprojection_error_px=reprojection_error_px,
                    reprojection_score=reprojection_score,
                    size_score=size_score,
                    perimeter_px=perimeter_px,
                    corners_px=corner_array,
                    axis_points_px=pose[4],
                    rvec=pose[0],
                    tvec_mm=pose[1],
                    rpy_deg=pose[3],
                )
            )
        frame_results[variant_frame.name] = VariantDetections(
            results=variant_results,
            rejected_corners=_normalize_rejected_corners(rejected),
        )
    frame_results["Fusion"] = _fuse_variant_detections(frame_results)
    return frame_results


def _estimate_pose(
    corners_px: np.ndarray,
    calibration: CameraCalibration,
    tag_size_mm: float,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    float,
    tuple[float, float, float] | None,
    np.ndarray | None,
]:
    half = float(tag_size_mm) * 0.5
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners_px,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, None, 999.0, None, None

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    projected_2d = projected.reshape(-1, 2)
    reprojection_error_px = float(np.mean(np.linalg.norm(projected_2d - corners_px, axis=1)))
    axis_length = half * 0.8
    axis_object_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, -axis_length],
        ],
        dtype=np.float32,
    )
    axis_projected, _ = cv2.projectPoints(
        axis_object_points,
        rvec,
        tvec,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    rpy_deg = _rotation_vector_to_rpy_deg(rvec)
    return (
        rvec,
        tvec.reshape(3),
        reprojection_error_px,
        rpy_deg,
        axis_projected.reshape(-1, 2),
    )


def _compute_template_score(
    detect_image: np.ndarray,
    corners_px: np.ndarray,
    dictionary,
    tag_id: int,
) -> float:
    ideal = cv2.aruco.generateImageMarker(
        dictionary,
        int(tag_id),
        DEFAULT_TEMPLATE_SIZE_PX,
        borderBits=1,
    )
    gray = _ensure_gray_image(detect_image)
    dst = np.array(
        [
            [0.0, 0.0],
            [DEFAULT_TEMPLATE_SIZE_PX - 1.0, 0.0],
            [DEFAULT_TEMPLATE_SIZE_PX - 1.0, DEFAULT_TEMPLATE_SIZE_PX - 1.0],
            [0.0, DEFAULT_TEMPLATE_SIZE_PX - 1.0],
        ],
        dtype=np.float32,
    )
    warp_matrix = cv2.getPerspectiveTransform(
        corners_px.astype(np.float32),
        dst,
    )
    warped = cv2.warpPerspective(
        gray,
        warp_matrix,
        (DEFAULT_TEMPLATE_SIZE_PX, DEFAULT_TEMPLATE_SIZE_PX),
        flags=cv2.INTER_LINEAR,
    )
    warped_f = warped.astype(np.float32) / 255.0
    ideal_f = ideal.astype(np.float32) / 255.0
    inv_ideal_f = 1.0 - ideal_f
    corr_a = _safe_normalized_correlation(warped_f, ideal_f)
    corr_b = _safe_normalized_correlation(warped_f, inv_ideal_f)
    return float(np.clip(max(corr_a, corr_b), 0.0, 1.0))


def _safe_normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 1e-6:
        return 0.0
    corr = float(np.sum(a_centered * b_centered) / denom)
    return (corr + 1.0) * 0.5


def _compute_reprojection_score(reprojection_error_px: float) -> float:
    if reprojection_error_px >= 999.0:
        return 0.0
    return float(np.exp(-reprojection_error_px / 2.5))


def _compute_size_score(
    perimeter_px: float,
    image_shape: tuple[int, int],
) -> float:
    image_diag = float(np.hypot(image_shape[1], image_shape[0]))
    if image_diag <= 1e-6:
        return 0.0
    normalized = perimeter_px / image_diag
    return float(np.clip(normalized / 0.35, 0.0, 1.0))


# endregion


# region 多色彩空间预览
def _build_variant_frames(
    undistorted_bgr: np.ndarray,
    clip_limit: float,
    clahe_grid: int,
) -> list[VariantFrame]:
    gray = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(clahe_grid), int(clahe_grid)),
    )
    contrast = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=40.0, sigmaSpace=40.0)
    blur_small = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    blur_large = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
    highpass = cv2.normalize(
        cv2.subtract(blur_small, blur_large),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    canny_contrast = cv2.Canny(contrast, 50, 150, apertureSize=3, L2gradient=True)
    hough_vis = _build_hough_preview(contrast, canny_contrast)

    return [
        VariantFrame(
            name="CLAHE",
            detect_images=_build_detect_inputs(contrast, bilateral, highpass),
            candidate_mode="canny",
            preview_image=_gray_to_bgr(contrast),
            edge_image=_gray_to_bgr(canny_contrast),
            source_bgr=undistorted_bgr,
        ),
        VariantFrame(
            name="HoughCompare",
            detect_images=_build_detect_inputs(contrast, bilateral, highpass),
            candidate_mode="canny",
            preview_image=hough_vis,
            edge_image=hough_vis,
            source_bgr=undistorted_bgr,
        ),
    ]


def _compose_preview(
    variant_frames: list[VariantFrame],
    frame_results: dict[str, VariantDetections],
    frame_index: int,
    elapsed_ms: float,
    session_dir: Path,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    for variant_frame in variant_frames:
        panel = variant_frame.preview_image.copy()
        detections = frame_results.get(
            variant_frame.name,
            VariantDetections(results=[], rejected_corners=[]),
        )
        _draw_detection_overlays(panel, detections, draw_pose=False)
        _draw_panel_header(
            panel=panel,
            title=variant_frame.name,
            subtitle=_build_variant_subtitle(detections),
        )
        panels.append(panel)

    fusion_panel = variant_frames[0].source_bgr.copy()
    fusion_detections = frame_results.get(
        "TemporalFusion",
        VariantDetections(results=[], rejected_corners=[]),
    )
    _draw_detection_overlays(fusion_panel, fusion_detections, draw_pose=True)
    _draw_panel_header(
        panel=fusion_panel,
        title="TemporalFusion",
        subtitle=_build_fusion_subtitle(frame_results, fusion_detections),
    )
    panels.append(fusion_panel)

    grid = _compose_panel_grid(panels, columns=4)
    footer_lines = [
        f"frame={frame_index} compute_ms={elapsed_ms:.2f}",
        "non-interactive capture until stable tags 3,4,5",
        f"output={session_dir.name}",
    ]
    return _append_footer(grid, footer_lines)


def _annotate_final_preview(preview: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    canvas = preview.copy()
    _draw_text(canvas, "FINAL STABLE TAGS: 3 / 4 / 5", (18, 34), scale=_panel_text_scale(canvas, 1.0))
    if not detections:
        _draw_text(canvas, "no stable targets detected", (18, 72), scale=_panel_text_scale(canvas, 0.85))
        return canvas
    for index, result in enumerate(detections[:3]):
        if result.corners_px is None:
            continue
        polygon = np.round(result.corners_px).astype(np.int32).reshape(-1, 1, 2)
        color = (0, 220, 0)
        cv2.polylines(canvas, [polygon], True, color, 3, cv2.LINE_AA)
        center = tuple(int(v) for v in np.round(np.mean(result.corners_px, axis=0)))
        _draw_text(canvas, f"target {result.tag_id}", (center[0] + 8, center[1] - 8), scale=_panel_text_scale(canvas, 0.75))
    return canvas


def _draw_detection_overlays(
    canvas: np.ndarray,
    detections: VariantDetections,
    draw_pose: bool,
) -> None:
    for rejected in detections.rejected_corners:
        polygon = np.round(rejected).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            canvas,
            [polygon],
            isClosed=True,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    for result in detections.results:
        if not result.detected or result.corners_px is None:
            continue
        polygon = np.round(result.corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            canvas,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        center = np.mean(result.corners_px, axis=0)
        anchor = (int(round(center[0])) + 6, int(round(center[1])) - 6)
        text = f"id={result.tag_id}"
        _draw_text(canvas, text, anchor, scale=_panel_text_scale(canvas, 0.95))
        if draw_pose:
            _draw_pose_axes(canvas, result)


def _draw_pose_axes(
    canvas: np.ndarray,
    result: DetectionResult,
) -> None:
    if result.axis_points_px is None or result.axis_points_px.shape[0] < 4:
        return
    axis_points = np.round(result.axis_points_px).astype(np.int32)
    origin = tuple(int(v) for v in axis_points[0])
    x_axis = tuple(int(v) for v in axis_points[1])
    y_axis = tuple(int(v) for v in axis_points[2])
    z_axis = tuple(int(v) for v in axis_points[3])
    cv2.line(canvas, origin, x_axis, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, origin, y_axis, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, origin, z_axis, (255, 0, 0), 2, cv2.LINE_AA)


def _draw_panel_header(panel: np.ndarray, title: str, subtitle: str) -> None:
    _draw_text(panel, title, (18, 34), scale=_panel_text_scale(panel, 1.10))
    _draw_text(panel, subtitle, (18, 66), scale=_panel_text_scale(panel, 0.78))


def _build_variant_subtitle(detections: VariantDetections) -> str:
    accepted = len(detections.results)
    rejected = len(detections.rejected_corners)
    if accepted == 0:
        return f"accepted=0 rejected={rejected}"
    ids_text = ",".join(str(item.tag_id) for item in detections.results[:3])
    return f"accepted={accepted} rejected={rejected} ids={ids_text}"


def _build_fusion_subtitle(
    frame_results: dict[str, VariantDetections],
    detections: VariantDetections,
) -> str:
    raw_count = sum(
        len(variant_detections.results)
        for variant_name, variant_detections in frame_results.items()
        if variant_name not in {"Fusion", "TemporalFusion"}
    )
    fused_ids = ",".join(str(item.tag_id) for item in detections.results[:4])
    return f"raw={raw_count} fused={len(detections.results)} ids={fused_ids}"


def _compose_panel_grid(panels: list[np.ndarray], columns: int) -> np.ndarray:
    if not panels:
        raise ValueError("panels must not be empty")
    rows = math.ceil(len(panels) / max(1, columns))
    blank = np.zeros_like(panels[0])
    row_images: list[np.ndarray] = []
    panel_index = 0
    for _ in range(rows):
        row_panels: list[np.ndarray] = []
        for _ in range(columns):
            if panel_index < len(panels):
                row_panels.append(panels[panel_index])
            else:
                row_panels.append(blank.copy())
            panel_index += 1
        row_images.append(np.hstack(row_panels))
    return np.vstack(row_images)


def _append_footer(image: np.ndarray, lines: list[str]) -> np.ndarray:
    footer_h = 96
    footer = np.zeros((footer_h, image.shape[1], 3), dtype=np.uint8)
    footer[:] = (20, 20, 20)
    for idx, line in enumerate(lines):
        _draw_text(footer, line, (16, 28 + idx * 28), scale=0.82)
    return np.vstack([image, footer])


# endregion


# region 采样与汇总
def _capture_pose(
    pose_index: int,
    frame_index: int,
    frame_results: dict[str, VariantDetections],
    rows: list[CaptureRow],
    session_dir: Path,
    preview_image: np.ndarray,
) -> None:
    timestamp_s = time.time()
    for variant_name, detections in frame_results.items():
        for result in detections.results:
            tx_mm = None if result.tvec_mm is None else float(result.tvec_mm[0])
            ty_mm = None if result.tvec_mm is None else float(result.tvec_mm[1])
            tz_mm = None if result.tvec_mm is None else float(result.tvec_mm[2])
            roll_deg = None if result.rpy_deg is None else float(result.rpy_deg[0])
            pitch_deg = None if result.rpy_deg is None else float(result.rpy_deg[1])
            yaw_deg = None if result.rpy_deg is None else float(result.rpy_deg[2])
            rows.append(
                CaptureRow(
                    pose_index=pose_index,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    variant_name=variant_name,
                    detection_index=result.detection_index,
                    tag_id=result.tag_id,
                    label=result.label,
                    color_signature=result.color_signature,
                    detected=result.detected,
                    score=float(result.score),
                    template_score=float(result.template_score),
                    reprojection_error_px=float(result.reprojection_error_px),
                    tx_mm=tx_mm,
                    ty_mm=ty_mm,
                    tz_mm=tz_mm,
                    roll_deg=roll_deg,
                    pitch_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                )
            )
    image_path = session_dir / f"pose_{pose_index:03d}_frame_{frame_index:06d}.png"
    cv2.imwrite(str(image_path), preview_image)
    logger.info(f"已记录 pose={pose_index} frame={frame_index} -> {image_path.name}")


def _save_frame_artifacts(
    session_dir: Path,
    frame_index: int,
    preview_image: np.ndarray,
    variant_frames: list[VariantFrame],
    frame_results: dict[str, VariantDetections],
) -> None:
    frame_dir = session_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_dir / "preview.png"), preview_image)
    for variant_frame in variant_frames:
        cv2.imwrite(str(frame_dir / f"{variant_frame.name}.png"), variant_frame.preview_image)
    summary_path = frame_dir / "summary.json"
    summary_payload = {
        "frame_index": int(frame_index),
        "variant_summary": {
            name: [int(item.tag_id) for item in detections.results]
            for name, detections in frame_results.items()
        },
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_final_result_payload(
    frame_index: int,
    elapsed_ms: float,
    frame_results: dict[str, VariantDetections],
) -> dict[str, object]:
    temporal = frame_results.get("TemporalFusion", VariantDetections(results=[], rejected_corners=[]))
    return {
        "frame_index": int(frame_index),
        "elapsed_ms": float(elapsed_ms),
        "temporal_tag_ids": [int(item.tag_id) for item in temporal.results],
        "temporal_labels": [str(item.label) for item in temporal.results],
        "temporal_scores": [float(item.score) for item in temporal.results],
    }


def _has_stable_target_tags(results: list[DetectionResult], target_tag_ids: tuple[int, ...]) -> bool:
    detected_ids = {int(item.tag_id) for item in results if item.detected}
    return set(int(tag_id) for tag_id in target_tag_ids).issubset(detected_ids)


def _write_outputs(session_dir: Path, rows: list[CaptureRow], final_state: dict[str, object]) -> None:
    detail_path = session_dir / "pose_samples.csv"
    _write_pose_rows(detail_path, rows)
    _write_summary_by_tag(session_dir / "summary_by_tag.csv", rows)
    _write_summary_by_variant(session_dir / "summary_by_variant.csv", rows)
    (session_dir / "final_state.json").write_text(
        json.dumps(final_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_pose_rows(path: Path, rows: list[CaptureRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pose_index",
                "frame_index",
                "timestamp_s",
                "variant_name",
                "detection_index",
                "tag_id",
                "label",
                "color_signature",
                "detected",
                "score",
                "template_score",
                "reprojection_error_px",
                "tx_mm",
                "ty_mm",
                "tz_mm",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.pose_index,
                    row.frame_index,
                    f"{row.timestamp_s:.6f}",
                    row.variant_name,
                    row.detection_index,
                    row.tag_id,
                    row.label,
                    row.color_signature,
                    int(row.detected),
                    f"{row.score:.6f}",
                    f"{row.template_score:.6f}",
                    f"{row.reprojection_error_px:.6f}",
                    _fmt_optional(row.tx_mm),
                    _fmt_optional(row.ty_mm),
                    _fmt_optional(row.tz_mm),
                    _fmt_optional(row.roll_deg),
                    _fmt_optional(row.pitch_deg),
                    _fmt_optional(row.yaw_deg),
                ]
            )


def _write_summary_by_tag(path: Path, rows: list[CaptureRow]) -> None:
    grouped: dict[str, list[CaptureRow]] = {}
    for row in rows:
        group_key = f"{row.label}|{row.color_signature}"
        grouped.setdefault(group_key, []).append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "color_signature",
                "tag_id",
                "pose_count",
                "detected_pose_count",
                "detected_pose_rate",
                "mean_best_score",
                "median_best_score",
                "best_variant_mode",
            ]
        )
        for group_key, label_rows in sorted(grouped.items()):
            rows_by_pose: dict[int, list[CaptureRow]] = {}
            for row in label_rows:
                rows_by_pose.setdefault(row.pose_index, []).append(row)

            best_scores: list[float] = []
            best_variants: list[str] = []
            detected_pose_count = 0
            label = label_rows[0].label
            color_signature = label_rows[0].color_signature
            tag_id = label_rows[0].tag_id
            for pose_rows in rows_by_pose.values():
                if not pose_rows:
                    continue
                best_row = max(pose_rows, key=lambda item: item.score)
                best_scores.append(best_row.score)
                best_variants.append(best_row.variant_name)
                if best_row.detected:
                    detected_pose_count += 1
            pose_count = len(rows_by_pose)
            detected_pose_rate = float(detected_pose_count) / float(max(1, pose_count))
            best_variant_mode = _mode_string(best_variants)
            writer.writerow(
                [
                    label,
                    color_signature,
                    tag_id,
                    pose_count,
                    detected_pose_count,
                    f"{detected_pose_rate:.6f}",
                    f"{float(np.mean(best_scores)) if best_scores else 0.0:.6f}",
                    f"{float(np.median(best_scores)) if best_scores else 0.0:.6f}",
                    best_variant_mode,
                ]
            )


def _write_summary_by_variant(path: Path, rows: list[CaptureRow]) -> None:
    grouped: dict[str, list[CaptureRow]] = {}
    for row in rows:
        grouped.setdefault(row.variant_name, []).append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant_name",
                "sample_count",
                "mean_score",
                "median_score",
                "mean_template_score",
                "mean_reprojection_error_px",
            ]
        )
        for variant_name, variant_rows in sorted(grouped.items()):
            writer.writerow(
                [
                    variant_name,
                    len(variant_rows),
                    f"{float(np.mean([row.score for row in variant_rows])) if variant_rows else 0.0:.6f}",
                    f"{float(np.median([row.score for row in variant_rows])) if variant_rows else 0.0:.6f}",
                    f"{float(np.mean([row.template_score for row in variant_rows])) if variant_rows else 0.0:.6f}",
                    f"{float(np.mean([row.reprojection_error_px for row in variant_rows])) if variant_rows else 0.0:.6f}",
                ]
            )


# endregion


# region 配置与 IO
def _load_tag_specs(path: Path | None) -> dict[int, TagSpec]:
    if path is None:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"tag spec 文件不存在：{resolved}")

    specs: dict[int, TagSpec] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tag_id = int(row["tag_id"])
            specs[tag_id] = TagSpec(
                tag_id=tag_id,
                label=str(row.get("label", f"tag_{tag_id}")).strip() or f"tag_{tag_id}",
                foreground=str(row.get("foreground", "")).strip(),
                background=str(row.get("background", "")).strip(),
                border=str(row.get("border", "")).strip(),
                note=str(row.get("note", "")).strip(),
            )
    return specs


def _read_camera_calibration(intrinsics_response) -> CameraCalibration:
    camera_matrix = np.array(
        [
            [float(intrinsics_response.fx), 0.0, float(intrinsics_response.cx)],
            [0.0, float(intrinsics_response.fy), float(intrinsics_response.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((8,), dtype=np.float64)
    return CameraCalibration(
        width=int(intrinsics_response.width),
        height=int(intrinsics_response.height),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )


def _create_session_dir(output_root: Path) -> Path:
    session_dir = Path(output_root) / "latest"
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _get_apriltag_dictionary(dictionary_name: str):
    name_to_id = {
        "DICT_APRILTAG_16H5": int(cv2.aruco.DICT_APRILTAG_16h5),
        "DICT_APRILTAG_25H9": int(cv2.aruco.DICT_APRILTAG_25h9),
        "DICT_APRILTAG_36H10": int(cv2.aruco.DICT_APRILTAG_36h10),
        "DICT_APRILTAG_36H11": int(cv2.aruco.DICT_APRILTAG_36h11),
    }
    if dictionary_name not in name_to_id:
        raise ValueError(f"不支持的 AprilTag 字典：{dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(name_to_id[dictionary_name])


def _build_template_bank(
    dictionary,
    tag_specs: dict[int, TagSpec],
    allowed_tag_ids: tuple[int, ...],
) -> TemplateBank:
    marker_size = int(dictionary.markerSize)
    total_cells = marker_size + 2
    if tag_specs:
        ordered_ids = np.array(
            [tag_id for tag_id in sorted(tag_specs.keys()) if tag_id in allowed_tag_ids],
            dtype=np.int32,
        )
    else:
        ordered_ids = np.array(sorted(set(int(tag_id) for tag_id in allowed_tag_ids)), dtype=np.int32)
    if ordered_ids.size == 0:
        raise ValueError("模板库为空，请检查 allowed_tag_ids 或 tag spec 配置。")

    templates: list[np.ndarray] = []
    template_size_px = total_cells * 8
    for tag_id in ordered_ids.tolist():
        marker_image = cv2.aruco.generateImageMarker(
            dictionary,
            int(tag_id),
            template_size_px,
            borderBits=1,
        )
        templates.append(_sample_marker_grid(marker_image, total_cells))
    flat_templates = np.stack(templates, axis=0).astype(np.uint8)
    return TemplateBank(
        tag_ids=ordered_ids,
        flat_templates=flat_templates,
        marker_size=marker_size,
        total_cells=total_cells,
    )


def _validate_runtime_requirements() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 不包含 aruco 模块，请安装 opencv-contrib-python。")
    if not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("当前 OpenCV 版本不支持 cv2.aruco.ArucoDetector。")


# endregion


# region 工具函数
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


def _ensure_gray_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _rotation_vector_to_rpy_deg(rvec: np.ndarray) -> tuple[float, float, float]:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        roll = float(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        pitch = float(np.arctan2(-rotation_matrix[2, 0], sy))
        yaw = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        roll = float(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        pitch = float(np.arctan2(-rotation_matrix[2, 0], sy))
        yaw = 0.0
    return (
        float(np.degrees(roll)),
        float(np.degrees(pitch)),
        float(np.degrees(yaw)),
    )


def _compute_preview_window_size(
    src_w: int,
    src_h: int,
    min_long_side: int,
) -> tuple[int, int]:
    long_side = max(src_w, src_h)
    if long_side <= 0:
        return min_long_side, min_long_side
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _panel_text_scale(panel: np.ndarray, base_scale: float) -> float:
    long_side = float(max(panel.shape[0], panel.shape[1]))
    return float(base_scale * max(0.75, long_side / 800.0))


def _normalize_rejected_corners(rejected) -> list[np.ndarray]:
    normalized: list[np.ndarray] = []
    if rejected is None:
        return normalized
    for corners in rejected:
        corner_array = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if corner_array.shape[0] >= 4:
            normalized.append(corner_array[:4])
    return normalized


def _build_detect_inputs(*images: np.ndarray) -> list[np.ndarray]:
    detect_inputs: list[np.ndarray] = []
    seen_keys: set[tuple[int, int, int, int]] = set()
    for image in images:
        gray = _ensure_gray_image(image)
        variants = [gray, cv2.bitwise_not(gray)]
        for variant in variants:
            key = (
                int(variant.shape[0]),
                int(variant.shape[1]),
                int(variant.dtype.itemsize),
                int(np.sum(variant[:: max(1, variant.shape[0] // 16), :: max(1, variant.shape[1] // 16)])),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            detect_inputs.append(variant)
    return detect_inputs


def _detect_candidates_and_decode_multi_input(
    detect_images: list[np.ndarray],
    template_bank: TemplateBank,
    candidate_mode: str,
):
    best_corners: list[np.ndarray] | None = None
    best_ids: np.ndarray | None = None
    best_rejected: list[np.ndarray] | None = None
    best_rejected_count = -1
    for detect_image in detect_images:
        corners_list, ids, rejected = _detect_candidates_and_decode_single(
            detect_image=detect_image,
            template_bank=template_bank,
            candidate_mode=candidate_mode,
        )
        if ids is not None and len(ids) > 0 and corners_list:
            return corners_list, ids, rejected
        rejected_count = 0 if rejected is None else len(rejected)
        if rejected_count > best_rejected_count:
            best_corners = corners_list
            best_ids = ids
            best_rejected = rejected
            best_rejected_count = rejected_count
    return best_corners, best_ids, best_rejected


def _detect_candidates_and_decode_single(
    detect_image: np.ndarray,
    template_bank: TemplateBank,
    candidate_mode: str,
) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
    gray = _ensure_gray_image(detect_image)
    accepted_corners: list[np.ndarray] = []
    accepted_ids: list[int] = []
    rejected_quads: list[np.ndarray] = []
    accepted_centers: list[np.ndarray] = []
    candidate_quads, rejected_quads = _extract_candidate_quads(
        gray=gray,
        candidate_mode=candidate_mode,
    )
    for ordered_quad in candidate_quads:
        center = np.mean(ordered_quad, axis=0)
        if any(np.linalg.norm(center - existing) < 8.0 for existing in accepted_centers):
            continue

        decode = _decode_candidate_quad(gray, ordered_quad, template_bank)
        if decode is None:
            rejected_quads.append(ordered_quad)
            continue

        accepted_corners.append(ordered_quad.reshape(1, 4, 2))
        accepted_ids.append(int(decode[0]))
        accepted_centers.append(center)

    ids = None if not accepted_ids else np.asarray(accepted_ids, dtype=np.int32).reshape(-1, 1)
    return accepted_corners, ids, rejected_quads


def _decode_candidate_quad(
    gray: np.ndarray,
    quad: np.ndarray,
    template_bank: TemplateBank,
) -> tuple[int, float] | None:
    warp_size = template_bank.total_cells * 8
    dst = np.array(
        [
            [0.0, 0.0],
            [warp_size - 1.0, 0.0],
            [warp_size - 1.0, warp_size - 1.0],
            [0.0, warp_size - 1.0],
        ],
        dtype=np.float32,
    )

    best_id = -1
    best_score = -1.0
    for rotated_quad in _quad_rotation_variants(quad):
        transform = cv2.getPerspectiveTransform(rotated_quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(gray, transform, (warp_size, warp_size), flags=cv2.INTER_LINEAR)
        _, thresholded = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        sampled = _sample_marker_grid(thresholded, template_bank.total_cells)
        sampled_inv = 1 - sampled
        score_a, tag_id_a = _match_sampled_bits(sampled, template_bank)
        score_b, tag_id_b = _match_sampled_bits(sampled_inv, template_bank)
        if score_a >= score_b:
            score = score_a
            tag_id = tag_id_a
        else:
            score = score_b
            tag_id = tag_id_b
        if score > best_score:
            best_score = score
            best_id = tag_id

    if best_id < 0 or best_score < 0.93:
        return None
    return best_id, best_score


def _sample_marker_grid(image: np.ndarray, total_cells: int) -> np.ndarray:
    gray = _ensure_gray_image(image)
    resized = cv2.resize(gray, (total_cells * 8, total_cells * 8), interpolation=cv2.INTER_AREA)
    sampled = np.zeros((total_cells, total_cells), dtype=np.uint8)
    for row in range(total_cells):
        for col in range(total_cells):
            cell = resized[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
            sampled[row, col] = 1 if float(np.mean(cell)) < 127.5 else 0
    return sampled.reshape(-1)


def _match_sampled_bits(sampled_flat: np.ndarray, template_bank: TemplateBank) -> tuple[float, int]:
    sampled_u8 = sampled_flat.astype(np.uint8).reshape(1, -1)
    distances = np.count_nonzero(template_bank.flat_templates != sampled_u8, axis=1)
    best_index = int(np.argmin(distances))
    total_bits = int(template_bank.flat_templates.shape[1])
    score = 1.0 - float(distances[best_index]) / float(max(1, total_bits))
    return score, int(template_bank.tag_ids[best_index])


def _order_quad_points_clockwise(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def _quad_rotation_variants(quad: np.ndarray) -> list[np.ndarray]:
    return [np.roll(quad, -shift, axis=0).astype(np.float32) for shift in range(4)]


def _build_candidate_image(gray: np.ndarray, candidate_mode: str) -> np.ndarray:
    if candidate_mode == "canny":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)
    if candidate_mode == "local_binary":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        binary_inv = cv2.bitwise_not(binary)
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    if candidate_mode == "global_otsu":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        binary_inv = cv2.bitwise_not(binary)
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    raise ValueError(f"不支持的 candidate_mode：{candidate_mode}")


def _build_hough_preview(gray: np.ndarray, canny_edges: np.ndarray) -> np.ndarray:
    canvas = _gray_to_bgr(canny_edges)
    lines = cv2.HoughLinesP(
        canny_edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=24,
        maxLineGap=8,
    )
    if lines is None:
        return canvas
    for line in lines[:80]:
        x1, y1, x2, y2 = (int(v) for v in line.reshape(4))
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _extract_candidate_quads(
    gray: np.ndarray,
    candidate_mode: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if candidate_mode == "lsd":
        return _extract_lsd_candidate_quads(gray)
    candidate_image = _build_candidate_image(gray=gray, candidate_mode=candidate_mode)
    contours, _ = cv2.findContours(
        candidate_image,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return _extract_contour_candidate_quads(gray, contours)


def _extract_contour_candidate_quads(
    gray: np.ndarray,
    contours,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    accepted_quads: list[np.ndarray] = []
    rejected_quads: list[np.ndarray] = []
    min_area = max(80.0, gray.shape[0] * gray.shape[1] * 0.00005)
    for contour in contours:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < 40.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        quad = np.asarray(approx, dtype=np.float32).reshape(4, 2)
        area = abs(float(cv2.contourArea(quad)))
        if area < min_area:
            continue
        ordered_quad = _order_quad_points_clockwise(quad)
        if not _is_reasonable_quad(ordered_quad):
            rejected_quads.append(ordered_quad)
            continue
        accepted_quads.append(ordered_quad)
    return accepted_quads, rejected_quads


def _extract_lsd_candidate_quads(gray: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)
    lines = detected[0]
    if lines is None:
        return [], []

    line_canvas = np.zeros_like(gray)
    for line in lines[:240]:
        x1, y1, x2, y2 = (int(round(v)) for v in line.reshape(4))
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length < 18.0:
            continue
        cv2.line(line_canvas, (x1, y1), (x2, y2), 255, 1, cv2.LINE_AA)

    kernel = np.ones((3, 3), dtype=np.uint8)
    merged = cv2.dilate(line_canvas, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _extract_contour_candidate_quads(gray, contours)


def _fuse_variant_detections(
    frame_results: dict[str, VariantDetections],
) -> VariantDetections:
    collected: list[DetectionResult] = []
    for variant_name, variant_detections in frame_results.items():
        if variant_name == "Fusion":
            continue
        collected.extend(variant_detections.results)

    if not collected:
        return VariantDetections(results=[], rejected_corners=[])

    sorted_results = sorted(collected, key=lambda item: item.score, reverse=True)
    fused_results: list[DetectionResult] = []
    for result in sorted_results:
        if not _is_duplicate_detection(result, fused_results):
            fused_results.append(result)

    normalized_results: list[DetectionResult] = []
    for detection_index, result in enumerate(fused_results, start=1):
        normalized_results.append(
            DetectionResult(
                variant_name="Fusion",
                detection_index=detection_index,
                tag_id=result.tag_id,
                label=result.label,
                color_signature=result.color_signature,
                detected=result.detected,
                score=result.score,
                template_score=result.template_score,
                reprojection_error_px=result.reprojection_error_px,
                reprojection_score=result.reprojection_score,
                size_score=result.size_score,
                perimeter_px=result.perimeter_px,
                corners_px=result.corners_px,
                axis_points_px=result.axis_points_px,
                rvec=result.rvec,
                tvec_mm=result.tvec_mm,
                rpy_deg=result.rpy_deg,
            )
        )
    return VariantDetections(results=normalized_results, rejected_corners=[])


def _fuse_temporal_detections(
    fusion_history: list[tuple[float, list[DetectionResult]]],
    window_s: float,
    min_support: int,
) -> VariantDetections:
    if not fusion_history:
        return VariantDetections(results=[], rejected_corners=[])
    latest_ts, latest_results = fusion_history[-1]
    if not latest_results:
        return VariantDetections(results=[], rejected_corners=[])

    temporal_results: list[DetectionResult] = []
    for latest_result in latest_results:
        cluster = [latest_result]
        for history_ts, history_results in reversed(fusion_history[:-1]):
            if latest_ts - history_ts > float(window_s):
                break
            matched = _find_temporal_match(latest_result, history_results)
            if matched is not None:
                cluster.append(matched)
        if len(cluster) < min_support:
            continue
        temporal_results.append(
            _build_temporal_detection_result(
                cluster=cluster,
                window_s=window_s,
                detection_index=len(temporal_results) + 1,
            )
        )
    return VariantDetections(results=temporal_results, rejected_corners=[])


def _find_temporal_match(
    anchor: DetectionResult,
    history_results: list[DetectionResult],
) -> DetectionResult | None:
    matched_results = [result for result in history_results if _detection_matches(anchor, result)]
    if not matched_results:
        return None
    return max(matched_results, key=lambda item: item.score)


def _build_temporal_detection_result(
    cluster: list[DetectionResult],
    window_s: float,
    detection_index: int,
) -> DetectionResult:
    representative = max(cluster, key=lambda item: item.score)
    support_ratio = float(len(cluster)) / float(max(1.0, window_s * 30.0))
    mean_score = float(np.mean([item.score for item in cluster]))
    temporal_score = float(0.7 * mean_score + 0.3 * support_ratio)
    mean_template_score = float(np.mean([item.template_score for item in cluster]))
    mean_reprojection_error = float(np.mean([item.reprojection_error_px for item in cluster]))
    mean_reprojection_score = float(np.mean([item.reprojection_score for item in cluster]))
    mean_size_score = float(np.mean([item.size_score for item in cluster]))
    mean_perimeter = float(np.mean([item.perimeter_px for item in cluster]))
    return DetectionResult(
        variant_name="TemporalFusion",
        detection_index=detection_index,
        tag_id=representative.tag_id,
        label=representative.label,
        color_signature=representative.color_signature,
        detected=True,
        score=temporal_score,
        template_score=mean_template_score,
        reprojection_error_px=mean_reprojection_error,
        reprojection_score=mean_reprojection_score,
        size_score=mean_size_score,
        perimeter_px=mean_perimeter,
        corners_px=representative.corners_px,
        axis_points_px=representative.axis_points_px,
        rvec=representative.rvec,
        tvec_mm=representative.tvec_mm,
        rpy_deg=representative.rpy_deg,
    )


def _prune_temporal_history(
    temporal_fusion_history: deque[tuple[float, list[DetectionResult]]],
    window_s: float,
) -> None:
    if not temporal_fusion_history:
        return
    latest_ts = temporal_fusion_history[-1][0]
    while temporal_fusion_history and latest_ts - temporal_fusion_history[0][0] > float(window_s):
        temporal_fusion_history.popleft()


def _is_duplicate_detection(
    candidate: DetectionResult,
    existing_results: list[DetectionResult],
) -> bool:
    if candidate.corners_px is None:
        return False
    candidate_center = np.mean(candidate.corners_px, axis=0)
    for existing in existing_results:
        if existing.tag_id != candidate.tag_id or existing.corners_px is None:
            continue
        existing_center = np.mean(existing.corners_px, axis=0)
        center_distance = float(np.linalg.norm(candidate_center - existing_center))
        corner_distance = float(np.mean(np.linalg.norm(candidate.corners_px - existing.corners_px, axis=1)))
        if center_distance <= 14.0 and corner_distance <= 18.0:
            return True
    return False


def _detection_matches(
    anchor: DetectionResult,
    candidate: DetectionResult,
) -> bool:
    if anchor.tag_id != candidate.tag_id or anchor.corners_px is None or candidate.corners_px is None:
        return False
    anchor_center = np.mean(anchor.corners_px, axis=0)
    candidate_center = np.mean(candidate.corners_px, axis=0)
    center_distance = float(np.linalg.norm(anchor_center - candidate_center))
    corner_distance = float(np.mean(np.linalg.norm(anchor.corners_px - candidate.corners_px, axis=1)))
    return center_distance <= 24.0 and corner_distance <= 28.0


def _is_reasonable_quad(quad: np.ndarray) -> bool:
    edge_lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if np.min(edge_lengths) < 10.0:
        return False
    width = float(0.5 * (edge_lengths[0] + edge_lengths[2]))
    height = float(0.5 * (edge_lengths[1] + edge_lengths[3]))
    aspect = width / max(height, 1e-6)
    if aspect < 0.78 or aspect > 1.22:
        return False
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    rect = cv2.minAreaRect(quad.astype(np.float32))
    box_area = max(1.0, float(rect[1][0] * rect[1][1]))
    fill_ratio = area / box_area
    return fill_ratio >= 0.78


def _fmt_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _mode_string(values: list[str]) -> str:
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _parse_allowed_tag_ids(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        return DEFAULT_ALLOWED_TAG_IDS
    return tuple(sorted(set(values)))


def _extract_color_signature(source_bgr: np.ndarray, corners_px: np.ndarray) -> str:
    center = np.mean(corners_px, axis=0)
    inner_corners = center + (corners_px - center) * 0.72
    mask = np.zeros(source_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(inner_corners).astype(np.int32), 255)
    pixels = source_bgr[mask > 0]
    if pixels.size == 0:
        return "unknown"
    median_bgr = np.median(pixels, axis=0)
    median_bgr_u8 = np.clip(np.round(median_bgr), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(median_bgr_u8, cv2.COLOR_BGR2HSV).reshape(3)
    hue_bucket = int(round(float(hsv[0]) / 10.0) * 10)
    sat_bucket = int(round(float(hsv[1]) / 32.0) * 32)
    val_bucket = int(round(float(hsv[2]) / 32.0) * 32)
    return f"h{hue_bucket:03d}_s{sat_bucket:03d}_v{val_bucket:03d}"


# endregion


# region CLI
def _parse_cli() -> AppConfig:
    parser = argparse.ArgumentParser(description="Gemini305 AprilTag 多色彩空间识别评估脚本")
    parser.add_argument(
        "--dictionary",
        type=str,
        default=DEFAULT_DICTIONARY_NAME,
        help="AprilTag 字典名，例如 DICT_APRILTAG_36H11",
    )
    parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=DEFAULT_TAG_SIZE_MM,
        help="AprilTag 实际边长（mm）",
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
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="实验结果输出根目录",
    )
    parser.add_argument(
        "--tag-spec-csv",
        type=Path,
        default=None,
        help="可选 CSV：tag_id,label,foreground,background,border,note",
    )
    parser.add_argument(
        "--allowed-tag-ids",
        type=str,
        default="0,1,2,3,4,5,6",
        help="允许接受的真实 tag id，逗号分隔；默认 0-6",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=DEFAULT_CLAHE_CLIP_LIMIT,
        help="CLAHE clip limit",
    )
    parser.add_argument(
        "--clahe-grid",
        type=int,
        default=DEFAULT_CLAHE_GRID,
        help="CLAHE tile grid 大小",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="最多处理多少帧，0 表示不限制",
    )
    args = parser.parse_args()
    return AppConfig(
        dictionary_name=str(args.dictionary),
        tag_size_mm=float(args.tag_size_mm),
        timeout_ms=int(args.timeout_ms),
        capture_fps=int(args.capture_fps),
        output_root=Path(args.output_root),
        tag_spec_path=None if args.tag_spec_csv is None else Path(args.tag_spec_csv),
        allowed_tag_ids=_parse_allowed_tag_ids(str(args.allowed_tag_ids)),
        clahe_clip_limit=float(args.clahe_clip_limit),
        clahe_grid=int(args.clahe_grid),
        max_frames=int(args.max_frames),
    )


# endregion


if __name__ == "__main__":
    try:
        main(_parse_cli())
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
    except Exception as exc:
        logger.exception(f"程序异常退出：{exc}")
        raise
