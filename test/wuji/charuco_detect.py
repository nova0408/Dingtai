from __future__ import annotations

# ruff: noqa: E402

"""Wuji ChArUco 识别测试页。

目标：
1. 订阅 Orin 相机彩色流并进行畸变校正。
2. 对同一帧使用多种前处理模式，提升板识别鲁棒性。
3. 基于 ChArUco 角点计算板在图像中的基准坐标系位姿。
4. 将板坐标轴、角点和调试信息绘制回原图，方便人工确认。
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraName, CameraPipelineClient
from src.calibration.charuco import CharucoPoseEstimator

# region 默认参数
DEFAULT_WINDOW_NAME = "Charuco Detect"
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_ORIN_SERVICE_ADDR = "tcp://192.168.1.128:6200"
DEFAULT_OUTPUT_ROOT = Path("test/wuji/.archive/charuco_detect_runs")
DEFAULT_MIN_CHARUCO_CORNERS = 2
DEFAULT_MAX_FRAMES = 0
DEFAULT_PREPROCESS_MODES = ("raw", "clahe", "bilateral_clahe", "unsharp_otsu")
DEFAULT_BOARD_DRAW_SIZE = (960, 720)
DEFAULT_DICTIONARY_NAME = "DICT_APRILTAG_16H5"
DEFAULT_SQUARES_X = 7
DEFAULT_SQUARES_Y = 4
DEFAULT_SQUARE_LENGTH_MM = 12.28
DEFAULT_MARKER_LENGTH_MM = 8.6
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """相机内参与畸变。"""

    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True, slots=True)
class CharucoVariantResult:
    """单个前处理分支的识别结果。"""

    mode: str
    image_bgr: np.ndarray
    marker_count: int
    charuco_count: int
    board_visible: bool
    reprojection_error_px: float | None
    marker_corners_px: list[np.ndarray]
    marker_ids: np.ndarray | None
    charuco_corners_px: np.ndarray | None
    charuco_ids: np.ndarray | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    transform_se3: np.ndarray | None


@dataclass(frozen=True, slots=True)
class AppConfig:
    """脚本配置。"""

    timeout_ms: int
    capture_fps: int
    output_root: Path
    min_charuco_corners: int
    max_frames: int


@dataclass(frozen=True, slots=True)
class BoardConfig:
    """ChArUco 板配置。"""

    dictionary_name: str
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float


# endregion


# region 主流程
def main(config: AppConfig) -> None:
    _validate_runtime_requirements()
    session_dir = _create_session_dir(config.output_root)
    board = _build_board()
    board_image = board.generateImage(DEFAULT_BOARD_DRAW_SIZE)
    cv2.imwrite(str(session_dir / "charuco_board_reference.png"), board_image)
    estimator = CharucoPoseEstimator(board)

    client = CameraPipelineClient(service_addr=DEFAULT_ORIN_SERVICE_ADDR, timeout_ms=30_000)
    try:
        timeout_s = float(config.timeout_ms) / 1000.0
        camera_name = CameraName(DEFAULT_CAMERA_NAME)
        summary_response = client.get_camera_summary(camera_name, timeout_s)
        status_response = client.get_camera_status(camera_name, timeout_s)
        intrinsics_response = client.get_camera_intrinsics(camera_name, timeout_s)
        calibration = _read_camera_calibration(intrinsics_response)
        logger.info(
            "Charuco eval Orin stream target: "
            f"camera={DEFAULT_CAMERA_NAME}, "
            f"summary_frame_id={summary_response.frame_id}, "
            f"calibration={calibration.width}x{calibration.height}, "
            f"status={status_response.camera_model}/{status_response.width}x{status_response.height}, "
        )
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        frame_index = 0
        latest_preview: np.ndarray | None = None

        for frame in client.subscribe_camera_frames(CameraName(DEFAULT_CAMERA_NAME)):
            color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
            if color_bgr.size == 0:
                logger.warning("Orin camera_frame 返回空图像，跳过本帧")
                continue

            frame_index += 1
            started = time.perf_counter()
            undistorted_bgr = cv2.undistort(
                color_bgr,
                calibration.camera_matrix,
                calibration.dist_coeffs,
            )
            variant_results = _evaluate_variants(
                undistorted_bgr=undistorted_bgr,
                calibration=calibration,
                estimator=estimator,
                min_charuco_corners=config.min_charuco_corners,
            )
            best_result = _select_best_result(variant_results)
            preview = _compose_preview(
                source_bgr=undistorted_bgr,
                variant_results=variant_results,
                best_result=best_result,
                calibration=calibration,
                frame_index=frame_index,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                session_dir=session_dir,
            )
            latest_preview = preview
            cv2.imshow(DEFAULT_WINDOW_NAME, preview)
            cv2.waitKey(1)
            _save_frame_artifacts(
                session_dir=session_dir,
                frame_index=frame_index,
                preview_image=preview,
                variant_results=variant_results,
                best_result=best_result,
            )

            if best_result is not None and best_result.board_visible:
                _write_pose_snapshot(session_dir, frame_index, best_result, variant_results)

            if int(config.max_frames) > 0 and frame_index >= int(config.max_frames):
                logger.warning("达到最大帧数 {}，停止采集。", int(config.max_frames))
                break
    finally:
        client.close()

    if latest_preview is not None:
        cv2.imwrite(str(session_dir / "final_preview.png"), latest_preview)
    cv2.destroyAllWindows()
    logger.success(f"评估结果输出目录：{session_dir}")


# endregion


# region ChArUco 评估
def _evaluate_variants(
    undistorted_bgr: np.ndarray,
    calibration: CameraCalibration,
    estimator: CharucoPoseEstimator,
    min_charuco_corners: int,
) -> list[CharucoVariantResult]:
    variants: list[tuple[str, np.ndarray]] = []
    gray = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(contrast, d=7, sigmaColor=35.0, sigmaSpace=35.0)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0.0)
    otsu_source = cv2.GaussianBlur(contrast, (5, 5), 0)
    _, otsu = cv2.threshold(otsu_source, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    otsu_bgr = cv2.cvtColor(cv2.bitwise_not(otsu), cv2.COLOR_GRAY2BGR)

    variants.append(("raw", undistorted_bgr))
    variants.append(("clahe", cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)))
    variants.append(("bilateral_clahe", cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)))
    variants.append(("unsharp_otsu", otsu_bgr if otsu_bgr.ndim == 3 else cv2.cvtColor(otsu_bgr, cv2.COLOR_GRAY2BGR)))
    variants.append(("unsharp", cv2.cvtColor(np.clip(sharp, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)))

    results: list[CharucoVariantResult] = []
    for mode, image_bgr in variants:
        result = _detect_charuco_variant(
            mode=mode,
            image_bgr=image_bgr,
            calibration=calibration,
            estimator=estimator,
            min_charuco_corners=min_charuco_corners,
        )
        results.append(result)
    return results


def _detect_charuco_variant(
    mode: str,
    image_bgr: np.ndarray,
    calibration: CameraCalibration,
    estimator: CharucoPoseEstimator,
    min_charuco_corners: int,
) -> CharucoVariantResult:
    result = estimator.estimate_pose(
        image_bgr=image_bgr,
        camera_matrix=calibration.camera_matrix,
        dist_coeffs=calibration.dist_coeffs,
        min_charuco_corners=int(min_charuco_corners),
    )
    marker_corners_list = [np.asarray(item, dtype=np.float32).reshape(4, 1, 2) for item in result.marker_corners_px]
    marker_ids_norm = (
        None if result.marker_ids is None else np.asarray(result.marker_ids, dtype=np.int32).reshape(-1, 1)
    )
    charuco_corners_norm = (
        None
        if result.charuco_corners_px is None
        else np.asarray(result.charuco_corners_px, dtype=np.float64).reshape(-1, 2)
    )
    charuco_ids_norm = (
        None if result.charuco_ids is None else np.asarray(result.charuco_ids, dtype=np.int32).reshape(-1, 1)
    )
    marker_count = int(result.marker_count)
    charuco_count = int(result.charuco_count)

    if (
        not result.board_visible
        or charuco_count < int(min_charuco_corners)
        or charuco_corners_norm is None
        or charuco_ids_norm is None
    ):
        return CharucoVariantResult(
            mode=mode,
            image_bgr=image_bgr,
            marker_count=marker_count,
            charuco_count=charuco_count,
            board_visible=False,
            reprojection_error_px=None,
            marker_corners_px=marker_corners_list,
            marker_ids=marker_ids_norm,
            charuco_corners_px=charuco_corners_norm,
            charuco_ids=charuco_ids_norm,
            rvec=None,
            tvec=None,
            transform_se3=None,
        )
    return CharucoVariantResult(
        mode=mode,
        image_bgr=image_bgr,
        marker_count=marker_count,
        charuco_count=charuco_count,
        board_visible=bool(result.board_visible),
        reprojection_error_px=result.reprojection_error_px,
        marker_corners_px=marker_corners_list,
        marker_ids=marker_ids_norm,
        charuco_corners_px=charuco_corners_norm,
        charuco_ids=charuco_ids_norm,
        rvec=None if result.rvec is None else np.asarray(result.rvec, dtype=np.float64).reshape(3),
        tvec=None if result.tvec is None else np.asarray(result.tvec, dtype=np.float64).reshape(3),
        transform_se3=result.transform_se3,
    )


def _select_best_result(results: list[CharucoVariantResult]) -> CharucoVariantResult | None:
    visible = [item for item in results if item.board_visible and item.transform_se3 is not None]
    if visible:
        return max(
            visible,
            key=lambda item: (
                int(item.charuco_count),
                -1.0 if item.reprojection_error_px is None else -float(item.reprojection_error_px),
            ),
        )
    partial = [item for item in results if item.charuco_count > 0]
    if partial:
        return max(partial, key=lambda item: int(item.charuco_count))
    return None


# endregion


# region 预览绘制
def _compose_preview(
    source_bgr: np.ndarray,
    variant_results: list[CharucoVariantResult],
    best_result: CharucoVariantResult | None,
    calibration: CameraCalibration,
    frame_index: int,
    elapsed_ms: float,
    session_dir: Path,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    for result in variant_results:
        panel = result.image_bgr.copy()
        _draw_variant_overlay(panel, result, calibration)
        _draw_panel_header(
            panel,
            title=result.mode,
            subtitle=_variant_subtitle(result),
        )
        panels.append(panel)

    board_panel = source_bgr.copy()
    if best_result is not None:
        _draw_board_overlay(board_panel, best_result, calibration)
    _draw_panel_header(
        board_panel,
        title="best_pose",
        subtitle=_best_subtitle(best_result),
    )
    panels.append(board_panel)

    grid = _compose_panel_grid(panels, columns=3)
    footer = [
        f"frame={frame_index} compute_ms={elapsed_ms:.2f}",
        "multi-preprocess charuco pose estimation",
        f"output={session_dir.name}",
    ]
    return _append_footer(grid, footer)


def _draw_variant_overlay(canvas: np.ndarray, result: CharucoVariantResult, calibration: CameraCalibration) -> None:
    if result.marker_corners_px:
        cv2.aruco.drawDetectedMarkers(
            canvas,
            [np.asarray(item, dtype=np.float32).reshape(4, 1, 2) for item in result.marker_corners_px],
            None if result.marker_ids is None else np.asarray(result.marker_ids, dtype=np.int32).reshape(-1, 1),
            borderColor=(255, 180, 0),
        )
    if result.charuco_corners_px is not None and result.charuco_ids is not None:
        board_corners = np.round(result.charuco_corners_px).astype(np.int32).reshape(-1, 1, 2)
        if board_corners.shape[0] >= 2:
            cv2.polylines(
                canvas, [board_corners], isClosed=False, color=(0, 220, 255), thickness=1, lineType=cv2.LINE_AA
            )
        for corner, charuco_id in zip(result.charuco_corners_px, result.charuco_ids.flatten(), strict=True):
            point = tuple(int(v) for v in np.round(corner))
            cv2.circle(canvas, point, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(int(charuco_id)),
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
    if result.board_visible and result.rvec is not None and result.tvec is not None:
        _draw_pose_axes(canvas, result.rvec, result.tvec, calibration.camera_matrix, calibration.dist_coeffs)
        _draw_pose_status(canvas, result, (20, 210, 20))
    else:
        _draw_pose_status(canvas, result, (0, 0, 255))


def _draw_board_overlay(canvas: np.ndarray, result: CharucoVariantResult, calibration: CameraCalibration) -> None:
    if result.marker_corners_px:
        cv2.aruco.drawDetectedMarkers(
            canvas,
            [np.asarray(item, dtype=np.float32).reshape(4, 1, 2) for item in result.marker_corners_px],
            None if result.marker_ids is None else np.asarray(result.marker_ids, dtype=np.int32).reshape(-1, 1),
            borderColor=(255, 180, 0),
        )
    if result.board_visible and result.rvec is not None and result.tvec is not None:
        _draw_pose_axes(canvas, result.rvec, result.tvec, calibration.camera_matrix, calibration.dist_coeffs)
        _draw_pose_status(canvas, result, (40, 220, 40))
        _draw_pose_metrics(canvas, result)
    else:
        _draw_pose_status(canvas, result, (0, 0, 255))
    if result.charuco_corners_px is not None and result.charuco_ids is not None:
        corners = np.round(result.charuco_corners_px).astype(np.int32)
        if corners.shape[0] >= 2:
            cv2.polylines(
                canvas,
                [corners.reshape(-1, 1, 2)],
                isClosed=False,
                color=(0, 220, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        for corner in corners:
            cv2.circle(canvas, tuple(int(v) for v in corner), 5, (0, 220, 0), -1, cv2.LINE_AA)
    if result.transform_se3 is not None:
        origin = result.transform_se3[:3, 3]
        text = f"t=[{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}]mm"
        _draw_text(canvas, text, (18, 34), scale=_panel_text_scale(canvas, 0.95))


def _draw_pose_axes(
    canvas: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
) -> None:
    axis_length = 30.0
    object_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, -axis_length],
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(_current_camera_matrix(canvas) if camera_matrix is None else camera_matrix, dtype=np.float64),
        np.asarray(_current_dist_coeffs(canvas) if dist_coeffs is None else dist_coeffs, dtype=np.float64),
    )
    points = np.round(projected.reshape(-1, 2)).astype(np.int32)
    origin = tuple(int(v) for v in points[0])
    cv2.line(canvas, origin, tuple(int(v) for v in points[1]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, origin, tuple(int(v) for v in points[2]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, origin, tuple(int(v) for v in points[3]), (255, 0, 0), 2, cv2.LINE_AA)


def _draw_pose_status(canvas: np.ndarray, result: CharucoVariantResult, color: tuple[int, int, int]) -> None:
    label = "POSE OK" if result.board_visible and result.rvec is not None and result.tvec is not None else "NO POSE"
    bar_h = 28
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], bar_h), color, -1)
    cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0.0, canvas)
    cv2.putText(
        canvas, label, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, _panel_text_scale(canvas, 0.7), color, 2, cv2.LINE_AA
    )


def _draw_pose_metrics(canvas: np.ndarray, result: CharucoVariantResult) -> None:
    lines = [
        f"charuco={result.charuco_count} markers={result.marker_count}",
        "reproj=" + ("n/a" if result.reprojection_error_px is None else f"{result.reprojection_error_px:.2f}px"),
    ]
    if result.tvec is not None:
        lines.append(f"tvec=[{result.tvec[0]:.1f}, {result.tvec[1]:.1f}, {result.tvec[2]:.1f}] mm")
    for idx, line in enumerate(lines):
        _draw_text(canvas, line, (18, 100 + idx * 28), scale=_panel_text_scale(canvas, 0.78))


def _variant_subtitle(result: CharucoVariantResult) -> str:
    if not result.board_visible:
        return f"markers={result.marker_count} charuco={result.charuco_count} not_visible"
    error_text = "n/a" if result.reprojection_error_px is None else f"{result.reprojection_error_px:.2f}px"
    return f"markers={result.marker_count} charuco={result.charuco_count} reproj={error_text}"


def _best_subtitle(result: CharucoVariantResult | None) -> str:
    if result is None:
        return "no stable pose"
    error_text = "n/a" if result.reprojection_error_px is None else f"{result.reprojection_error_px:.2f}px"
    tvec = "n/a" if result.tvec is None else f"[{result.tvec[0]:.1f}, {result.tvec[1]:.1f}, {result.tvec[2]:.1f}]"
    return f"mode={result.mode} charuco={result.charuco_count} reproj={error_text} t={tvec}"


def _draw_panel_header(panel: np.ndarray, title: str, subtitle: str) -> None:
    _draw_text(panel, title, (18, 34), scale=_panel_text_scale(panel, 1.05))
    _draw_text(panel, subtitle, (18, 66), scale=_panel_text_scale(panel, 0.78))


def _compose_panel_grid(panels: list[np.ndarray], columns: int) -> np.ndarray:
    rows = int(np.ceil(len(panels) / max(1, columns)))
    blank = np.zeros_like(panels[0])
    row_images: list[np.ndarray] = []
    index = 0
    for _ in range(rows):
        row_panels: list[np.ndarray] = []
        for _ in range(columns):
            row_panels.append(panels[index] if index < len(panels) else blank.copy())
            index += 1
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


# region 采样与输出
def _save_frame_artifacts(
    session_dir: Path,
    frame_index: int,
    preview_image: np.ndarray,
    variant_results: list[CharucoVariantResult],
    best_result: CharucoVariantResult | None,
) -> None:
    frame_dir = session_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_dir / "preview.png"), preview_image)
    for result in variant_results:
        cv2.imwrite(str(frame_dir / f"{result.mode}.png"), result.image_bgr)
    summary = {
        "frame_index": int(frame_index),
        "best_mode": None if best_result is None else best_result.mode,
        "variants": [
            {
                "mode": item.mode,
                "marker_count": int(item.marker_count),
                "charuco_count": int(item.charuco_count),
                "board_visible": bool(item.board_visible),
                "reprojection_error_px": (
                    None if item.reprojection_error_px is None else float(item.reprojection_error_px)
                ),
                "tvec_mm": None if item.tvec is None else [float(v) for v in item.tvec],
            }
            for item in variant_results
        ],
    }
    (frame_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_pose_snapshot(
    session_dir: Path,
    frame_index: int,
    best_result: CharucoVariantResult,
    variant_results: list[CharucoVariantResult],
) -> None:
    payload = {
        "frame_index": int(frame_index),
        "best_mode": best_result.mode,
        "marker_count": int(best_result.marker_count),
        "charuco_count": int(best_result.charuco_count),
        "reprojection_error_px": (
            None if best_result.reprojection_error_px is None else float(best_result.reprojection_error_px)
        ),
        "tvec_mm": None if best_result.tvec is None else [float(v) for v in best_result.tvec],
        "transform_se3": None if best_result.transform_se3 is None else best_result.transform_se3.tolist(),
        "all_modes": [item.mode for item in variant_results],
    }
    (session_dir / "latest_pose.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "frame={} best_mode={} charuco={} reproj={} tvec={}",
        frame_index,
        best_result.mode,
        best_result.charuco_count,
        "n/a" if best_result.reprojection_error_px is None else f"{best_result.reprojection_error_px:.2f}",
        "n/a" if best_result.tvec is None else np.array2string(best_result.tvec, precision=2),
    )


# endregion


# region 相机与工具
def _create_session_dir(output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir = output_root / time.strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _validate_runtime_requirements() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 未安装 aruco 模块，无法运行 ChArUco 检测")


def _read_camera_calibration(intrinsics_response) -> CameraCalibration:
    camera_matrix = np.array(
        [
            [float(intrinsics_response.fx), 0.0, float(intrinsics_response.cx)],
            [0.0, float(intrinsics_response.fy), float(intrinsics_response.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intrinsics_response.distortion, dtype=np.float64).reshape(-1)
    return CameraCalibration(
        width=int(intrinsics_response.width),
        height=int(intrinsics_response.height),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )


def _build_board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(int(cv2.aruco.DICT_APRILTAG_16h5))
    return cv2.aruco.CharucoBoard(
        (int(DEFAULT_SQUARES_X), int(DEFAULT_SQUARES_Y)),
        float(DEFAULT_SQUARE_LENGTH_MM),
        float(DEFAULT_MARKER_LENGTH_MM),
        dictionary,
    )


def _current_camera_matrix(canvas: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, canvas.shape[1] / 2.0],
            [0.0, 1.0, canvas.shape[0] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _current_dist_coeffs(_: np.ndarray) -> np.ndarray:
    return np.zeros((5,), dtype=np.float64)


def _panel_text_scale(panel: np.ndarray, ratio: float) -> float:
    return float(np.clip(min(panel.shape[0], panel.shape[1]) / 700.0 * ratio, 0.45, 1.2))


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _parse_cli() -> AppConfig:
    parser = argparse.ArgumentParser(description="Wuji ChArUco 识别测试页")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="等待超时（ms）")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="期望采样帧率（fps）")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=DEFAULT_MIN_CHARUCO_CORNERS,
        help="进入位姿估计的最小 ChArUco 角点数量",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="最多处理帧数，0 表示不限制")
    args = parser.parse_args()
    return AppConfig(
        timeout_ms=int(args.timeout_ms),
        capture_fps=int(args.capture_fps),
        output_root=Path(args.output_root),
        min_charuco_corners=int(args.min_charuco_corners),
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
