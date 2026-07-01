from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pyorbbecsdk import OBFormat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.charuco import CHARUCO_200_12_9, CharucoPoseEstimator
from src.rgbd_camera import Gemini305, OrbbecSession, SessionOptions

# region 默认参数
DEFAULT_WINDOW_NAME = "Charuco Board Viewer"
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
# endregion


# region 数据结构
class BoardDetectionResult:
    """单帧 ChArUco 识别结果。"""

    marker_count: int
    charuco_count: int
    reprojection_error_px: float | None
    board_visible: bool
    corners_px: np.ndarray | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    marker_corners_px: list[np.ndarray]
    marker_ids: np.ndarray | None
    mode_name: str


# endregion


# region 主流程
def main() -> None:
    _validate_runtime_requirements()
    _run_viewer()


def _run_viewer() -> None:
    options = SessionOptions(timeout=DEFAULT_TIMEOUT_MS, preferred_capture_fps=DEFAULT_CAPTURE_FPS)
    estimator = CharucoPoseEstimator(CHARUCO_200_12_9)

    with Gemini305(options=options) as session:
        width, height, camera_matrix, dist_coeffs = _read_color_calibration(session)
        logger.info(f"color={width}x{height} " f"fx={camera_matrix[0,0]:.2f} fy={camera_matrix[1,1]:.2f}")
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        frame_index = 0
        try:
            while True:
                frames = session.wait_for_frames()
                if frames is None:
                    continue
                color_bgr = _decode_color_frame_bgr(frames.get_color_frame())
                if color_bgr is None:
                    continue
                frame_index += 1
                started = time.perf_counter()
                undistorted = cv2.undistort(
                    color_bgr,
                    camera_matrix,
                    dist_coeffs,
                )
                result = estimator.estimate_pose(undistorted, camera_matrix, dist_coeffs)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                preview = _draw_preview(
                    undistorted, result, estimator, camera_matrix, dist_coeffs, frame_index, elapsed_ms
                )
                cv2.imshow(DEFAULT_WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("l"), ord("L")):
                    board = estimator.board
                    rebuilt = cv2.aruco.CharucoBoard(
                        board.getChessboardSize(),
                        float(board.getSquareLength()),
                        float(board.getMarkerLength()),
                        board.getDictionary(),
                    )
                    rebuilt.setLegacyPattern(not board.getLegacyPattern())
                    estimator = CharucoPoseEstimator(rebuilt)
                if cv2.getWindowProperty(DEFAULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()


# endregion


def _draw_preview(
    image_bgr: np.ndarray,
    result,
    estimator: CharucoPoseEstimator,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_index: int,
    elapsed_ms: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    _draw_marker_corners(canvas, result.marker_corners_px, result.marker_ids)
    if result.charuco_corners_px is not None:
        pts = np.round(result.charuco_corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        for pt in result.charuco_corners_px:
            cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 0, 255), -1, cv2.LINE_AA)
    if result.rvec is not None and result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            camera_matrix,
            dist_coeffs,
            result.rvec.reshape(3, 1),
            result.tvec.reshape(3, 1),
            float(CHARUCO_200_12_9.getSquareLength()) * 1.8,
        )

    lines = [
        f"frame={frame_index} compute={elapsed_ms:.2f} ms",
        f"mode={'legacy' if estimator.legacy_pattern else 'modern'}",
        f"marker={result.marker_count} charuco={result.charuco_count}",
        f"visible={result.board_visible} pose={'yes' if result.rvec is not None else 'no'} reproj={_fmt_float(result.reprojection_error_px)}",
        "L: toggle legacy  Esc/Q: quit",
    ]
    _draw_text_block(canvas, lines, (18, 30))
    theory = _draw_theoretical_board_panel(estimator)
    return _stack_side_by_side(canvas, theory)


def _draw_theoretical_board_panel(estimator: CharucoPoseEstimator) -> np.ndarray:
    board_width_px = 840
    board_height_px = 1260
    panel = estimator.generate_board_image((board_width_px, board_height_px))
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    label_lines = [
        "theoretical board",
        f"mode={'legacy' if estimator.legacy_pattern else 'modern'}",
        f"size_x={estimator.board.getChessboardSize()[0]} size_y={estimator.board.getChessboardSize()[1]}",
        f"square={estimator.board.getSquareLength():.2f} marker={estimator.board.getMarkerLength():.2f}",
        "compare pattern / orientation / row parity",
    ]
    _draw_text_block(panel, label_lines, (18, 30))
    return panel


def _stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    target_height = max(int(left.shape[0]), int(right.shape[0]))
    left_resized = _resize_to_height(left, target_height)
    right_resized = _resize_to_height(right, target_height)
    separator = np.full((target_height, 12, 3), 40, dtype=np.uint8)
    return np.hstack([left_resized, separator, right_resized])


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    scale = float(target_height) / float(image.shape[0])
    target_width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _draw_marker_corners(
    canvas: np.ndarray,
    marker_corners_px: list[np.ndarray],
    marker_ids: np.ndarray | None,
) -> None:
    if not marker_corners_px:
        return
    for idx, corners in enumerate(marker_corners_px):
        pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        pts_i32 = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_i32], True, (255, 255, 0), 2, cv2.LINE_AA)
        center = np.mean(pts, axis=0)
        label = "" if marker_ids is None or idx >= len(marker_ids) else str(int(marker_ids[idx]))
        _draw_single_text(canvas, f"M{label}", (int(round(center[0])), int(round(center[1]))))


def _draw_text_block(canvas: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def _draw_single_text(canvas: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


# endregion


# region SDK 读取
def _read_color_calibration(session: OrbbecSession) -> tuple[int, int, np.ndarray, np.ndarray]:
    camera_param = session.get_camera_param()
    rgb_intrinsic = camera_param.rgb_intrinsic
    rgb_distortion = camera_param.rgb_distortion
    camera_matrix = np.asarray(
        [
            [float(rgb_intrinsic.fx), 0.0, float(rgb_intrinsic.cx)],
            [0.0, float(rgb_intrinsic.fy), float(rgb_intrinsic.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(
        [
            float(rgb_distortion.k1),
            float(rgb_distortion.k2),
            float(rgb_distortion.p1),
            float(rgb_distortion.p2),
            float(rgb_distortion.k3),
            float(rgb_distortion.k4),
            float(rgb_distortion.k5),
            float(rgb_distortion.k6),
        ],
        dtype=np.float64,
    ).reshape(-1, 1)
    return int(rgb_intrinsic.width), int(rgb_intrinsic.height), camera_matrix, dist_coeffs


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
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.YUYV:
        yuy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.YUY2:
        yuy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if color_format == OBFormat.I420:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    raise RuntimeError(f"不支持的 color format: {color_format}")


def _fmt_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


# endregion


# region 校验
def _validate_runtime_requirements() -> None:
    _ = cv2.aruco.ArucoDetector
    _ = cv2.aruco.CharucoBoard


# endregion


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
    except Exception as exc:
        logger.exception(f"程序异常退出：{exc}")
        raise
