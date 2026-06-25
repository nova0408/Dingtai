from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import cv2  # type: ignore[reportMissingImports]
import numpy as np
from loguru import logger
from pyorbbecsdk import OBFormat  # type: ignore[reportMissingImports]
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import calibrate_hand_eye_from_pose_sequences  # noqa: E402
from src.calibration.hand_eye import PoseLike  # noqa: E402
from src.rgbd_camera import Gemini305, OrbbecSession, SessionOptions  # noqa: E402
from src.utils.datas import Quaternion, Transform, Translation  # noqa: E402

# region 默认参数
DEFAULT_WINDOW_NAME = "Charuco Hand-Eye Test"
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_OUTPUT_ROOT = Path("experiments/hand_eye/runs")
DEFAULT_DICT_NAME = "DICT_5X5_1000"
DEFAULT_SQUARE_LENGTH_MM = 15.0
DEFAULT_MARKER_LENGTH_MM = 11.25
DEFAULT_SQUARES_X = 9
DEFAULT_SQUARES_Y = 12
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class CameraCalibration:
    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True, slots=True)
class CharucoConfig:
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    dictionary_name: str


@dataclass(frozen=True, slots=True)
class RobotPoseRow:
    sample_index: int
    pose: Transform


@dataclass(frozen=True, slots=True)
class SampleRow:
    sample_index: int
    timestamp_s: float
    frame_index: int
    board_visible: bool
    detected_corners: int
    reprojection_error_px: float | None
    board_pose: Transform | None
    robot_pose: Transform | None


# endregion


# region 主流程
def main(
    config: CharucoConfig,
    timeout_ms: int,
    capture_fps: int,
    output_root: Path,
    robot_pose_csv: Path | None,
) -> None:
    _launch_qt_app(config, timeout_ms, capture_fps, output_root, robot_pose_csv)


def _launch_qt_app(
    config: CharucoConfig,
    timeout_ms: int,
    capture_fps: int,
    output_root: Path,
    robot_pose_csv: Path | None,
) -> None:
    _validate_charuco_config(config)
    _validate_runtime_requirements()
    app = QApplication.instance() or QApplication(sys.argv)
    window = _CharucoMainWindow(
        config=config,
        timeout_ms=timeout_ms,
        capture_fps=capture_fps,
        output_root=output_root,
        robot_pose_csv=robot_pose_csv,
    )
    window.show()
    app.exec()


# endregion


# region 标定板检测
@dataclass(frozen=True, slots=True)
class BoardDetectionResult:
    board_visible: bool
    detected_corners: int
    reprojection_error_px: float | None
    board_pose: Transform | None
    corners_px: np.ndarray | None


class _CharucoCaptureWorker(QObject):
    frame_ready = Signal(QImage, str)
    sample_logged = Signal(str)
    finished = Signal(str)

    def __init__(
        self,
        config: CharucoConfig,
        timeout_ms: int,
        capture_fps: int,
        session_dir: Path,
        robot_pose_rows: list[RobotPoseRow],
        saved_frames_dir: Path,
    ) -> None:
        super().__init__()
        self._config = config
        self._timeout_ms = timeout_ms
        self._capture_fps = capture_fps
        self._session_dir = session_dir
        self._robot_pose_rows = robot_pose_rows
        self._saved_frames_dir = saved_frames_dir
        self._samples: list[SampleRow] = []
        self._frame_index = 0
        self._sample_index = 0
        self._robot_pose_cursor = 0
        self._latest_preview: np.ndarray | None = None
        self._latest_board_result: BoardDetectionResult | None = None
        self._capture_requested = False
        self._running = False

    @Slot()
    def request_capture(self) -> None:
        self._capture_requested = True

    @Slot()
    def stop(self) -> None:
        self._running = False

    def run_forever(self) -> None:
        self._running = True
        dictionary = _build_dictionary(self._config.dictionary_name)
        board = _build_charuco_board(self._config, dictionary)
        charuco_detector = _build_charuco_detector(board)
        options = SessionOptions(
            timeout=int(self._timeout_ms),
            preferred_capture_fps=max(1, int(self._capture_fps)),
        )
        with Gemini305(options=options) as session:
            calibration = _read_camera_calibration(session)
            detector = _build_detector(dictionary)
            try:
                while self._running:
                    frames = session.wait_for_frames()
                    if frames is None:
                        continue
                    color_bgr = _decode_color_frame_bgr(frames.get_color_frame())
                    if color_bgr is None:
                        continue
                    self._frame_index += 1
                    started = time.perf_counter()
                    calibrated_bgr = cv2.undistort(
                        color_bgr,
                        calibration.camera_matrix,
                        calibration.dist_coeffs,
                    )
                    board_result = _detect_charuco_pose(
                        image_bgr=calibrated_bgr,
                        detector=detector,
                        board=board,
                        charuco_detector=charuco_detector,
                        calibration=calibration,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    preview = _draw_preview(
                        image_bgr=calibrated_bgr,
                        board_result=board_result,
                        frame_index=self._frame_index,
                        elapsed_ms=elapsed_ms,
                        config=self._config,
                    )
                    self._latest_preview = preview
                    self._latest_board_result = board_result
                    self.frame_ready.emit(
                        _to_qimage(preview),
                        self._build_status_text(board_result, elapsed_ms),
                    )
                    if self._capture_requested:
                        self._capture_requested = False
                        self._capture_current_sample(preview, board_result)
            finally:
                self._write_outputs()
                self.finished.emit(str(self._session_dir))

    def _capture_current_sample(self, preview: np.ndarray, board_result: BoardDetectionResult) -> None:
        self._sample_index += 1
        robot_pose = _consume_robot_pose(self._robot_pose_rows, self._robot_pose_cursor, self._sample_index)
        if robot_pose is not None:
            self._robot_pose_cursor += 1
        self._samples.append(
            SampleRow(
                sample_index=self._sample_index,
                timestamp_s=time.time(),
                frame_index=self._frame_index,
                board_visible=board_result.board_visible,
                detected_corners=board_result.detected_corners,
                reprojection_error_px=board_result.reprojection_error_px,
                board_pose=board_result.board_pose,
                robot_pose=robot_pose,
            )
        )
        _save_sample_frame(self._saved_frames_dir, self._sample_index, preview)
        self.sample_logged.emit(
            f"已记录样本 #{self._sample_index}，board_visible={board_result.board_visible} "
            f"corners={board_result.detected_corners} reproj={board_result.reprojection_error_px}"
        )

    def _build_status_text(self, board_result: BoardDetectionResult, elapsed_ms: float) -> str:
        return (
            f"frame={self._frame_index} compute={elapsed_ms:.2f}ms "
            f"visible={board_result.board_visible} corners={board_result.detected_corners}"
        )

    def _write_outputs(self) -> None:
        _write_samples_csv(self._session_dir / "samples.csv", self._samples)
        _write_board_detection_csv(self._session_dir / "board_detections.csv", self._samples)
        if _has_calibration_samples(self._samples):
            _run_calibration(self._session_dir, self._samples)


class _CharucoMainWindow(QMainWindow):
    def __init__(
        self,
        config: CharucoConfig,
        timeout_ms: int,
        capture_fps: int,
        output_root: Path,
        robot_pose_csv: Path | None,
    ) -> None:
        super().__init__()
        self._config = config
        self._timeout_ms = timeout_ms
        self._capture_fps = capture_fps
        self._output_root = output_root
        self._robot_pose_csv = robot_pose_csv
        self._worker: _CharucoCaptureWorker | None = None
        self._thread: QThread | None = None
        self._image_label = QLabel()
        self._status_label = QLabel()
        self._session_path_label = QLabel()
        self._setup_ui()
        self._start_worker()

    def _setup_ui(self) -> None:
        self.setWindowTitle(DEFAULT_WINDOW_NAME)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(1280, 720)
        self._image_label.setStyleSheet("background: #111; color: white;")
        self._status_label.setText("等待相机数据...")
        self._session_path_label.setText("输出目录初始化中...")
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self._image_label, stretch=1)
        layout.addWidget(self._status_label)
        layout.addWidget(self._session_path_label)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("空格键采集一帧，Esc 退出")

    def _start_worker(self) -> None:
        session_dir = _create_session_dir(self._output_root)
        robot_pose_rows = _load_robot_pose_rows(self._robot_pose_csv) if self._robot_pose_csv is not None else []
        self._thread = QThread(self)
        self._worker = _CharucoCaptureWorker(
            config=self._config,
            timeout_ms=self._timeout_ms,
            capture_fps=self._capture_fps,
            session_dir=session_dir,
            robot_pose_rows=robot_pose_rows,
            saved_frames_dir=session_dir / "frames",
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run_forever)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.sample_logged.connect(self._on_sample_logged)
        self._worker.finished.connect(self._on_worker_finished)
        self._thread.start()
        self._session_path_label.setText(f"输出目录：{session_dir}")

    @Slot(QImage, str)
    def _on_frame_ready(self, image: QImage, status_text: str) -> None:
        self._image_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._status_label.setText(status_text)

    @Slot(str)
    def _on_sample_logged(self, text: str) -> None:
        self.statusBar().showMessage(text, 5000)

    @Slot(str)
    def _on_worker_finished(self, session_dir: str) -> None:
        self.statusBar().showMessage(f"采集结束：{session_dir}", 10000)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and self._worker is not None:
            self._worker.request_capture()
            self.statusBar().showMessage("已请求记录当前帧", 2000)
            return
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)


def _detect_charuco_pose(
    image_bgr: np.ndarray,
    detector,
    board,
    charuco_detector,
    calibration: CameraCalibration,
) -> BoardDetectionResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return BoardDetectionResult(False, 0, None, None, None)

    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    charuco_count = 0 if charuco_ids is None else len(charuco_ids)
    if charuco_ids is None or charuco_corners is None or int(charuco_count) < 4:
        return BoardDetectionResult(False, int(charuco_count or 0), None, None, None)

    pose = _estimate_charuco_board_pose(
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        board=board,
        calibration=calibration,
    )
    if pose is None:
        return BoardDetectionResult(True, int(charuco_count), None, None, charuco_corners)

    return BoardDetectionResult(
        board_visible=True,
        detected_corners=int(charuco_count),
        reprojection_error_px=pose[2],
        board_pose=pose[0],
        corners_px=charuco_corners,
    )


def _estimate_charuco_board_pose(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board,
    calibration: CameraCalibration,
) -> tuple[Transform, np.ndarray, float] | None:
    obj_points, img_points = _charuco_object_image_points(charuco_corners, charuco_ids, board)
    if len(obj_points) < 4:
        return None
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    return _pose_from_rvec_tvec(rvec, tvec, charuco_corners, charuco_ids, board, calibration)


def _pose_from_rvec_tvec(
    rvec: np.ndarray,
    tvec: np.ndarray,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board,
    calibration: CameraCalibration,
) -> tuple[Transform, np.ndarray, float]:
    obj_points, img_points = _charuco_object_image_points(charuco_corners, charuco_ids, board)
    projected, _ = cv2.projectPoints(
        obj_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    projected_2d = projected.reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(projected_2d - img_points, axis=1)))
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = Transform(
        translation=Translation(*np.asarray(tvec, dtype=np.float64).reshape(3).tolist()),
        rotation=Quaternion.from_SO3(rotation_matrix),
    )
    return transform, projected_2d, reprojection_error


def _charuco_object_image_points(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board,
) -> tuple[np.ndarray, np.ndarray]:
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    img_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    obj_points = board_corners[ids_flat]
    return obj_points, img_points


# endregion


# region 绘制与输出
def _draw_preview(
    image_bgr: np.ndarray,
    board_result: BoardDetectionResult,
    frame_index: int,
    elapsed_ms: float,
    config: CharucoConfig,
) -> np.ndarray:
    canvas = image_bgr.copy()
    status = "visible" if board_result.board_visible else "missing"
    text_lines = [
        f"frame={frame_index} compute={elapsed_ms:.2f}ms status={status}",
        f"corners={board_result.detected_corners} reproj={board_result.reprojection_error_px}",
        f"board={config.squares_x}x{config.squares_y} square={config.square_length_mm:.2f}mm marker={config.marker_length_mm:.2f}mm",
    ]
    if board_result.corners_px is not None:
        pts = np.round(board_result.corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        for pt in board_result.corners_px.reshape(-1, 2):
            cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 0, 255), -1, cv2.LINE_AA)
    if board_result.board_pose is not None:
        text_lines.append(board_result.board_pose.as_string(with_name=False))

    _draw_text_block(canvas, text_lines, (18, 28))
    return canvas


def _draw_text_block(canvas: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def _to_qimage(image_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    bytes_per_line = int(rgb.strides[0])
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def _save_sample_frame(frames_dir: Path, sample_index: int, image_bgr: np.ndarray) -> None:
    output_path = frames_dir / f"sample_{sample_index:03d}.png"
    cv2.imwrite(str(output_path), image_bgr)


def _write_samples_csv(csv_path: Path, samples: list[SampleRow]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "sample_index",
                "timestamp_s",
                "frame_index",
                "board_visible",
                "detected_corners",
                "reprojection_error_px",
                "board_tx_mm",
                "board_ty_mm",
                "board_tz_mm",
                "board_qw",
                "board_qx",
                "board_qy",
                "board_qz",
                "robot_tx_mm",
                "robot_ty_mm",
                "robot_tz_mm",
                "robot_qw",
                "robot_qx",
                "robot_qy",
                "robot_qz",
            ]
        )
        for item in samples:
            writer.writerow(
                [
                    item.sample_index,
                    f"{item.timestamp_s:.6f}",
                    item.frame_index,
                    int(item.board_visible),
                    item.detected_corners,
                    "" if item.reprojection_error_px is None else f"{item.reprojection_error_px:.6f}",
                    *_transform_csv_fields(item.board_pose),
                    *_transform_csv_fields(item.robot_pose),
                ]
            )


def _write_board_detection_csv(csv_path: Path, samples: list[SampleRow]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["sample_index", "board_visible", "detected_corners", "reprojection_error_px"])
        for item in samples:
            writer.writerow(
                [
                    item.sample_index,
                    int(item.board_visible),
                    item.detected_corners,
                    "" if item.reprojection_error_px is None else f"{item.reprojection_error_px:.6f}",
                ]
            )


def _transform_csv_fields(transform: Transform | None) -> list[str]:
    if transform is None:
        return [""] * 7
    translation = transform.translation
    rotation = transform.rotation
    return [
        f"{translation.x:.6f}",
        f"{translation.y:.6f}",
        f"{translation.z:.6f}",
        f"{rotation.w:.8f}",
        f"{rotation.x:.8f}",
        f"{rotation.y:.8f}",
        f"{rotation.z:.8f}",
    ]


# endregion


# region 机器人位姿与标定
def _load_robot_pose_rows(csv_path: Path) -> list[RobotPoseRow]:
    rows: list[RobotPoseRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                RobotPoseRow(
                    sample_index=int(row["sample_index"]),
                    pose=_transform_from_row(row, prefix="robot_"),
                )
            )
    return rows


def _consume_robot_pose(
    robot_pose_rows: list[RobotPoseRow],
    cursor: int,
    sample_index: int,
) -> Transform | None:
    if cursor >= len(robot_pose_rows):
        return None
    row = robot_pose_rows[cursor]
    if row.sample_index != sample_index:
        logger.warning(f"机器人位姿样本索引不匹配：当前采样 {sample_index}，CSV 为 {row.sample_index}")
    return row.pose


def _run_calibration(session_dir: Path, samples: list[SampleRow]) -> None:
    board_poses: list[PoseLike] = []
    robot_poses: list[PoseLike] = []
    for item in samples:
        if item.board_pose is None or item.robot_pose is None:
            continue
        board_poses.append(item.board_pose)
        robot_poses.append(item.robot_pose)
    if len(board_poses) < 3:
        logger.warning("可用于标定的样本不足，已跳过手眼求解。")
        return
    result = calibrate_hand_eye_from_pose_sequences(
        group_a_poses=robot_poses,
        group_b_poses=board_poses,
        pair_mode="all",
    )
    output_path = session_dir / "hand_eye_result.txt"
    output_path.write_text(
        "\n".join(
            [
                result.transform.as_string(with_name=True),
                f"sample_count={result.residual.sample_count}",
                f"rotation_rmse_deg={result.residual.rotation_rmse_deg:.6f}",
                f"rotation_max_deg={result.residual.rotation_max_deg:.6f}",
                f"translation_rmse={result.residual.translation_rmse:.6f}",
                f"translation_max={result.residual.translation_max:.6f}",
            ]
        ),
        encoding="utf-8",
    )
    logger.success(f"手眼结果已写入：{output_path}")


# endregion


# region 工具
def _build_dictionary(dictionary_name: str):
    name_to_id = {
        "DICT_5X5_50": int(cv2.aruco.DICT_5X5_50),
        "DICT_5X5_100": int(cv2.aruco.DICT_5X5_100),
        "DICT_5X5_250": int(cv2.aruco.DICT_5X5_250),
        "DICT_5X5_1000": int(cv2.aruco.DICT_5X5_1000),
    }
    if dictionary_name not in name_to_id:
        raise ValueError(f"不支持的字典：{dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(name_to_id[dictionary_name])


def _build_charuco_board(config: CharucoConfig, dictionary):
    return cv2.aruco.CharucoBoard(
        (int(config.squares_x), int(config.squares_y)),
        float(config.square_length_mm),
        float(config.marker_length_mm),
        dictionary,
    )


def _build_detector(dictionary):
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def _build_charuco_detector(board):
    return cv2.aruco.CharucoDetector(board)


def _validate_charuco_config(config: CharucoConfig) -> None:
    if config.squares_x < 2 or config.squares_y < 2:
        raise ValueError("Charuco 棋盘至少需要 2x2 个棋格。")
    if config.square_length_mm <= 0.0:
        raise ValueError("square_length_mm 必须大于 0。")
    if config.marker_length_mm <= 0.0:
        raise ValueError("marker_length_mm 必须大于 0。")
    if config.marker_length_mm >= config.square_length_mm:
        raise ValueError("marker_length_mm 必须小于 square_length_mm。")


def _validate_runtime_requirements() -> None:
    _ = cv2.aruco.ArucoDetector
    _ = cv2.aruco.CharucoDetector
    _ = cv2.aruco.CharucoBoard


def _create_session_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _read_camera_calibration(session: OrbbecSession) -> CameraCalibration:
    calibration = session.get_color_intrinsics()
    return CameraCalibration(
        width=int(calibration.width),
        height=int(calibration.height),
        camera_matrix=np.asarray(calibration.camera_matrix(), dtype=np.float64).reshape(3, 3),
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
    )


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
    raise RuntimeError(f"当前 color format 暂未支持直接预览：{color_format}")


def _transform_from_row(row: dict[str, str], prefix: str) -> Transform:
    tx = float(row[f"{prefix}tx_mm"])
    ty = float(row[f"{prefix}ty_mm"])
    tz = float(row[f"{prefix}tz_mm"])
    qw = float(row[f"{prefix}qw"])
    qx = float(row[f"{prefix}qx"])
    qy = float(row[f"{prefix}qy"])
    qz = float(row[f"{prefix}qz"])
    return Transform(
        translation=Translation(tx, ty, tz),
        rotation=Quaternion(w=qw, x=qx, y=qy, z=qz),
    )


def _has_calibration_samples(samples: list[SampleRow]) -> bool:
    return any(item.board_pose is not None and item.robot_pose is not None for item in samples)


# endregion


if __name__ == "__main__":
    try:
        config_arg = CharucoConfig(
            squares_x=7,
            squares_y=5,
            square_length_mm=DEFAULT_SQUARE_LENGTH_MM,
            marker_length_mm=DEFAULT_MARKER_LENGTH_MM,
            dictionary_name=DEFAULT_DICT_NAME,
        )
        main(
            config=config_arg,
            timeout_ms=DEFAULT_TIMEOUT_MS,
            capture_fps=DEFAULT_CAPTURE_FPS,
            output_root=DEFAULT_OUTPUT_ROOT,
            robot_pose_csv=None,
        )
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
    except Exception as exc:
        logger.exception(f"程序异常退出：{exc}")
        raise
