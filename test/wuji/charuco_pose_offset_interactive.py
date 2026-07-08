from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient  # noqa: E402
from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from src.calibration import CHARUCO_200_12_9, CharucoPoseEstimator  # noqa: E402
from src.utils.datas import Quaternion, Transform, Translation  # noqa: E402
from test.wuji.ball_pose_offset_interactive import (  # noqa: E402
    DEFAULT_ARM_IP,
    DEFAULT_BODY_FONT_SIZE,
    DEFAULT_CAMERA_NAME,
    DEFAULT_TITLE_FONT_SIZE,
    RobotSnapshot,
    _create_session_dir,
    _format_triplet,
    _load_font,
    _print_sdk_result,
    _shutdown_robot,
    _vector_norm,
)
from test.wuji.charuco_detect import (  # noqa: E402
    DEFAULT_MIN_CHARUCO_CORNERS,
    DEFAULT_ORIN_SERVICE_ADDR,
    _read_camera_calibration,
    _validate_runtime_requirements,
)
from test.wuji.xcoresdk_arm_cli_test import (  # noqa: E402
    _ensure_nrt_motion_ready,
)


# region 默认参数
DEFAULT_WINDOW_NAME = "Charuco Board Pose Drag Verification"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test" / "wuji" / ".archive" / "charuco_pose_offset_interactive"
DEFAULT_HAND_EYE_PATH = (
    PROJECT_ROOT / "experiments" / "hand_eye" / "runs" / "20260707_154334" / "base_board_flange_camera_joint_result.txt"
)
DEFAULT_CAMERA_TIMEOUT_MS = 30_000
DEFAULT_CAPTURE_TIMEOUT_S = 3.0
DEFAULT_DETECTION_SOURCE = "charuco_pose_estimator"
DEFAULT_REFERENCE_FRAME_LABEL = "flangeInBase"
DEFAULT_DRAG_POWER_OFF_TIMEOUT_S = 5.0
DEFAULT_REFRESH_SLEEP_S = 0.03
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    recorded_at_iso: str
    flange_pose_in_base: Transform
    flange_translation_mm: tuple[float, float, float]
    flange_rpy_degrees: tuple[float, float, float]
    sdk_pose_has_elbow: bool
    sdk_pose_elbow_deg: float
    sdk_pose_conf_data: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class JointSolveResult:
    flange_camera: Transform
    base_board: Transform
    rotation_rmse_deg: float
    translation_rmse_mm: float
    max_rotation_deg: float
    max_translation_mm: float


@dataclass(frozen=True, slots=True)
class ComparisonSnapshot:
    actual_robot_delta_mm: tuple[float, float, float]
    actual_robot_delta_distance_mm: float
    actual_pose_camera_board: Transform
    actual_delta_camera_mm: tuple[float, float, float]
    actual_delta_distance_mm: float
    predicted_with_hand_eye_pose_camera_board: Transform
    predicted_with_hand_eye_delta_camera_mm: tuple[float, float, float]
    predicted_with_hand_eye_error_mm: tuple[float, float, float]
    predicted_with_hand_eye_error_distance_mm: float
    predicted_without_hand_eye_pose_camera_board: Transform
    predicted_without_hand_eye_delta_camera_mm: tuple[float, float, float]
    predicted_without_hand_eye_error_mm: tuple[float, float, float]
    predicted_without_hand_eye_error_distance_mm: float
    actual_delta_rotation_deg: float
    predicted_with_hand_eye_error_rotation_deg: float
    predicted_without_hand_eye_error_rotation_deg: float


@dataclass(frozen=True, slots=True)
class JointVerificationSnapshot:
    current_robot_pose_flange: Transform
    current_pose_camera_board: Transform
    estimated_base_board: Transform
    reference_base_board: Transform
    translation_error_mm: tuple[float, float, float]
    translation_error_distance_mm: float
    rotation_error_deg: float


@dataclass(frozen=True, slots=True)
class ReferenceSnapshot:
    recorded_at_iso: str
    robot_snapshot: RobotSnapshot
    detection_snapshot: DetectionSnapshot


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    sample_count: int
    valid_sample_count: int
    rotation_rmse_deg: float
    rotation_max_deg: float
    translation_rmse_mm: float
    translation_max_mm: float
    result_rotation_rmse_deg: float | None
    result_rotation_max_deg: float | None
    result_translation_rmse_mm: float | None
    result_translation_max_mm: float | None


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    frame_id: int
    timestamp_ms: float
    pose_camera_board: Transform
    pose_translation_mm: tuple[float, float, float]
    reprojection_error_px: float | None
    marker_count: int
    charuco_count: int
    detection_source: str


@dataclass(frozen=True, slots=True)
class FrameDetectionResult:
    preview_base_bgr: np.ndarray
    charuco_result: Any


@dataclass(slots=True)
class RuntimeState:
    latest_verification: JointVerificationSnapshot | None = None
    last_action_text: str = "等待首个有效 ChArUco 标定板位姿"


# endregion


# region 主流程
def main(
    service_addr: str = DEFAULT_ORIN_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    arm_ip: str = DEFAULT_ARM_IP,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    hand_eye_path: Path = DEFAULT_HAND_EYE_PATH,
    min_charuco_corners: int = DEFAULT_MIN_CHARUCO_CORNERS,
) -> int:
    logger.info("交互式 ChArUco 标定板位姿偏移验证启动")
    logger.info("服务地址 {} 相机 {}", service_addr, camera_name)
    logger.info("机械臂 {}", arm_ip)
    logger.info("验证边界：依赖真实相机、真实机械臂与 Orin 相机流；当前仅做运行与静态验证，未实连硬件")

    _validate_runtime_requirements()
    session_dir = _create_session_dir(output_root)
    latest_json_path = session_dir / "latest_comparison.json"
    hand_eye_result = _load_hand_eye_result(hand_eye_path)
    estimator = CharucoPoseEstimator(CHARUCO_200_12_9)
    state = RuntimeState()

    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(str(arm_ip))
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=DEFAULT_CAMERA_TIMEOUT_MS)
    try:
        intrinsics_response = client.get_camera_intrinsics(timeout_s=DEFAULT_CAPTURE_TIMEOUT_S)
        calibration = _read_camera_calibration(intrinsics_response)

        robot_info = robot.robotInfo(ec)
        _print_sdk_result("robotInfo", ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"连接机械臂失败: ip={arm_ip}")
        logger.success("机械臂已连接 type={} uid={}", robot_info.type, robot_info.id)
        if not _ensure_nrt_motion_ready(robot, ec):
            raise RuntimeError("机械臂运动准备失败")
        _enable_drag(robot, ec)
        frame_stream = client.subscribe_camera_color_frames(camera_name)
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            frame = next(frame_stream)
            frame_detection = _detect_charuco_frame(
                frame_color_bgr=np.asarray(frame.color_bgr, dtype=np.uint8),
                calibration=calibration,
                estimator=estimator,
                min_charuco_corners=min_charuco_corners,
            )
            detection_result = frame_detection.charuco_result
            robot_snapshot = _read_robot_snapshot(robot, ec)
            detection_snapshot = _build_detection_snapshot(frame_id=int(frame.frame_id), timestamp_ms=float(frame.timestamp_ms), result=detection_result)
            if detection_snapshot is not None:
                if state.latest_verification is None:
                    state.last_action_text = f"已进入实时验算 frame={detection_snapshot.frame_id}"
                state.latest_verification = _build_joint_verification_snapshot(
                    current_robot_snapshot=robot_snapshot,
                    current_detection_snapshot=detection_snapshot,
                    hand_eye_result=hand_eye_result,
                )
                _write_latest_comparison_json(latest_json_path, state.latest_verification)
            else:
                state.last_action_text = "等待首个有效 ChArUco 标定板位姿"

            preview_bgr = _build_preview_image(
                preview_base_bgr=frame_detection.preview_base_bgr,
                detection_result=detection_result,
                robot_snapshot=robot_snapshot,
                verification_snapshot=state.latest_verification,
                last_action_text=state.last_action_text,
            )
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            time.sleep(DEFAULT_REFRESH_SLEEP_S)
        return 0
    finally:
        client.close()
        cv2.destroyAllWindows()
        _shutdown_robot(robot, ec)


# endregion


# region ChArUco 检测
def _detect_charuco_frame(
    frame_color_bgr: np.ndarray,
    calibration: Any,
    estimator: CharucoPoseEstimator,
    min_charuco_corners: int,
) -> FrameDetectionResult:
    if frame_color_bgr.size == 0:
        logger.warning("相机返回空图像，跳过本帧")
        return FrameDetectionResult(
            preview_base_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
            charuco_result=None,
        )
    charuco_result = estimator.estimate_pose(
        image_bgr=frame_color_bgr,
        camera_matrix=calibration.camera_matrix,
        dist_coeffs=calibration.dist_coeffs,
        min_charuco_corners=int(min_charuco_corners),
    )
    return FrameDetectionResult(
        preview_base_bgr=frame_color_bgr.copy(),
        charuco_result=charuco_result,
    )


def _build_detection_snapshot(
    frame_id: int,
    timestamp_ms: float,
    result: Any,
) -> DetectionSnapshot | None:
    if result is None or not result.board_visible or result.transform_se3 is None:
        return None
    pose_transform = Transform.from_SE3(np.asarray(result.transform_se3, dtype=np.float64))
    return DetectionSnapshot(
        frame_id=int(frame_id),
        timestamp_ms=float(timestamp_ms),
        pose_camera_board=pose_transform,
        pose_translation_mm=(
            float(pose_transform.translation.x),
            float(pose_transform.translation.y),
            float(pose_transform.translation.z),
        ),
        reprojection_error_px=None if result.reprojection_error_px is None else float(result.reprojection_error_px),
        marker_count=int(result.marker_count),
        charuco_count=int(result.charuco_count),
        detection_source=DEFAULT_DETECTION_SOURCE,
    )


def _enable_drag(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _print_sdk_result("setMotionControlMode(NrtCommandMode)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("切换非实时运动模式失败")
    robot.setPowerState(False, ec)
    _print_sdk_result("setPowerState(False)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前下电失败")
    if not _wait_for_power_off(robot, ec):
        raise RuntimeError("拖动前未在超时内确认下电")
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _print_sdk_result("setOperateMode(manual)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前切换手动模式失败")
    robot.moveReset(ec)
    _print_sdk_result("moveReset", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("拖动前 moveReset 失败")
    robot.enableDrag(
        int(xCoreSDK_python.DragParameterSpace.cartesianSpace),
        int(xCoreSDK_python.DragParameterType.freely),
        ec,
        enable_drag_button=False,
    )
    _print_sdk_result("enableDrag(cartesianSpace, freely)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("开启拖动失败")


def _wait_for_power_off(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object], timeout_s: float = DEFAULT_DRAG_POWER_OFF_TIMEOUT_S) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        power_state = robot.powerState(ec)
        _print_sdk_result("powerState", ec)
        if ec.get("ec", 0) != 0:
            return False
        if power_state == xCoreSDK_python.PowerState.off:
            return True
        time.sleep(0.2)
    return False


def _predict_current_camera_board_pose(
    reference_robot_pose_flange: Transform,
    current_robot_pose_flange: Transform,
    reference_pose_camera_board: Transform,
    tool_camera_transform: Transform,
) -> Transform:
    ref_camera_reference = reference_robot_pose_flange @ tool_camera_transform
    ref_camera_current = current_robot_pose_flange @ tool_camera_transform
    ref_board = ref_camera_reference @ reference_pose_camera_board
    current_camera_ref = _inverse_transform(ref_camera_current)
    return current_camera_ref @ ref_board


def _inverse_transform(transform: Transform) -> Transform:
    return Transform.from_SE3(np.linalg.inv(transform.as_SE3()))


def _translation_delta(current_transform: Transform, reference_transform: Transform) -> tuple[float, float, float]:
    return (
        float(current_transform.translation.x - reference_transform.translation.x),
        float(current_transform.translation.y - reference_transform.translation.y),
        float(current_transform.translation.z - reference_transform.translation.z),
    )


def _relative_rotation_deg(reference_transform: Transform, current_transform: Transform) -> float:
    delta = _inverse_transform(reference_transform) @ current_transform
    return _rotation_angle_deg(delta.rotation.as_SE3()[:3, :3])


def _relative_rotation_error_deg(actual_transform: Transform, predicted_transform: Transform) -> float:
    delta = _inverse_transform(actual_transform) @ predicted_transform
    return _rotation_angle_deg(delta.rotation.as_SE3()[:3, :3])


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    trace_value = float(np.trace(np.asarray(rotation_matrix, dtype=np.float64)))
    cosine_value = float(np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_value)))


# endregion


# region 预览与记录
def _build_preview_image(
    preview_base_bgr: np.ndarray,
    detection_result: Any,
    robot_snapshot: RobotSnapshot,
    verification_snapshot: JointVerificationSnapshot | None,
    last_action_text: str,
) -> np.ndarray:
    canvas = _build_overlay_canvas(preview_base_bgr, detection_result)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(DEFAULT_TITLE_FONT_SIZE)
    body_font = _load_font(DEFAULT_BODY_FONT_SIZE)

    lines = [
        "ChArUco 标定板位姿偏移交互验证",
        "空格=手动重置基准  C=清空  Q=退出  当前模式=拖动采样 + 实时验算",
        f"当前 robot xyz(mm)=({_format_triplet(robot_snapshot.flange_translation_mm)})",
        f"当前 robot rpy(deg)=({_format_triplet(robot_snapshot.flange_rpy_degrees)})",
        _build_detection_status_text(detection_result),
        f"状态: {last_action_text}",
    ]
    if verification_snapshot is not None:
        lines.extend(
            [
                "当前帧验算 base_board_est(mm)=("
                f"{_format_triplet(verification_snapshot.estimated_base_board.translation.to_list())})",
                "base_board_error(mm)=("
                f"{_format_triplet(verification_snapshot.translation_error_mm)}) "
                f"|err|={verification_snapshot.translation_error_distance_mm:.3f} "
                f"rot_err={verification_snapshot.rotation_error_deg:.3f} deg",
            ]
        )

    _draw_info_panel(draw, lines, title_font=title_font, body_font=body_font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _build_overlay_canvas(preview_base_bgr: np.ndarray, detection_result: Any) -> np.ndarray:
    canvas = np.asarray(preview_base_bgr, dtype=np.uint8).copy()
    if detection_result is None:
        return canvas
    if detection_result.marker_corners_px:
        cv2.aruco.drawDetectedMarkers(canvas, detection_result.marker_corners_px, detection_result.marker_ids, borderColor=(255, 180, 0))
    if detection_result.charuco_corners_px is not None:
        for corner_index, corner in enumerate(detection_result.charuco_corners_px):
            point = tuple(int(v) for v in np.round(corner))
            cv2.circle(canvas, point, 4, (0, 255, 255), -1, cv2.LINE_AA)
            if detection_result.charuco_ids is not None and corner_index < len(detection_result.charuco_ids):
                charuco_id = int(detection_result.charuco_ids.flatten()[corner_index])
                cv2.putText(canvas, str(charuco_id), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    if detection_result.board_visible and detection_result.rvec is not None and detection_result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            _current_camera_matrix(canvas),
            np.zeros((5,), dtype=np.float64),
            detection_result.rvec.reshape(3, 1),
            detection_result.tvec.reshape(3, 1),
            30.0,
            2,
        )
    return canvas


def _build_detection_status_text(detection_result: Any) -> str:
    if detection_result is None:
        return "当前检测: 无有效图像"
    return (
        f"当前检测 source={DEFAULT_DETECTION_SOURCE} markers={int(detection_result.marker_count)} "
        f"charuco={int(detection_result.charuco_count)} "
        f"reproj_px={'None' if detection_result.reprojection_error_px is None else f'{float(detection_result.reprojection_error_px):.3f}'} "
        f"visible={bool(detection_result.board_visible)}"
    )


def _read_robot_snapshot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> RobotSnapshot:
    cartesian_pose = robot.cartPosture(xCoreSDK_python.flangeInBase, ec)
    _print_sdk_result("cartPosture(flangeInBase)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取机械臂法兰位姿失败")
    translation_mm = (
        float(cartesian_pose.trans[0]) * 1000.0,
        float(cartesian_pose.trans[1]) * 1000.0,
        float(cartesian_pose.trans[2]) * 1000.0,
    )
    rpy_degrees = (
        float(np.degrees(float(cartesian_pose.rpy[0]))),
        float(np.degrees(float(cartesian_pose.rpy[1]))),
        float(np.degrees(float(cartesian_pose.rpy[2]))),
    )
    return RobotSnapshot(
        recorded_at_iso=datetime.now().isoformat(timespec="milliseconds"),
        flange_pose_in_base=_sdk_pose_to_transform(translation_mm, rpy_degrees),
        flange_translation_mm=translation_mm,
        flange_rpy_degrees=rpy_degrees,
        sdk_pose_has_elbow=bool(cartesian_pose.hasElbow),
        sdk_pose_elbow_deg=float(np.degrees(float(cartesian_pose.elbow))),
        sdk_pose_conf_data=tuple(int(value) for value in list(cartesian_pose.confData)),
    )


def _draw_info_panel(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    title_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    body_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x0 = 18
    y0 = 18
    for index, text in enumerate(lines):
        font = title_font if index == 0 else body_font
        fill = (255, 255, 255) if index == 0 else (230, 230, 230)
        draw.text((x0, y0 + index * 28), text, font=font, fill=fill, stroke_width=2, stroke_fill=(20, 20, 20))


def _current_camera_matrix(canvas: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, canvas.shape[1] / 2.0],
            [0.0, 1.0, canvas.shape[0] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _sdk_pose_to_transform(
    translation_mm: tuple[float, float, float],
    rpy_degrees: tuple[float, float, float],
) -> Transform:
    from scipy.spatial.transform import Rotation as Rotation3D

    rotation_matrix = Rotation3D.from_euler("XYZ", list(rpy_degrees), degrees=True).as_matrix()
    return Transform(
        translation=Translation(*translation_mm),
        rotation=Quaternion.from_SO3(rotation_matrix),
    )


def _load_hand_eye_result(hand_eye_path: Path) -> JointSolveResult:
    if not hand_eye_path.is_file():
        raise FileNotFoundError(f"手眼标定结果文件不存在: {hand_eye_path}")
    lines = hand_eye_path.read_text(encoding="utf-8").splitlines()
    matrix_rows: list[list[float]] = []
    base_board_rows: list[list[float]] = []
    collecting = False
    collecting_base_board = False
    rotation_rmse_deg = 0.0
    translation_rmse_mm = 0.0
    max_rotation_deg = 0.0
    max_translation_mm = 0.0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("rotation_rmse_deg="):
            rotation_rmse_deg = float(stripped.split("=", 1)[1])
        elif stripped.startswith("translation_rmse_mm="):
            translation_rmse_mm = float(stripped.split("=", 1)[1])
        elif stripped.startswith("max_rotation_deg="):
            max_rotation_deg = float(stripped.split("=", 1)[1])
        elif stripped.startswith("max_translation_mm="):
            max_translation_mm = float(stripped.split("=", 1)[1])
        elif stripped == "T_flange_camera:":
            collecting = True
            collecting_base_board = False
            continue
        elif stripped == "T_base_board:":
            collecting = False
            collecting_base_board = True
            continue
        if collecting:
            if stripped == "":
                break
            values = _parse_matrix_row(stripped)
            if len(values) != 4:
                raise ValueError(f"手眼矩阵行格式错误: {line}")
            matrix_rows.append(values)
            if len(matrix_rows) == 4:
                collecting = False
        elif collecting_base_board:
            if stripped == "":
                break
            values = _parse_matrix_row(stripped)
            if len(values) != 4:
                raise ValueError(f"手眼矩阵行格式错误: {line}")
            base_board_rows.append(values)
            if len(base_board_rows) == 4:
                collecting_base_board = False
    flange_matrix = np.asarray(matrix_rows, dtype=np.float64)
    base_board_matrix = np.asarray(base_board_rows, dtype=np.float64)
    if flange_matrix.shape != (4, 4) or base_board_matrix.shape != (4, 4):
        raise ValueError(
            f"手眼矩阵维度错误: flange_shape={flange_matrix.shape}, base_board_shape={base_board_matrix.shape}, path={hand_eye_path}"
        )
    logger.info("已加载手眼标定结果 {}", hand_eye_path)
    return JointSolveResult(
        flange_camera=Transform.from_SE3(flange_matrix),
        base_board=Transform.from_SE3(base_board_matrix),
        rotation_rmse_deg=rotation_rmse_deg,
        translation_rmse_mm=translation_rmse_mm,
        max_rotation_deg=max_rotation_deg,
        max_translation_mm=max_translation_mm,
    )


def _parse_matrix_row(line: str) -> list[float]:
    cleaned = line.replace(",", " ")
    tokens = [token for token in cleaned.split() if token]
    return [float(token) for token in tokens]


def _build_joint_verification_snapshot(
    current_robot_snapshot: RobotSnapshot,
    current_detection_snapshot: DetectionSnapshot,
    hand_eye_result: JointSolveResult,
) -> JointVerificationSnapshot | None:
    current_pose_camera_board = current_detection_snapshot.pose_camera_board
    if current_pose_camera_board is None:
        return None
    estimated_base_board = current_robot_snapshot.flange_pose_in_base @ hand_eye_result.flange_camera @ current_pose_camera_board
    reference_base_board = hand_eye_result.base_board
    delta = _inverse_transform(reference_base_board) @ estimated_base_board
    translation_error_mm = (
        float(delta.translation.x),
        float(delta.translation.y),
        float(delta.translation.z),
    )
    return JointVerificationSnapshot(
        current_robot_pose_flange=current_robot_snapshot.flange_pose_in_base,
        current_pose_camera_board=current_pose_camera_board,
        estimated_base_board=estimated_base_board,
        reference_base_board=reference_base_board,
        translation_error_mm=translation_error_mm,
        translation_error_distance_mm=_vector_norm(translation_error_mm),
        rotation_error_deg=_rotation_angle_deg(delta.rotation.as_SE3()[:3, :3]),
    )


def _write_reference_snapshot_json(path: Path, reference_snapshot: ReferenceSnapshot) -> None:
    payload = {
        "recorded_at_iso": reference_snapshot.recorded_at_iso,
        "robot_flange_translation_mm": list(reference_snapshot.robot_snapshot.flange_translation_mm),
        "robot_flange_rpy_degrees": list(reference_snapshot.robot_snapshot.flange_rpy_degrees),
        "robot_flange_pose_in_base": reference_snapshot.robot_snapshot.flange_pose_in_base.as_SE3().tolist(),
        "frame_id": reference_snapshot.detection_snapshot.frame_id,
        "timestamp_ms": reference_snapshot.detection_snapshot.timestamp_ms,
        "pose_camera_board": reference_snapshot.detection_snapshot.pose_camera_board.as_SE3().tolist(),
        "pose_translation_mm": list(reference_snapshot.detection_snapshot.pose_translation_mm),
        "reprojection_error_px": reference_snapshot.detection_snapshot.reprojection_error_px,
        "marker_count": reference_snapshot.detection_snapshot.marker_count,
        "charuco_count": reference_snapshot.detection_snapshot.charuco_count,
        "detection_source": reference_snapshot.detection_snapshot.detection_source,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_latest_comparison_json(path: Path, verification_snapshot: JointVerificationSnapshot | None) -> None:
    if verification_snapshot is None:
        path.write_text("{}", encoding="utf-8")
        return
    payload = {
        "current_robot_pose_flange": verification_snapshot.current_robot_pose_flange.as_SE3().tolist(),
        "current_pose_camera_board": verification_snapshot.current_pose_camera_board.as_SE3().tolist(),
        "estimated_base_board": verification_snapshot.estimated_base_board.as_SE3().tolist(),
        "reference_base_board": verification_snapshot.reference_base_board.as_SE3().tolist(),
        "translation_error_mm": list(verification_snapshot.translation_error_mm),
        "translation_error_distance_mm": verification_snapshot.translation_error_distance_mm,
        "rotation_error_deg": verification_snapshot.rotation_error_deg,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# endregion


# region CLI
def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="interactive charuco board pose offset validation")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_ORIN_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--arm-ip", type=str, default=DEFAULT_ARM_IP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hand-eye-path", type=Path, default=DEFAULT_HAND_EYE_PATH)
    parser.add_argument("--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_args = _parse_cli(sys.argv[1:])
        raise SystemExit(
            main(
                service_addr=str(cli_args.service_addr),
                camera_name=str(cli_args.camera_name),
                arm_ip=str(cli_args.arm_ip),
                output_root=Path(cli_args.output_root),
                hand_eye_path=Path(cli_args.hand_eye_path),
                min_charuco_corners=int(cli_args.min_charuco_corners),
            )
        )
    raise SystemExit(main())
