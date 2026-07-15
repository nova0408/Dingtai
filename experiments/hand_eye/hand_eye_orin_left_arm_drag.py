from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rotation3D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.protocol import CameraColorFramePacket
from camera_pipeline.client import CameraPipelineClient
from sdk.xcoresdk import xCoreSDK_python
from src.calibration import CHARUCO_200_12_9, CharucoPoseEstimator

# region 默认参数
DEFAULT_WINDOW_NAME = "Left Arm Hand-Eye Calibration"
DEFAULT_WINDOW_WIDTH = 1680
DEFAULT_WINDOW_HEIGHT = 960
DEFAULT_ORIN_SERVICE_ADDR = "tcp://192.168.1.121:6200"
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_LEFT_ARM_IP = "192.168.1.161"
DEFAULT_OUTPUT_ROOT = Path("experiments/hand_eye/runs")
DEFAULT_CAMERA_TIMEOUT_S = 10.0
DEFAULT_MIN_CHARUCO_CORNERS = 6
EXPECTED_LEFT_ARM_TYPE = "AR5-5_0.8L-W4C1C9-ZY2"
DEFAULT_TOOL_NAME = "g_tool_0"
DEFAULT_WOBJ_NAME = "g_wobj_0"
AXIS_DRAW_LENGTH_MM = 28.0
DEFAULT_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_FONT_SIZE = 20
TRANSLATION_SPAN_TARGETS_MM = {
    "x": 120.0,
    "y": 120.0,
    "z": 80.0,
}
ROTATION_SPAN_TARGETS_DEG = {
    "roll": 35.0,
    "pitch": 35.0,
    "yaw": 45.0,
}
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_name: str
    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    host_timestamp_iso: str
    joint_degrees: tuple[float, ...]
    toolset_end_frame: xCoreSDK_python.Frame
    toolset_ref_frame: xCoreSDK_python.Frame
    end_pose_in_ref: xCoreSDK_python.CartesianPosition
    end_translation_mm: tuple[float, float, float]
    end_rpy_degrees: tuple[float, float, float]
    sdk_pose_has_elbow: bool
    sdk_pose_elbow_deg: float
    sdk_pose_conf_data: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_index: int
    host_timestamp_iso: str
    host_timestamp_s: float
    frame_id: int
    camera_timestamp_ms: float
    board_visible: bool
    marker_count: int
    charuco_count: int
    reprojection_error_px: float | None
    robot_snapshot: RobotSnapshot
    board_pose_camera_board: np.ndarray | None
    raw_frame_path: Path
    preview_frame_path: Path


@dataclass(frozen=True, slots=True)
class ConnectedLeftArm:
    robot_ip: str
    robot: xCoreSDK_python.xMateErProRobot
    robot_type: str
    robot_uid: str
    ec: dict[str, object]


@dataclass(frozen=True, slots=True)
class SamplingCoverageSummary:
    sample_count: int
    span_x_mm: float
    span_y_mm: float
    span_z_mm: float
    span_roll_deg: float
    span_pitch_deg: float
    span_yaw_deg: float


@dataclass(frozen=True, slots=True)
class BoardInBaseStats:
    mean_translation_m: np.ndarray
    std_translation_m: np.ndarray
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray
    translation_mean_error_m: float
    translation_max_error_m: float
    rotation_mean_error_deg: float
    rotation_max_error_deg: float


@dataclass(frozen=True, slots=True)
class HandEyeCalibrationResult:
    method_name: str
    robot_source: str
    camera_source: str
    toolset_source: str
    tool_cam: np.ndarray
    base_board_list: tuple[np.ndarray, ...]
    stats: BoardInBaseStats
    used_indices: tuple[int, ...]


# endregion


# region 主流程
def main() -> int:
    args = _parse_cli()
    if args.replay_run_dir:
        _run_replay(Path(args.replay_run_dir))
        return 0
    _validate_runtime_requirements()
    session_dir = _create_session_dir(Path(args.output_root))
    logger.info(f"输出目录：{session_dir}")

    left_arm = _connect_left_arm(
        robot_ip=str(args.left_arm_ip),
    )
    try:
        _enable_left_arm_drag(left_arm)
        _run_calibration_session(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            session_dir=session_dir,
            left_arm=left_arm,
            min_charuco_corners=int(args.min_charuco_corners),
        )
        return 0
    finally:
        _shutdown_left_arm(left_arm)


def _run_calibration_session(
    service_addr: str,
    camera_name: str,
    session_dir: Path,
    left_arm: ConnectedLeftArm,
    min_charuco_corners: int,
) -> None:
    client = CameraPipelineClient(
        service_addr=service_addr,
        timeout_ms=int(DEFAULT_CAMERA_TIMEOUT_S * 1000.0),
    )
    estimator = CharucoPoseEstimator(CHARUCO_200_12_9)
    calibration = _read_camera_calibration(client)
    raw_frames_dir = session_dir / "frames_raw"
    preview_frames_dir = session_dir / "frames_preview"
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    preview_frames_dir.mkdir(parents=True, exist_ok=True)
    extrinsic_result_path = session_dir / "hand_eye_result.txt"
    samples: list[SampleRecord] = []
    last_result: HandEyeCalibrationResult | None = None
    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

    try:
        for frame_packet in client.subscribe_camera_color_frames(camera_name):
            robot_snapshot = _read_left_arm_snapshot(left_arm)
            frame_bgr = np.asarray(frame_packet.color_bgr, dtype=np.uint8).copy()
            started = time.perf_counter()
            charuco_result = estimator.estimate_pose(
                image_bgr=frame_bgr,
                camera_matrix=calibration.camera_matrix,
                dist_coeffs=calibration.dist_coeffs,
                min_charuco_corners=min_charuco_corners,
            )
            compute_ms = (time.perf_counter() - started) * 1000.0
            board_pose_camera_board = None
            if charuco_result.transform_se3 is not None:
                board_pose_camera_board = np.asarray(
                    charuco_result.transform_se3, dtype=np.float64
                ).reshape(4, 4)
            preview_bgr = _draw_preview(
                frame_bgr=frame_bgr,
                frame_packet=frame_packet,
                robot_snapshot=robot_snapshot,
                charuco_result=charuco_result,
                calibration_result=last_result,
                camera_calibration=calibration,
                compute_ms=compute_ms,
                recorded_samples=samples,
            )
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (13, 32, ord("p"), ord("P")):
                sample_record = _capture_sample(
                    session_dir=session_dir,
                    raw_frames_dir=raw_frames_dir,
                    preview_frames_dir=preview_frames_dir,
                    frame_packet=frame_packet,
                    raw_frame_bgr=frame_bgr,
                    preview_frame_bgr=preview_bgr,
                    robot_snapshot=robot_snapshot,
                    charuco_result=charuco_result,
                    board_pose_camera_board=board_pose_camera_board,
                    next_sample_index=len(samples) + 1,
                )
                samples.append(sample_record)
                _write_samples_csv(
                    session_dir / "samples.csv", samples, calibration_result=last_result
                )
                _write_sample_guidance_file(
                    session_dir / "sampling_guidance.txt", samples
                )
                last_result = _maybe_update_hand_eye_result(
                    output_path=extrinsic_result_path,
                    samples=samples,
                )
                logger.success(
                    "已记录样本 #{} visible={} charuco={} reproj={}",
                    sample_record.sample_index,
                    sample_record.board_visible,
                    sample_record.charuco_count,
                    _format_optional_float(
                        sample_record.reprojection_error_px, digits=4
                    ),
                )
            if cv2.getWindowProperty(DEFAULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        client.close()
        cv2.destroyAllWindows()
        _write_samples_csv(
            session_dir / "samples.csv", samples, calibration_result=last_result
        )
        _write_sample_guidance_file(session_dir / "sampling_guidance.txt", samples)
        _maybe_update_hand_eye_result(
            output_path=extrinsic_result_path,
            samples=samples,
        )


# endregion


# region handeyeCali 同款手眼标定
def _run_replay(run_dir: Path) -> None:
    csv_path = run_dir / "samples.csv"
    output_dir = run_dir / "handeyeCali_replay"
    logger.info("使用 handeyeCali 同款链路离线复算：{}", csv_path)
    base_tool_list, cam_board_list, used_indices, robot_source = _load_replay_samples(
        csv_path
    )
    result = _solve_park_hand_eye(
        base_tool_list=base_tool_list,
        cam_board_list=cam_board_list,
        used_indices=used_indices,
        robot_source=robot_source,
        toolset_source="not_recorded_in_csv",
    )
    _save_hand_eye_outputs(
        output_dir=output_dir,
        result=result,
    )
    summary_path = output_dir / "board_in_base_mean.txt"
    logger.success("离线复算完成，结果已写入：{}", summary_path)
    print(summary_path.read_text(encoding="utf-8"))


def _load_replay_samples(
    csv_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 samples.csv: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    fieldnames = [] if reader.fieldnames is None else list(reader.fieldnames)
    has_end_columns = all(
        column in fieldnames
        for column in (
            "robot_end_x_mm",
            "robot_end_y_mm",
            "robot_end_z_mm",
            "robot_end_roll_deg",
            "robot_end_pitch_deg",
            "robot_end_yaw_deg",
        )
    )
    if not has_end_columns:
        raise ValueError(
            "samples.csv 缺少 robot_end_*，当前已验证链路只接受 xCoreSDK_python.endInRef"
        )
    robot_prefix = "robot_end"

    base_tool_list: list[np.ndarray] = []
    cam_board_list: list[np.ndarray] = []
    used_indices: list[int] = []

    for row in rows:
        sample_index = int(_require_csv_value(row, "sample_index"))
        if int(_require_csv_value(row, "board_visible")) != 1:
            continue
        base_tool = _read_robot_transform_from_csv_row(row, robot_prefix)
        cam_board = _read_camera_board_transform_from_csv_row(row)
        base_tool_list.append(base_tool)
        cam_board_list.append(cam_board)
        used_indices.append(sample_index)

    if len(used_indices) < 6:
        raise RuntimeError(f"有效样本太少，无法按旧算法复算：{len(used_indices)}")
    return base_tool_list, cam_board_list, used_indices, robot_prefix


def _require_csv_value(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"CSV 缺少有效字段值：{key}")
    return value


def _read_robot_transform_from_csv_row(
    row: dict[str, str | None], robot_prefix: str
) -> np.ndarray:
    translation = (
        np.array(
            [
                float(_require_csv_value(row, f"{robot_prefix}_x_mm")),
                float(_require_csv_value(row, f"{robot_prefix}_y_mm")),
                float(_require_csv_value(row, f"{robot_prefix}_z_mm")),
            ],
            dtype=np.float64,
        )
        * 0.001
    )
    rotation = Rotation3D.from_euler(
        "xyz",
        [
            float(_require_csv_value(row, f"{robot_prefix}_roll_deg")),
            float(_require_csv_value(row, f"{robot_prefix}_pitch_deg")),
            float(_require_csv_value(row, f"{robot_prefix}_yaw_deg")),
        ],
        degrees=True,
    ).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _read_camera_board_transform_from_csv_row(row: dict[str, str | None]) -> np.ndarray:
    return _transform_from_csv_quaternion_fields(
        row=row,
        xyz_columns=("camera_board_x_mm", "camera_board_y_mm", "camera_board_z_mm"),
        quat_columns=(
            "camera_board_qw",
            "camera_board_qx",
            "camera_board_qy",
            "camera_board_qz",
        ),
    )


def _transform_from_csv_quaternion_fields(
    row: dict[str, str | None],
    xyz_columns: tuple[str, str, str],
    quat_columns: tuple[str, str, str, str],
) -> np.ndarray:
    translation = (
        np.array(
            [float(_require_csv_value(row, column)) for column in xyz_columns],
            dtype=np.float64,
        )
        * 0.001
    )
    qw, qx, qy, qz = (float(_require_csv_value(row, column)) for column in quat_columns)
    rotation = Rotation3D.from_quat([qx, qy, qz, qw]).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _solve_park_hand_eye(
    base_tool_list: list[np.ndarray],
    cam_board_list: list[np.ndarray],
    used_indices: list[int],
    robot_source: str,
    toolset_source: str,
) -> HandEyeCalibrationResult:
    rotations_gripper_to_base = [transform[:3, :3] for transform in base_tool_list]
    translations_gripper_to_base = [
        transform[:3, 3].reshape(3, 1) for transform in base_tool_list
    ]
    rotations_target_to_cam = [transform[:3, :3] for transform in cam_board_list]
    translations_target_to_cam = [
        transform[:3, 3].reshape(3, 1) for transform in cam_board_list
    ]
    rotation_cam_to_gripper, translation_cam_to_gripper = cv2.calibrateHandEye(
        R_gripper2base=rotations_gripper_to_base,
        t_gripper2base=translations_gripper_to_base,
        R_target2cam=rotations_target_to_cam,
        t_target2cam=translations_target_to_cam,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    tool_cam = np.eye(4, dtype=np.float64)
    tool_cam[:3, :3] = np.asarray(rotation_cam_to_gripper, dtype=np.float64).reshape(
        3, 3
    )
    tool_cam[:3, 3] = np.asarray(translation_cam_to_gripper, dtype=np.float64).reshape(
        3
    )
    base_board_list = tuple(
        base_tool @ tool_cam @ cam_board
        for base_tool, cam_board in zip(base_tool_list, cam_board_list, strict=True)
    )
    return HandEyeCalibrationResult(
        method_name="PARK",
        robot_source=robot_source,
        camera_source="camera_board",
        toolset_source=toolset_source,
        tool_cam=tool_cam,
        base_board_list=base_board_list,
        stats=_compute_board_in_base_stats(list(base_board_list)),
        used_indices=tuple(used_indices),
    )


def _compute_board_in_base_stats(base_board_list: list[np.ndarray]) -> BoardInBaseStats:
    translations = np.asarray(
        [transform[:3, 3] for transform in base_board_list], dtype=np.float64
    )
    mean_translation = np.mean(translations, axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)
    reference_rotation = base_board_list[0][:3, :3]
    rotation_errors_deg: list[float] = []
    for transform in base_board_list:
        rotation_delta = reference_rotation.T @ transform[:3, :3]
        rotation_errors_deg.append(
            float(
                np.linalg.norm(Rotation3D.from_matrix(rotation_delta).as_rotvec())
                * 180.0
                / np.pi
            )
        )
    rotation_errors = np.asarray(rotation_errors_deg, dtype=np.float64)
    return BoardInBaseStats(
        mean_translation_m=mean_translation,
        std_translation_m=np.std(translations, axis=0),
        translation_errors_m=translation_errors,
        rotation_errors_deg=rotation_errors,
        translation_mean_error_m=float(np.mean(translation_errors)),
        translation_max_error_m=float(np.max(translation_errors)),
        rotation_mean_error_deg=float(np.mean(rotation_errors)),
        rotation_max_error_deg=float(np.max(rotation_errors)),
    )


def _save_hand_eye_outputs(
    output_dir: Path,
    result: HandEyeCalibrationResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tool_cam = np.asarray(result.tool_cam, dtype=np.float64).reshape(4, 4)
    records: list[dict[str, float | int]] = []
    for sample_index, base_board in zip(
        result.used_indices, result.base_board_list, strict=True
    ):
        quat_xyzw = Rotation3D.from_matrix(base_board[:3, :3]).as_quat()
        rpy_deg = Rotation3D.from_matrix(base_board[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        records.append(
            {
                "sample_index": sample_index,
                "base_board_x_m": float(base_board[0, 3]),
                "base_board_y_m": float(base_board[1, 3]),
                "base_board_z_m": float(base_board[2, 3]),
                "base_board_x_mm": float(base_board[0, 3] * 1000.0),
                "base_board_y_mm": float(base_board[1, 3] * 1000.0),
                "base_board_z_mm": float(base_board[2, 3] * 1000.0),
                "base_board_qw": float(quat_xyzw[3]),
                "base_board_qx": float(quat_xyzw[0]),
                "base_board_qy": float(quat_xyzw[1]),
                "base_board_qz": float(quat_xyzw[2]),
                "base_board_roll_deg": float(rpy_deg[0]),
                "base_board_pitch_deg": float(rpy_deg[1]),
                "base_board_yaw_deg": float(rpy_deg[2]),
            }
        )
    with (output_dir / "board_in_base.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    mean_translation = result.stats.mean_translation_m
    std_translation = result.stats.std_translation_m
    tool_cam_rpy_deg = Rotation3D.from_matrix(tool_cam[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    summary_lines = [
        "Board pose in robot base frame",
        "Formula:",
        "T_base_board = T_base_tool @ T_tool_cam @ T_cam_board",
        "",
        f"robot_source = {result.robot_source}",
        f"camera_source = {result.camera_source}",
        f"method = {result.method_name}",
        f"translation_mean_error_m = {result.stats.translation_mean_error_m:.10f}",
        f"translation_max_error_m = {result.stats.translation_max_error_m:.10f}",
        f"rotation_mean_error_deg = {result.stats.rotation_mean_error_deg:.10f}",
        f"rotation_max_error_deg = {result.stats.rotation_max_error_deg:.10f}",
        "",
        "Mean board position in base:",
        f"x_m = {mean_translation[0]:.10f}",
        f"y_m = {mean_translation[1]:.10f}",
        f"z_m = {mean_translation[2]:.10f}",
        "",
        "Mean board position in base, unit mm:",
        f"x_mm = {mean_translation[0] * 1000.0:.6f}",
        f"y_mm = {mean_translation[1] * 1000.0:.6f}",
        f"z_mm = {mean_translation[2] * 1000.0:.6f}",
        "",
        "Std board position in base, unit mm:",
        f"x_std_mm = {std_translation[0] * 1000.0:.6f}",
        f"y_std_mm = {std_translation[1] * 1000.0:.6f}",
        f"z_std_mm = {std_translation[2] * 1000.0:.6f}",
        "",
        "Per-sample results are saved in board_in_base.csv",
        "",
        "T_tool_cam:",
        np.array2string(tool_cam, precision=10, suppress_small=False),
        "",
        "T_tool_cam_rpy_deg:",
        f"roll = {tool_cam_rpy_deg[0]:.6f}",
        f"pitch = {tool_cam_rpy_deg[1]:.6f}",
        f"yaw = {tool_cam_rpy_deg[2]:.6f}",
        "",
        "Used samples:",
        ", ".join(str(sample_index) for sample_index in result.used_indices),
    ]
    (output_dir / "board_in_base_mean.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


# endregion


# region 机器人连接与拖动
def _connect_left_arm(robot_ip: str) -> ConnectedLeftArm:
    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(robot_ip)
    robot_info = robot.robotInfo(ec)
    _print_sdk_result(f"robotInfo({robot_ip})", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"读取左臂机器人信息失败：ip={robot_ip}")
    if str(robot_info.type) != EXPECTED_LEFT_ARM_TYPE:
        raise RuntimeError(
            "连接到的机器人不是左臂控制器："
            f"expected={EXPECTED_LEFT_ARM_TYPE}, actual={robot_info.type}"
        )
    _apply_fixed_toolset(robot, ec)
    logger.success(
        f"左臂已连接：ip={robot_ip}, type={robot_info.type}, uid={robot_info.id}"
    )
    return ConnectedLeftArm(
        robot_ip=robot_ip,
        robot=robot,
        robot_type=str(robot_info.type),
        robot_uid=str(robot_info.id),
        ec=ec,
    )


def _enable_left_arm_drag(left_arm: ConnectedLeftArm) -> None:
    robot = left_arm.robot
    ec = left_arm.ec
    _apply_fixed_toolset(robot, ec)
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
    logger.success("左臂拖动已开启，程序启动后即可直接拖动示教。")


def _read_left_arm_snapshot(left_arm: ConnectedLeftArm) -> RobotSnapshot:
    robot = left_arm.robot
    ec = left_arm.ec
    _apply_fixed_toolset(robot, ec)
    joint_values_rad = [float(value) for value in robot.jointPos(ec)]
    _print_sdk_result("jointPos", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取左臂关节角失败")
    toolset = robot.toolset(ec)
    _print_sdk_result("toolset", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取左臂当前 toolset 失败")
    end_pose_in_ref = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取左臂末端位姿失败")
    joint_degrees = tuple(float(np.degrees(value)) for value in joint_values_rad)
    end_translation_mm = (
        float(end_pose_in_ref.trans[0]) * 1000.0,
        float(end_pose_in_ref.trans[1]) * 1000.0,
        float(end_pose_in_ref.trans[2]) * 1000.0,
    )
    end_rpy_degrees = (
        float(np.degrees(float(end_pose_in_ref.rpy[0]))),
        float(np.degrees(float(end_pose_in_ref.rpy[1]))),
        float(np.degrees(float(end_pose_in_ref.rpy[2]))),
    )
    return RobotSnapshot(
        host_timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        joint_degrees=joint_degrees,
        toolset_end_frame=toolset.end,
        toolset_ref_frame=toolset.ref,
        end_pose_in_ref=end_pose_in_ref,
        end_translation_mm=end_translation_mm,
        end_rpy_degrees=end_rpy_degrees,
        sdk_pose_has_elbow=bool(end_pose_in_ref.hasElbow),
        sdk_pose_elbow_deg=float(np.degrees(float(end_pose_in_ref.elbow))),
        sdk_pose_conf_data=tuple(
            int(value) for value in list(end_pose_in_ref.confData)
        ),
    )


def _apply_fixed_toolset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> xCoreSDK_python.Toolset:
    """无条件强制设置手眼标定使用的 tool/wobj。"""

    toolset = robot.setToolset(DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME, ec)
    _print_sdk_result(f"setToolset({DEFAULT_TOOL_NAME}, {DEFAULT_WOBJ_NAME})", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(
            "设置固定 toolset 失败："
            f"tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}"
        )
    return toolset


def _format_frame_values(frame: xCoreSDK_python.Frame) -> str:
    translation_mm = tuple(float(value) * 1000.0 for value in frame.trans)
    rotation_deg = tuple(float(np.degrees(float(value))) for value in frame.rpy)
    return f"xyz_mm=({translation_mm[0]:.3f}, {translation_mm[1]:.3f}, {translation_mm[2]:.3f}) rpy_deg=({rotation_deg[0]:.3f}, {rotation_deg[1]:.3f}, {rotation_deg[2]:.3f})"


def _wait_for_power_off(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float = 3.0,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        power_state = robot.powerState(ec)
        if power_state == xCoreSDK_python.PowerState.off:
            return True
        time.sleep(0.1)
    return False


def _shutdown_left_arm(left_arm: ConnectedLeftArm) -> None:
    robot = left_arm.robot
    ec = left_arm.ec
    try:
        robot.disableDrag(ec)
        _print_sdk_result("disableDrag", ec)
    except Exception:
        pass
    try:
        robot.stop(ec)
    except Exception:
        pass
    try:
        robot.setPowerState(False, ec)
    except Exception:
        pass
    try:
        robot.disconnectFromRobot(ec)
    except Exception:
        pass


def _print_sdk_result(action: str, ec: dict[str, object]) -> None:
    code = ec.get("ec", 0)
    message = str(ec.get("message", ""))
    logger.debug(f"{action}: ec={code}, message={message}")


# endregion


# region 相机与 ChArUco
def _read_camera_calibration(client: CameraPipelineClient) -> CameraCalibration:
    response = client.get_camera_intrinsics(timeout_s=DEFAULT_CAMERA_TIMEOUT_S)
    distortion = np.asarray(response.distortion, dtype=np.float64).reshape(-1, 1)
    if distortion.size == 0:
        distortion = np.zeros((5, 1), dtype=np.float64)
    return CameraCalibration(
        camera_name=str(response.camera_name),
        width=int(response.width),
        height=int(response.height),
        camera_matrix=np.asarray(
            [
                [float(response.fx), 0.0, float(response.cx)],
                [0.0, float(response.fy), float(response.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        dist_coeffs=distortion,
    )


# endregion


# region 采样与保存
def _capture_sample(
    session_dir: Path,
    raw_frames_dir: Path,
    preview_frames_dir: Path,
    frame_packet: CameraColorFramePacket,
    raw_frame_bgr: np.ndarray,
    preview_frame_bgr: np.ndarray,
    robot_snapshot: RobotSnapshot,
    charuco_result,
    board_pose_camera_board: np.ndarray | None,
    next_sample_index: int,
) -> SampleRecord:
    host_timestamp_s = time.time()
    sample_name = f"sample_{next_sample_index:03d}"
    raw_frame_path = raw_frames_dir / f"{sample_name}.png"
    preview_frame_path = preview_frames_dir / f"{sample_name}.png"
    cv2.imwrite(str(raw_frame_path), raw_frame_bgr)
    cv2.imwrite(str(preview_frame_path), preview_frame_bgr)
    return SampleRecord(
        sample_index=next_sample_index,
        host_timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        host_timestamp_s=host_timestamp_s,
        frame_id=int(frame_packet.frame_id),
        camera_timestamp_ms=float(frame_packet.timestamp_ms),
        board_visible=bool(charuco_result.board_visible),
        marker_count=int(charuco_result.marker_count),
        charuco_count=int(charuco_result.charuco_count),
        reprojection_error_px=(
            None
            if charuco_result.reprojection_error_px is None
            else float(charuco_result.reprojection_error_px)
        ),
        robot_snapshot=robot_snapshot,
        board_pose_camera_board=board_pose_camera_board,
        raw_frame_path=raw_frame_path.relative_to(session_dir),
        preview_frame_path=preview_frame_path.relative_to(session_dir),
    )


def _write_samples_csv(
    csv_path: Path,
    samples: list[SampleRecord],
    calibration_result: HandEyeCalibrationResult | None = None,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "host_timestamp_iso",
                "host_timestamp_s",
                "frame_id",
                "camera_timestamp_ms",
                "board_visible",
                "marker_count",
                "charuco_count",
                "reprojection_error_px",
                "robot_joint_j1_deg",
                "robot_joint_j2_deg",
                "robot_joint_j3_deg",
                "robot_joint_j4_deg",
                "robot_joint_j5_deg",
                "robot_joint_j6_deg",
                "robot_joint_j7_deg",
                "toolset_source",
                "toolset_end_x_mm",
                "toolset_end_y_mm",
                "toolset_end_z_mm",
                "toolset_end_roll_deg",
                "toolset_end_pitch_deg",
                "toolset_end_yaw_deg",
                "toolset_ref_x_mm",
                "toolset_ref_y_mm",
                "toolset_ref_z_mm",
                "toolset_ref_roll_deg",
                "toolset_ref_pitch_deg",
                "toolset_ref_yaw_deg",
                "robot_end_x_mm",
                "robot_end_y_mm",
                "robot_end_z_mm",
                "robot_end_roll_deg",
                "robot_end_pitch_deg",
                "robot_end_yaw_deg",
                "robot_end_qw",
                "robot_end_qx",
                "robot_end_qy",
                "robot_end_qz",
                "robot_has_elbow",
                "robot_elbow_deg",
                "robot_conf_data",
                "camera_board_x_mm",
                "camera_board_y_mm",
                "camera_board_z_mm",
                "camera_board_qw",
                "camera_board_qx",
                "camera_board_qy",
                "camera_board_qz",
                "base_board_x_mm",
                "base_board_y_mm",
                "base_board_z_mm",
                "base_board_qw",
                "base_board_qx",
                "base_board_qy",
                "base_board_qz",
                "raw_frame_path",
                "preview_frame_path",
            ]
        )
        for sample in samples:
            end_transform = _cartesian_pose_to_matrix_m(
                sample.robot_snapshot.end_pose_in_ref
            )
            end_quat_wxyz = _rotation_matrix_to_quaternion_wxyz(end_transform[:3, :3])
            toolset_end_fields = _frame_csv_fields(
                sample.robot_snapshot.toolset_end_frame
            )
            toolset_ref_fields = _frame_csv_fields(
                sample.robot_snapshot.toolset_ref_frame
            )
            camera_board_fields = _transform_csv_fields(sample.board_pose_camera_board)
            base_board_fields = _transform_csv_fields(
                _compute_base_pose_board(
                    robot_transform=end_transform,
                    tool_cam=(
                        None
                        if calibration_result is None
                        else calibration_result.tool_cam
                    ),
                    cam_board=sample.board_pose_camera_board,
                ),
                translation_scale=1000.0,
            )

            # 这里显式展开每一个字段，而不是把复杂对象直接序列化成一列字符串。
            # 这样做的目的有两个：
            # 1. 后续如果要在 Excel、Pandas 或其它标定调试脚本中筛选异常样本，可以直接按列过滤。
            # 2. 对手眼标定这种“采样一次代价高、复查频率高”的数据，列式展开最利于人工追查单位、
            #    方向和同步问题，避免后续再去解析嵌套字符串。
            writer.writerow(
                [
                    sample.sample_index,
                    sample.host_timestamp_iso,
                    f"{sample.host_timestamp_s:.6f}",
                    sample.frame_id,
                    f"{sample.camera_timestamp_ms:.3f}",
                    int(sample.board_visible),
                    sample.marker_count,
                    sample.charuco_count,
                    _format_optional_float(sample.reprojection_error_px, digits=6),
                    *[f"{value:.6f}" for value in sample.robot_snapshot.joint_degrees],
                    "sdk.toolset(ec)",
                    *toolset_end_fields,
                    *toolset_ref_fields,
                    f"{sample.robot_snapshot.end_translation_mm[0]:.6f}",
                    f"{sample.robot_snapshot.end_translation_mm[1]:.6f}",
                    f"{sample.robot_snapshot.end_translation_mm[2]:.6f}",
                    f"{sample.robot_snapshot.end_rpy_degrees[0]:.6f}",
                    f"{sample.robot_snapshot.end_rpy_degrees[1]:.6f}",
                    f"{sample.robot_snapshot.end_rpy_degrees[2]:.6f}",
                    f"{end_quat_wxyz[0]:.8f}",
                    f"{end_quat_wxyz[1]:.8f}",
                    f"{end_quat_wxyz[2]:.8f}",
                    f"{end_quat_wxyz[3]:.8f}",
                    int(sample.robot_snapshot.sdk_pose_has_elbow),
                    f"{sample.robot_snapshot.sdk_pose_elbow_deg:.6f}",
                    str(list(sample.robot_snapshot.sdk_pose_conf_data)),
                    *camera_board_fields,
                    *base_board_fields,
                    str(sample.raw_frame_path),
                    str(sample.preview_frame_path),
                ]
            )


def _write_hand_eye_result_file(
    output_path: Path, result: HandEyeCalibrationResult
) -> None:
    tool_cam_rpy_deg = Rotation3D.from_matrix(result.tool_cam[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    mean_translation = result.stats.mean_translation_m
    std_translation = result.stats.std_translation_m
    toolset_lines = _build_toolset_result_lines(result.toolset_source)
    lines = [
        "Hand-eye calibration result",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "Formula:",
        "T_base_board = T_base_tool @ T_tool_cam @ T_cam_board",
        "",
        f"robot_source = {result.robot_source}",
        f"camera_source = {result.camera_source}",
        f"toolset_source = {result.toolset_source}",
        f"method = {result.method_name}",
        f"valid_sample_count = {len(result.used_indices)}",
        "valid_sample_indices = "
        + ", ".join(str(index) for index in result.used_indices),
        "",
        *toolset_lines,
        "",
        "T_tool_cam:",
        np.array2string(result.tool_cam, precision=10, suppress_small=False),
        "",
        "T_tool_cam_rpy_deg:",
        f"roll = {tool_cam_rpy_deg[0]:.6f}",
        f"pitch = {tool_cam_rpy_deg[1]:.6f}",
        f"yaw = {tool_cam_rpy_deg[2]:.6f}",
        "",
        "Mean board position in base:",
        f"x_m = {mean_translation[0]:.10f}",
        f"y_m = {mean_translation[1]:.10f}",
        f"z_m = {mean_translation[2]:.10f}",
        "",
        "Mean board position in base, unit mm:",
        f"x_mm = {mean_translation[0] * 1000.0:.6f}",
        f"y_mm = {mean_translation[1] * 1000.0:.6f}",
        f"z_mm = {mean_translation[2] * 1000.0:.6f}",
        "",
        "Std board position in base, unit mm:",
        f"x_std_mm = {std_translation[0] * 1000.0:.6f}",
        f"y_std_mm = {std_translation[1] * 1000.0:.6f}",
        f"z_std_mm = {std_translation[2] * 1000.0:.6f}",
        "",
        f"translation_mean_error_m = {result.stats.translation_mean_error_m:.10f}",
        f"translation_max_error_m = {result.stats.translation_max_error_m:.10f}",
        f"rotation_mean_error_deg = {result.stats.rotation_mean_error_deg:.10f}",
        f"rotation_max_error_deg = {result.stats.rotation_max_error_deg:.10f}",
        "",
        "[per_sample_base_board]",
    ]
    for sample_index, base_board in zip(
        result.used_indices, result.base_board_list, strict=True
    ):
        lines.append(f"sample_{sample_index:03d} " + _format_pose_text(base_board))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_toolset_result_lines(toolset_source: str) -> list[str]:
    return [
        "Toolset note:",
        "xCoreSDK_python.endInRef is defined by the current SDK toolset.",
        f"toolset_record = {toolset_source}",
    ]


def _write_sample_guidance_file(output_path: Path, samples: list[SampleRecord]) -> None:
    guidance_lines = _build_sampling_guidance_lines(samples)
    coverage = _summarize_sampling_coverage(samples)
    lines = [
        "拖动采样建议",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"sample_count={len(samples)}",
    ]
    if coverage is not None:
        lines.extend(
            [
                f"span_x_mm={coverage.span_x_mm:.2f}",
                f"span_y_mm={coverage.span_y_mm:.2f}",
                f"span_z_mm={coverage.span_z_mm:.2f}",
                f"span_roll_deg={coverage.span_roll_deg:.2f}",
                f"span_pitch_deg={coverage.span_pitch_deg:.2f}",
                f"span_yaw_deg={coverage.span_yaw_deg:.2f}",
            ]
        )
    lines.append("")
    lines.extend(guidance_lines)

    # 这份建议文件不是算法输入，而是给现场采样者的“即时操作说明”。
    # 单独落盘有几个实际好处：
    # 1. 即使预览窗口已经关闭，现场人员仍然可以复盘当时为什么提示“多做俯仰”或“多做左右展开”；
    # 2. 如果一次采样中断，下一次重新打开目录时，可以快速知道上一次还缺什么激励；
    # 3. 对预研阶段来说，把采样建议固化下来也方便后续把这套经验迁移到 GUI 或自动流程中。
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_update_hand_eye_result(
    output_path: Path,
    samples: list[SampleRecord],
) -> HandEyeCalibrationResult | None:
    valid_samples = [
        sample for sample in samples if sample.board_pose_camera_board is not None
    ]
    if len(valid_samples) < 6:
        return None
    base_tool_list: list[np.ndarray] = []
    cam_board_list: list[np.ndarray] = []
    used_indices: list[int] = []
    for sample in valid_samples:
        try:
            base_tool = _robot_end_transform_m_from_snapshot(sample.robot_snapshot)
            cam_board = _camera_board_matrix_m(sample.board_pose_camera_board)
        except ValueError as exc:
            logger.warning("跳过样本 #{}: {}", sample.sample_index, exc)
            continue
        base_tool_list.append(base_tool)
        cam_board_list.append(cam_board)
        used_indices.append(sample.sample_index)
    if len(used_indices) < 6:
        logger.warning("有效右手系样本不足 6 个，暂不计算手眼：{}", used_indices)
        return None
    try:
        result = _solve_park_hand_eye(
            base_tool_list=base_tool_list,
            cam_board_list=cam_board_list,
            used_indices=used_indices,
            robot_source="robot_end",
            toolset_source=_format_toolset_source(valid_samples[0].robot_snapshot),
        )
    except cv2.error as exc:
        logger.warning("PARK 手眼求解失败，继续采样：{}", exc)
        return None
    except ValueError as exc:
        logger.warning("PARK 手眼求解输入无效，继续采样：{}", exc)
        return None
    _write_hand_eye_result_file(output_path=output_path, result=result)
    logger.info(
        "手眼结果 PARK t_mean={:.3f}mm std=({:.3f}, {:.3f}, {:.3f})mm",
        result.stats.translation_mean_error_m * 1000.0,
        result.stats.std_translation_m[0] * 1000.0,
        result.stats.std_translation_m[1] * 1000.0,
        result.stats.std_translation_m[2] * 1000.0,
    )
    return result


def _transform_csv_fields(
    transform: np.ndarray | None, translation_scale: float = 1.0
) -> list[str]:
    if transform is None:
        return [""] * 7
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    quat_wxyz = _rotation_matrix_to_quaternion_wxyz(matrix[:3, :3])
    return [
        f"{float(matrix[0, 3] * translation_scale):.6f}",
        f"{float(matrix[1, 3] * translation_scale):.6f}",
        f"{float(matrix[2, 3] * translation_scale):.6f}",
        f"{quat_wxyz[0]:.8f}",
        f"{quat_wxyz[1]:.8f}",
        f"{quat_wxyz[2]:.8f}",
        f"{quat_wxyz[3]:.8f}",
    ]


def _frame_csv_fields(frame: xCoreSDK_python.Frame) -> list[str]:
    translation_mm = [float(value) * 1000.0 for value in frame.trans]
    rpy_deg = [float(np.degrees(float(value))) for value in frame.rpy]
    return [
        f"{translation_mm[0]:.6f}",
        f"{translation_mm[1]:.6f}",
        f"{translation_mm[2]:.6f}",
        f"{rpy_deg[0]:.6f}",
        f"{rpy_deg[1]:.6f}",
        f"{rpy_deg[2]:.6f}",
    ]


def _format_toolset_source(robot_snapshot: RobotSnapshot) -> str:
    end_text = _format_frame_values(robot_snapshot.toolset_end_frame)
    ref_text = _format_frame_values(robot_snapshot.toolset_ref_frame)
    return f"sdk.toolset(ec); end={end_text}; ref={ref_text}"


# endregion


# region 预览绘制
def _draw_preview(
    frame_bgr: np.ndarray,
    frame_packet: CameraColorFramePacket,
    robot_snapshot: RobotSnapshot,
    charuco_result,
    calibration_result,
    camera_calibration: CameraCalibration,
    compute_ms: float,
    recorded_samples: list[SampleRecord],
) -> np.ndarray:
    canvas = frame_bgr.copy()
    _draw_marker_corners(
        canvas, charuco_result.marker_corners_px, charuco_result.marker_ids
    )
    if charuco_result.charuco_corners_px is not None:
        points = (
            np.round(charuco_result.charuco_corners_px)
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )
        cv2.polylines(canvas, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
        for point in charuco_result.charuco_corners_px:
            cv2.circle(
                canvas,
                (int(round(point[0])), int(round(point[1]))),
                4,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )
    if charuco_result.rvec is not None and charuco_result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            camera_calibration.camera_matrix,
            camera_calibration.dist_coeffs,
            np.asarray(charuco_result.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(charuco_result.tvec, dtype=np.float64).reshape(3, 1),
            AXIS_DRAW_LENGTH_MM,
        )

    coverage = _summarize_sampling_coverage(recorded_samples)
    overlay_lines = _build_overlay_lines(
        frame_packet=frame_packet,
        robot_snapshot=robot_snapshot,
        charuco_result=charuco_result,
        calibration_result=calibration_result,
        coverage=coverage,
        sample_count=len(recorded_samples),
        compute_ms=compute_ms,
    )
    guidance_lines = _build_sampling_guidance_lines(recorded_samples)
    canvas = _draw_text_block(
        canvas, overlay_lines, (18, 28), color=(255, 255, 255), line_gap=22
    )
    canvas = _draw_text_block(
        canvas, guidance_lines, (18, 412), color=(80, 255, 255), line_gap=22
    )
    return canvas


def _build_overlay_lines(
    frame_packet: CameraColorFramePacket,
    robot_snapshot: RobotSnapshot,
    charuco_result,
    calibration_result: HandEyeCalibrationResult | None,
    coverage: SamplingCoverageSummary | None,
    sample_count: int,
    compute_ms: float,
) -> list[str]:
    joint_text = ", ".join(
        f"J{index + 1}={value:.1f}"
        for index, value in enumerate(robot_snapshot.joint_degrees)
    )
    end_matrix = _robot_end_transform_m_from_snapshot(robot_snapshot)
    board_pose_camera_board = None
    if charuco_result.transform_se3 is not None:
        board_pose_camera_board = np.asarray(
            charuco_result.transform_se3, dtype=np.float64
        ).reshape(4, 4)
    lines = [
        f"camera_frame={int(frame_packet.frame_id)} camera_ts_ms={float(frame_packet.timestamp_ms):.1f} compute_ms={compute_ms:.2f}",
        "drag=ON arm=left calc=endInRef + T_cam_board + PARK",
        f"board_visible={bool(charuco_result.board_visible)} marker={int(charuco_result.marker_count)} charuco={int(charuco_result.charuco_count)} reproj={_format_optional_float(charuco_result.reprojection_error_px, digits=4)}",
        (
            f"end_mm=({robot_snapshot.end_translation_mm[0]:.1f}, {robot_snapshot.end_translation_mm[1]:.1f}, "
            f"{robot_snapshot.end_translation_mm[2]:.1f}) end_rpy_deg=({robot_snapshot.end_rpy_degrees[0]:.1f}, "
            f"{robot_snapshot.end_rpy_degrees[1]:.1f}, {robot_snapshot.end_rpy_degrees[2]:.1f})"
        ),
        f"robot_joints_deg={joint_text}",
        f"sample_count={sample_count} host_time={robot_snapshot.host_timestamp_iso}",
        "capture: Enter/Space/P    quit: Esc/Q",
    ]
    try:
        cam_board_text = _format_pose_text(
            None
            if board_pose_camera_board is None
            else _camera_board_matrix_m(board_pose_camera_board)
        )
    except ValueError as exc:
        cam_board_text = f"invalid({exc})"
    lines.append("T_cam_board=" + cam_board_text)
    if coverage is not None:
        lines.append(
            "coverage_mm_deg="
            f"x:{coverage.span_x_mm:.1f} y:{coverage.span_y_mm:.1f} z:{coverage.span_z_mm:.1f} "
            f"roll:{coverage.span_roll_deg:.1f} pitch:{coverage.span_pitch_deg:.1f} yaw:{coverage.span_yaw_deg:.1f}"
        )
    if calibration_result is not None:
        tool_cam = np.asarray(calibration_result.tool_cam, dtype=np.float64).reshape(
            4, 4
        )
        tool_cam_rpy = Rotation3D.from_matrix(tool_cam[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        lines.append(
            f"T_tool_cam_mm=({tool_cam[0, 3] * 1000.0:.1f}, {tool_cam[1, 3] * 1000.0:.1f}, {tool_cam[2, 3] * 1000.0:.1f}) "
            f"rpy=({tool_cam_rpy[0]:.1f}, {tool_cam_rpy[1]:.1f}, {tool_cam_rpy[2]:.1f})"
        )
        lines.append(
            "board_mean_mm="
            f"({calibration_result.stats.mean_translation_m[0] * 1000.0:.1f}, "
            f"{calibration_result.stats.mean_translation_m[1] * 1000.0:.1f}, "
            f"{calibration_result.stats.mean_translation_m[2] * 1000.0:.1f})"
        )
        lines.append(
            "board_std_mm="
            f"({calibration_result.stats.std_translation_m[0] * 1000.0:.2f}, "
            f"{calibration_result.stats.std_translation_m[1] * 1000.0:.2f}, "
            f"{calibration_result.stats.std_translation_m[2] * 1000.0:.2f})"
        )
        try:
            base_board = _compute_base_pose_board(
                robot_transform=end_matrix,
                tool_cam=calibration_result.tool_cam,
                cam_board=board_pose_camera_board,
            )
            lines.append("current_T_base_board=" + _format_pose_text(base_board))
        except ValueError as exc:
            lines.append(f"current_T_base_board=invalid({exc})")
    return lines


def _draw_marker_corners(
    canvas: np.ndarray,
    marker_corners_px: list[np.ndarray],
    marker_ids: np.ndarray | None,
) -> None:
    if not marker_corners_px:
        return
    for marker_index, corners in enumerate(marker_corners_px):
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        points_i32 = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points_i32], True, (255, 255, 0), 2, cv2.LINE_AA)
        center = np.mean(points, axis=0)
        marker_label = (
            ""
            if marker_ids is None or marker_index >= len(marker_ids)
            else str(int(marker_ids[marker_index]))
        )
        _draw_single_text(
            canvas,
            f"M{marker_label}",
            (int(round(center[0])), int(round(center[1]))),
            (0, 255, 255),
        )


def _draw_text_block(
    canvas: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    line_gap: int,
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(DEFAULT_FONT_SIZE)
    x, y = origin
    for line in lines:
        _draw_stroked_text(
            draw,
            (x, y),
            line,
            font=font,
            fill=(int(color[2]), int(color[1]), int(color[0])),
            stroke_fill=(0, 0, 0),
            stroke_width=2,
        )
        y += line_gap
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _draw_single_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    _draw_stroked_text(
        draw,
        origin,
        text,
        font=_load_font(18),
        fill=(int(color[2]), int(color[1]), int(color[0])),
        stroke_fill=(0, 0, 0),
        stroke_width=2,
    )
    canvas[:, :, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(DEFAULT_FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_stroked_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    stroke_fill: tuple[int, int, int],
    stroke_width: int,
) -> None:
    draw.text(
        position,
        text,
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=stroke_width,
        anchor="la",
    )


# endregion


# region 采样建议
def _summarize_sampling_coverage(
    samples: list[SampleRecord],
) -> SamplingCoverageSummary | None:
    valid_samples = [
        sample for sample in samples if sample.board_pose_camera_board is not None
    ]
    if not valid_samples:
        return None
    translations = np.asarray(
        [sample.robot_snapshot.end_translation_mm for sample in valid_samples],
        dtype=np.float64,
    )
    rpy_values = np.asarray(
        [sample.robot_snapshot.end_rpy_degrees for sample in valid_samples],
        dtype=np.float64,
    )
    spans_translation = np.ptp(translations, axis=0)
    spans_rpy = np.ptp(rpy_values, axis=0)
    return SamplingCoverageSummary(
        sample_count=len(valid_samples),
        span_x_mm=float(spans_translation[0]),
        span_y_mm=float(spans_translation[1]),
        span_z_mm=float(spans_translation[2]),
        span_roll_deg=float(spans_rpy[0]),
        span_pitch_deg=float(spans_rpy[1]),
        span_yaw_deg=float(spans_rpy[2]),
    )


def _build_sampling_guidance_lines(samples: list[SampleRecord]) -> list[str]:
    coverage = _summarize_sampling_coverage(samples)
    lines = ["sampling_guidance:"]
    if coverage is None:
        lines.extend(
            [
                "1. 先在板清晰可见的位置记录第 1 帧，作为基准位。",
                "2. 然后依次做左右、前后、上下和翻腕动作，每次停稳后再采样。",
                "3. 保持左臂拖动时标定板始终在画面内，避免只做平移不做旋转。",
            ]
        )
        return lines

    pending_lines: list[str] = []
    if coverage.span_x_mm < TRANSLATION_SPAN_TARGETS_MM["x"]:
        pending_lines.append(
            f"补左右横向展开，当前 X 跨度 {coverage.span_x_mm:.1f} mm，建议至少 {TRANSLATION_SPAN_TARGETS_MM['x']:.1f} mm。"
        )
    if coverage.span_y_mm < TRANSLATION_SPAN_TARGETS_MM["y"]:
        pending_lines.append(
            f"补前后距离变化，当前 Y 跨度 {coverage.span_y_mm:.1f} mm，建议至少 {TRANSLATION_SPAN_TARGETS_MM['y']:.1f} mm。"
        )
    if coverage.span_z_mm < TRANSLATION_SPAN_TARGETS_MM["z"]:
        pending_lines.append(
            f"补上下高度变化，当前 Z 跨度 {coverage.span_z_mm:.1f} mm，建议至少 {TRANSLATION_SPAN_TARGETS_MM['z']:.1f} mm。"
        )
    if coverage.span_roll_deg < ROTATION_SPAN_TARGETS_DEG["roll"]:
        pending_lines.append(
            f"补 roll 翻腕，当前 roll 跨度 {coverage.span_roll_deg:.1f} deg，建议至少 {ROTATION_SPAN_TARGETS_DEG['roll']:.1f} deg。"
        )
    if coverage.span_pitch_deg < ROTATION_SPAN_TARGETS_DEG["pitch"]:
        pending_lines.append(
            f"补 pitch 抬头/低头，当前 pitch 跨度 {coverage.span_pitch_deg:.1f} deg，建议至少 {ROTATION_SPAN_TARGETS_DEG['pitch']:.1f} deg。"
        )
    if coverage.span_yaw_deg < ROTATION_SPAN_TARGETS_DEG["yaw"]:
        pending_lines.append(
            f"补 yaw 左右摆头，当前 yaw 跨度 {coverage.span_yaw_deg:.1f} deg，建议至少 {ROTATION_SPAN_TARGETS_DEG['yaw']:.1f} deg。"
        )

    if not pending_lines:
        lines.extend(
            [
                "采样覆盖已经比较充分，可以补充 3-5 个斜向组合姿态做收尾。",
                "优先补“边平移边转动”的组合位，避免所有样本都围绕同一个姿态小范围抖动。",
                "若残差仍偏大，优先重采重投影误差高或姿态过于相似的样本。",
            ]
        )
        return lines

    lines.extend(f"{index + 1}. {text}" for index, text in enumerate(pending_lines[:3]))
    lines.append("停稳后再按采样键，优先让板保持清晰、无遮挡、非纯正视。")
    return lines


# endregion


# region 通用工具
def _create_session_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _format_optional_float(value: float | None, digits: int) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="左臂拖动示教 + Orin 左手相机流手眼标定实采页"
    )
    parser.add_argument(
        "--replay-run-dir",
        type=str,
        default="",
        help="离线复算模式：指定 runs 目录，按 handeyeCali.py 同款链路重算",
    )
    parser.add_argument(
        "--service-addr",
        type=str,
        default=DEFAULT_ORIN_SERVICE_ADDR,
        help="Orin camera_pipeline_service 地址",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=DEFAULT_CAMERA_NAME,
        help="逻辑相机名，默认使用左手相机",
    )
    parser.add_argument(
        "--left-arm-ip", type=str, default=DEFAULT_LEFT_ARM_IP, help="左臂控制器 IP"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="运行输出根目录",
    )
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=DEFAULT_MIN_CHARUCO_CORNERS,
        help="进入位姿估计所需的最小 ChArUco 角点数",
    )
    return parser.parse_args()


def _validate_runtime_requirements() -> None:
    _ = cv2.aruco.ArucoDetector
    _ = cv2.aruco.CharucoBoard


def _cartesian_pose_to_matrix_m(
    cartesian_pose: xCoreSDK_python.CartesianPosition,
) -> np.ndarray:
    rotation = Rotation3D.from_euler(
        "xyz",
        np.asarray(cartesian_pose.rpy, dtype=np.float64).reshape(3),
        degrees=False,
    ).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(cartesian_pose.trans, dtype=np.float64).reshape(3)
    return matrix


def _robot_end_transform_m_from_snapshot(robot_snapshot: RobotSnapshot) -> np.ndarray:
    rotation = Rotation3D.from_euler(
        "xyz",
        [
            robot_snapshot.end_rpy_degrees[0],
            robot_snapshot.end_rpy_degrees[1],
            robot_snapshot.end_rpy_degrees[2],
        ],
        degrees=True,
    ).as_matrix()
    _validate_right_handed_rotation("robot_end", rotation)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (
        np.asarray(robot_snapshot.end_translation_mm, dtype=np.float64).reshape(3)
        * 0.001
    )
    return matrix


def _camera_board_matrix_m(camera_board_mm: np.ndarray | None) -> np.ndarray:
    if camera_board_mm is None:
        raise ValueError("缺少 T_cam_board，无法参与手眼标定")
    source = np.asarray(camera_board_mm, dtype=np.float64).reshape(4, 4)
    rotation = source[:3, :3]
    _validate_right_handed_rotation("camera_board", rotation)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = source[:3, 3] * 0.001
    return matrix


def _validate_right_handed_rotation(name: str, rotation: np.ndarray) -> None:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    determinant = float(np.linalg.det(matrix))
    if determinant <= 0.0:
        raise ValueError(f"{name} 旋转矩阵不是右手系：det={determinant:.9f}")
    orthogonality_error = float(
        np.linalg.norm(matrix.T @ matrix - np.eye(3, dtype=np.float64))
    )
    if orthogonality_error > 1e-3:
        raise ValueError(f"{name} 旋转矩阵不正交：error={orthogonality_error:.9f}")


def _rotation_matrix_to_quaternion_wxyz(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    quat_xyzw = Rotation3D.from_matrix(
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    ).as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _format_pose_text(transform: np.ndarray | None) -> str:
    if transform is None:
        return "NA"
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rpy_deg = Rotation3D.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    return (
        f"t_mm=({float(matrix[0, 3] * 1000.0):.1f}, {float(matrix[1, 3] * 1000.0):.1f}, {float(matrix[2, 3] * 1000.0):.1f}) "
        f"rpy_deg=({float(rpy_deg[0]):.1f}, {float(rpy_deg[1]):.1f}, {float(rpy_deg[2]):.1f})"
    )


def _compute_base_pose_board(
    *,
    robot_transform: np.ndarray,
    tool_cam: np.ndarray | None,
    cam_board: np.ndarray | None,
) -> np.ndarray | None:
    if tool_cam is None or cam_board is None:
        return None
    robot_matrix = np.asarray(robot_transform, dtype=np.float64).reshape(4, 4)
    tool_cam_matrix = np.asarray(tool_cam, dtype=np.float64).reshape(4, 4)
    cam_board_matrix = _camera_board_matrix_m(cam_board)
    return robot_matrix @ tool_cam_matrix @ cam_board_matrix


# endregion


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
        raise
