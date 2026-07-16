#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false

"""左右臂眼在手外拖动标定。

运行后先在交互式 CLI 中选择 l/r，再固定头部姿态、连接对应机械臂并开启拖动。
在预览窗口中按 Space/Enter/P 记录样本，按 Q/Esc 结束。
"""

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
from scipy.spatial.transform import Rotation as Rotation3D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WUJI_TEST_ROOT = PROJECT_ROOT / "test" / "wuji"
for import_root in (PROJECT_ROOT, WUJI_TEST_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from camera_pipeline.client import CameraPipelineClient
from camera_pipeline.protocol import CameraColorFramePacket
from common import (
    DEFAULT_PORT,
    SshTunnelGroup,
    close_wuyou_channel,
    create_wuyou_channel,
    stop_ssh_process,
)
from sdk.xcoresdk import xCoreSDK_python
from src.calibration import CharucoPoseEstimator, CharucoPoseResult
from src.wuji.head_client import WujiHeadClient

# region 默认参数
DEFAULT_ORIN_SERVICE_ADDR = "tcp://192.168.1.128:6200"
DEFAULT_CAMERA_NAME = "head_camera"
DEFAULT_LEFT_ARM_IP = "192.168.1.161"
DEFAULT_RIGHT_ARM_IP = "192.168.1.160"
DEFAULT_OUTPUT_ROOT = Path("experiments/hand_eye/runs")
DEFAULT_CAMERA_TIMEOUT_S = 10.0
DEFAULT_HEAD_YAW_DEG = 60.0  # 头部固定 yaw 角度，单位 deg
DEFAULT_HEAD_PITCH_DEG = 45.0  # 头部固定 pitch 角度，单位 deg
DEFAULT_HEAD_SETTLE_S = 1.0
DEFAULT_DICTIONARY_NAME = "DICT_APRILTAG_16H5"
DEFAULT_SQUARES_X = 4
DEFAULT_SQUARES_Y = 4
DEFAULT_SQUARE_LENGTH_MM = 20
DEFAULT_MARKER_LENGTH_MM = 14
DEFAULT_MIN_CHARUCO_CORNERS = 6
DEFAULT_MIN_CALIBRATION_SAMPLES = 6
DEFAULT_TOOL_NAME = "g_tool_0"
DEFAULT_WOBJ_NAME = "g_wobj_0"
DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 900
EXPECTED_ARM_TYPES = {
    "left": "AR5-5_0.8L-W4C1C9-ZY2",
    "right": "AR5-5_0.8R-W4C1C9-ZY2",
}
ARM_LABELS = {"left": "左臂", "right": "右臂"}
RESULT_PREFIXES = {"left": "L_EtH_", "right": "R_EtH_"}
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """头部相机内参与畸变参数。"""

    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(slots=True)
class ConnectedArm:
    """已连接并可拖动的单侧机械臂。"""

    arm_side: str
    robot_ip: str
    robot: xCoreSDK_python.xMateErProRobot
    robot_type: str
    robot_uid: str
    ec: dict[str, object]


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    """机械臂采样瞬间的绝对位姿；矩阵平移单位固定为 m。"""

    timestamp_iso: str
    joint_degrees: tuple[float, ...]
    base_gripper_pose_m: np.ndarray
    translation_mm: tuple[float, float, float]
    rpy_degrees: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """一次同步保存的机器人位姿与标定板观测。"""

    sample_index: int
    timestamp_iso: str
    frame_id: int
    camera_timestamp_ms: float
    marker_count: int
    charuco_count: int
    reprojection_error_px: float
    robot_snapshot: RobotSnapshot
    camera_board_pose_m: np.ndarray
    raw_frame_path: Path
    preview_frame_path: Path


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    """逐样本 T_base_camera 围绕平均位姿的离散统计。"""

    translation_mean_m: np.ndarray
    translation_std_m: np.ndarray
    translation_errors_m: np.ndarray
    translation_mean_error_m: float
    translation_max_error_m: float
    rotation_vector_std_deg: np.ndarray
    rotation_errors_deg: np.ndarray
    rotation_mean_error_deg: float
    rotation_max_error_deg: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """眼在手外结果；所有内部变换矩阵的平移单位均为 m。"""

    park_base_camera_pose_m: np.ndarray
    base_camera_pose_m: np.ndarray
    base_camera_poses_m: tuple[np.ndarray, ...]
    camera_board_poses_m: tuple[np.ndarray, ...]
    mean_gripper_board_pose_m: np.ndarray
    gripper_board_poses_m: tuple[np.ndarray, ...]
    used_sample_indices: tuple[int, ...]
    stats: CalibrationStats


@dataclass(frozen=True, slots=True)
class AppConfig:
    """命令行可覆盖的运行配置。"""

    service_addr: str
    left_arm_ip: str
    right_arm_ip: str
    output_root: Path
    min_charuco_corners: int


# endregion


# region 主流程
def main() -> int:
    config = _parse_cli()
    arm_side = _select_arm_interactively()
    prefix = RESULT_PREFIXES[arm_side]
    session_dir = _create_session_dir(config.output_root, prefix)
    logger.info("本次标定侧别：{}；输出目录：{}", ARM_LABELS[arm_side], session_dir)

    head_tunnel: SshTunnelGroup | None = None
    head_channel: object | None = None
    arm: ConnectedArm | None = None
    try:
        head_tunnel, head_channel = create_wuyou_channel(DEFAULT_PORT)
        _set_head_fixed_pose(WujiHeadClient(head_channel))
        robot_ip = config.left_arm_ip if arm_side == "left" else config.right_arm_ip
        arm = _connect_arm(arm_side, robot_ip)
        _enable_arm_drag(arm)
        _run_calibration_session(
            config=config,
            arm=arm,
            session_dir=session_dir,
            result_prefix=prefix,
        )
        return 0
    finally:
        if arm is not None:
            _shutdown_arm(arm)
        if head_channel is not None:
            close_wuyou_channel(head_channel)
        if head_tunnel is not None:
            stop_ssh_process(head_tunnel)


def _run_calibration_session(
    *,
    config: AppConfig,
    arm: ConnectedArm,
    session_dir: Path,
    result_prefix: str,
) -> None:
    client = CameraPipelineClient(
        service_addr=config.service_addr,
        timeout_ms=int(DEFAULT_CAMERA_TIMEOUT_S * 1000.0),
    )
    estimator = CharucoPoseEstimator(_build_board())
    calibration = _read_head_camera_calibration(client)
    raw_dir = session_dir / "frames_raw"
    preview_dir = session_dir / "frames_preview"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    samples: list[SampleRecord] = []
    last_result: CalibrationResult | None = None
    window_name = f"{ARM_LABELS[arm.arm_side]} Eye-to-Hand Calibration"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    logger.success("开始订阅头部相机。拖动{}，停稳后按 Space/Enter/P 采样。", ARM_LABELS[arm.arm_side])
    try:
        for frame_packet in client.subscribe_head_camera_color_frames():
            frame_bgr = np.asarray(frame_packet.color_bgr, dtype=np.uint8).copy()
            pose_result = estimator.estimate_pose(
                image_bgr=frame_bgr,
                camera_matrix=calibration.camera_matrix,
                dist_coeffs=calibration.dist_coeffs,
                min_charuco_corners=config.min_charuco_corners,
            )
            preview_bgr = _draw_preview(
                frame_bgr=frame_bgr,
                pose_result=pose_result,
                sample_count=len(samples),
                result=last_result,
                arm_side=arm.arm_side,
                calibration=calibration,
            )
            cv2.imshow(window_name, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (13, 32, ord("p"), ord("P")):
                sample = _capture_sample(
                    arm=arm,
                    frame_packet=frame_packet,
                    frame_bgr=frame_bgr,
                    preview_bgr=preview_bgr,
                    pose_result=pose_result,
                    sample_index=len(samples) + 1,
                    session_dir=session_dir,
                    raw_dir=raw_dir,
                    preview_dir=preview_dir,
                )
                if sample is None:
                    logger.warning("当前帧未获得有效 ChArUco 位姿，未记录样本。")
                    continue
                samples.append(sample)
                last_result = _update_outputs(
                    samples=samples,
                    session_dir=session_dir,
                    result_prefix=result_prefix,
                )
                logger.success(
                    "已记录样本 #{}，角点={}，重投影误差={:.3f}px",
                    sample.sample_index,
                    sample.charuco_count,
                    sample.reprojection_error_px,
                )
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        client.close()
        cv2.destroyAllWindows()
        _write_samples_csv(session_dir / f"{result_prefix}samples.csv", samples)
        if len(samples) >= DEFAULT_MIN_CALIBRATION_SAMPLES:
            _update_outputs(
                samples=samples,
                session_dir=session_dir,
                result_prefix=result_prefix,
            )


# endregion


# region 交互、头部与机械臂
def _select_arm_interactively() -> str:
    print("\n眼在手外标定：请选择需要标定的机械臂")
    print("  l - 左臂")
    print("  r - 右臂")
    while True:
        choice = input("请输入 l 或 r: ").strip().lower()
        if choice == "l":
            return "left"
        if choice == "r":
            return "right"
        print("输入无效，请只输入 l 或 r。")


def _set_head_fixed_pose(head: WujiHeadClient) -> None:
    logger.info(
        "固定头部姿态：yaw={:.1f}deg，pitch={:.1f}deg",
        DEFAULT_HEAD_YAW_DEG,
        DEFAULT_HEAD_PITCH_DEG,
    )
    head.set_head_yaw(DEFAULT_HEAD_YAW_DEG)
    head.set_head_pitch(DEFAULT_HEAD_PITCH_DEG)
    time.sleep(DEFAULT_HEAD_SETTLE_S)
    yaw_deg = float(head.get_head_yaw() or 0.0)
    pitch_deg = float(head.get_head_pitch() or 0.0)
    logger.success("头部已固定：yaw={:.1f}deg，pitch={:.1f}deg", yaw_deg, pitch_deg)


def _connect_arm(arm_side: str, robot_ip: str) -> ConnectedArm:
    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(robot_ip)
    robot_info = robot.robotInfo(ec)
    _raise_for_sdk_error(ec, f"读取{ARM_LABELS[arm_side]}机器人信息")
    actual_type = str(robot_info.type)
    expected_type = EXPECTED_ARM_TYPES[arm_side]
    if actual_type != expected_type:
        raise RuntimeError(
            f"机器人型号与所选侧别不一致：expected={expected_type}, actual={actual_type}"
        )
    _apply_fixed_toolset(robot, ec)
    logger.success(
        "{}已连接：ip={}，type={}，uid={}",
        ARM_LABELS[arm_side],
        robot_ip,
        actual_type,
        robot_info.id,
    )
    return ConnectedArm(
        arm_side=arm_side,
        robot_ip=robot_ip,
        robot=robot,
        robot_type=actual_type,
        robot_uid=str(robot_info.id),
        ec=ec,
    )


def _enable_arm_drag(arm: ConnectedArm) -> None:
    robot = arm.robot
    ec = arm.ec
    _apply_fixed_toolset(robot, ec)
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _raise_for_sdk_error(ec, "切换 NrtCommandMode")
    robot.setPowerState(False, ec)
    _raise_for_sdk_error(ec, "拖动前下电")
    if not _wait_for_power_off(robot, ec):
        raise RuntimeError("拖动前未在超时内确认机械臂下电")
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _raise_for_sdk_error(ec, "切换手动模式")
    robot.moveReset(ec)
    _raise_for_sdk_error(ec, "moveReset")
    robot.enableDrag(
        int(xCoreSDK_python.DragParameterSpace.cartesianSpace),
        int(xCoreSDK_python.DragParameterType.freely),
        ec,
        enable_drag_button=False,
    )
    _raise_for_sdk_error(ec, "开启笛卡尔自由拖动")
    logger.success("{}拖动已开启。", ARM_LABELS[arm.arm_side])


def _read_robot_snapshot(arm: ConnectedArm) -> RobotSnapshot:
    _apply_fixed_toolset(arm.robot, arm.ec)
    joint_values_rad = tuple(float(value) for value in arm.robot.jointPos(arm.ec))
    _raise_for_sdk_error(arm.ec, "读取关节角")
    pose = arm.robot.cartPosture(xCoreSDK_python.endInRef, arm.ec)
    _raise_for_sdk_error(arm.ec, "读取 cartPosture(endInRef)")

    translation_m = np.asarray(pose.trans, dtype=np.float64).reshape(3)
    rpy_rad = np.asarray(pose.rpy, dtype=np.float64).reshape(3)
    base_gripper_pose_m = np.eye(4, dtype=np.float64)
    base_gripper_pose_m[:3, :3] = Rotation3D.from_euler(
        "xyz", rpy_rad, degrees=False
    ).as_matrix()
    base_gripper_pose_m[:3, 3] = translation_m
    base_gripper_pose_m = _validate_transform_m(
        "SDK.T_base_gripper", base_gripper_pose_m
    )
    translation_mm = (
        float(translation_m[0] * 1000.0),
        float(translation_m[1] * 1000.0),
        float(translation_m[2] * 1000.0),
    )
    rpy_degrees = (
        float(np.degrees(rpy_rad[0])),
        float(np.degrees(rpy_rad[1])),
        float(np.degrees(rpy_rad[2])),
    )
    return RobotSnapshot(
        timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        joint_degrees=tuple(float(np.degrees(value)) for value in joint_values_rad),
        base_gripper_pose_m=base_gripper_pose_m,
        translation_mm=translation_mm,
        rpy_degrees=rpy_degrees,
    )


def _apply_fixed_toolset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> None:
    robot.setToolset(DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME, ec)
    _raise_for_sdk_error(ec, f"setToolset({DEFAULT_TOOL_NAME}, {DEFAULT_WOBJ_NAME})")


def _wait_for_power_off(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float = 3.0,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if robot.powerState(ec) == xCoreSDK_python.PowerState.off:
            return True
        time.sleep(0.1)
    return False


def _shutdown_arm(arm: ConnectedArm) -> None:
    try:
        arm.robot.disableDrag(arm.ec)
    except Exception:
        pass
    try:
        arm.robot.stop(arm.ec)
    except Exception:
        pass
    try:
        arm.robot.setPowerState(False, arm.ec)
    except Exception:
        pass
    try:
        arm.robot.disconnectFromRobot(arm.ec)
    except Exception:
        pass


def _raise_for_sdk_error(ec: dict[str, object], action: str) -> None:
    code = ec.get("ec", 0)
    if code != 0:
        raise RuntimeError(f"{action}失败：ec={code}, message={ec.get('message', '')}")


# endregion


# region 标定计算
def _solve_eye_to_hand(samples: list[SampleRecord]) -> CalibrationResult:
    base_gripper_poses = [
        _validate_transform_m(
            f"sample_{sample.sample_index}.T_base_gripper",
            sample.robot_snapshot.base_gripper_pose_m,
        )
        for sample in samples
    ]
    camera_board_poses = [
        _validate_transform_m(
            f"sample_{sample.sample_index}.T_camera_board",
            sample.camera_board_pose_m,
        )
        for sample in samples
    ]
    base_to_gripper_poses = [np.linalg.inv(pose) for pose in base_gripper_poses]

    rotation_base_to_gripper = [pose[:3, :3] for pose in base_to_gripper_poses]
    translation_base_to_gripper_m = [
        pose[:3, 3].reshape(3, 1) for pose in base_to_gripper_poses
    ]
    rotation_board_to_camera = [pose[:3, :3] for pose in camera_board_poses]
    translation_board_to_camera_m = [
        pose[:3, 3].reshape(3, 1) for pose in camera_board_poses
    ]
    rotation_camera_to_base, translation_camera_to_base = cv2.calibrateHandEye(
        R_gripper2base=rotation_base_to_gripper,
        t_gripper2base=translation_base_to_gripper_m,
        R_target2cam=rotation_board_to_camera,
        t_target2cam=translation_board_to_camera_m,
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    park_base_camera_pose_m = np.eye(4, dtype=np.float64)
    park_base_camera_pose_m[:3, :3] = np.asarray(
        rotation_camera_to_base, dtype=np.float64
    ).reshape(3, 3)
    park_base_camera_pose_m[:3, 3] = np.asarray(
        translation_camera_to_base, dtype=np.float64
    ).reshape(3)
    park_base_camera_pose_m = _validate_transform_m(
        "PARK.T_base_camera", park_base_camera_pose_m
    )

    initial_gripper_board_poses_m = tuple(
        np.linalg.inv(base_gripper) @ park_base_camera_pose_m @ camera_board
        for base_gripper, camera_board in zip(
            base_gripper_poses, camera_board_poses, strict=True
        )
    )
    mean_gripper_board_pose_m = _mean_transform_m(initial_gripper_board_poses_m)
    base_camera_poses_m = tuple(
        base_gripper @ mean_gripper_board_pose_m @ np.linalg.inv(camera_board)
        for base_gripper, camera_board in zip(
            base_gripper_poses, camera_board_poses, strict=True
        )
    )
    base_camera_pose_m = _mean_transform_m(base_camera_poses_m)
    gripper_board_poses_m = tuple(
        np.linalg.inv(base_gripper) @ base_camera_pose_m @ camera_board
        for base_gripper, camera_board in zip(
            base_gripper_poses, camera_board_poses, strict=True
        )
    )
    return CalibrationResult(
        park_base_camera_pose_m=park_base_camera_pose_m,
        base_camera_pose_m=base_camera_pose_m,
        base_camera_poses_m=base_camera_poses_m,
        camera_board_poses_m=tuple(camera_board_poses),
        mean_gripper_board_pose_m=mean_gripper_board_pose_m,
        gripper_board_poses_m=gripper_board_poses_m,
        used_sample_indices=tuple(sample.sample_index for sample in samples),
        stats=_compute_calibration_stats(base_camera_poses_m),
    )


def _compute_calibration_stats(
    base_camera_poses_m: tuple[np.ndarray, ...],
) -> CalibrationStats:
    translations = np.asarray(
        [pose[:3, 3] for pose in base_camera_poses_m], dtype=np.float64
    )
    mean_translation = np.mean(translations, axis=0)
    translation_errors_m = np.linalg.norm(translations - mean_translation, axis=1)
    mean_rotation = Rotation3D.from_matrix(
        np.asarray([pose[:3, :3] for pose in base_camera_poses_m])
    ).mean()
    rotation_vectors_deg = np.asarray(
        [
            np.degrees(
                (
                    mean_rotation.inv() * Rotation3D.from_matrix(pose[:3, :3])
                ).as_rotvec()
            )
            for pose in base_camera_poses_m
        ],
        dtype=np.float64,
    )
    rotation_errors_deg = np.linalg.norm(rotation_vectors_deg, axis=1)
    return CalibrationStats(
        translation_mean_m=mean_translation,
        translation_std_m=np.std(translations, axis=0),
        translation_errors_m=translation_errors_m,
        translation_mean_error_m=float(np.mean(translation_errors_m)),
        translation_max_error_m=float(np.max(translation_errors_m)),
        rotation_vector_std_deg=np.std(rotation_vectors_deg, axis=0),
        rotation_errors_deg=rotation_errors_deg,
        rotation_mean_error_deg=float(np.mean(rotation_errors_deg)),
        rotation_max_error_deg=float(np.max(rotation_errors_deg)),
    )


def _mean_transform_m(transforms_m: tuple[np.ndarray, ...]) -> np.ndarray:
    """计算 SE(3) 位姿均值；平移在 m 中求算术均值，旋转使用 SciPy Rotation.mean。"""

    if not transforms_m:
        raise ValueError("至少需要一个变换矩阵才能计算均值")
    validated = tuple(
        _validate_transform_m(f"mean_input_{index}", transform)
        for index, transform in enumerate(transforms_m, start=1)
    )
    mean_pose_m = np.eye(4, dtype=np.float64)
    mean_pose_m[:3, :3] = Rotation3D.from_matrix(
        np.asarray([pose[:3, :3] for pose in validated], dtype=np.float64)
    ).mean().as_matrix()
    mean_pose_m[:3, 3] = np.mean(
        np.asarray([pose[:3, 3] for pose in validated], dtype=np.float64), axis=0
    )
    return mean_pose_m


def _validate_transform_m(name: str, transform_m: np.ndarray) -> np.ndarray:
    """校验内部米制 SE(3) 矩阵，不执行任何隐式单位换算。"""

    matrix = np.asarray(transform_m, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 包含非有限数值")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} 齐次矩阵末行无效：{matrix[3]}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} 旋转矩阵不正交")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-6):
        raise ValueError(f"{name} 旋转矩阵不是右手系：det={determinant:.9f}")
    return matrix


def _update_outputs(
    *,
    samples: list[SampleRecord],
    session_dir: Path,
    result_prefix: str,
) -> CalibrationResult | None:
    _write_samples_csv(session_dir / f"{result_prefix}samples.csv", samples)
    if len(samples) < DEFAULT_MIN_CALIBRATION_SAMPLES:
        return None
    try:
        result = _solve_eye_to_hand(samples)
    except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("PARK 眼在手外求解失败，继续补充样本：{}", exc)
        return None
    _save_calibration_result(session_dir, result_prefix, result)
    logger.info(
        "EtH PARK：有效样本={}，逐帧 T_base_camera 平移 std_xyz=({:.3f}, {:.3f}, {:.3f})mm，"
        "旋转 mean/max={:.3f}/{:.3f}deg",
        len(result.used_sample_indices),
        result.stats.translation_std_m[0] * 1000.0,
        result.stats.translation_std_m[1] * 1000.0,
        result.stats.translation_std_m[2] * 1000.0,
        result.stats.rotation_mean_error_deg,
        result.stats.rotation_max_error_deg,
    )
    return result


# endregion


# region 相机、采样与保存
def _build_board() -> cv2.aruco.CharucoBoard:
    dictionary_code = int(cv2.aruco.DICT_APRILTAG_16h5)
    if DEFAULT_DICTIONARY_NAME != "DICT_APRILTAG_16H5":
        raise ValueError(f"不支持的字典配置：{DEFAULT_DICTIONARY_NAME}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_code)
    return cv2.aruco.CharucoBoard(
        (DEFAULT_SQUARES_X, DEFAULT_SQUARES_Y),
        float(DEFAULT_SQUARE_LENGTH_MM),
        float(DEFAULT_MARKER_LENGTH_MM),
        dictionary,
    )


def _camera_board_pose_mm_to_m(camera_board_pose_mm: np.ndarray) -> np.ndarray:
    """把 ChArUco 的 T_camera_board 从 mm 转为 m，且只在采样边界转换一次。"""

    source_mm = np.asarray(camera_board_pose_mm, dtype=np.float64).reshape(4, 4)
    camera_board_pose_m = source_mm.copy()
    camera_board_pose_m[:3, 3] = source_mm[:3, 3] / 1000.0
    return _validate_transform_m("T_camera_board", camera_board_pose_m)


def _read_head_camera_calibration(client: CameraPipelineClient) -> CameraCalibration:
    response = client.get_head_camera_intrinsics(timeout_s=DEFAULT_CAMERA_TIMEOUT_S)
    distortion = np.asarray(response.distortion, dtype=np.float64).reshape(-1, 1)
    if distortion.size == 0:
        distortion = np.zeros((5, 1), dtype=np.float64)
    logger.info(
        "头部相机内参：camera={}，size={}x{}，fx={:.3f}，fy={:.3f}",
        response.camera_name,
        response.width,
        response.height,
        response.fx,
        response.fy,
    )
    return CameraCalibration(
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


def _capture_sample(
    *,
    arm: ConnectedArm,
    frame_packet: CameraColorFramePacket,
    frame_bgr: np.ndarray,
    preview_bgr: np.ndarray,
    pose_result: CharucoPoseResult,
    sample_index: int,
    session_dir: Path,
    raw_dir: Path,
    preview_dir: Path,
) -> SampleRecord | None:
    if pose_result.transform_se3 is None or pose_result.reprojection_error_px is None:
        return None
    robot_snapshot = _read_robot_snapshot(arm)
    camera_board_pose_m = _camera_board_pose_mm_to_m(pose_result.transform_se3)
    sample_name = f"sample_{sample_index:03d}"
    raw_path = raw_dir / f"{sample_name}.png"
    preview_path = preview_dir / f"{sample_name}.png"
    if not cv2.imwrite(str(raw_path), frame_bgr):
        raise RuntimeError(f"保存原始图像失败：{raw_path}")
    if not cv2.imwrite(str(preview_path), preview_bgr):
        raise RuntimeError(f"保存预览图像失败：{preview_path}")
    return SampleRecord(
        sample_index=sample_index,
        timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        frame_id=int(frame_packet.frame_id),
        camera_timestamp_ms=float(frame_packet.timestamp_ms),
        marker_count=int(pose_result.marker_count),
        charuco_count=int(pose_result.charuco_count),
        reprojection_error_px=float(pose_result.reprojection_error_px),
        robot_snapshot=robot_snapshot,
        camera_board_pose_m=camera_board_pose_m,
        raw_frame_path=raw_path.relative_to(session_dir),
        preview_frame_path=preview_path.relative_to(session_dir),
    )


def _write_samples_csv(csv_path: Path, samples: list[SampleRecord]) -> None:
    fieldnames = [
        "sample_index",
        "timestamp_iso",
        "frame_id",
        "camera_timestamp_ms",
        "marker_count",
        "charuco_count",
        "reprojection_error_px",
        "robot_x_mm",
        "robot_y_mm",
        "robot_z_mm",
        "robot_roll_deg",
        "robot_pitch_deg",
        "robot_yaw_deg",
        "raw_frame_path",
        "preview_frame_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_index": sample.sample_index,
                    "timestamp_iso": sample.timestamp_iso,
                    "frame_id": sample.frame_id,
                    "camera_timestamp_ms": f"{sample.camera_timestamp_ms:.3f}",
                    "marker_count": sample.marker_count,
                    "charuco_count": sample.charuco_count,
                    "reprojection_error_px": f"{sample.reprojection_error_px:.6f}",
                    "robot_x_mm": f"{sample.robot_snapshot.translation_mm[0]:.6f}",
                    "robot_y_mm": f"{sample.robot_snapshot.translation_mm[1]:.6f}",
                    "robot_z_mm": f"{sample.robot_snapshot.translation_mm[2]:.6f}",
                    "robot_roll_deg": f"{sample.robot_snapshot.rpy_degrees[0]:.6f}",
                    "robot_pitch_deg": f"{sample.robot_snapshot.rpy_degrees[1]:.6f}",
                    "robot_yaw_deg": f"{sample.robot_snapshot.rpy_degrees[2]:.6f}",
                    "raw_frame_path": str(sample.raw_frame_path),
                    "preview_frame_path": str(sample.preview_frame_path),
                }
            )


def _save_calibration_result(
    session_dir: Path,
    result_prefix: str,
    result: CalibrationResult,
) -> None:
    matrix_path = session_dir / f"{result_prefix}T_base_camera.npy"
    park_matrix_path = session_dir / f"{result_prefix}T_base_camera_PARK_raw.npy"
    text_path = session_dir / f"{result_prefix}T_base_camera.txt"
    base_camera_samples_path = (
        session_dir / f"{result_prefix}base_camera_per_sample.csv"
    )
    residual_path = session_dir / f"{result_prefix}gripper_board_residuals.csv"
    np.save(matrix_path, result.base_camera_pose_m)
    np.save(park_matrix_path, result.park_base_camera_pose_m)
    _write_base_camera_samples_csv(base_camera_samples_path, result)
    rpy_deg = Rotation3D.from_matrix(result.base_camera_pose_m[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    park_rpy_deg = Rotation3D.from_matrix(
        result.park_base_camera_pose_m[:3, :3]
    ).as_euler("xyz", degrees=True)
    translation_mean_m = result.stats.translation_mean_m
    translation_std_m = result.stats.translation_std_m
    rotation_vector_std_deg = result.stats.rotation_vector_std_deg
    lines = [
        "Eye-to-hand calibration result",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "method=PARK",
        "result_semantics=T_base_camera_mean (camera coordinates -> robot base coordinates)",
        "saved_npy=T_base_camera_mean computed from per-sample T_base_camera",
        "solve_input_robot=inv(T_base_gripper) reconstructed from SDK trans(m)+rpy(rad)",
        'robot_rpy_reconstruction=Rotation.from_euler("xyz", sdk_rpy_rad, degrees=False)',
        "solve_input_board=T_camera_board converted once from ChArUco mm to m",
        "internal_translation_unit=m",
        "display_translation_unit=mm",
        "per_sample_formula=T_base_camera_i = T_base_gripper_i @ mean(T_gripper_board) @ inv(T_camera_board_i)",
        "gripper_board_formula=T_gripper_board_i=inv(T_base_gripper_i) @ T_base_camera @ T_camera_board_i",
        f"head_camera={DEFAULT_CAMERA_NAME}",
        f"head_yaw_deg={DEFAULT_HEAD_YAW_DEG:.6f}",
        f"head_pitch_deg={DEFAULT_HEAD_PITCH_DEG:.6f}",
        f"dictionary={DEFAULT_DICTIONARY_NAME}",
        f"board={DEFAULT_SQUARES_X}x{DEFAULT_SQUARES_Y}",
        f"square_length_mm={DEFAULT_SQUARE_LENGTH_MM}",
        f"marker_length_mm={DEFAULT_MARKER_LENGTH_MM}",
        f"valid_sample_count={len(result.used_sample_indices)}",
        "valid_sample_indices=" + ",".join(str(value) for value in result.used_sample_indices),
        "",
        "Per-sample T_base_camera translation mean, unit m:",
        f"x_mean_m={translation_mean_m[0]:.10f}",
        f"y_mean_m={translation_mean_m[1]:.10f}",
        f"z_mean_m={translation_mean_m[2]:.10f}",
        "Per-sample T_base_camera translation mean, unit mm:",
        f"x_mean_mm={translation_mean_m[0] * 1000.0:.6f}",
        f"y_mean_mm={translation_mean_m[1] * 1000.0:.6f}",
        f"z_mean_mm={translation_mean_m[2] * 1000.0:.6f}",
        "Per-sample T_base_camera translation std, unit mm:",
        f"x_std_mm={translation_std_m[0] * 1000.0:.6f}",
        f"y_std_mm={translation_std_m[1] * 1000.0:.6f}",
        f"z_std_mm={translation_std_m[2] * 1000.0:.6f}",
        "Per-sample T_base_camera local rotation-vector std, unit deg:",
        f"rx_std_deg={rotation_vector_std_deg[0]:.6f}",
        f"ry_std_deg={rotation_vector_std_deg[1]:.6f}",
        f"rz_std_deg={rotation_vector_std_deg[2]:.6f}",
        f"translation_mean_error_mm={result.stats.translation_mean_error_m * 1000.0:.9f}",
        f"translation_max_error_mm={result.stats.translation_max_error_m * 1000.0:.9f}",
        f"rotation_mean_error_deg={result.stats.rotation_mean_error_deg:.9f}",
        f"rotation_max_error_deg={result.stats.rotation_max_error_deg:.9f}",
        "",
        "T_base_camera_mean:",
        np.array2string(result.base_camera_pose_m, precision=10, suppress_small=False),
        "",
        "T_base_camera_mean translation_mm:",
        np.array2string(result.base_camera_pose_m[:3, 3] * 1000.0, precision=6),
        "T_base_camera_mean rpy_deg:",
        np.array2string(rpy_deg, precision=6),
        "",
        "T_base_camera_PARK_raw:",
        np.array2string(
            result.park_base_camera_pose_m, precision=10, suppress_small=False
        ),
        "T_base_camera_PARK_raw translation_mm:",
        np.array2string(
            result.park_base_camera_pose_m[:3, 3] * 1000.0, precision=6
        ),
        "T_base_camera_PARK_raw rpy_deg:",
        np.array2string(park_rpy_deg, precision=6),
        "",
        f"per_sample_records={base_camera_samples_path.name}",
        "",
        *_build_camera_board_result_lines(result),
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with residual_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "gripper_board_x_mm",
                "gripper_board_y_mm",
                "gripper_board_z_mm",
                "gripper_board_roll_deg",
                "gripper_board_pitch_deg",
                "gripper_board_yaw_deg",
            ]
        )
        for sample_index, pose in zip(
            result.used_sample_indices, result.gripper_board_poses_m, strict=True
        ):
            pose_rpy_deg = Rotation3D.from_matrix(pose[:3, :3]).as_euler(
                "xyz", degrees=True
            )
            writer.writerow(
                [
                    sample_index,
                    f"{pose[0, 3] * 1000.0:.6f}",
                    f"{pose[1, 3] * 1000.0:.6f}",
                    f"{pose[2, 3] * 1000.0:.6f}",
                    f"{pose_rpy_deg[0]:.6f}",
                    f"{pose_rpy_deg[1]:.6f}",
                    f"{pose_rpy_deg[2]:.6f}",
                ]
            )


def _build_camera_board_result_lines(result: CalibrationResult) -> list[str]:
    """构造逐样本 T_camera_board 文本；内部矩阵为 m，另附 mm/deg 展示值。"""

    lines = [
        "[per_sample_T_camera_board]",
        "matrix_translation_unit=m",
        "display_translation_unit=mm",
        "rotation_display_convention=as_euler(\"xyz\", degrees=True)",
    ]
    for sample_index, camera_board_pose_m in zip(
        result.used_sample_indices, result.camera_board_poses_m, strict=True
    ):
        matrix = _validate_transform_m(
            f"sample_{sample_index}.T_camera_board", camera_board_pose_m
        )
        rpy_deg = Rotation3D.from_matrix(matrix[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        lines.extend(
            [
                "",
                f"sample_{sample_index:03d}.T_camera_board:",
                np.array2string(matrix, precision=10, suppress_small=False),
                f"sample_{sample_index:03d}.translation_mm:",
                np.array2string(matrix[:3, 3] * 1000.0, precision=6),
                f"sample_{sample_index:03d}.rpy_deg:",
                np.array2string(rpy_deg, precision=6),
            ]
        )
    return lines


def _write_base_camera_samples_csv(
    output_path: Path,
    result: CalibrationResult,
) -> None:
    """逐样本记录完整 T_base_camera；矩阵平移为 m，另附 mm/deg 展示列。"""

    matrix_fields = (
        ("r00", 0, 0),
        ("r01", 0, 1),
        ("r02", 0, 2),
        ("tx_m", 0, 3),
        ("r10", 1, 0),
        ("r11", 1, 1),
        ("r12", 1, 2),
        ("ty_m", 1, 3),
        ("r20", 2, 0),
        ("r21", 2, 1),
        ("r22", 2, 2),
        ("tz_m", 2, 3),
        ("h30", 3, 0),
        ("h31", 3, 1),
        ("h32", 3, 2),
        ("h33", 3, 3),
    )
    fieldnames = [
        "sample_index",
        *[fieldname for fieldname, _, _ in matrix_fields],
        "x_mm",
        "y_mm",
        "z_mm",
        "qw",
        "qx",
        "qy",
        "qz",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "translation_error_mm",
        "rotation_error_deg",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, pose_m, translation_error_m, rotation_error_deg in zip(
            result.used_sample_indices,
            result.base_camera_poses_m,
            result.stats.translation_errors_m,
            result.stats.rotation_errors_deg,
            strict=True,
        ):
            matrix = _validate_transform_m(
                f"sample_{sample_index}.T_base_camera", pose_m
            )
            quat_xyzw = Rotation3D.from_matrix(matrix[:3, :3]).as_quat()
            rpy_deg = Rotation3D.from_matrix(matrix[:3, :3]).as_euler(
                "xyz", degrees=True
            )
            row: dict[str, float | int | str] = {
                "sample_index": sample_index,
                "x_mm": f"{matrix[0, 3] * 1000.0:.6f}",
                "y_mm": f"{matrix[1, 3] * 1000.0:.6f}",
                "z_mm": f"{matrix[2, 3] * 1000.0:.6f}",
                "qw": f"{quat_xyzw[3]:.9f}",
                "qx": f"{quat_xyzw[0]:.9f}",
                "qy": f"{quat_xyzw[1]:.9f}",
                "qz": f"{quat_xyzw[2]:.9f}",
                "roll_deg": f"{rpy_deg[0]:.6f}",
                "pitch_deg": f"{rpy_deg[1]:.6f}",
                "yaw_deg": f"{rpy_deg[2]:.6f}",
                "translation_error_mm": f"{translation_error_m * 1000.0:.6f}",
                "rotation_error_deg": f"{rotation_error_deg:.6f}",
            }
            for fieldname, matrix_row, matrix_column in matrix_fields:
                row[fieldname] = f"{matrix[matrix_row, matrix_column]:.10f}"
            writer.writerow(row)


# endregion


# region 可视化与通用工具
def _draw_preview(
    *,
    frame_bgr: np.ndarray,
    pose_result: CharucoPoseResult,
    sample_count: int,
    result: CalibrationResult | None,
    arm_side: str,
    calibration: CameraCalibration,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if pose_result.marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(
            canvas, pose_result.marker_corners_px, pose_result.marker_ids
        )
    if pose_result.charuco_corners_px is not None and pose_result.charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            canvas,
            pose_result.charuco_corners_px.reshape(-1, 1, 2).astype(np.float32),
            pose_result.charuco_ids,
        )
    if pose_result.rvec is not None and pose_result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            pose_result.rvec,
            pose_result.tvec,
            float(DEFAULT_SQUARE_LENGTH_MM * 1.5),
            3,
        )

    status_color = (40, 220, 40) if pose_result.transform_se3 is not None else (30, 30, 240)
    reprojection = (
        "NA"
        if pose_result.reprojection_error_px is None
        else f"{pose_result.reprojection_error_px:.3f}px"
    )
    lines = [
        f"Eye-to-Hand | arm={arm_side} | camera={DEFAULT_CAMERA_NAME}",
        f"head yaw={DEFAULT_HEAD_YAW_DEG:.1f}deg pitch={DEFAULT_HEAD_PITCH_DEG:.1f}deg",
        f"markers={pose_result.marker_count} charuco={pose_result.charuco_count} reproj={reprojection}",
        f"samples={sample_count} | Space/Enter/P capture | Q/Esc quit",
    ]
    if result is None:
        lines.append(f"solve after {DEFAULT_MIN_CALIBRATION_SAMPLES} valid samples")
    else:
        lines.append(
            "T_base_camera std xyz: "
            f"({result.stats.translation_std_m[0] * 1000.0:.2f}, "
            f"{result.stats.translation_std_m[1] * 1000.0:.2f}, "
            f"{result.stats.translation_std_m[2] * 1000.0:.2f})mm | "
            f"rot mean={result.stats.rotation_mean_error_deg:.3f}deg"
        )
    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, 10), (940, 172), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0, canvas)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (25, 40 + index * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            status_color if index == 2 else (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _create_session_dir(output_root: Path, prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir = output_root / f"{prefix}{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _parse_cli() -> AppConfig:
    parser = argparse.ArgumentParser(description="左右臂头部相机眼在手外拖动标定")
    parser.add_argument("--service-addr", default=DEFAULT_ORIN_SERVICE_ADDR)
    parser.add_argument("--left-arm-ip", default=DEFAULT_LEFT_ARM_IP)
    parser.add_argument("--right-arm-ip", default=DEFAULT_RIGHT_ARM_IP)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS
    )
    args = parser.parse_args()
    return AppConfig(
        service_addr=str(args.service_addr),
        left_arm_ip=str(args.left_arm_ip),
        right_arm_ip=str(args.right_arm_ip),
        output_root=Path(args.output_root),
        min_charuco_corners=int(args.min_charuco_corners),
    )


# endregion


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
        raise SystemExit(130) from None
