from __future__ import annotations

import ast
import csv
import gc
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.ball_pose_detection.protocol import BallPoseDetectionRequest  # noqa: E402
from camera_pipeline.client import CameraPipelineClient  # noqa: E402
from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from test.wuji.ball_pose_detection import (  # noqa: E402
    DEFAULT_CAMERA_NAME as DEFAULT_BALL_POSE_CAMERA_NAME,
    DEFAULT_PRIOR_CAPTURE_PATH,
    DEFAULT_SERVICE_ADDR as DEFAULT_BALL_POSE_SERVICE_ADDR,
    _build_three_ball_basis_transform,
    _build_priors_from_capture,
    _load_prior_capture,
)
from test.wuji.common import (  # noqa: E402
    DEFAULT_PORT,
    GRIPPER_PORT,
    SshTunnelGroup,
    close_wuyou_channel,
    create_wuyou_channel,
    stop_ssh_process,
)
from test.wuji.xcoresdk_arm_cli_test import (  # noqa: E402
    DEFAULT_CARTESIAN_ZONE,
    DEFAULT_JOINT_ZONE,
    DEFAULT_TOOL_NAME,
    DEFAULT_WOBJ_NAME,
    LEFT_ARM_IP,
    M11_ROOT_ACTUATOR_IDS,
    M11_TIP_ACTUATOR_IDS,
    RIGHT_ARM_IP,
    ConnectedArm,
    DahuanGripperClient,
    WujiRightHandClient,
    WujiBodyClient,
    _apply_named_toolset,
    _copy_cartesian_pose_context,
    _detect_arm_side,
    _ensure_nrt_motion_ready,
    _format_sequence,
    _parse_cartesian_pose_input,
    _mm_to_m,
    _m_to_mm,
    _deg_to_rad,
    _print_sdk_result,
    _rad_to_deg,
    _shutdown_robot,
    _validate_cartesian_target,
    _wait_until_idle,
)

DEFAULT_LEFT_RECORD_DIR = PROJECT_ROOT / "record_left"
"默认左臂拖动示教 CSV 目录。"

DEFAULT_RIGHT_RECORD_DIR = PROJECT_ROOT / "record_right"
"默认右臂拖动示教 CSV 目录。"

DEFAULT_RECORD_DIR = DEFAULT_LEFT_RECORD_DIR
"默认拖动示教 CSV 目录。"

DEFAULT_ARM_SIDE = "left"
"默认回放机械臂侧别。"

DEFAULT_MAX_FILES: int | None = None
"默认加载的 CSV 文件数量；`None` 表示全部。"

DEFAULT_REPLAY_JOINT_SPEED = 400.0
"回放关节空间速度，单位 deg/s。"

DEFAULT_REPLAY_CARTESIAN_SPEED = 20.0
"回放笛卡尔空间速度，单位 mm/s。"

DEFAULT_REPLAY_LIFT_SETTLE_DELAY_S = 1.0
"回放 lift 单次等待时间，单位 s。"

DEFAULT_REPLAY_LIFT_RETRY_COUNT = 4
"回放 lift 最大重试次数。"

DEFAULT_REPLAY_LIFT_HEIGHT_TOLERANCE_MM = 4.0
"回放 lift 到位误差容忍，单位 mm。"

CSV_CARTESIAN_OFFSET_TARGETS: list[int] = [4,6]
"需要应用全局笛卡尔纠偏的 CSV 序号列表。"

CSV_CARTESIAN_OFFSET_CALCULATE_AT:int = 3
"在该 CSV 的最后一个 arm pose 处计算一次全局笛卡尔纠偏。"

DEFAULT_OFFSET_SERVICE_ADDR = DEFAULT_BALL_POSE_SERVICE_ADDR
"计算全局 offset 时使用的球位姿检测服务地址。"

DEFAULT_OFFSET_CAMERA_NAME = DEFAULT_BALL_POSE_CAMERA_NAME
"计算全局 offset 时使用的相机名称。"

DEFAULT_OFFSET_PRIOR_CAPTURE_PATH = DEFAULT_PRIOR_CAPTURE_PATH
"计算全局 offset 时使用的先验采集结果路径。"


# region 数据结构


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """单条 CSV 回放记录。"""

    csv_name: str
    row_index: int
    action_type: str
    joints_text: str
    pose_text: str


@dataclass(frozen=True, slots=True)
class ParsedArmPose:
    """CSV 中单条笛卡尔目标。"""

    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    has_elbow: bool | None
    elbow_deg: float | None
    conf_data: tuple[int, ...] | None


@dataclass(slots=True)
class ReplayRuntime:
    """回放执行期上下文。"""

    connected_arm: ConnectedArm
    hand_process: SshTunnelGroup
    hand_channel: object
    body_process: SshTunnelGroup
    body_channel: object
    body: WujiBodyClient
    gripper: DahuanGripperClient | None = None
    right_hand: WujiRightHandClient | None = None
    global_cartesian_offset: tuple[tuple[float, float, float, float], ...] | None = None
    offset_service_addr: str = DEFAULT_OFFSET_SERVICE_ADDR
    offset_camera_name: str = DEFAULT_OFFSET_CAMERA_NAME
    offset_prior_capture_path: Path = DEFAULT_OFFSET_PRIOR_CAPTURE_PATH
    joint_speed_deg_s: float = DEFAULT_REPLAY_JOINT_SPEED
    cartesian_speed_mm_s: float = DEFAULT_REPLAY_CARTESIAN_SPEED
    auto_execute_remaining: bool = False


# endregion


# region CSV 解析

def _discover_csv_paths(record_dir: Path, max_files: int | None) -> list[Path]:
    if not record_dir.is_dir():
        raise FileNotFoundError(f"CSV 目录不存在: {record_dir}")
    csv_paths = sorted(path for path in record_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    if max_files is not None:
        return csv_paths[:max_files]
    return csv_paths


def _load_replay_rows(csv_path: Path) -> list[ReplayRow]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[ReplayRow] = []
        for row_index, row in enumerate(reader, start=1):
            action_type = str(row.get("type", "")).strip().lower()
            joints_text = str(row.get("joints", "")).strip()
            pose_text = str(row.get("pose", "")).strip()
            if action_type == "":
                raise ValueError(f"CSV 缺少 type: file={csv_path}, row={row_index}")
            rows.append(
                ReplayRow(
                    csv_name=csv_path.name,
                    row_index=row_index,
                    action_type=action_type,
                    joints_text=joints_text,
                    pose_text=pose_text,
                )
            )
    if not rows:
        raise ValueError(f"CSV 没有可执行数据: {csv_path}")
    return rows


def _parse_joint_values(joints_text: str, expected_len: int = 7) -> list[float]:
    if joints_text.strip().lower() == "nan":
        raise ValueError("关节列为 NaN，不能解析为关节目标")
    parsed = ast.literal_eval(joints_text)
    if not isinstance(parsed, list) or len(parsed) != expected_len:
        raise ValueError(f"关节列长度无效: {joints_text}")
    return [float(value) for value in parsed]


def _parse_pose_values(pose_text: str) -> ParsedArmPose:
    if pose_text.strip().lower() == "nan":
        raise ValueError("pose 列为 NaN，不能解析为笛卡尔目标")
    parsed_pose = _parse_cartesian_pose_input(pose_text)
    return ParsedArmPose(
        xyz_mm=parsed_pose.xyz_mm,
        rpy_deg=parsed_pose.rpy_deg,
        has_elbow=parsed_pose.has_elbow,
        elbow_deg=parsed_pose.elbow_deg,
        conf_data=parsed_pose.conf_data,
    )


def _extract_csv_sequence(csv_name: str) -> int:
    prefix = csv_name.split("_", maxsplit=1)[0]
    return int(prefix)


def _format_optional_csv_sequence(sequence: int | None) -> str:
    if sequence is None:
        return "None"
    return f"{sequence:02d}"


# endregion


# region 连接与执行


def _resolve_record_dir(arm_side: str, record_dir: Path | None) -> Path:
    if record_dir is not None and Path(record_dir) != DEFAULT_RECORD_DIR:
        return Path(record_dir)
    if arm_side == "right":
        return DEFAULT_RIGHT_RECORD_DIR
    return DEFAULT_LEFT_RECORD_DIR


def _connect_arm(arm_side: str) -> ConnectedArm:
    ec: dict[str, object] = {}
    robot_ip = LEFT_ARM_IP if arm_side == "left" else RIGHT_ARM_IP
    robot = xCoreSDK_python.xMateErProRobot(robot_ip)
    robot_info = robot.robotInfo(ec)
    _print_sdk_result(f"robotInfo({robot_ip})", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"读取机械臂机器人信息失败: arm_side={arm_side}, ip={robot_ip}")
    if _apply_named_toolset(robot, ec) is None:
        raise RuntimeError(
            f"设置默认工具/工件失败: ip={robot_ip}, tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}"
        )
    detected_arm_side = _detect_arm_side(robot_info.type)
    if detected_arm_side != arm_side:
        raise RuntimeError(f"连接到的机械臂侧别不匹配: expected={arm_side}, ip={robot_ip}, actual={detected_arm_side}")
    logger.success(
        "已连接机械臂 arm_side={} ip={} type={} uid={} tool={} wobj={}",
        arm_side,
        robot_ip,
        robot_info.type,
        robot_info.id,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    return ConnectedArm(
        arm_side=detected_arm_side,
        robot_ip=robot_ip,
        robot=robot,
        robot_type=robot_info.type,
        robot_uid=robot_info.id,
        ec=ec,
    )


def _create_runtime(arm_side: str) -> ReplayRuntime:
    connected_arm = _connect_arm(arm_side)
    hand_port = GRIPPER_PORT if arm_side == "left" else DEFAULT_PORT
    hand_process, hand_channel = create_wuyou_channel(hand_port)
    body_process, body_channel = create_wuyou_channel(DEFAULT_PORT)
    runtime = ReplayRuntime(
        connected_arm=connected_arm,
        hand_process=hand_process,
        hand_channel=hand_channel,
        body_process=body_process,
        body_channel=body_channel,
        body=WujiBodyClient(body_channel),
    )
    if arm_side == "left":
        runtime.gripper = DahuanGripperClient(hand_channel)
    else:
        runtime.right_hand = WujiRightHandClient(hand_channel)
    return runtime


def _prepare_runtime(runtime: ReplayRuntime) -> None:
    if not _ensure_nrt_motion_ready(runtime.connected_arm.robot, runtime.connected_arm.ec):
        raise RuntimeError(f"{runtime.connected_arm.arm_side} 臂未准备到可执行回放的 NRT 状态")
    runtime.body.lift.set_enable(True)
    logger.info(
        "已确认机械臂侧别={} 基坐标采用 tool={} wobj={}",
        runtime.connected_arm.arm_side,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )


def _execute_joint_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    target_joint_deg = _parse_joint_values(row.joints_text)
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    cmd_id = xCoreSDK_python.PyString()
    target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(target_joint_deg))
    robot.moveReset(ec)
    _print_sdk_result("moveReset(replay-joint)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放关节 moveReset 失败")
    robot.moveAppend(
        [xCoreSDK_python.MoveAbsJCommand(target_joint, runtime.joint_speed_deg_s, DEFAULT_JOINT_ZONE)],
        cmd_id,
        ec,
    )
    _print_sdk_result("moveAppend(MoveAbsJCommand)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放关节 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(replay-joint)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放关节 moveStart 失败")
    logger.info(
        "已下发关节运动 file={} row={} joints(deg)=[{}] cmd_id={}",
        row.csv_name,
        row.row_index,
        _format_sequence(target_joint_deg),
        cmd_id.content(),
    )
    if not _wait_until_idle(robot, ec, "等待回放关节运动"):
        logger.warning(
            "回放关节运动等待超时，继续后续流程 file={} row={} cmd_id={}",
            row.csv_name,
            row.row_index,
            cmd_id.content(),
        )


def _build_cartesian_target(runtime: ReplayRuntime, row: ReplayRow) -> xCoreSDK_python.CartesianPosition:
    parsed_pose = _parse_pose_values(row.pose_text)
    target_xyz_mm = list(parsed_pose.xyz_mm)
    target_rpy_deg = list(parsed_pose.rpy_deg)
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    current_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取当前笛卡尔位姿失败")
    target_pose = xCoreSDK_python.CartesianPosition(_mm_to_m(target_xyz_mm) + _deg_to_rad(target_rpy_deg))
    _copy_cartesian_pose_context(current_pose, target_pose)
    if parsed_pose.has_elbow is not None:
        target_pose.hasElbow = parsed_pose.has_elbow
    if parsed_pose.elbow_deg is not None:
        target_pose.elbow = _deg_to_rad([parsed_pose.elbow_deg])[0]
    if parsed_pose.conf_data is not None:
        target_pose.confData = list(parsed_pose.conf_data)
    csv_sequence = _extract_csv_sequence(row.csv_name)
    if csv_sequence in CSV_CARTESIAN_OFFSET_TARGETS:
        target_pose = _apply_global_cartesian_offset(runtime, row, target_pose)
    return target_pose


def _frame_to_homogeneous_matrix(frame: xCoreSDK_python.CartesianPosition) -> list[list[float]]:
    rx, ry, rz = (float(value) for value in frame.rpy)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    matrix = [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(frame.trans[0])],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(frame.trans[1])],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(frame.trans[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matrix


def _multiply_homogeneous_matrix(
    left: tuple[tuple[float, float, float, float], ...],
    right: list[list[float]],
) -> list[list[float]]:
    result = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for row_index in range(4):
        for col_index in range(4):
            result[row_index][col_index] = sum(
                float(left[row_index][term_index]) * float(right[term_index][col_index])
                for term_index in range(4)
            )
    return result


def _homogeneous_matrix_to_rpy(matrix: list[list[float]]) -> tuple[float, float, float]:
    sy = -float(matrix[2][0])
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-9:
        rx = math.atan2(float(matrix[2][1]), float(matrix[2][2]))
        ry = math.atan2(sy, cy)
        rz = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
        return rx, ry, rz
    rx = math.atan2(-float(matrix[1][2]), float(matrix[1][1]))
    ry = math.atan2(sy, cy)
    rz = 0.0
    return rx, ry, rz


def _homogeneous_matrix_to_cartesian_position(
    source_pose: xCoreSDK_python.CartesianPosition,
    matrix: list[list[float]],
) -> xCoreSDK_python.CartesianPosition:
    xyz_m = [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
    rpy_rad = list(_homogeneous_matrix_to_rpy(matrix))
    target_pose = xCoreSDK_python.CartesianPosition(xyz_m + rpy_rad)
    target_pose.hasElbow = source_pose.hasElbow
    target_pose.elbow = source_pose.elbow
    target_pose.confData = list(source_pose.confData)
    return target_pose


def _apply_global_cartesian_offset(
    runtime: ReplayRuntime,
    row: ReplayRow,
    target_pose: xCoreSDK_python.CartesianPosition,
) -> xCoreSDK_python.CartesianPosition:
    if runtime.global_cartesian_offset is None:
        raise RuntimeError(
            f"CSV {row.csv_name} 需要使用全局笛卡尔纠偏，但当前尚未在 "
            f"{_format_optional_csv_sequence(CSV_CARTESIAN_OFFSET_CALCULATE_AT)}_*.csv 末尾计算 offset"
        )
    original_matrix = _frame_to_homogeneous_matrix(target_pose)
    offset_matrix_m = _offset_matrix_mm_to_pose_matrix_m(runtime.global_cartesian_offset)
    corrected_matrix = _multiply_homogeneous_matrix(offset_matrix_m, original_matrix)
    corrected_pose = _homogeneous_matrix_to_cartesian_position(target_pose, corrected_matrix)
    logger.info(
        "已对笛卡尔目标应用全局左乘纠偏 file={} row={} base=tool:{} wobj:{}",
        row.csv_name,
        row.row_index,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    return corrected_pose


def _load_prior_three_ball_basis_transform(prior_capture_path: Path) -> np.ndarray:
    prior_capture = _load_prior_capture(prior_capture_path)
    recorded_balls = prior_capture.get("balls", {}).get("ballinfo", [])
    if not isinstance(recorded_balls, list) or len(recorded_balls) < 3:
        raise RuntimeError(f"先验文件缺少三球位置: {prior_capture_path}")
    detections = [{"center_mm": item.get("position_camera_mm")} for item in recorded_balls[:3]]
    basis_transform = _build_three_ball_basis_transform(detections)
    if basis_transform is None:
        raise RuntimeError(f"先验三球基础坐标系构造失败: {prior_capture_path}")
    return basis_transform


def _detect_current_three_ball_basis_transform(
    service_addr: str,
    camera_name: str,
    prior_capture_path: Path,
) -> np.ndarray:
    prior_capture = _load_prior_capture(prior_capture_path)
    priors = _build_priors_from_capture(prior_capture)
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=30_000)
    try:
        response = client.request_ball_pose_detection(
            BallPoseDetectionRequest(
                request_id=1,
                camera_name=str(camera_name),
                frame_id=-1,
                enable_debug=True,
                priors=tuple(priors),
            )
        )
    finally:
        client.close()
    if response.error is not None:
        raise RuntimeError(f"ball pose detection 返回错误: {response.error}")
    if response.matched_count < 3:
        raise RuntimeError("ball pose detection 未返回足够的三球检测结果")
    basis_transform = _build_three_ball_basis_transform(response.detections)
    if basis_transform is None:
        raise RuntimeError("当前三球基础坐标系构造失败")
    logger.info(
        "ball pose detection 完成 frame_id={} matched_count={} camera={}",
        response.frame_id,
        response.matched_count,
        camera_name,
    )
    return basis_transform


def _calculate_global_cartesian_offset(
    runtime: ReplayRuntime,
    csv_path: Path,
) -> tuple[tuple[float, float, float, float], ...]:
    prior_three_ball_basis_transform = _load_prior_three_ball_basis_transform(runtime.offset_prior_capture_path)
    current_three_ball_basis_transform = _detect_current_three_ball_basis_transform(
        service_addr=runtime.offset_service_addr,
        camera_name=runtime.offset_camera_name,
        prior_capture_path=runtime.offset_prior_capture_path,
    )
    offset_matrix = prior_three_ball_basis_transform @ np.linalg.inv(current_three_ball_basis_transform)
    logger.success(
        "全局 offset 已计算 file={} trigger=csv_end_state base=tool:{} wobj:{}",
        csv_path.name,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    logger.info(
        "offset 来源=先验三球基础坐标系 左乘 当前三球基础坐标系逆 translation(mm)=[{}]",
        _format_sequence(
            [
                float(offset_matrix[0, 3]),
                float(offset_matrix[1, 3]),
                float(offset_matrix[2, 3]),
            ]
        ),
    )
    logger.info(
        "offset rotation(deg)=[{}]",
        _format_sequence(_rotation_matrix_to_rpy_deg(offset_matrix[:3, :3])),
    )
    offset_distance_mm = float(np.linalg.norm(offset_matrix[:3, 3]))
    logger.info("offset distance={:.3f} mm", offset_distance_mm)
    if offset_distance_mm > 50.0:
        logger.warning("offset 平移明显偏大，请检查拍摄时机/三球识别/先验是否一致 distance={:.3f} mm", offset_distance_mm)
    return (
        (
            float(offset_matrix[0, 0]),
            float(offset_matrix[0, 1]),
            float(offset_matrix[0, 2]),
            float(offset_matrix[0, 3]),
        ),
        (
            float(offset_matrix[1, 0]),
            float(offset_matrix[1, 1]),
            float(offset_matrix[1, 2]),
            float(offset_matrix[1, 3]),
        ),
        (
            float(offset_matrix[2, 0]),
            float(offset_matrix[2, 1]),
            float(offset_matrix[2, 2]),
            float(offset_matrix[2, 3]),
        ),
        (
            float(offset_matrix[3, 0]),
            float(offset_matrix[3, 1]),
            float(offset_matrix[3, 2]),
            float(offset_matrix[3, 3]),
        ),
    )


def _offset_matrix_mm_to_pose_matrix_m(
    offset_matrix_mm: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    matrix = np.asarray(offset_matrix_mm, dtype=np.float64).copy()
    matrix[:3, 3] = matrix[:3, 3] * 0.001
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3])),
        (float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3])),
    )


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    return _rad_to_deg(list(_homogeneous_matrix_to_rpy(matrix.tolist())))


def _execute_cartesian_row(
    runtime: ReplayRuntime,
    row: ReplayRow,
) -> None:
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    target_pose = _build_cartesian_target(runtime, row)
    robot_model = robot.model()
    toolset = robot.toolset(ec)
    _print_sdk_result("toolset(replay-cartesian)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放笛卡尔读取 toolset 失败")
    target_joint = robot_model.calcIk(target_pose, toolset, ec)
    _print_sdk_result("calcIk(replay-cartesian)", ec)
    if ec.get("ec", 0) != 0:
        fallback_joint_deg: list[float] | None = None
        try:
            fallback_joint_deg = _parse_joint_values(row.joints_text)
        except ValueError:
            fallback_joint_deg = None
        if fallback_joint_deg is not None:
            logger.warning(
                "回放笛卡尔目标逆解失败，改用记录关节值兜底 file={} row={} ec={} message={}",
                row.csv_name,
                row.row_index,
                ec.get("ec", 0),
                ec.get("message", ""),
            )
            target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(fallback_joint_deg))
        else:
            logger.warning(
                "回放笛卡尔目标逆解失败 file={} row={} ec={} message={}",
                row.csv_name,
                row.row_index,
                ec.get("ec", 0),
                ec.get("message", ""),
            )
            raise RuntimeError("回放笛卡尔目标逆解失败，且无可用 joints 兜底")
    cmd_id = xCoreSDK_python.PyString()
    should_fallback_to_move_abs_j = bool(ec.get("ec", 0) != 0) or not _validate_cartesian_target(robot, ec, target_pose)
    if should_fallback_to_move_abs_j:
        logger.warning(
            "回放 MoveL 路径检查失败，回退 MoveAbsJ file={} row={} xyz(mm)=[{}] rpy(deg)=[{}]",
            row.csv_name,
            row.row_index,
            _format_sequence(_m_to_mm(target_pose.trans)),
            _format_sequence(_rad_to_deg(target_pose.rpy)),
        )
    robot.moveReset(ec)
    _print_sdk_result("moveReset(replay-cartesian)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放笛卡尔 moveReset 失败")
    if should_fallback_to_move_abs_j:
        robot.moveAppend(
            [xCoreSDK_python.MoveAbsJCommand(target_joint, runtime.joint_speed_deg_s, DEFAULT_JOINT_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveAbsJ)", ec)
    else:
        robot.moveAppend(
            [xCoreSDK_python.MoveLCommand(target_pose, runtime.cartesian_speed_mm_s, DEFAULT_CARTESIAN_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveL)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放笛卡尔 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(replay-cartesian)", ec)
    if ec.get("ec", 0) != 0:
        logger.warning(
            "回放运动启动失败 file={} row={} motion={} ec={} message={}",
            row.csv_name,
            row.row_index,
            "MoveAbsJ" if should_fallback_to_move_abs_j else "MoveL",
            ec.get("ec", 0),
            ec.get("message", ""),
        )
        raise RuntimeError("回放笛卡尔 moveStart 失败")
    logger.info(
        "已下发笛卡尔运动 file={} row={} motion={} xyz(mm)=[{}] rpy(deg)=[{}] cmd_id={}",
        row.csv_name,
        row.row_index,
        "moveabsj" if should_fallback_to_move_abs_j else "movel",
        _format_sequence(_m_to_mm(target_pose.trans)),
        _format_sequence(_rad_to_deg(target_pose.rpy)),
        cmd_id.content(),
    )
    if not _wait_until_idle(robot, ec, "等待回放笛卡尔运动"):
        logger.warning(
            "回放笛卡尔运动等待超时，继续后续流程 file={} row={} motion={} cmd_id={}",
            row.csv_name,
            row.row_index,
            "moveabsj" if should_fallback_to_move_abs_j else "movel",
            cmd_id.content(),
        )


def _execute_gripper_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    if runtime.gripper is None:
        raise RuntimeError("当前 runtime 未配置左手夹爪客户端")
    target_value = int(round(float(row.pose_text)))
    if not runtime.gripper.set_pos(target_value):
        raise RuntimeError("夹爪 set_pos 下发失败")
    logger.info("已下发夹爪目标 file={} row={} pos={}，当前策略不等待到位", row.csv_name, row.row_index, target_value)
    deadline_hint = runtime.gripper.get_status()
    logger.info("夹爪当前状态 pos={} calibrated={}", deadline_hint.position, bool(deadline_hint.calibrated))


def _get_right_hand_positions(runtime: ReplayRuntime) -> list[float]:
    if runtime.right_hand is None:
        raise RuntimeError("当前 runtime 未配置右手 m11 客户端")
    state = runtime.right_hand.get_hand_state(include_tactile=False)
    if state is None:
        raise RuntimeError("右手状态不可用")
    actuators = state.get("actuators")
    if not isinstance(actuators, list):
        raise RuntimeError("右手状态格式异常：actuators 不是 list")
    positions: list[float] = []
    for index, actuator in enumerate(actuators):
        if not isinstance(actuator, dict):
            raise RuntimeError(f"右手状态格式异常：actuators[{index}] 不是 dict")
        position = actuator.get("position")
        if not isinstance(position, int | float):
            raise RuntimeError(f"右手状态格式异常：actuators[{index}].position 非数值")
        positions.append(float(position))
    return positions


def _execute_m11_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    if runtime.right_hand is None:
        raise RuntimeError("当前 runtime 未配置右手 m11 客户端")
    target_positions = _parse_joint_values(row.joints_text, expected_len=11)
    current_positions = _get_right_hand_positions(runtime)
    required_max_id = max(*M11_ROOT_ACTUATOR_IDS, *M11_TIP_ACTUATOR_IDS)
    if len(current_positions) <= required_max_id:
        raise RuntimeError(
            f"右手状态执行器数量不足以覆盖 m11 索引 required_max_id={required_max_id}, actual_len={len(current_positions)}"
        )
    for actuator_id, target_value in enumerate(target_positions):
        current_positions[actuator_id] = float(target_value)
    if not runtime.right_hand.set_hand_state(current_positions):
        raise RuntimeError("右手 m11 下发失败")
    logger.info(
        "已下发右手 m11 目标 file={} row={} root=[{}] tip=[{}]",
        row.csv_name,
        row.row_index,
        _format_sequence([current_positions[actuator_id] for actuator_id in M11_ROOT_ACTUATOR_IDS], decimals=4),
        _format_sequence([current_positions[actuator_id] for actuator_id in M11_TIP_ACTUATOR_IDS], decimals=4),
    )


def _read_lift_height_mm(result: object) -> float:
    if isinstance(result, tuple) and len(result) == 2:
        first_value = result[0]
        if isinstance(first_value, int | float):
            return float(first_value)
        raise TypeError(f"lift 返回值首元素类型无效: {type(first_value)!r}")
    if isinstance(result, int | float):
        return float(result)
    raise TypeError(f"lift 返回值类型无效: {type(result)!r}")


def _wait_replay_lift_until_near_target(body: WujiBodyClient, target_height_mm: int) -> float:
    lift = body.lift
    time.sleep(DEFAULT_REPLAY_LIFT_SETTLE_DELAY_S)
    attempt_index = 0
    current_height_mm = _read_lift_height_mm(lift.get_lift_physical_height())
    while attempt_index < DEFAULT_REPLAY_LIFT_RETRY_COUNT:
        if current_height_mm == -1.0:
            logger.warning("lift 返回值 -1，视为无效读数，立即重试且不计次数")
            current_height_mm = _read_lift_height_mm(lift.get_lift_physical_height())
            continue
        current_error_mm = abs(current_height_mm - float(target_height_mm))
        logger.info(
            "lift 到位检查 {}/{}: target={} mm actual={:.1f} mm error={:.1f} mm",
            attempt_index + 1,
            DEFAULT_REPLAY_LIFT_RETRY_COUNT,
            target_height_mm,
            current_height_mm,
            current_error_mm,
        )
        if current_error_mm <= DEFAULT_REPLAY_LIFT_HEIGHT_TOLERANCE_MM:
            return current_height_mm
        attempt_index += 1
        if attempt_index < DEFAULT_REPLAY_LIFT_RETRY_COUNT:
            lift.set_lift_physical_height(target_height_mm)
            logger.warning("lift 超出误差，重新下发目标高度 {} mm", target_height_mm)
            time.sleep(DEFAULT_REPLAY_LIFT_SETTLE_DELAY_S)
            current_height_mm = _read_lift_height_mm(lift.get_lift_physical_height())
    return current_height_mm


def _execute_lift_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    target_height_mm = int(round(float(row.pose_text)))
    if target_height_mm < 0:
        raise ValueError(f"lift 目标高度非法: {target_height_mm}")
    runtime.body.lift.set_lift_physical_height(target_height_mm)
    actual_height_mm = _wait_replay_lift_until_near_target(runtime.body, target_height_mm)
    logger.info(
        "lift 已执行 file={} row={} target={} mm actual={:.1f} mm",
        row.csv_name,
        row.row_index,
        target_height_mm,
        actual_height_mm,
    )


def _execute_row(
    runtime: ReplayRuntime,
    row: ReplayRow,
) -> None:
    logger.info(
        "开始执行 file={} row={} type={} joints={} pose={}",
        row.csv_name,
        row.row_index,
        row.action_type,
        row.joints_text,
        row.pose_text,
    )
    if row.action_type == "arm":
        if row.pose_text.strip().lower() == "nan":
            _execute_joint_row(runtime, row)
            return
        _execute_cartesian_row(runtime, row)
        return
    if row.action_type == "gripper":
        _execute_gripper_row(runtime, row)
        return
    if row.action_type == "m11":
        _execute_m11_row(runtime, row)
        return
    if row.action_type == "lift":
        _execute_lift_row(runtime, row)
        return
    raise ValueError(f"当前脚本暂不支持的记录类型: {row.action_type}")


def _cleanup_runtime(runtime: ReplayRuntime | None) -> None:
    if runtime is None:
        return
    try:
        _shutdown_robot(runtime.connected_arm.robot, runtime.connected_arm.ec)
    finally:
        preserved_hand_process = runtime.hand_process
        preserved_body_process = runtime.body_process
        close_wuyou_channel(runtime.hand_channel)
        close_wuyou_channel(runtime.body_channel)
        stop_ssh_process(preserved_hand_process)
        stop_ssh_process(preserved_body_process)
        del runtime
        gc.collect()


# endregion


# region 交互流程


def _toggle_arm_side(current_arm_side: str) -> str:
    if current_arm_side == "left":
        return "right"
    return "left"


def _confirm_runtime_config(
    arm_side: str,
    record_dir: Path,
    max_files: int | None,
    offset_service_addr: str,
    offset_camera_name: str,
    offset_prior_capture_path: Path,
    joint_speed_deg_s: float,
    cartesian_speed_mm_s: float,
) -> str:
    while True:
        print("")
        print("========== 回放配置 ==========")
        print(f"当前机械臂侧别: {arm_side}")
        print(f"当前 CSV 目录: {record_dir}")
        print(f"当前最大文件数: {'全部' if max_files is None else max_files}")
        print(f"当前关节回放速度: {joint_speed_deg_s:.2f} deg/s")
        print(f"当前笛卡尔回放速度: {cartesian_speed_mm_s:.2f} mm/s")
        print(f"当前 offset 服务: {offset_service_addr}")
        print(f"当前 offset 相机: {offset_camera_name}")
        print(f"当前 offset 先验: {offset_prior_capture_path}")
        print("输入回车确认当前配置并继续")
        print("输入 a 使用当前配置全自动开始")
        print("输入 l 切换左右臂")
        print("输入 s 调整初始速度")
        print("输入 q 退出")
        choice = input("请选择: ").strip().lower()
        if choice == "q":
            return "quit"
        if choice == "a":
            return "auto"
        if choice == "l":
            return "toggle-arm"
        if choice == "s":
            return "speed"
        if choice == "":
            return "confirm"
        print(f"未知输入: {choice}")


def _print_csv_summary(csv_paths: list[Path], joint_speed_deg_s: float, cartesian_speed_mm_s: float) -> None:
    print("本次将按以下顺序执行 CSV：")
    for index, csv_path in enumerate(csv_paths, start=1):
        print(f"  {index:02d}. {csv_path.name}")
    print(f"关节回放速度: {joint_speed_deg_s:.1f} deg/s")
    print(f"笛卡尔回放速度: {cartesian_speed_mm_s:.1f} mm/s")
    print(
        "全局笛卡尔纠偏配置: "
        f"calculate_at={_format_optional_csv_sequence(CSV_CARTESIAN_OFFSET_CALCULATE_AT)}, "
        f"targets={[f'{value:02d}' for value in CSV_CARTESIAN_OFFSET_TARGETS]}"
    )


def _prompt_positive_speed(current_value: float, label: str, unit_text: str) -> float:
    while True:
        raw_text = input(f"请输入新的{label}速度，当前 {current_value:.2f} {unit_text}，输入 q 返回: ").strip().lower()
        if raw_text == "q":
            return current_value
        try:
            new_value = float(raw_text)
        except ValueError:
            print("速度输入无效")
            continue
        if new_value <= 0:
            print("速度必须大于 0")
            continue
        return new_value


def _configure_replay_speeds(runtime: ReplayRuntime) -> None:
    print(f"当前关节回放速度: {runtime.joint_speed_deg_s:.2f} deg/s")
    print(f"当前笛卡尔回放速度: {runtime.cartesian_speed_mm_s:.2f} mm/s")
    runtime.joint_speed_deg_s = _prompt_positive_speed(runtime.joint_speed_deg_s, "关节回放", "deg/s")
    runtime.cartesian_speed_mm_s = _prompt_positive_speed(runtime.cartesian_speed_mm_s, "笛卡尔回放", "mm/s")
    logger.info(
        "回放速度已更新 joint={:.2f} deg/s cartesian={:.2f} mm/s",
        runtime.joint_speed_deg_s,
        runtime.cartesian_speed_mm_s,
    )


def _configure_speed_values(current_joint_speed_deg_s: float, current_cartesian_speed_mm_s: float) -> tuple[float, float]:
    print(f"当前关节回放速度: {current_joint_speed_deg_s:.2f} deg/s")
    print(f"当前笛卡尔回放速度: {current_cartesian_speed_mm_s:.2f} mm/s")
    new_joint_speed_deg_s = _prompt_positive_speed(current_joint_speed_deg_s, "关节回放", "deg/s")
    new_cartesian_speed_mm_s = _prompt_positive_speed(current_cartesian_speed_mm_s, "笛卡尔回放", "mm/s")
    logger.info(
        "初始回放速度已更新 joint={:.2f} deg/s cartesian={:.2f} mm/s",
        new_joint_speed_deg_s,
        new_cartesian_speed_mm_s,
    )
    return new_joint_speed_deg_s, new_cartesian_speed_mm_s


def _configure_initial_replay_speeds(runtime: ReplayRuntime, auto_start: bool) -> None:
    print(f"初始关节回放速度: {runtime.joint_speed_deg_s:.2f} deg/s")
    print(f"初始笛卡尔回放速度: {runtime.cartesian_speed_mm_s:.2f} mm/s")
    if auto_start:
        logger.info(
            "自动启动模式保留默认速度 joint={:.2f} deg/s cartesian={:.2f} mm/s",
            runtime.joint_speed_deg_s,
            runtime.cartesian_speed_mm_s,
        )
        return
    raw_text = input("输入回车保留默认速度，输入 s 立即调整速度，输入 q 退出: ").strip().lower()
    if raw_text == "q":
        raise RuntimeError("用户在启动阶段取消执行")
    if raw_text == "s":
        _configure_replay_speeds(runtime)


def _confirm_start(csv_paths: list[Path], auto_start: bool, joint_speed_deg_s: float, cartesian_speed_mm_s: float) -> str:
    _print_csv_summary(csv_paths, joint_speed_deg_s, cartesian_speed_mm_s)
    arm_side = "right" if "record_right" in str(csv_paths[0].parent) else "left"
    print(f"{arm_side} 臂基坐标固定为 tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    print("arm 动作策略: pose=NaN -> MoveAbsJ；否则 MoveL，失败自动回退 MoveAbsJ")
    if arm_side == "left":
        print("gripper 动作策略: 仅下发，不等待到位")
    else:
        print("m11 动作策略: 读取当前 11 轴状态后整体下发，不等待到位")
    print(
        "lift 动作策略: 等待到位后才允许继续下一步，"
        f"delay={DEFAULT_REPLAY_LIFT_SETTLE_DELAY_S:.1f}s "
        f"retry={DEFAULT_REPLAY_LIFT_RETRY_COUNT} "
        f"tolerance={DEFAULT_REPLAY_LIFT_HEIGHT_TOLERANCE_MM:.1f}mm"
    )
    if auto_start:
        return "auto"
    raw_text = input("输入回车开始单步模式，输入 a 全自动执行全部 CSV，输入 q 退出: ").strip().lower()
    if raw_text == "q":
        return "quit"
    if raw_text == "a":
        return "auto"
    return "manual"


def _confirm_each_file(csv_path: Path) -> str:
    print("")
    print(f"准备执行 {csv_path.name}")
    return input("输入回车执行该文件，输入 s 跳过，输入 q 终止: ").strip().lower()


def _confirm_each_action(runtime: ReplayRuntime, row: ReplayRow) -> str:
    print("")
    print(f"下一动作: file={row.csv_name} row={row.row_index} type={row.action_type}")
    print(
        f"当前速度: joint={runtime.joint_speed_deg_s:.2f} deg/s, "
        f"cartesian={runtime.cartesian_speed_mm_s:.2f} mm/s"
    )
    if row.action_type == "arm":
        if row.pose_text.strip().lower() == "nan":
            print(f"目标 joints(deg): {row.joints_text}")
        else:
            print(f"目标 pose(mm/deg): {row.pose_text}")
    elif row.action_type == "gripper":
        print(f"目标 gripper: {row.pose_text}")
    elif row.action_type == "m11":
        print(f"目标 m11 joints: {row.joints_text}")
    elif row.action_type == "lift":
        print(f"目标 lift(mm): {row.pose_text}")
    else:
        print(f"目标值: joints={row.joints_text} pose={row.pose_text}")
    return input("输入回车执行下一动作，输入 a 全自动执行，输入 s 修改速度，输入 j 跳过该动作，输入 q 终止: ").strip().lower()


def main(
    arm_side: str = DEFAULT_ARM_SIDE,
    record_dir: Path = DEFAULT_RECORD_DIR,
    max_files: int | None = DEFAULT_MAX_FILES,
    auto_start: bool = False,
    offset_service_addr: str = DEFAULT_OFFSET_SERVICE_ADDR,
    offset_camera_name: str = DEFAULT_OFFSET_CAMERA_NAME,
    offset_prior_capture_path: Path = DEFAULT_OFFSET_PRIOR_CAPTURE_PATH,
) -> int:
    selected_arm_side = str(arm_side)
    selected_record_dir = Path(record_dir)
    selected_joint_speed_deg_s = DEFAULT_REPLAY_JOINT_SPEED
    selected_cartesian_speed_mm_s = DEFAULT_REPLAY_CARTESIAN_SPEED
    selected_auto_start = bool(auto_start)
    while True:
        resolved_record_dir = _resolve_record_dir(selected_arm_side, selected_record_dir)
        config_choice = _confirm_runtime_config(
            arm_side=selected_arm_side,
            record_dir=resolved_record_dir,
            max_files=max_files,
            offset_service_addr=offset_service_addr,
            offset_camera_name=offset_camera_name,
            offset_prior_capture_path=offset_prior_capture_path,
            joint_speed_deg_s=selected_joint_speed_deg_s,
            cartesian_speed_mm_s=selected_cartesian_speed_mm_s,
        )
        if config_choice == "quit":
            logger.info("用户在配置阶段取消执行")
            return 0
        if config_choice == "auto":
            selected_auto_start = True
            break
        if config_choice == "toggle-arm":
            selected_arm_side = _toggle_arm_side(selected_arm_side)
            selected_record_dir = DEFAULT_RECORD_DIR
            continue
        if config_choice == "speed":
            selected_joint_speed_deg_s, selected_cartesian_speed_mm_s = _configure_speed_values(
                selected_joint_speed_deg_s,
                selected_cartesian_speed_mm_s,
            )
            continue
        break
    logger.info("拖动示教自动回放 CLI 启动 arm_side={} record_dir={}", selected_arm_side, resolved_record_dir)
    csv_paths = _discover_csv_paths(resolved_record_dir, max_files)
    if not csv_paths:
        raise RuntimeError(f"没有在目录中发现 CSV: {record_dir}")
    start_mode = _confirm_start(
        csv_paths,
        selected_auto_start,
        joint_speed_deg_s=selected_joint_speed_deg_s,
        cartesian_speed_mm_s=selected_cartesian_speed_mm_s,
    )
    if start_mode == "quit":
        logger.info("用户取消执行")
        return 0

    runtime: ReplayRuntime | None = None
    try:
        runtime = _create_runtime(selected_arm_side)
        runtime.offset_service_addr = str(offset_service_addr)
        runtime.offset_camera_name = str(offset_camera_name)
        runtime.offset_prior_capture_path = Path(offset_prior_capture_path)
        runtime.joint_speed_deg_s = selected_joint_speed_deg_s
        runtime.cartesian_speed_mm_s = selected_cartesian_speed_mm_s
        runtime.auto_execute_remaining = start_mode == "auto"
        _configure_initial_replay_speeds(runtime, selected_auto_start)
        _prepare_runtime(runtime)
        for csv_path in csv_paths:
            if not selected_auto_start and not runtime.auto_execute_remaining:
                file_choice = _confirm_each_file(csv_path)
                if file_choice == "q":
                    logger.warning("用户终止执行")
                    return 0
                if file_choice == "s":
                    logger.warning("跳过文件 {}", csv_path.name)
                    continue
            rows = _load_replay_rows(csv_path)
            logger.info("开始执行文件 {}，共 {} 行", csv_path.name, len(rows))
            for row in rows:
                if not selected_auto_start and not runtime.auto_execute_remaining:
                    while True:
                        action_choice = _confirm_each_action(runtime, row)
                        if action_choice == "q":
                            logger.warning("用户终止执行")
                            return 0
                        if action_choice == "a":
                            runtime.auto_execute_remaining = True
                            logger.success("已切换为全自动执行模式，后续动作将连续执行到结束")
                            break
                        if action_choice == "s":
                            _configure_replay_speeds(runtime)
                            continue
                        if action_choice == "j":
                            logger.warning("跳过动作 file={} row={} type={}", row.csv_name, row.row_index, row.action_type)
                            break
                        break
                    if action_choice == "j":
                        continue
                _execute_row(runtime, row)
            csv_sequence = _extract_csv_sequence(csv_path.name)
            if csv_sequence == CSV_CARTESIAN_OFFSET_CALCULATE_AT:
                runtime.global_cartesian_offset = _calculate_global_cartesian_offset(runtime, csv_path)
                logger.success("已更新全局笛卡尔纠偏矩阵，后续目标 CSV 将按左乘方式应用")
            logger.success("文件执行完成 {}", csv_path.name)
        logger.success("全部 CSV 执行完成")
        return 0
    finally:
        _cleanup_runtime(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
