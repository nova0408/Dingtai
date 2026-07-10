from __future__ import annotations

import ast
import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.ball_pose_detection.protocol import BallPoseDetectionRequest  # noqa: E402
from camera_pipeline.client import CameraPipelineClient  # noqa: E402
from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from test.wuji.ball_pose_detection import (  # noqa: E402
    DEFAULT_CAMERA_NAME as DEFAULT_BALL_POSE_CAMERA_NAME,
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

DEFAULT_REPLAY_JOINT_SPEED = 600.0
"回放关节空间速度，单位 deg/s。"

DEFAULT_REPLAY_CARTESIAN_SPEED = 30.0
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

DEFAULT_OFFSET_PRIOR_CAPTURE_PATH = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture" / "summary.json"
"计算全局 offset 时使用的先验采集结果路径。"

DEFAULT_HAND_EYE_RESULT_PATH = PROJECT_ROOT / "experiments" / "hand_eye" / "runs" / "20260708_152829" / "hand_eye_result.txt"
"计算全局 offset 时使用的手眼标定结果文件。"

OFFSET_CAPTURE_SETTLE_DELAY_S = 3.0
"到达 offset 触发 CSV 后，等待机械臂和相机画面稳定的时间。"

OFFSET_BALL_CAPTURE_SAMPLE_COUNT = 10
"计算 offset 时连续采集三球坐标的次数。"

OFFSET_TRIGGER_TEMP_JOINT_SPEED_DEG_S = 200.0
"执行 offset 触发 CSV 时临时使用的关节速度，单位 deg/s。"

OFFSET_TRIGGER_TEMP_CARTESIAN_SPEED_MM_S = 20.0
"执行 offset 触发 CSV 时临时使用的笛卡尔速度，单位 mm/s。"

OFFSET_BALL_OUTLIER_MAD_SCALE = 3.5
"三球 9 维坐标鲁棒剔除的 MAD 倍数阈值。"

OFFSET_BALL_OUTLIER_MIN_THRESHOLD_MM = 2.0
"三球坐标鲁棒剔除的最小距离阈值，避免 MAD 过小时误删正常样本。"

# 统一使用米单位：
# T_prior_base_ball = T_tcp @ T_tool_cam @ T_cam_ball
# T_off = T_tcp @ T_tool_cam @ T_cam_ball @ inv(T_prior_base_ball)
# T_new_tcp = T_off @ T_tcp

# region 数据结构
# 实测最终应用公式统一使用：T_new_tcp = T_off @ T_tcp

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
    hand_eye_result_path: Path = DEFAULT_HAND_EYE_RESULT_PATH
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


def _should_apply_global_cartesian_offset(csv_name: str) -> bool:
    return _extract_csv_sequence(csv_name) in CSV_CARTESIAN_OFFSET_TARGETS


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
    if _should_apply_global_cartesian_offset(row.csv_name):
        target_pose = _apply_global_cartesian_offset(runtime, row, target_pose)
    return target_pose


def _frame_to_homogeneous_matrix(frame: xCoreSDK_python.CartesianPosition) -> list[list[float]]:
    """把 SDK 返回的 `CartesianPosition` 转为内部计算矩阵。

    来源与单位：
    - `frame.trans`：SDK 原始单位 `m`
    - `frame.rpy`：SDK 原始单位 `rad`
    - 欧拉解释：与 hand_eye_orin_left_arm_drag.py 的实际手眼求解链路一致，使用 scipy `from_euler("xyz")`
    - 返回矩阵平移：`m`
    - 返回矩阵姿态：无量纲旋转矩阵
    """

    rotation = Rotation.from_euler(
        "xyz",
        np.asarray(frame.rpy, dtype=np.float64).reshape(3),
        degrees=False,
    ).as_matrix()
    matrix = [
        [rotation[0][0], rotation[0][1], rotation[0][2], float(frame.trans[0])],
        [rotation[1][0], rotation[1][1], rotation[1][2], float(frame.trans[1])],
        [rotation[2][0], rotation[2][1], rotation[2][2], float(frame.trans[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matrix


def _frame_to_homogeneous_matrix_m(frame: xCoreSDK_python.CartesianPosition) -> np.ndarray:
    """返回 `m` 单位的 4x4 齐次矩阵。"""

    return np.asarray(_frame_to_homogeneous_matrix(frame), dtype=np.float64)


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


def _homogeneous_matrix_to_rpy(matrix: np.ndarray | list[list[float]]) -> tuple[float, float, float]:
    matrix_np = np.asarray(matrix, dtype=np.float64)
    rpy_rad = Rotation.from_matrix(matrix_np[:3, :3]).as_euler("xyz", degrees=False)
    return float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2])


def _homogeneous_matrix_to_cartesian_position(
    source_pose: xCoreSDK_python.CartesianPosition,
    matrix: np.ndarray | list[list[float]],
) -> xCoreSDK_python.CartesianPosition:
    """把内部 `m` 矩阵回写成 SDK 可接收的 `CartesianPosition`。

    说明：
    - 输入矩阵平移必须是 `m`
    - 输入矩阵姿态必须是标准旋转矩阵
    - 返回给 SDK 的 `trans` 仍是 `m`
    - 返回给 SDK 的 `rpy` 仍是 `rad`，欧拉顺序与手眼求解链路保持 `"xyz"`
    """

    matrix_np = np.asarray(matrix, dtype=np.float64)
    xyz_m = [float(matrix_np[0][3]), float(matrix_np[1][3]), float(matrix_np[2][3])]
    rpy_rad = list(_homogeneous_matrix_to_rpy(matrix_np))
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
    """将全局 offset 左乘到目标 TCP 上。

    链路：
    - 目标 TCP 由 CSV 目标构造，内部以 `m/rad` 进入 `CartesianPosition`
    - `runtime.global_cartesian_offset` 以 `m` 保存
    - 应用公式固定为 `T_new_tcp = T_off @ T_tcp`
    - 最终返回给 SDK 的 `CartesianPosition` 仍保持 `m/rad`
    """

    if runtime.global_cartesian_offset is None:
        raise RuntimeError(
            f"CSV {row.csv_name} 需要使用全局笛卡尔纠偏，但当前尚未在 "
            f"{_format_optional_csv_sequence(CSV_CARTESIAN_OFFSET_CALCULATE_AT)}_*.csv 末尾计算 offset"
    )
    original_matrix = _frame_to_homogeneous_matrix(target_pose)
    offset_matrix_m = np.asarray(runtime.global_cartesian_offset, dtype=np.float64)
    corrected_matrix = offset_matrix_m @ np.asarray(original_matrix, dtype=np.float64)
    corrected_pose = _homogeneous_matrix_to_cartesian_position(target_pose, corrected_matrix)
    logger.info(
        "已对笛卡尔目标应用全局左乘纠偏矩阵 T_new_tcp=T_off@T_tcp file={} row={} base=tool:{} wobj:{} {}",
        row.csv_name,
        row.row_index,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
        _format_matrix_xyzrpy_mm_deg("T_new_tcp", corrected_matrix),
    )
    return corrected_pose


def _load_prior_base_ball_transform(prior_capture_path: Path, hand_eye_result_path: Path) -> np.ndarray:
    """从先验捕获文件重建 `T_prior_base_ball`，单位统一为 `m`。

    文件来源：
    - `tcp_pose_matrix`：记录时的 `T_tcp`，由 SDK `cartPosture(endInRef)` 的 `trans(m)+rpy(rad)` 重建，已经是 `m`
    - `local_pose_transform`：记录时的 `T_cam_ball`，文件落盘为 `mm`
    - `hand_eye_result.txt`：记录时使用的 `T_tool_cam`，原始单位 `m`
    """

    prior_capture = _load_prior_capture(prior_capture_path)
    tcp_pose_matrix = prior_capture.get("tcp_pose_matrix")
    local_pose_transform = prior_capture.get("local_pose_transform")
    if tcp_pose_matrix is None:
        raise RuntimeError(
            f"先验文件缺少 tcp_pose_matrix: {prior_capture_path}。"
            "请先重新运行 ball_pose_detection.py 生成包含 tcp_pose_matrix 的 summary.json"
        )
    if local_pose_transform is None:
        raise RuntimeError(f"先验文件缺少 local_pose_transform: {prior_capture_path}")
    tcp_matrix_m = np.asarray(tcp_pose_matrix, dtype=np.float64)
    ball_matrix_m = np.asarray(local_pose_transform, dtype=np.float64)
    if tcp_matrix_m.shape != (4, 4) or not np.all(np.isfinite(tcp_matrix_m)):
        raise RuntimeError(f"先验 tcp_pose_matrix 格式无效: {prior_capture_path}")
    if ball_matrix_m.shape != (4, 4) or not np.all(np.isfinite(ball_matrix_m)):
        raise RuntimeError(f"先验 local_pose_transform 格式无效: {prior_capture_path}")
    ball_matrix_m = ball_matrix_m.copy()
    tcp_matrix_m = tcp_matrix_m.copy()
    # tcp_pose_matrix 已经是内部计算单位 m，不能再次做 mm -> m 缩放。
    ball_matrix_m[:3, 3] *= 0.001
    tool_camera_matrix_m = _load_tool_camera_transform_m(hand_eye_result_path)
    prior_base_ball_transform = tcp_matrix_m @ tool_camera_matrix_m @ ball_matrix_m
    return prior_base_ball_transform


def _load_tool_camera_transform_m(hand_eye_result_path: Path) -> np.ndarray:
    """从 hand-eye 结果文件加载 `T_tool_cam`，单位保持为 `m`。"""

    if not hand_eye_result_path.is_file():
        raise FileNotFoundError(f"手眼结果文件不存在: {hand_eye_result_path}")
    lines = hand_eye_result_path.read_text(encoding="utf-8").splitlines()
    matrix_rows: list[list[float]] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped == "T_tool_cam:":
            collecting = True
            continue
        if collecting:
            if stripped == "":
                break
            cleaned = stripped.replace("[", " ").replace("]", " ")
            values = [float(token) for token in cleaned.split() if token]
            if len(values) != 4:
                raise ValueError(f"手眼矩阵行格式错误: {line}")
            matrix_rows.append(values)
            if len(matrix_rows) == 4:
                break
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"手眼矩阵维度错误: shape={matrix.shape}, path={hand_eye_result_path}")
    return matrix


def _detect_current_three_ball_basis_transform(
    service_addr: str,
    camera_name: str,
    prior_capture_path: Path,
) -> np.ndarray:
    """连续采样三球坐标，剔除异常值后返回均值 `T_cam_ball`。

    检测服务返回的 `center_mm` 单位是 `mm`；本函数最终返回矩阵平移统一为 `m`。
    """

    prior_capture = _load_prior_capture(prior_capture_path)
    priors = _build_priors_from_capture(prior_capture)
    samples_mm: list[np.ndarray] = []
    frame_ids: list[int] = []
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=30_000)
    try:
        for sample_index in range(OFFSET_BALL_CAPTURE_SAMPLE_COUNT):
            response = client.request_ball_pose_detection(
                BallPoseDetectionRequest(
                    request_id=sample_index + 1,
                    camera_name=str(camera_name),
                    frame_id=-1,
                    enable_debug=True,
                    priors=tuple(priors),
                )
            )
            logger.info(
                "ball pose detection sample response {}/{} frame_id={} matched_count={} valid_so_far={} error={}",
                sample_index + 1,
                OFFSET_BALL_CAPTURE_SAMPLE_COUNT,
                response.frame_id,
                response.matched_count,
                len(samples_mm),
                response.error,
            )
            if response.error is not None:
                logger.warning("ball pose detection sample failed index={} error={}", sample_index + 1, response.error)
                continue
            if response.matched_count < 3:
                logger.warning(
                    "ball pose detection sample insufficient index={} matched_count={}",
                    sample_index + 1,
                    response.matched_count,
                )
                continue
            sample_mm = _extract_ordered_three_ball_centers_mm(response.detections)
            if sample_mm is None:
                logger.warning("ball pose detection sample basis invalid index={}", sample_index + 1)
                continue
            samples_mm.append(sample_mm)
            frame_ids.append(int(response.frame_id))
            logger.info(
                "ball pose detection sample accepted {}/{} valid={} {}",
                sample_index + 1,
                OFFSET_BALL_CAPTURE_SAMPLE_COUNT,
                len(samples_mm),
                _format_three_ball_centers_mm(sample_mm),
            )
    finally:
        client.close()
    if not samples_mm:
        raise RuntimeError("ball pose detection 连续采样未得到可用三球检测结果")
    mean_detections, kept_count, rejected_count, mean_distance_mm, max_distance_mm = _build_mean_three_ball_detections(
        samples_mm
    )
    basis_transform = _build_three_ball_basis_transform(mean_detections)
    if basis_transform is None:
        raise RuntimeError("均值三球基础坐标系构造失败")
    # ball_pose_detection 的检测结果以 mm 输出，这里统一转换为 m 再参与链路。
    basis_transform = basis_transform.copy()
    basis_transform[:3, 3] *= 0.001
    logger.info(
        "ball pose detection 均值采样完成 camera={} requested={} valid={} kept={} rejected={} "
        "frame_first={} frame_last={} mean_dist_mm={:.3f} max_dist_mm={:.3f}",
        camera_name,
        OFFSET_BALL_CAPTURE_SAMPLE_COUNT,
        len(samples_mm),
        kept_count,
        rejected_count,
        frame_ids[0] if frame_ids else "NA",
        frame_ids[-1] if frame_ids else "NA",
        mean_distance_mm,
        max_distance_mm,
    )
    return basis_transform


def _extract_ordered_three_ball_centers_mm(detections: object) -> np.ndarray | None:
    if not isinstance(detections, list | tuple):
        return None
    by_color: dict[str, np.ndarray] = {}
    for item in detections:
        if not isinstance(item, dict):
            continue
        color_hex = str(item.get("color_hex"))
        center = np.asarray(item.get("center_mm"), dtype=np.float64)
        if center.shape == (3,) and np.all(np.isfinite(center)):
            by_color[color_hex] = center
    ordered_centers = [by_color.get(color_hex) for color_hex in ("#ffff00", "#ff0000", "#ff00ff")]
    if any(center is None for center in ordered_centers):
        return None
    return np.stack([np.asarray(center, dtype=np.float64) for center in ordered_centers], axis=0)


def _format_three_ball_centers_mm(centers_mm: np.ndarray) -> str:
    centers = np.asarray(centers_mm, dtype=np.float64).reshape(3, 3)
    labels = ("yellow", "red", "purple")
    values = []
    for label, center in zip(labels, centers, strict=True):
        values.append(f"{label}_xyz_mm=[{_format_sequence(center.tolist())}]")
    return " ".join(values)


def _build_mean_three_ball_detections(
    samples_mm: list[np.ndarray],
) -> tuple[list[dict[str, object]], int, int, float, float]:
    sample_stack = np.stack(samples_mm, axis=0)
    flattened = sample_stack.reshape(sample_stack.shape[0], 9)
    median = np.median(flattened, axis=0)
    distances = np.linalg.norm(flattened - median.reshape(1, 9), axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = max(OFFSET_BALL_OUTLIER_MIN_THRESHOLD_MM, median_distance + OFFSET_BALL_OUTLIER_MAD_SCALE * mad)
    keep_mask = distances <= threshold
    if not np.any(keep_mask):
        keep_mask[int(np.argmin(distances))] = True
    kept_samples = sample_stack[keep_mask]
    mean_centers = np.mean(kept_samples, axis=0)
    kept_distances = distances[keep_mask]
    detections = [
        {"color_hex": "#ffff00", "center_mm": mean_centers[0].tolist()},
        {"color_hex": "#ff0000", "center_mm": mean_centers[1].tolist()},
        {"color_hex": "#ff00ff", "center_mm": mean_centers[2].tolist()},
    ]
    kept_count = int(np.count_nonzero(keep_mask))
    return (
        detections,
        kept_count,
        int(sample_stack.shape[0] - kept_count),
        float(np.mean(kept_distances)),
        float(np.max(kept_distances)),
    )


def _calculate_global_cartesian_offset(
    runtime: ReplayRuntime,
    csv_path: Path,
) -> tuple[tuple[float, float, float, float], ...]:
    """严格按下式计算全局偏移矩阵，所有矩阵统一为 `m`：

    - `T_prior_base_ball = T_prior_tcp @ T_tool_cam @ T_prior_cam_ball`
    - `T_off = T_curr_tcp @ T_tool_cam @ T_curr_cam_ball @ inv(T_prior_base_ball)`
    - `T_new_tcp = T_off @ T_curr_tcp`
    """

    prior_base_ball_transform = _load_prior_base_ball_transform(
        runtime.offset_prior_capture_path,
        runtime.hand_eye_result_path,
    )
    current_cam_ball_transform = _detect_current_three_ball_basis_transform(
        service_addr=runtime.offset_service_addr,
        camera_name=runtime.offset_camera_name,
        prior_capture_path=runtime.offset_prior_capture_path,
    )
    if _apply_named_toolset(runtime.connected_arm.robot, runtime.connected_arm.ec) is None:
        raise RuntimeError(f"设置固定 toolset 失败: tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    tcp_pose = runtime.connected_arm.robot.cartPosture(xCoreSDK_python.endInRef, runtime.connected_arm.ec)
    _print_sdk_result("cartPosture(endInRef, offset-calc)", runtime.connected_arm.ec)
    if runtime.connected_arm.ec.get("ec", 0) != 0:
        raise RuntimeError("读取当前 TCP 位姿失败，无法计算全局 offset")
    # SDK 原始输出：trans(m), rpy(rad)，这里重建成内部计算矩阵(m)。
    tcp_matrix_m = _frame_to_homogeneous_matrix_m(tcp_pose)
    tool_camera_matrix_m = _load_tool_camera_transform_m(runtime.hand_eye_result_path)
    current_base_ball_m = tcp_matrix_m @ tool_camera_matrix_m @ current_cam_ball_transform
    offset_matrix_m = current_base_ball_m @ np.linalg.inv(prior_base_ball_transform)
    new_tcp_matrix_m = offset_matrix_m @ tcp_matrix_m
    logger.success(
        "全局 offset 已计算 file={} trigger=csv_end_state base=tool:{} wobj:{}",
        csv_path.name,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    logger.info(
        "offset 来源=T_tcp@T_tool_cam@T_cam_ball@inv(T_prior_base_ball) {}",
        _format_matrix_xyzrpy_mm_deg("T_off", offset_matrix_m),
    )
    logger.info(
        "{}",
        _format_matrix_xyzrpy_mm_deg("T_prior_base_ball", prior_base_ball_transform),
    )
    logger.info(
        "T_current_base_ball=T_tcp@T_tool_cam@T_cam_ball {}",
        _format_matrix_xyzrpy_mm_deg("T_current_base_ball", current_base_ball_m),
    )
    logger.info(
        "{}",
        _format_matrix_xyzrpy_mm_deg("T_new_tcp=T_off@T_tcp", new_tcp_matrix_m),
    )
    offset_distance_m = float(np.linalg.norm(offset_matrix_m[:3, 3]))
    logger.info("offset_norm_mm={:.3f} {}", offset_distance_m * 1000.0, _format_matrix_xyzrpy_mm_deg("T_off", offset_matrix_m))
    if offset_distance_m > 0.05:
        logger.warning(
            "offset 平移明显偏大，请检查拍摄时机/三球识别/先验是否一致 distance_mm={:.3f} {}",
            offset_distance_m * 1000.0,
            _format_matrix_xyzrpy_mm_deg("T_off", offset_matrix_m),
        )
    return (
        (
            float(offset_matrix_m[0, 0]),
            float(offset_matrix_m[0, 1]),
            float(offset_matrix_m[0, 2]),
            float(offset_matrix_m[0, 3]),
        ),
        (
            float(offset_matrix_m[1, 0]),
            float(offset_matrix_m[1, 1]),
            float(offset_matrix_m[1, 2]),
            float(offset_matrix_m[1, 3]),
        ),
        (
            float(offset_matrix_m[2, 0]),
            float(offset_matrix_m[2, 1]),
            float(offset_matrix_m[2, 2]),
            float(offset_matrix_m[2, 3]),
        ),
        (
            float(offset_matrix_m[3, 0]),
            float(offset_matrix_m[3, 1]),
            float(offset_matrix_m[3, 2]),
            float(offset_matrix_m[3, 3]),
        ),
    )


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    return _rad_to_deg(list(_homogeneous_matrix_to_rpy(matrix.tolist())))


def _format_matrix_xyzrpy_mm_deg(name: str, matrix: np.ndarray) -> str:
    matrix_np = np.asarray(matrix, dtype=np.float64)
    xyz_mm = [
        float(matrix_np[0, 3]) * 1000.0,
        float(matrix_np[1, 3]) * 1000.0,
        float(matrix_np[2, 3]) * 1000.0,
    ]
    rpy_deg = _rad_to_deg(list(_homogeneous_matrix_to_rpy(matrix_np)))
    return f"{name} xyzrpy(mm,deg)=[{_format_sequence(xyz_mm + rpy_deg)}]"


def _execute_cartesian_row(
    runtime: ReplayRuntime,
    row: ReplayRow,
) -> None:
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    applies_offset = _should_apply_global_cartesian_offset(row.csv_name)
    # 只有 CSV_CARTESIAN_OFFSET_TARGETS 指定的 CSV 才执行：
    # T_new_tcp = T_off @ T_tcp，其余 CSV 保持原始 TCP。
    target_pose = _build_cartesian_target(runtime, row)
    cmd_id = xCoreSDK_python.PyString()
    target_joint: xCoreSDK_python.JointPosition | None = None
    should_fallback_to_move_abs_j = not _validate_cartesian_target(robot, ec, target_pose)
    if should_fallback_to_move_abs_j:
        logger.warning(
            "回放 MoveL 路径检查失败，将对 {} 计算逆解并回退 MoveAbsJ "
            "file={} row={} xyz(mm)=[{}] rpy(deg)=[{}]",
            "offset 后 T_new_tcp" if applies_offset else "原始 T_tcp",
            row.csv_name,
            row.row_index,
            _format_sequence(_m_to_mm(target_pose.trans)),
            _format_sequence(_rad_to_deg(target_pose.rpy)),
        )
        robot_model = robot.model()
        toolset = robot.toolset(ec)
        _print_sdk_result("toolset(replay-cartesian-fallback)", ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError("回放笛卡尔读取 toolset 失败")
        target_joint = robot_model.calcIk(target_pose, toolset, ec)
        _print_sdk_result("calcIk(replay-cartesian-fallback-new-tcp)", ec)
        if ec.get("ec", 0) != 0:
            failed_ec = ec.get("ec", 0)
            failed_message = ec.get("message", "")
            if not applies_offset:
                try:
                    fallback_joint_deg = _parse_joint_values(row.joints_text)
                except ValueError as exc:
                    logger.warning(
                        "原始笛卡尔目标逆解失败，且 CSV 记录关节值不可用 file={} row={} ec={} message={} parse_error={}",
                        row.csv_name,
                        row.row_index,
                        failed_ec,
                        failed_message,
                        exc,
                    )
                    raise RuntimeError("原始笛卡尔目标 MoveL/IK 均失败，且 CSV joints 兜底不可用") from exc
                target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(fallback_joint_deg))
                ec["ec"] = 0
                ec["message"] = "原始笛卡尔目标逆解失败，已改用 CSV 记录关节值"
                logger.warning(
                    "原始笛卡尔目标逆解失败，最终兜底为 CSV joints MoveAbsJ file={} row={} "
                    "ik_ec={} ik_message={} joints(deg)=[{}]",
                    row.csv_name,
                    row.row_index,
                    failed_ec,
                    failed_message,
                    _format_sequence(fallback_joint_deg),
                )
            else:
                logger.warning(
                    "offset 后 T_new_tcp 逆解失败，无法回退 MoveAbsJ file={} row={} ec={} message={}",
                    row.csv_name,
                    row.row_index,
                    ec.get("ec", 0),
                    ec.get("message", ""),
                )
                raise RuntimeError("offset 后 T_new_tcp 逆解失败，无法回退 MoveAbsJ")
    robot.moveReset(ec)
    _print_sdk_result("moveReset(replay-cartesian)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放笛卡尔 moveReset 失败")
    if should_fallback_to_move_abs_j:
        if target_joint is None:
            if applies_offset:
                raise RuntimeError("MoveAbsJ 回退缺少 offset 后 T_new_tcp 的逆解结果")
            raise RuntimeError("MoveAbsJ 回退缺少原始 T_tcp 的逆解或 CSV joints 兜底结果")
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
    hand_eye_result_path: Path,
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
        print(f"当前手眼结果: {hand_eye_result_path}")
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
    hand_eye_result_path: Path = DEFAULT_HAND_EYE_RESULT_PATH,
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
            hand_eye_result_path=hand_eye_result_path,
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
        runtime.hand_eye_result_path = Path(hand_eye_result_path)
        runtime.joint_speed_deg_s = selected_joint_speed_deg_s
        runtime.cartesian_speed_mm_s = selected_cartesian_speed_mm_s
        runtime.auto_execute_remaining = start_mode == "auto"
        _configure_initial_replay_speeds(runtime, selected_auto_start)
        _prepare_runtime(runtime)
        for csv_path in csv_paths:
            csv_sequence = _extract_csv_sequence(csv_path.name)
            is_offset_trigger_csv = csv_sequence == CSV_CARTESIAN_OFFSET_CALCULATE_AT
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
            original_joint_speed_deg_s = runtime.joint_speed_deg_s
            original_cartesian_speed_mm_s = runtime.cartesian_speed_mm_s
            if is_offset_trigger_csv:
                runtime.joint_speed_deg_s = OFFSET_TRIGGER_TEMP_JOINT_SPEED_DEG_S
                runtime.cartesian_speed_mm_s = OFFSET_TRIGGER_TEMP_CARTESIAN_SPEED_MM_S
                logger.info(
                    "offset 触发 CSV 临时速度调整 file={} joint_speed {:.2f}->{:.2f} deg/s "
                    "cartesian_speed {:.2f}->{:.2f} mm/s",
                    csv_path.name,
                    original_joint_speed_deg_s,
                    OFFSET_TRIGGER_TEMP_JOINT_SPEED_DEG_S,
                    original_cartesian_speed_mm_s,
                    OFFSET_TRIGGER_TEMP_CARTESIAN_SPEED_MM_S,
                )
            try:
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
                                logger.warning(
                                    "跳过动作 file={} row={} type={}",
                                    row.csv_name,
                                    row.row_index,
                                    row.action_type,
                                )
                                break
                            break
                        if action_choice == "j":
                            continue
                    _execute_row(runtime, row)
                if is_offset_trigger_csv:
                    logger.info(
                        "已到达 offset 触发 CSV，等待 {:.1f}s 后开始连续采集三球坐标 file={}",
                        OFFSET_CAPTURE_SETTLE_DELAY_S,
                        csv_path.name,
                    )
                    time.sleep(OFFSET_CAPTURE_SETTLE_DELAY_S)
                    runtime.global_cartesian_offset = _calculate_global_cartesian_offset(runtime, csv_path)
                    logger.success("已更新全局笛卡尔纠偏矩阵，后续目标 CSV 将按 T_off@T_tcp 左乘方式应用")
            finally:
                if is_offset_trigger_csv:
                    runtime.joint_speed_deg_s = original_joint_speed_deg_s
                    runtime.cartesian_speed_mm_s = original_cartesian_speed_mm_s
                    logger.info(
                        "offset 触发 CSV 结束，恢复回放速度 file={} joint_speed={:.2f} deg/s "
                        "cartesian_speed={:.2f} mm/s",
                        csv_path.name,
                        runtime.joint_speed_deg_s,
                        runtime.cartesian_speed_mm_s,
                    )
            logger.success("文件执行完成 {}", csv_path.name)
        logger.success("全部 CSV 执行完成")
        return 0
    finally:
        _cleanup_runtime(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
