from __future__ import annotations

import ast
import csv
import gc
import json
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq
from loguru import logger
from qmlinker import create_channel
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.wuji.ball_pose_detection import (
    DEFAULT_CAMERA_NAME as DEFAULT_BALL_POSE_CAMERA_NAME,
)
from test.wuji.ball_pose_detection import (
    DEFAULT_SERVICE_ADDR as DEFAULT_BALL_POSE_SERVICE_ADDR,
)
from test.wuji.common import (
    AGV_HOST,
    DEFAULT_PORT,
    GRIPPER_PORT,
    SSH_ALIAS,
    WUYOU_QMLINKER_HOST,
    close_wuyou_channel,
    stop_ssh_process,
)
from test.wuji.prior_record import (
    DEFAULT_DICTIONARY_NAME,
    DEFAULT_HEAD_PITCH_DEG,
    DEFAULT_HEAD_SETTLE_S,
    DEFAULT_HEAD_YAW_DEG,
    DEFAULT_MARKER_LENGTH_MM,
    DEFAULT_MIN_CHARUCO_CORNERS,
    DEFAULT_SQUARE_LENGTH_MM,
    DEFAULT_SQUARES_X,
    DEFAULT_SQUARES_Y,
    _load_prior_capture,
)
from test.wuji.xcoresdk_arm_cli_test import (
    ARM_SSH_FORWARD_PORTS,
    DEFAULT_JOINT_ZONE,
    DEFAULT_TOOL_NAME,
    DEFAULT_WOBJ_NAME,
    LEFT_ARM_CONTROLLER_IP,
    LEFT_ARM_IP,
    RIGHT_ARM_CONTROLLER_IP,
    RIGHT_ARM_IP,
    ConnectedArm,
    DahuanGripperClient,
    WujiBodyClient,
    WujiRightHandClient,
    _apply_named_toolset,
    _copy_cartesian_pose_context,
    _deg_to_rad,
    _detect_arm_side,
    _ensure_nrt_motion_ready,
    _format_sequence,
    _mm_to_m,
    _parse_cartesian_pose_input,
    _print_sdk_result,
    _rad_to_deg,
    _shutdown_robot,
)

from camera_pipeline.client import CameraName, CameraPipelineClient
from camera_pipeline.service.protocol import CharucoDetectionRequest
from record_replay.offset_detection import camera_ball_transform_m
from record_replay.offset_detector_gateway import (
    CameraPipelineThreeBallDetector,
    load_three_ball_priors,
)
from record_replay.settings import ReplayOffsetSettings
from sdk.xcoresdk import xCoreSDK_python
from src.wuji.agv_client import WujiAgvClient
from src.wuji.head_client import WujiHeadClient

# region CSV 序号配置

LEFT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE: dict[int, float] = {
    -1: 1000.0,
    4: 200.0,
}
"左臂各 CSV 的 MoveAbsJ 末端线速度，单位 mm/s；-1 为左臂默认值。"

RIGHT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE: dict[int, float] = {
    -1: 1000.0,
}
"右臂各 CSV 的 MoveAbsJ 末端线速度，单位 mm/s；-1 为右臂默认值。"

LEFT_REPLAY_MOVE_ABS_J_ZONE_MM_BY_CSV_SEQUENCE: dict[int, float] = {
    -1: DEFAULT_JOINT_ZONE,
    2:40.0,
    4:0,
    15:40.0,
}
"左臂各 CSV 连续 MoveAbsJ 中间点的转弯区半径，单位 mm；-1 为左臂默认值。"

RIGHT_REPLAY_MOVE_ABS_J_ZONE_MM_BY_CSV_SEQUENCE: dict[int, float] = {
    -1: DEFAULT_JOINT_ZONE,
}
"右臂各 CSV 连续 MoveAbsJ 中间点的转弯区半径，单位 mm；-1 为右臂默认值。"

CSV_CARTESIAN_OFFSET_TARGETS: list[int] = [5, 12]
"需要应用三球全局笛卡尔纠偏的 CSV 序号列表。"

CSV_CARTESIAN_OFFSET_CALCULATE_AT: int = 4
"在该 CSV 的最后一个 arm pose 处计算一次全局笛卡尔纠偏。"

CHARUCO_OFFSET_LEFT_CSV_SEQUENCE: list[int] = [2, 15]
"需要应用头部 ChArUco offset 的左臂 CSV 序号列表；空列表表示不启用。"

CHARUCO_OFFSET_RIGHT_CSV_SEQUENCE: list[int] = [2, 3]
"需要应用头部 ChArUco offset 的右臂 CSV 序号列表；空列表表示不启用。"

# endregion

# region 默认配置

DEFAULT_LEFT_RECORD_DIR = PROJECT_ROOT / "record_left"
"默认左臂拖动示教 CSV 目录。"

DEFAULT_RIGHT_RECORD_DIR = PROJECT_ROOT / "record_right"
"默认右臂拖动示教 CSV 目录。"

DEFAULT_RECORD_DIR = DEFAULT_LEFT_RECORD_DIR
"默认拖动示教 CSV 目录。"

DEFAULT_ARM_SIDE = "left"
"默认回放机械臂侧别。"

DEFAULT_EXECUTION_MODE = "single"
"默认仅执行单臂回放；与 DEFAULT_ARM_SIDE 组合后默认只调试左臂。"

DEFAULT_AGV_POINT = "1"
"全自动回放开始前导航到的 AGV 地图站点名称。"

DEFAULT_ENABLE_AGV_NAVIGATION = False
"是否在自动回放开始前执行 AGV 导航；人工测试默认关闭。"

DEFAULT_AGV_NAVIGATION_TIMEOUT_S = 600.0
"等待 AGV 导航结束的超时时间，单位 s。"

DEFAULT_AGV_NAVIGATION_POLL_INTERVAL_S = 0.2
"AGV 导航状态轮询间隔，单位 s。"

SHARED_TUNNEL_BODY_PORT = DEFAULT_PORT - 1
"共享 SSH 隧道中 M11、body 与 head 的本地端口。"

SHARED_TUNNEL_GRIPPER_PORT = GRIPPER_PORT - 1
"共享 SSH 隧道中 gripper 的本地端口。"

SHARED_TUNNEL_AGV_PORT = DEFAULT_PORT + 1
"共享 SSH 隧道中 AGV 的本地端口。"

SHARED_TUNNEL_READY_TIMEOUT_S = 5.0
"等待共享 SSH 隧道全部本地端口就绪的超时时间，单位 s。"

GRIPPER_CALIBRATION_WAIT_S = 3.0
"夹爪校准命令发出后的固定等待时间，单位 s。"

GRIPPER_ZERO_POSITION = 0
"未校准夹爪完成校准后必须到达的初始位置。"

GRIPPER_ZERO_POLL_INTERVAL_S = 0.2
"等待夹爪运动到初始位置的状态轮询间隔，单位 s。"

DEFAULT_MAX_FILES: int | None = None
"默认加载的 CSV 文件数量；`None` 表示全部。"

MOVE_ABS_J_MIN_END_LINEAR_SPEED_MM_S = 5.0
"协作机器人 MoveAbsJ 末端线速度的最小有效值，单位 mm/s。"

MOVE_ABS_J_MAX_END_LINEAR_SPEED_MM_S = 4000.0
"协作机器人 MoveAbsJ 末端线速度的最大有效值，单位 mm/s。"

DEFAULT_RESET_READY_TIMEOUT_S = 2.0
"等待机械臂 reset 后进入 idle 状态的默认超时时间，单位 s。"

DEFAULT_RESET_READY_STABLE_IDLE_CHECKS = 2
"判定机械臂 reset 就绪所需的连续 idle 状态次数。"

RESET_READY_POLL_INTERVAL_S = 0.2
"等待机械臂 reset 就绪时的状态轮询间隔，单位 s。"

MOTION_STATE_POLL_INTERVAL_S = 0.1
"等待机械臂运动结束时的状态轮询间隔，单位 s。"

DEFAULT_REPLAY_LIFT_TIMEOUT_S = 15.0
"回放 lift 从首次下发到实际到位的总超时时间，单位 s。"

DEFAULT_REPLAY_LIFT_PULSE_INTERVAL_S = 0.5
"回放 lift 每次脉冲下发后读取高度并决定是否再次下发的间隔，单位 s。"

DEFAULT_REPLAY_LIFT_HEIGHT_TOLERANCE_MM = 4.0
"回放 lift 到位误差容忍，单位 mm。"

LEFT_HEAD_BASE_CAMERA_PATH: Path  = (
    PROJECT_ROOT / "experiments" / "hand_eye" / "runs" / "L_EtH_20260717_141031" / "L_EtH_T_base_camera.npy"
)
"左臂基坐标系下的 T_base_camera.npy；启用左臂 ChArUco offset 时必须配置 Path。"

RIGHT_HEAD_BASE_CAMERA_PATH: Path  = (
    PROJECT_ROOT / "experiments" / "hand_eye" / "runs" / "R_EtH_20260717_142024" / "R_EtH_T_base_camera.npy"
)
"右臂基坐标系下的 T_base_camera.npy；启用右臂 ChArUco offset 时必须配置 Path。"

DEFAULT_OFFSET_SERVICE_ADDR = DEFAULT_BALL_POSE_SERVICE_ADDR
"计算全局 offset 时使用的球位姿检测服务地址。"

DEFAULT_OFFSET_CAMERA_NAME = DEFAULT_BALL_POSE_CAMERA_NAME
DEFAULT_HEAD_CAMERA_NAME = "head_camera"
"计算全局 offset 时使用的相机名称。"

DEFAULT_PRIOR_RECORD_DIR = PROJECT_ROOT / "record_replay" / "prior_data"
"GUI 保存的三球与 ChArUco 先验记录所在目录。"

DEFAULT_OFFSET_PRIOR_CAPTURE_PATH = DEFAULT_PRIOR_RECORD_DIR / "ball_pose_prior.json"
"计算全局 offset 时使用的先验采集结果路径。"

DEFAULT_CHARUCO_PRIOR_PATH = DEFAULT_PRIOR_RECORD_DIR / "charuco_board_prior.json"
"头部 ChArUco 的 T_camera_board 先验文件。"

DEFAULT_CHARUCO_OFFSET_HISTORY_PATH = DEFAULT_PRIOR_RECORD_DIR / "charuco_offset_history.csv"
"人工确认的 ChArUco offset 历史样本；运行时只读，不自动追加。"

CHARUCO_OFFSET_HISTORY_MIN_ACCEPTED_SAMPLES = 6
"允许启用统计范围所需的同侧机械臂最少有效历史样本数。"

CHARUCO_OFFSET_SIGMA_LIMIT = 4.0
"ChArUco offset 各 xyz/rpy 分量允许偏离历史均值的标准差倍数。"

CHARUCO_OFFSET_MAX_TRANSLATION_NORM_MM = 60.0
"ChArUco offset 平移模长的绝对安全上限，单位 mm。"

CHARUCO_OFFSET_MAX_ROTATION_NORM_DEG = 5.0
"ChArUco offset 旋转向量模长的绝对安全上限，单位 deg。"

CHARUCO_OFFSET_HISTORY_FIELDS: tuple[str, ...] = (
    "source_file",
    "captured_at",
    "arm_side",
    "x_mm",
    "y_mm",
    "z_mm",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "translation_norm_mm",
    "rotation_norm_deg",
    "accepted",
    "decision_reason",
)
"ChArUco offset 历史 CSV 的固定字段顺序。"

DEFAULT_HAND_EYE_RESULT_PATH = (
    PROJECT_ROOT / "experiments" / "hand_eye" / "runs" / "20260708_152829" / "hand_eye_result.txt"
)
"计算全局 offset 时使用的手眼标定结果文件。"

OFFSET_RECORD_NAME_PREFIX = "offset_compare"
"offset 记录文件名前缀；每轮新建带 MMDD_HHMMSS 后缀的文件，不覆盖历史记录。"

OFFSET_CAPTURE_SETTLE_DELAY_S = 0.0
"到达 offset 触发 CSV 后，等待机械臂和相机画面稳定的时间。"

OFFSET_BALL_CAPTURE_SAMPLE_COUNT = 2
"计算 offset 时连续采集三球坐标的次数。"

OFFSET_BALL_DETECTION_TIMEOUT_MS = 30_000
"计算 offset 时单次球位姿检测 RPC 的超时时间，单位 ms。"

CHARUCO_DETECTION_SERVICE_ADDR = DEFAULT_BALL_POSE_SERVICE_ADDR
"头部 ChArUco 检测使用的 camera_pipeline 服务地址。"

CHARUCO_DETECTION_CAMERA_TIMEOUT_S = 10.0
"头部相机每次等待稳定帧的超时时间，单位 s。"

CHARUCO_DETECTION_MAX_FRAME_COUNT = 5
"单次 ChArUco offset 计算最多检查的头部相机帧数。"

CHARUCO_DETECTION_MIN_CORNERS = DEFAULT_MIN_CHARUCO_CORNERS
"目标板位姿检测要求的最少 ChArUco 角点数量。"

CHARUCO_DETECTION_RPC_TIMEOUT_S = (
    CHARUCO_DETECTION_CAMERA_TIMEOUT_S * CHARUCO_DETECTION_MAX_FRAME_COUNT + 5.0
)
"ChArUco RPC 接收超时时间，覆盖服务端完整稳定帧检测窗口，单位 s。"

CHARUCO_DETECTION_TIMEOUT_RETRY_COUNT = 3
"ChArUco RPC 发生 ZMQ 超时时的最大请求次数。"

CHARUCO_DETECTION_TIMEOUT_RETRY_DELAY_S = 1.0
"ChArUco RPC 超时后的重试间隔，单位 s。"

CHARUCO_OFFSET_SAFETY_ATTEMPT_COUNT = 3
"ChArUco offset 未通过历史安全检查时重新检测并计算的总尝试次数。"

CHARUCO_OFFSET_SAFETY_RETRY_DELAY_S = 1.0
"ChArUco offset 历史安全检查拒绝后再次检测前的等待时间，单位 s。"

OFFSET_TRANSLATION_WARNING_THRESHOLD_MM = 50.0
"全局 offset 平移超过该值时输出风险警告，单位 mm。"

NON_MOTION_RETRY_COUNT = 3
"非直接运动指令的重试次数。"

NON_MOTION_RETRY_DELAY_S = 0.5
"非直接运动指令的重试间隔，单位 s。"

M11_STATE_READ_TIMEOUT_S = 10.0
"读取至少 11 个有效右手执行器状态的总超时时间，单位 s。"

M11_STATE_READ_POLL_INTERVAL_S = 0.1
"右手状态内容无效时的重新读取间隔，单位 s。"

RIGHT_HAND_M11_ROOT_ACTUATOR_IDS: tuple[int, ...] = (3, 5, 7, 9)
"右手 M11 四指根部执行器索引。"

RIGHT_HAND_M11_TIP_ACTUATOR_IDS: tuple[int, ...] = (4, 6, 8, 10)
"右手 M11 四指指尖执行器索引。"

LIFT_ENABLE_STATE_TIMEOUT_S = 10.0
"下发 lift enable 后等待 get_enable() 状态为 True 的总超时时间，单位 s。"

LIFT_ENABLE_RETRY_INTERVAL_S = 0.2
"lift enable 状态尚未生效时的重新下发间隔，单位 s。"

# 左手眼在手上，内部统一使用 m：
# T_prior_base_ball = T_tcp @ T_tool_cam @ T_cam_ball
# T_ball_off = T_tcp @ T_tool_cam @ T_cam_ball @ inv(T_prior_base_ball)
# T_ball_new_tcp = T_ball_off @ T_tcp

# 头部眼在手外，内部统一使用 m：
# T_board_off=T_base_cam@T_cam_board@inv(T_base_cam@T_prior_cam_board)
# T_board_new_tcp = T_board_off @ T_tcp

# endregion


# region 数据结构


@dataclass(frozen=True, slots=True)
class ParsedArmPose:
    """CSV 中单条笛卡尔目标。"""

    translation_m: tuple[float, float, float]
    pose_rpy_rad: tuple[float, float, float]
    has_elbow: bool | None
    elbow_rad: float | None
    conf_data: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """启动阶段已完成字段解析的单条 CSV 回放记录。"""

    csv_name: str
    row_index: int
    action_type: str
    joints_text: str
    pose_text: str
    joint_values: tuple[float, ...] | None
    arm_joint_rad: tuple[float, ...] | None
    arm_pose: ParsedArmPose | None
    pose_value: float | None

    def __str__(self) -> str:
        return f"{self.csv_name} {self.row_index} {self.action_type} {self.joints_text} {self.pose_text}"


@dataclass(frozen=True, slots=True)
class ArmMoveAbsJTarget:
    """单条 arm 记录最终用于 MoveAbsJ 的关节目标。"""

    row: ReplayRow
    joint: xCoreSDK_python.JointPosition
    source: str


@dataclass(slots=True)
class ReplayRuntime:
    """回放执行期上下文。"""

    connected_arm: ConnectedArm
    hand_channel: object
    body_channel: object
    body: WujiBodyClient
    gripper: DahuanGripperClient | None = None
    right_hand: WujiRightHandClient | None = None
    global_cartesian_offset: tuple[tuple[float, float, float, float], ...] | None = None
    charuco_cartesian_offset: tuple[tuple[float, float, float, float], ...] | None = None
    offset_record_path: Path | None = None
    offset_service_addr: str = DEFAULT_OFFSET_SERVICE_ADDR
    offset_camera_name: str = DEFAULT_OFFSET_CAMERA_NAME
    offset_prior_capture_path: Path = DEFAULT_OFFSET_PRIOR_CAPTURE_PATH
    hand_eye_result_path: Path = DEFAULT_HAND_EYE_RESULT_PATH
    stop_event: threading.Event | None = None
    move_abs_j_end_linear_speed_mm_s: float = (
        LEFT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE[-1]
    )
    pending_arm_rows: list[ReplayRow] = field(default_factory=list)
    preloaded_rows_by_path: dict[Path, tuple[ReplayRow, ...]] = field(default_factory=dict)


@dataclass(slots=True)
class ReplaySharedTunnelGroup:
    """持有双臂、AGV、gripper、M11/body 共用的单一 SSH 隧道及三个 channel。"""

    process: subprocess.Popen[bytes]
    gripper_channel: object
    body_channel: object
    agv_channel: object

    def close(self) -> None:
        """关闭全部 channel，再停止唯一的 SSH 隧道进程。"""

        close_wuyou_channel(self.agv_channel)
        close_wuyou_channel(self.gripper_channel)
        close_wuyou_channel(self.body_channel)
        stop_ssh_process(self.process)


@dataclass(frozen=True, slots=True)
class CsvExecutionPlan:
    """单个左臂阶段与右臂阶段的执行计划。"""

    left_csv_path: Path
    right_start_csv_path: Path | None = None
    right_pre_stage_csv_paths: tuple[Path, ...] = ()
    right_sync_csv_path: Path | None = None
    right_post_stage_csv_paths: tuple[Path, ...] = ()
    start_together: bool = False

    @property
    def has_sync(self) -> bool:
        return self.right_sync_csv_path is not None

    @property
    def has_parallel_start(self) -> bool:
        return self.start_together or self.has_sync


# endregion


# region CSV 解析


def _discover_csv_paths(record_dir: Path, max_files: int | None) -> list[Path]:
    if not record_dir.is_dir():
        raise FileNotFoundError(f"CSV 目录不存在：{record_dir}")
    numbered_paths = [
        (int(path.name.split("_", maxsplit=1)[0]), path.name, path)
        for path in record_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.name.split("_", maxsplit=1)[0].isdigit()
    ]
    numbered_paths.sort(key=lambda item: (item[0], item[1]))
    csv_paths = [path for _, _, path in numbered_paths]
    if max_files is not None:
        return csv_paths[:max_files]
    return csv_paths


def _load_replay_rows(csv_path: Path) -> list[ReplayRow]:
    """预解析 CSV；零字节或只有表头的占位文件返回空列表。"""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[ReplayRow] = []
        for row_index, row in enumerate(reader, start=1):
            action_type = str(row.get("type", "")).strip().lower()
            joints_text = str(row.get("joints", "")).strip()
            pose_text = str(row.get("pose", "")).strip()
            if action_type == "":
                raise ValueError(f"CSV 缺少 type: file={csv_path}, row={row_index}")
            joint_values: tuple[float, ...] | None = None
            arm_joint_rad: tuple[float, ...] | None = None
            arm_pose: ParsedArmPose | None = None
            pose_value: float | None = None
            if action_type == "arm":
                joint_values = tuple(_parse_joint_values(joints_text))
                arm_joint_rad = tuple(_deg_to_rad(list(joint_values)))
                if pose_text.lower() != "nan":
                    arm_pose = _parse_pose_values(pose_text)
            elif action_type == "m11":
                joint_values = tuple(_parse_joint_values(joints_text, expected_len=11))
            elif action_type in ("gripper", "lift"):
                pose_value = float(pose_text)
            rows.append(
                ReplayRow(
                    csv_name=csv_path.name,
                    row_index=row_index,
                    action_type=action_type,
                    joints_text=joints_text,
                    pose_text=pose_text,
                    joint_values=joint_values,
                    arm_joint_rad=arm_joint_rad,
                    arm_pose=arm_pose,
                    pose_value=pose_value,
                )
            )
    return rows


def _collect_execution_csv_paths(plans: list[CsvExecutionPlan]) -> list[Path]:
    ordered_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for plan in plans:
        plan_paths = (
            plan.left_csv_path,
            plan.right_start_csv_path,
            *plan.right_pre_stage_csv_paths,
            plan.right_sync_csv_path,
            *plan.right_post_stage_csv_paths,
        )
        for csv_path in plan_paths:
            if csv_path is None or csv_path in seen_paths:
                continue
            seen_paths.add(csv_path)
            ordered_paths.append(csv_path)
    return ordered_paths


def _preload_replay_rows(csv_paths: list[Path]) -> dict[Path, tuple[ReplayRow, ...]]:
    started_at = time.perf_counter()
    rows_by_path = {csv_path: tuple(_load_replay_rows(csv_path)) for csv_path in csv_paths}
    total_rows = sum(len(rows) for rows in rows_by_path.values())
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    logger.success(
        "全部 CSV 已预读取并预解析 files={} rows={} elapsed={:.2f} ms",
        len(rows_by_path),
        total_rows,
        elapsed_ms,
    )
    return rows_by_path


def _parse_joint_values(joints_text: str, expected_len: int = 7) -> list[float]:
    if joints_text.strip().lower() == "nan":
        raise ValueError("关节列为 NaN，不能解析为关节目标")
    parsed = ast.literal_eval(joints_text)
    if not isinstance(parsed, list) or len(parsed) != expected_len:
        raise ValueError(f"关节列长度无效：{joints_text}")
    return [float(value) for value in parsed]


def _parse_pose_values(pose_text: str) -> ParsedArmPose:
    if pose_text.strip().lower() == "nan":
        raise ValueError("pose 列为 NaN，不能解析为笛卡尔目标")
    parsed_pose = _parse_cartesian_pose_input(pose_text)
    translation_m_values = _mm_to_m(list(parsed_pose.xyz_mm))
    pose_rpy_rad_values = _deg_to_rad(list(parsed_pose.rpy_deg))
    return ParsedArmPose(
        translation_m=(
            translation_m_values[0],
            translation_m_values[1],
            translation_m_values[2],
        ),
        pose_rpy_rad=(
            pose_rpy_rad_values[0],
            pose_rpy_rad_values[1],
            pose_rpy_rad_values[2],
        ),
        has_elbow=parsed_pose.has_elbow,
        elbow_rad=(
            None if parsed_pose.elbow_deg is None else _deg_to_rad([parsed_pose.elbow_deg])[0]
        ),
        conf_data=parsed_pose.conf_data,
    )


def _extract_csv_sequence(csv_name: str) -> int:
    prefix = csv_name.split("_", maxsplit=1)[0]
    return int(prefix)


def _extract_sync_csv_sequence(csv_name: str) -> int | None:
    parts = csv_name.split("_")
    if len(parts) < 2:
        return None
    match = re.fullmatch(r"S(\d+)", parts[1])
    if match is None:
        return None
    return int(match.group(1))


def _should_apply_global_cartesian_offset(csv_name: str) -> bool:
    return _extract_csv_sequence(csv_name) in CSV_CARTESIAN_OFFSET_TARGETS


def _charuco_offset_csv_sequences(arm_side: str) -> list[int]:
    if arm_side == "left":
        return CHARUCO_OFFSET_LEFT_CSV_SEQUENCE
    return CHARUCO_OFFSET_RIGHT_CSV_SEQUENCE


def _should_apply_charuco_cartesian_offset(runtime: ReplayRuntime, csv_name: str) -> bool:
    configured_sequences = _charuco_offset_csv_sequences(runtime.connected_arm.arm_side)
    return _extract_csv_sequence(csv_name) in configured_sequences


def _resolve_cartesian_offset(
    runtime: ReplayRuntime,
    csv_name: str,
) -> tuple[np.ndarray | None, str]:
    if _should_apply_charuco_cartesian_offset(runtime, csv_name):
        if runtime.charuco_cartesian_offset is None:
            raise RuntimeError(f"CSV {csv_name} 需要 ChArUco offset，但当前尚未完成目标板检测")
        return np.asarray(runtime.charuco_cartesian_offset, dtype=np.float64), "charuco"
    if _should_apply_global_cartesian_offset(csv_name):
        if runtime.global_cartesian_offset is None:
            raise RuntimeError(f"CSV {csv_name} 需要三球 offset，但当前尚未完成三球检测")
        return np.asarray(runtime.global_cartesian_offset, dtype=np.float64), "three-ball"
    return None, "none"


def _should_trigger_offset_calculation(runtime: ReplayRuntime, csv_name: str) -> bool:
    return (
        runtime.connected_arm.arm_side == "left"
        and _extract_csv_sequence(csv_name) == CSV_CARTESIAN_OFFSET_CALCULATE_AT
    )


def _replay_move_abs_j_end_linear_speed_config(arm_side: str) -> dict[int, float]:
    if arm_side == "left":
        return LEFT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE
    return RIGHT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE


def _replay_move_abs_j_zone_config(arm_side: str) -> dict[int, float]:
    if arm_side == "left":
        return LEFT_REPLAY_MOVE_ABS_J_ZONE_MM_BY_CSV_SEQUENCE
    return RIGHT_REPLAY_MOVE_ABS_J_ZONE_MM_BY_CSV_SEQUENCE


def _resolve_replay_move_abs_j_end_linear_speed_mm_s(
    runtime: ReplayRuntime,
    csv_name: str,
) -> float:
    csv_sequence = _extract_csv_sequence(csv_name)
    speed_config = _replay_move_abs_j_end_linear_speed_config(runtime.connected_arm.arm_side)
    return speed_config.get(csv_sequence, runtime.move_abs_j_end_linear_speed_mm_s)


def _resolve_replay_move_abs_j_zone_mm(runtime: ReplayRuntime, csv_name: str) -> float:
    csv_sequence = _extract_csv_sequence(csv_name)
    zone_config = _replay_move_abs_j_zone_config(runtime.connected_arm.arm_side)
    return zone_config.get(
        csv_sequence,
        zone_config[-1],
    )


def _format_optional_csv_sequence(sequence: int | None) -> str:
    if sequence is None:
        return "None"
    return f"{sequence:02d}"


def _retry_non_motion_call(label: str, func):
    last_exc: BaseException | None = None
    for attempt in range(1, NON_MOTION_RETRY_COUNT + 1):
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "{} 失败，准备重试 attempt={}/{} delay={:.1f}s exc={}",
                label,
                attempt,
                NON_MOTION_RETRY_COUNT,
                NON_MOTION_RETRY_DELAY_S,
                exc,
            )
            if attempt < NON_MOTION_RETRY_COUNT:
                time.sleep(NON_MOTION_RETRY_DELAY_S)
    raise RuntimeError(f"{label} 连续失败 {NON_MOTION_RETRY_COUNT} 次") from last_exc


def _ensure_lift_enabled(body: WujiBodyClient, label: str) -> None:
    lift = body.lift
    deadline = time.monotonic() + LIFT_ENABLE_STATE_TIMEOUT_S
    attempt = 0
    last_enable_state: object = None
    while True:
        attempt += 1
        try:
            command_result = _retry_non_motion_call(label, lambda: lift.set_enable(True))
        except Exception as exc:
            command_result = f"调用异常：{exc}"
        try:
            last_enable_state = _retry_non_motion_call("lift.get_enable(wait)", lift.get_enable)
        except Exception as exc:
            last_enable_state = f"读取异常：{exc}"
        if last_enable_state is True:
            logger.success("lift enable 状态已生效 label={} attempt={}", label, attempt)
            return
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f"等待 lift enable 状态超时 label={label} timeout={LIFT_ENABLE_STATE_TIMEOUT_S:.1f} s "
                f"attempt={attempt} last_state={last_enable_state!r}"
            )
        logger.warning(
            "lift enable 尚未生效，等待后重新下发 label={} attempt={} "
            "command_return={} state={} remaining={:.1f} s",
            label,
            attempt,
            command_result,
            last_enable_state,
            remaining_s,
        )
        time.sleep(min(LIFT_ENABLE_RETRY_INTERVAL_S, remaining_s))


def _format_robot_runtime_state(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> str:
    operate_mode = robot.operateMode(ec)
    operation_state = robot.operationState(ec)
    power_state = robot.powerState(ec)
    return f"operate_mode={operate_mode} operation_state={operation_state} power_state={power_state}"


def _wait_until_reset_ready(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    label: str,
    timeout_s: float = DEFAULT_RESET_READY_TIMEOUT_S,
    stable_idle_checks: int = DEFAULT_RESET_READY_STABLE_IDLE_CHECKS,
) -> None:
    deadline = time.time() + timeout_s
    idle_count = 0
    last_state_text = ""
    while time.time() < deadline:
        operation_state = robot.operationState(ec)
        operate_mode = robot.operateMode(ec)
        power_state = robot.powerState(ec)
        last_state_text = f"operate_mode={operate_mode} operation_state={operation_state} power_state={power_state}"
        if operation_state == xCoreSDK_python.OperationState.idle:
            idle_count += 1
            if idle_count >= stable_idle_checks:
                return
        else:
            idle_count = 0
        time.sleep(RESET_READY_POLL_INTERVAL_S)
    logger.warning("{} 等待 reset 就绪超时，继续尝试 moveReset {}", label, last_state_text)


def _wait_until_motion_finished(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    label: str,
) -> None:
    """按 SDK 示例持续等待运动进入 idle 或 unknown。"""

    while True:
        time.sleep(MOTION_STATE_POLL_INTERVAL_S)
        operation_state = robot.operationState(ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"{label} 查询运动状态失败 ec={ec.get('ec', 0)} message={ec.get('message', '')}")
        if operation_state in (
            xCoreSDK_python.OperationState.idle,
            xCoreSDK_python.OperationState.unknown,
        ):
            logger.info("{} 已结束 state={}", label, operation_state)
            return


def _build_csv_execution_plans(
    left_csv_paths: list[Path],
    right_csv_paths: list[Path],
) -> list[CsvExecutionPlan]:
    right_csv_by_sequence = {_extract_csv_sequence(path.name): path for path in right_csv_paths}
    right_sequences = sorted(right_csv_by_sequence)
    consumed_right_sequences: set[int] = set()
    stage_specs: list[tuple[Path, Path | None, tuple[Path, ...], Path | None, bool]] = []
    for left_index, left_csv_path in enumerate(left_csv_paths):
        left_sequence = _extract_csv_sequence(left_csv_path.name)
        right_sync_sequence = _extract_sync_csv_sequence(left_csv_path.name)
        right_pre_stage_sequences: list[int] = []
        right_sync_csv_path = None
        start_together = False
        right_start_csv_path = None

        if left_index == 0 and left_sequence in right_csv_by_sequence:
            start_together = True
            right_start_csv_path = right_csv_by_sequence[left_sequence]
            consumed_right_sequences.add(left_sequence)

        if right_sync_sequence is not None:
            right_sync_csv_path = right_csv_by_sequence.get(right_sync_sequence)
            if right_sync_csv_path is None:
                raise RuntimeError(
                    "左臂 CSV 声明了同步右臂文件，但右臂目录中不存在对应序号："
                    f" left={left_csv_path.name} right_seq={right_sync_sequence:02d}"
                )
            for right_sequence in right_sequences:
                if right_sequence in consumed_right_sequences:
                    continue
                if right_sequence >= right_sync_sequence:
                    break
                right_pre_stage_sequences.append(right_sequence)
                consumed_right_sequences.add(right_sequence)
            consumed_right_sequences.add(right_sync_sequence)
        elif left_index > 0:
            next_sync_sequence = None
            for future_left_csv_path in left_csv_paths[left_index + 1 :]:
                next_sync_sequence = _extract_sync_csv_sequence(future_left_csv_path.name)
                if next_sync_sequence is not None:
                    break
            upper_bound = left_sequence if next_sync_sequence is None else next_sync_sequence
            for right_sequence in right_sequences:
                if right_sequence in consumed_right_sequences:
                    continue
                if right_sequence >= upper_bound:
                    break
                right_pre_stage_sequences.append(right_sequence)
                consumed_right_sequences.add(right_sequence)

        stage_specs.append(
            (
                left_csv_path,
                right_start_csv_path,
                tuple(right_csv_by_sequence[sequence] for sequence in right_pre_stage_sequences),
                right_sync_csv_path,
                start_together,
            )
        )
    trailing_right_csv_paths = tuple(
        right_csv_by_sequence[sequence] for sequence in right_sequences if sequence not in consumed_right_sequences
    )
    plans: list[CsvExecutionPlan] = []
    for plan_index, (
        left_csv_path,
        right_start_csv_path,
        right_pre_stage_csv_paths,
        right_sync_csv_path,
        start_together,
    ) in enumerate(stage_specs):
        right_post_stage_csv_paths = trailing_right_csv_paths if plan_index == len(stage_specs) - 1 else ()
        plans.append(
            CsvExecutionPlan(
                left_csv_path=left_csv_path,
                right_start_csv_path=right_start_csv_path,
                right_pre_stage_csv_paths=right_pre_stage_csv_paths,
                right_sync_csv_path=right_sync_csv_path,
                right_post_stage_csv_paths=right_post_stage_csv_paths,
                start_together=start_together,
            )
        )
    return plans


# endregion


# region 连接与运行时


def _create_replay_shared_tunnel_group() -> ReplaySharedTunnelGroup:
    """创建同时承载双臂、AGV、gripper 与 M11/body 的单一 SSH 隧道。"""

    forwards = (
        ("127.0.0.1", SHARED_TUNNEL_BODY_PORT, WUYOU_QMLINKER_HOST, DEFAULT_PORT),
        ("127.0.0.1", SHARED_TUNNEL_GRIPPER_PORT, WUYOU_QMLINKER_HOST, GRIPPER_PORT),
        ("127.0.0.1", SHARED_TUNNEL_AGV_PORT, AGV_HOST, DEFAULT_PORT),
        *(
            (local_ip, port, controller_ip, port)
            for local_ip, controller_ip in (
                (LEFT_ARM_IP, LEFT_ARM_CONTROLLER_IP),
                (RIGHT_ARM_IP, RIGHT_ARM_CONTROLLER_IP),
            )
            for port in ARM_SSH_FORWARD_PORTS
        ),
    )
    command = [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-N",
    ]
    for local_host, local_port, remote_host, remote_port in forwards:
        command.extend(("-L", f"{local_host}:{local_port}:{remote_host}:{remote_port}"))
    command.append(SSH_ALIAS)
    logger.info(
        "启动共享 SSH 隧道 left_arm={} right_arm={} arm_ports={} body/m11={} gripper={} agv={} alias={}",
        LEFT_ARM_CONTROLLER_IP,
        RIGHT_ARM_CONTROLLER_IP,
        ARM_SSH_FORWARD_PORTS,
        SHARED_TUNNEL_BODY_PORT,
        SHARED_TUNNEL_GRIPPER_PORT,
        SHARED_TUNNEL_AGV_PORT,
        SSH_ALIAS,
    )
    process = subprocess.Popen(command, stderr=subprocess.PIPE)
    try:
        _wait_for_replay_shared_tunnel(
            process,
            tuple((local_host, local_port) for local_host, local_port, _, _ in forwards),
        )
        return ReplaySharedTunnelGroup(
            process=process,
            gripper_channel=create_channel(f"127.0.0.1:{SHARED_TUNNEL_GRIPPER_PORT}"),
            body_channel=create_channel(f"127.0.0.1:{SHARED_TUNNEL_BODY_PORT}"),
            agv_channel=create_channel(f"127.0.0.1:{SHARED_TUNNEL_AGV_PORT}"),
        )
    except BaseException:
        stop_ssh_process(process)
        raise


def _wait_for_replay_shared_tunnel(
    process: subprocess.Popen[bytes],
    local_endpoints: tuple[tuple[str, int], ...],
) -> None:
    """等待共享 SSH 隧道的全部本地地址与端口可连接。"""

    deadline = time.monotonic() + SHARED_TUNNEL_READY_TIMEOUT_S
    pending_endpoints = set(local_endpoints)
    while pending_endpoints and time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = b"" if process.stderr is None else process.stderr.read()
            error_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"共享 SSH 隧道提前退出：{error_text or '无错误信息'}")
        for local_host, local_port in tuple(pending_endpoints):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex((local_host, local_port)) == 0:
                    pending_endpoints.remove((local_host, local_port))
        if pending_endpoints:
            time.sleep(0.1)
    if pending_endpoints:
        endpoints_text = ", ".join(f"{host}:{port}" for host, port in sorted(pending_endpoints))
        raise RuntimeError(f"共享 SSH 隧道地址未就绪：{endpoints_text}")


def _navigate_agv_before_replay(tunnel_group: ReplaySharedTunnelGroup) -> None:
    """导航到默认 AGV 站点，等待状态从 busy 恢复为 idel。"""

    logger.info("全自动回放开始前导航 AGV target={}", DEFAULT_AGV_POINT)
    client = WujiAgvClient(tunnel_group.agv_channel)
    client.navigate_to(DEFAULT_AGV_POINT)
    deadline = time.monotonic() + DEFAULT_AGV_NAVIGATION_TIMEOUT_S
    observed_busy = False
    while time.monotonic() < deadline:
        raw_status = str(client.get_runtime_info().get("agv_navi_status", "")).strip().lower()
        logger.info("AGV 导航状态 target={} raw_status={!r}", DEFAULT_AGV_POINT, raw_status)
        if raw_status == "busy":
            observed_busy = True
        elif observed_busy and raw_status == "idel":
            logger.success("AGV 导航结束 target={} raw_status={!r}", DEFAULT_AGV_POINT, raw_status)
            return
        time.sleep(DEFAULT_AGV_NAVIGATION_POLL_INTERVAL_S)
    raise TimeoutError(
        f"AGV 导航到位超时：target={DEFAULT_AGV_POINT}, observed_busy={observed_busy}"
    )


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
    robot_info = _retry_non_motion_call(f"robotInfo({robot_ip})", lambda: robot.robotInfo(ec))
    _print_sdk_result(f"robotInfo({robot_ip})", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"读取机械臂机器人信息失败：arm_side={arm_side}, ip={robot_ip}")
    if (
        _retry_non_motion_call(
            f"apply_named_toolset({robot_ip})",
            lambda: _apply_named_toolset(robot, ec),
        )
        is None
    ):
        raise RuntimeError(f"设置默认工具/工件失败：ip={robot_ip}, tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    detected_arm_side = _detect_arm_side(robot_info.type)
    if detected_arm_side != arm_side:
        raise RuntimeError(f"连接到的机械臂侧别不匹配：expected={arm_side}, ip={robot_ip}, actual={detected_arm_side}")
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


def _create_runtime(arm_side: str, tunnel_group: ReplaySharedTunnelGroup) -> ReplayRuntime:
    connected_arm = _connect_arm(arm_side)
    hand_channel = tunnel_group.gripper_channel if arm_side == "left" else tunnel_group.body_channel
    body_channel = tunnel_group.body_channel
    runtime = ReplayRuntime(
        connected_arm=connected_arm,
        hand_channel=hand_channel,
        body_channel=body_channel,
        body=WujiBodyClient(body_channel),
    )
    if arm_side == "left":
        runtime.gripper = DahuanGripperClient(hand_channel)
    else:
        runtime.right_hand = WujiRightHandClient(hand_channel)
    return runtime


def _prepare_gripper_before_replay(gripper: DahuanGripperClient) -> None:
    """检查夹爪校准状态，必要时校准并回到初始位置。"""

    status = gripper.get_status()
    logger.info(
        "回放前夹爪状态 calibrated={} position={}",
        bool(status.calibrated),
        status.position,
    )
    ready_status = status
    if bool(status.calibrated):
        logger.success("夹爪已校准，继续确认回放前位置")
    else:
        logger.warning("夹爪尚未校准，开始执行回放前校准")
        if not gripper.calibrate():
            raise RuntimeError("回放前夹爪校准命令下发失败")
        logger.info("夹爪校准命令已发出，等待 {:.1f} s", GRIPPER_CALIBRATION_WAIT_S)
        time.sleep(GRIPPER_CALIBRATION_WAIT_S)

        ready_status = gripper.get_status()
        if not bool(ready_status.calibrated):
            raise RuntimeError("夹爪校准命令发出并等待 3 s 后仍未进入已校准状态")

    current_position = int(ready_status.position or 0)
    command_count = 0
    while current_position != GRIPPER_ZERO_POSITION:
        command_count += 1
        if not gripper.set_pos(GRIPPER_ZERO_POSITION):
            raise RuntimeError("回放前夹爪运动到位置 0 的命令下发失败")
        logger.info(
            "持续下发夹爪回零命令 count={} current_position={} target_position={}",
            command_count,
            current_position,
            GRIPPER_ZERO_POSITION,
        )
        time.sleep(GRIPPER_ZERO_POLL_INTERVAL_S)
        current_status = gripper.get_status()
        current_position = int(current_status.position or 0)
    logger.success(
        "夹爪回放前状态已就绪 calibrated=True position={} command_count={}",
        GRIPPER_ZERO_POSITION,
        command_count,
    )


def _prepare_runtime(runtime: ReplayRuntime) -> None:
    if not _retry_non_motion_call(
        f"ensure_nrt_motion_ready({runtime.connected_arm.arm_side})",
        lambda: _ensure_nrt_motion_ready(runtime.connected_arm.robot, runtime.connected_arm.ec),
    ):
        raise RuntimeError(f"{runtime.connected_arm.arm_side} 臂未准备到可执行回放的 NRT 状态")
    _ensure_lift_enabled(
        runtime.body,
        f"lift.set_enable({runtime.connected_arm.arm_side})",
    )
    logger.info(
        "已确认机械臂侧别={} 基坐标采用 tool={} wobj={}",
        runtime.connected_arm.arm_side,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )


# endregion


# region 位姿与 offset


def _build_cartesian_target(runtime: ReplayRuntime, row: ReplayRow) -> xCoreSDK_python.CartesianPosition:
    parsed_pose = row.arm_pose
    if parsed_pose is None:
        raise RuntimeError(f"arm pose 未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    current_pose = _retry_non_motion_call(
        f"cartPosture(endInRef:{row.csv_name}:{row.row_index})",
        lambda: robot.cartPosture(xCoreSDK_python.endInRef, ec),
    )
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取当前笛卡尔位姿失败")
    target_pose = xCoreSDK_python.CartesianPosition(
        list(parsed_pose.translation_m + parsed_pose.pose_rpy_rad)
    )
    _copy_cartesian_pose_context(current_pose, target_pose)
    if parsed_pose.has_elbow is not None:
        target_pose.hasElbow = parsed_pose.has_elbow
    if parsed_pose.elbow_rad is not None:
        target_pose.elbow = parsed_pose.elbow_rad
    if parsed_pose.conf_data is not None:
        target_pose.confData = list(parsed_pose.conf_data)
    offset_matrix_m, offset_source = _resolve_cartesian_offset(runtime, row.csv_name)
    if offset_matrix_m is not None:
        target_pose = _apply_cartesian_offset(row, target_pose, offset_matrix_m, offset_source)
    return target_pose


def _build_move_abs_j_target(runtime: ReplayRuntime, row: ReplayRow) -> ArmMoveAbsJTarget:
    if row.arm_pose is None:
        if row.arm_joint_rad is None:
            raise RuntimeError(f"arm joints 未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
        return ArmMoveAbsJTarget(
            row=row,
            joint=xCoreSDK_python.JointPosition(list(row.arm_joint_rad)),
            source="csv-joints",
        )

    target_pose = _build_cartesian_target(runtime, row)
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    offset_matrix_m, offset_source = _resolve_cartesian_offset(runtime, row.csv_name)
    applies_offset = offset_matrix_m is not None
    robot_model = robot.model()
    toolset = _retry_non_motion_call(
        f"toolset(replay-arm-ik:{row.csv_name}:{row.row_index})",
        lambda: robot.toolset(ec),
    )
    _print_sdk_result("toolset(replay-arm-ik)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放 arm 读取 toolset 失败")
    raw_target_joint = robot_model.calcIk(target_pose, toolset, ec)
    _print_sdk_result("calcIk(replay-arm-moveabsj)", ec)
    if ec.get("ec", 0) == 0:
        if isinstance(raw_target_joint, xCoreSDK_python.JointPosition):
            target_joint_rad = [float(value) for value in raw_target_joint.joints]
        elif isinstance(raw_target_joint, np.ndarray):
            target_joint_rad = [float(value) for value in raw_target_joint.reshape(-1).tolist()]
        elif isinstance(raw_target_joint, (list, tuple)):
            target_joint_rad = [float(value) for value in raw_target_joint]
        else:
            raise RuntimeError(f"calcIk 成功，但返回值类型不支持：{type(raw_target_joint).__name__}")
        if len(target_joint_rad) != 7:
            raise RuntimeError(f"calcIk 成功，但返回关节数异常：len={len(target_joint_rad)}")
        target_joint = xCoreSDK_python.JointPosition(target_joint_rad)
        return ArmMoveAbsJTarget(
            row=row,
            joint=target_joint,
            source=f"tcp-ik-offset-{offset_source}" if applies_offset else "tcp-ik",
        )

    failed_ec = ec.get("ec", 0)
    failed_message = ec.get("message", "")
    if applies_offset:
        if offset_matrix_m is None:
            raise RuntimeError("offset 应用状态异常")
        corrected_tcp_matrix_m = _frame_to_homogeneous_matrix_m(target_pose)
        original_tcp_matrix_m = np.linalg.inv(offset_matrix_m) @ corrected_tcp_matrix_m
        logger.warning(
            "offset 后 T_new_tcp 逆解失败，将使用该行 CSV joints 且不实施 offset "
            "file={} row={} ec={} message={} {} {} {}",
            row.csv_name,
            row.row_index,
            failed_ec,
            failed_message,
            _format_matrix_xyzrpy_mm_deg("T_tcp", original_tcp_matrix_m),
            _format_matrix_xyzrpy_mm_deg("T_off", offset_matrix_m),
            _format_matrix_xyzrpy_mm_deg("T_new_tcp", corrected_tcp_matrix_m),
        )

    if row.joint_values is None or row.arm_joint_rad is None:
        raise RuntimeError(f"CSV joints 兜底未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    fallback_joint_deg = list(row.joint_values)
    ec["ec"] = 0
    fallback_reason = "offset 后 T_new_tcp 逆解失败，本行未实施 offset" if applies_offset else "原始 TCP 逆解失败"
    ec["message"] = f"{fallback_reason}，已改用 CSV 记录关节值"
    logger.warning(
        "{}，最终兜底为 CSV joints MoveAbsJ file={} row={} " "ik_ec={} ik_message={} joints(deg)=[{}]",
        fallback_reason,
        row.csv_name,
        row.row_index,
        failed_ec,
        failed_message,
        _format_sequence(fallback_joint_deg),
    )
    return ArmMoveAbsJTarget(
        row=row,
        joint=xCoreSDK_python.JointPosition(list(row.arm_joint_rad)),
        source="csv-joints-fallback-offset-ik" if applies_offset else "csv-joints-fallback",
    )


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
                float(left[row_index][term_index]) * float(right[term_index][col_index]) for term_index in range(4)
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


def _apply_cartesian_offset(
    row: ReplayRow,
    target_pose: xCoreSDK_python.CartesianPosition,
    offset_matrix_m: np.ndarray,
    offset_source: str,
) -> xCoreSDK_python.CartesianPosition:
    """将运行时 offset 左乘到目标 TCP 上。

    链路：
    - 目标 TCP 由 CSV 目标构造，内部以 `m/rad` 进入 `CartesianPosition`
    - `offset_matrix_m` 以 `m` 保存
    - 应用公式固定为 `T_new_tcp = T_off @ T_tcp`
    - 最终返回给 SDK 的 `CartesianPosition` 仍保持 `m/rad`
    """

    original_matrix = _frame_to_homogeneous_matrix(target_pose)
    corrected_matrix = offset_matrix_m @ np.asarray(original_matrix, dtype=np.float64)
    corrected_pose = _homogeneous_matrix_to_cartesian_position(target_pose, corrected_matrix)
    logger.info(
        "已应用左乘纠偏 T_new_tcp=T_off@T_tcp source={} file={} row={} base=tool:{} wobj:{}",
        offset_source,
        row.csv_name,
        row.row_index,
        DEFAULT_TOOL_NAME,
        DEFAULT_WOBJ_NAME,
    )
    logger.info(
        "offset 矩阵 file={} row={} {}",
        row.csv_name,
        row.row_index,
        _format_matrix_xyzrpy_mm_deg("T_off", offset_matrix_m),
    )
    logger.info(
        "纠偏后目标矩阵 file={} row={} {}",
        row.csv_name,
        row.row_index,
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
        raise RuntimeError(f"先验 tcp_pose_matrix 格式无效：{prior_capture_path}")
    if ball_matrix_m.shape != (4, 4) or not np.all(np.isfinite(ball_matrix_m)):
        raise RuntimeError(f"先验 local_pose_transform 格式无效：{prior_capture_path}")
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
        raise FileNotFoundError(f"手眼结果文件不存在：{hand_eye_result_path}")
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
                raise ValueError(f"手眼矩阵行格式错误：{line}")
            matrix_rows.append(values)
            if len(matrix_rows) == 4:
                break
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"手眼矩阵维度错误：shape={matrix.shape}, path={hand_eye_result_path}")
    return matrix


def _detect_current_three_ball_basis_transform(
    service_addr: str,
    camera_name: str,
    prior_capture_path: Path,
) -> np.ndarray:
    """使用正式服务的宽窄分级策略返回当前 `T_cam_ball`。

    宽 HSV 首先保证三球可检出；窄 HSV 结果与宽 HSV 球心一致时优先采用窄结果，
    否则回退宽结果。最终矩阵平移单位为 m。
    """

    settings = ReplayOffsetSettings(
        sample_count=OFFSET_BALL_CAPTURE_SAMPLE_COUNT,
        detection_timeout_ms=OFFSET_BALL_DETECTION_TIMEOUT_MS,
    )
    detector = CameraPipelineThreeBallDetector(
        camera_name=CameraName(camera_name),
        priors=load_three_ball_priors(prior_capture_path),
        settings=settings,
        service_addr=str(service_addr),
    )
    samples_mm = detector.capture_samples(OFFSET_BALL_CAPTURE_SAMPLE_COUNT)
    basis_transform = camera_ball_transform_m(samples_mm, settings)
    logger.info(
        "ball pose detection 宽窄分级采样完成 camera={} requested={} valid={}",
        camera_name,
        OFFSET_BALL_CAPTURE_SAMPLE_COUNT,
        len(samples_mm),
    )
    return basis_transform


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
    if (
        _retry_non_motion_call(
            f"apply_named_toolset(offset:{runtime.connected_arm.arm_side})",
            lambda: _apply_named_toolset(runtime.connected_arm.robot, runtime.connected_arm.ec),
        )
        is None
    ):
        raise RuntimeError(f"设置固定 toolset 失败：tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    tcp_pose = _retry_non_motion_call(
        f"cartPosture(endInRef, offset-calc:{runtime.connected_arm.arm_side})",
        lambda: runtime.connected_arm.robot.cartPosture(xCoreSDK_python.endInRef, runtime.connected_arm.ec),
    )
    _print_sdk_result("cartPosture(endInRef, offset-calc)", runtime.connected_arm.ec)
    if runtime.connected_arm.ec.get("ec", 0) != 0:
        raise RuntimeError("读取当前 TCP 位姿失败，无法计算全局 offset")
    # SDK 原始输出：trans(m), rpy(rad)，这里重建成内部计算矩阵 (m)。
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
    offset_distance_mm = offset_distance_m * 1000.0
    logger.info("offset_norm_mm={:.3f} {}", offset_distance_mm, _format_matrix_xyzrpy_mm_deg("T_off", offset_matrix_m))
    if offset_distance_mm > OFFSET_TRANSLATION_WARNING_THRESHOLD_MM:
        logger.warning(
            "offset 平移明显偏大，请检查拍摄时机/三球识别/先验是否一致 distance_mm={:.3f} {}",
            offset_distance_mm,
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


def _matrix_to_runtime_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    validated = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(validated)):
        raise ValueError("offset 矩阵包含非有限数值")
    return (
        (float(validated[0, 0]), float(validated[0, 1]), float(validated[0, 2]), float(validated[0, 3])),
        (float(validated[1, 0]), float(validated[1, 1]), float(validated[1, 2]), float(validated[1, 3])),
        (float(validated[2, 0]), float(validated[2, 1]), float(validated[2, 2]), float(validated[2, 3])),
        (float(validated[3, 0]), float(validated[3, 1]), float(validated[3, 2]), float(validated[3, 3])),
    )


def _head_base_camera_path(arm_side: str) -> Path:
    configured_path = LEFT_HEAD_BASE_CAMERA_PATH if arm_side == "left" else RIGHT_HEAD_BASE_CAMERA_PATH
    if configured_path is None:
        raise RuntimeError(
            f"{arm_side} 臂已配置 ChArUco offset CSV，但对应 T_base_camera 路径仍为 None；"
            "请在文件顶部填写 Path"
        )
    return configured_path


def _load_head_base_camera_transform_m(arm_side: str) -> np.ndarray:
    path = _head_base_camera_path(arm_side)
    if not path.is_file():
        raise FileNotFoundError(f"{arm_side} 臂 T_base_camera.npy 不存在：{path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{arm_side} 臂 T_base_camera.npy 格式无效：shape={matrix.shape}, path={path}")
    return matrix


def _load_prior_camera_board_transform_m() -> np.ndarray:
    if not DEFAULT_CHARUCO_PRIOR_PATH.is_file():
        raise FileNotFoundError(f"ChArUco 先验文件不存在：{DEFAULT_CHARUCO_PRIOR_PATH}")
    payload = json.loads(DEFAULT_CHARUCO_PRIOR_PATH.read_text(encoding="utf-8"))
    matrix = np.asarray(payload.get("camera_board_transform"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"ChArUco 先验缺少有效 camera_board_transform：{DEFAULT_CHARUCO_PRIOR_PATH}")
    translation_unit = payload.get("translation_unit")
    if translation_unit != "mm":
        raise ValueError(f"ChArUco 先验平移单位不是 mm：{translation_unit!r}")
    matrix_m = matrix.copy()
    matrix_m[:3, 3] *= 0.001
    return matrix_m


def _set_head_charuco_detection_pose(head: WujiHeadClient) -> None:
    head.set_head_yaw(DEFAULT_HEAD_YAW_DEG)
    head.set_head_pitch(DEFAULT_HEAD_PITCH_DEG)
    time.sleep(DEFAULT_HEAD_SETTLE_S)
    logger.info(
        "头部 ChArUco 检测姿态已设置 yaw={:.1f} deg pitch={:.1f} deg",
        float(head.get_head_yaw() or 0.0),
        float(head.get_head_pitch() or 0.0),
    )


def _detect_current_camera_board_transform_m(head_channel: object) -> np.ndarray:
    client = CameraPipelineClient(
        service_addr=CHARUCO_DETECTION_SERVICE_ADDR,
        timeout_ms=int(CHARUCO_DETECTION_RPC_TIMEOUT_S * 1000.0),
    )
    try:
        _set_head_charuco_detection_pose(WujiHeadClient(head_channel))
        request = CharucoDetectionRequest(
            camera_name=CameraName(DEFAULT_HEAD_CAMERA_NAME),
            dictionary_name=DEFAULT_DICTIONARY_NAME,
            squares_x=DEFAULT_SQUARES_X,
            squares_y=DEFAULT_SQUARES_Y,
            square_length_mm=float(DEFAULT_SQUARE_LENGTH_MM),
            marker_length_mm=float(DEFAULT_MARKER_LENGTH_MM),
            min_charuco_corners=CHARUCO_DETECTION_MIN_CORNERS,
            max_frames=CHARUCO_DETECTION_MAX_FRAME_COUNT,
            stable_timeout_s=CHARUCO_DETECTION_CAMERA_TIMEOUT_S,
        )
        response = None
        last_timeout: zmq.Again | None = None
        for attempt in range(1, CHARUCO_DETECTION_TIMEOUT_RETRY_COUNT + 1):
            try:
                response = client.detect_charuco(request)
                break
            except zmq.Again as exc:
                last_timeout = exc
                if attempt == CHARUCO_DETECTION_TIMEOUT_RETRY_COUNT:
                    break
                logger.warning(
                    "ChArUco RPC 接收超时，等待后重试 attempt={}/{} "
                    "rpc_timeout={:.1f}s delay={:.1f}s",
                    attempt,
                    CHARUCO_DETECTION_TIMEOUT_RETRY_COUNT,
                    CHARUCO_DETECTION_RPC_TIMEOUT_S,
                    CHARUCO_DETECTION_TIMEOUT_RETRY_DELAY_S,
                )
                time.sleep(CHARUCO_DETECTION_TIMEOUT_RETRY_DELAY_S)
        if response is None:
            raise TimeoutError(
                "ChArUco RPC 连续超时 "
                f"attempts={CHARUCO_DETECTION_TIMEOUT_RETRY_COUNT} "
                f"timeout={CHARUCO_DETECTION_RPC_TIMEOUT_S:.1f}s"
            ) from last_timeout
        if response.status == "detected" and len(response.t_cam_board_mm) == 4:
            matrix_m = np.asarray(response.t_cam_board_mm, dtype=np.float64).reshape(4, 4)
            matrix_m = matrix_m.copy()
            matrix_m[:3, 3] *= 0.001
            logger.success(
                "头部 ChArUco 检测成功 corners={} reprojection_error_px={}",
                response.charuco_num,
                response.error_px,
            )
            return matrix_m
    finally:
        client.close()
    raise RuntimeError(
        f"连续 {CHARUCO_DETECTION_MAX_FRAME_COUNT} 帧未检测到有效 ChArUco 目标板位姿"
    )


def _calculate_charuco_cartesian_offset(
    runtime: ReplayRuntime,
    current_camera_board_m: np.ndarray,
    context_label: str,
) -> tuple[tuple[float, float, float, float], ...]:
    if runtime.charuco_cartesian_offset is not None:
        raise RuntimeError(f"{runtime.connected_arm.arm_side} 臂本轮 ChArUco offset 已计算，拒绝重复覆盖")
    arm_side = runtime.connected_arm.arm_side
    base_camera_m = _load_head_base_camera_transform_m(arm_side)
    prior_camera_board_m = _load_prior_camera_board_transform_m()
    prior_base_board_m = base_camera_m @ prior_camera_board_m
    current_base_board_m = base_camera_m @ current_camera_board_m
    offset_matrix_m = current_base_board_m @ np.linalg.inv(prior_base_board_m)
    accepted, decision_reason = _evaluate_charuco_offset(
        offset_matrix_m,
        arm_side,
        DEFAULT_CHARUCO_OFFSET_HISTORY_PATH,
    )
    if not accepted:
        raise RuntimeError(f"ChArUco offset 安全检查拒绝执行：{decision_reason}")
    logger.info("ChArUco offset 安全检查通过 {}", decision_reason)
    runtime_offset = _matrix_to_runtime_tuple(offset_matrix_m)
    record_path = _create_offset_record(runtime_offset)
    runtime.charuco_cartesian_offset = runtime_offset
    runtime.offset_record_path = record_path
    logger.success(
        "ChArUco offset 已在内存中更新 arm_side={} context={} {}",
        arm_side,
        context_label,
        _format_matrix_xyzrpy_mm_deg("T_off", offset_matrix_m),
    )
    logger.info(
        "CHARUCO_OFFSET_READY {}",
        _format_matrix_xyzrpy_mm_deg("T_charuco_off", offset_matrix_m),
    )
    logger.info("{}", _format_matrix_xyzrpy_mm_deg("T_prior_base_board", prior_base_board_m))
    logger.info("{}", _format_matrix_xyzrpy_mm_deg("T_current_base_board", current_base_board_m))
    return runtime_offset


def _precheck_charuco_cartesian_offset(
    runtime: ReplayRuntime,
    current_camera_board_m: np.ndarray,
) -> tuple[bool, str]:
    """计算候选 ChArUco offset，并在写入运行时前执行历史安全检查。"""

    arm_side = runtime.connected_arm.arm_side
    base_camera_m = _load_head_base_camera_transform_m(arm_side)
    prior_camera_board_m = _load_prior_camera_board_transform_m()
    prior_base_board_m = base_camera_m @ prior_camera_board_m
    current_base_board_m = base_camera_m @ current_camera_board_m
    offset_matrix_m = current_base_board_m @ np.linalg.inv(prior_base_board_m)
    return _evaluate_charuco_offset(
        offset_matrix_m,
        arm_side,
        DEFAULT_CHARUCO_OFFSET_HISTORY_PATH,
    )


def _evaluate_charuco_offset(
    offset_matrix_m: np.ndarray,
    arm_side: str,
    history_path: Path,
) -> tuple[bool, str]:
    """依据同侧历史有效样本判断 ChArUco offset 是否可安全使用。

    每个 xyz/rpy 分量使用历史均值 ±4σ；平移和旋转模长同时受历史 4σ 上界
    与绝对安全上限约束。绝对上限用于阻止历史样本逐步漂移后放宽到危险范围。
    """

    values = _charuco_offset_xyzrpy_mm_deg(offset_matrix_m)
    accepted_history = _load_accepted_charuco_offset_history(history_path, arm_side)
    sample_count = accepted_history.shape[0]
    if sample_count < CHARUCO_OFFSET_HISTORY_MIN_ACCEPTED_SAMPLES:
        return (
            False,
            f"{arm_side} 臂有效历史样本不足："
            f"{sample_count} < {CHARUCO_OFFSET_HISTORY_MIN_ACCEPTED_SAMPLES}",
        )

    means = np.mean(accepted_history, axis=0)
    standard_deviations = np.std(accepted_history, axis=0, ddof=1)
    lower_bounds = means - CHARUCO_OFFSET_SIGMA_LIMIT * standard_deviations
    upper_bounds = means + CHARUCO_OFFSET_SIGMA_LIMIT * standard_deviations
    labels = ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")
    violations = [
        f"{label}={value:.3f} 不在 [{lower:.3f}, {upper:.3f}]"
        for label, value, lower, upper in zip(
            labels,
            values,
            lower_bounds,
            upper_bounds,
            strict=True,
        )
        if value < lower or value > upper
    ]

    history_translation_norms = np.linalg.norm(accepted_history[:, :3], axis=1)
    history_rotation_norms = np.linalg.norm(accepted_history[:, 3:], axis=1)
    translation_limit_mm = min(
        float(
            np.mean(history_translation_norms)
            + CHARUCO_OFFSET_SIGMA_LIMIT * np.std(history_translation_norms, ddof=1)
        ),
        CHARUCO_OFFSET_MAX_TRANSLATION_NORM_MM,
    )
    rotation_limit_deg = min(
        float(
            np.mean(history_rotation_norms)
            + CHARUCO_OFFSET_SIGMA_LIMIT * np.std(history_rotation_norms, ddof=1)
        ),
        CHARUCO_OFFSET_MAX_ROTATION_NORM_DEG,
    )
    translation_norm_mm = float(np.linalg.norm(values[:3]))
    rotation_norm_deg = float(np.linalg.norm(values[3:]))
    if translation_norm_mm > translation_limit_mm:
        violations.append(
            f"translation_norm_mm={translation_norm_mm:.3f} > {translation_limit_mm:.3f}"
        )
    if rotation_norm_deg > rotation_limit_deg:
        violations.append(
            f"rotation_norm_deg={rotation_norm_deg:.3f} > {rotation_limit_deg:.3f}"
        )
    summary = (
        f"arm_side={arm_side} history_count={sample_count} sigma={CHARUCO_OFFSET_SIGMA_LIMIT:.1f} "
        f"translation_limit_mm={translation_limit_mm:.3f} "
        f"rotation_limit_deg={rotation_limit_deg:.3f}"
    )
    if violations:
        return False, f"{summary}; violations={'; '.join(violations)}"
    return True, f"{summary}; within_normal_range"


def _load_accepted_charuco_offset_history(history_path: Path, arm_side: str) -> np.ndarray:
    """读取指定机械臂侧别的有效 ChArUco offset 历史向量。"""

    if not history_path.is_file():
        raise FileNotFoundError(f"ChArUco offset 历史文件不存在：{history_path}")
    values: list[list[float]] = []
    with history_path.open("r", encoding="utf-8", newline="") as history_file:
        reader = csv.DictReader(history_file)
        if tuple(reader.fieldnames or ()) != CHARUCO_OFFSET_HISTORY_FIELDS:
            raise RuntimeError(f"ChArUco offset 历史 CSV 字段不符合约定：{history_path}")
        for row in reader:
            if row["arm_side"] != arm_side or row["accepted"].strip().lower() != "true":
                continue
            values.append(
                [
                    float(row["x_mm"]),
                    float(row["y_mm"]),
                    float(row["z_mm"]),
                    float(row["roll_deg"]),
                    float(row["pitch_deg"]),
                    float(row["yaw_deg"]),
                ]
            )
    if not values:
        return np.empty((0, 6), dtype=np.float64)
    history = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(history)):
        raise RuntimeError(f"ChArUco offset 历史 CSV 包含非有限数值：{history_path}")
    return history


def _initialize_charuco_cartesian_offsets(runtimes: list[ReplayRuntime]) -> None:
    """检测目标板并在安全检查通过后为所有活动机械臂初始化 offset。"""

    if not runtimes:
        raise ValueError("初始化 ChArUco offset 时缺少活动 runtime")
    if any(runtime.charuco_cartesian_offset is not None for runtime in runtimes):
        raise RuntimeError("本轮 ChArUco offset 已初始化，拒绝重复检测目标板")
    logger.info(
        "开始 ChArUco 目标板检测与 offset 安全检查 active_arms={} attempts={}",
        [runtime.connected_arm.arm_side for runtime in runtimes],
        CHARUCO_OFFSET_SAFETY_ATTEMPT_COUNT,
    )
    rejected_attempts: list[str] = []
    for attempt in range(1, CHARUCO_OFFSET_SAFETY_ATTEMPT_COUNT + 1):
        current_camera_board_m = _detect_current_camera_board_transform_m(
            runtimes[0].body_channel
        )
        rejection_reasons: list[str] = []
        for runtime in runtimes:
            accepted, decision_reason = _precheck_charuco_cartesian_offset(
                runtime,
                current_camera_board_m,
            )
            if not accepted:
                rejection_reasons.append(decision_reason)
        if rejection_reasons:
            attempt_reason = (
                f"attempt={attempt}/{CHARUCO_OFFSET_SAFETY_ATTEMPT_COUNT} "
                f"rejections={' | '.join(rejection_reasons)}"
            )
            rejected_attempts.append(attempt_reason)
            if attempt < CHARUCO_OFFSET_SAFETY_ATTEMPT_COUNT:
                logger.warning(
                    "ChArUco offset 安全检查未通过，等待后重新检测 {} delay={:.1f}s",
                    attempt_reason,
                    CHARUCO_OFFSET_SAFETY_RETRY_DELAY_S,
                )
                time.sleep(CHARUCO_OFFSET_SAFETY_RETRY_DELAY_S)
                continue
            raise RuntimeError(
                "ChArUco offset 连续安全检查均被拒绝："
                f"attempts={CHARUCO_OFFSET_SAFETY_ATTEMPT_COUNT}; "
                f"details={'; '.join(rejected_attempts)}"
            )

        for runtime in runtimes:
            _calculate_charuco_cartesian_offset(
                runtime,
                current_camera_board_m,
                "replay-cycle-initialization",
            )
        logger.success(
            "本轮 ChArUco offset 已完成初始化 attempt={}/{}，后续 CSV 仅使用缓存结果",
            attempt,
            CHARUCO_OFFSET_SAFETY_ATTEMPT_COUNT,
        )
        return

    raise RuntimeError("ChArUco offset 安全检查尝试流程意外结束")


def _create_offset_record(
    charuco_offset: tuple[tuple[float, float, float, float], ...],
) -> Path:
    charuco_matrix_m = np.asarray(charuco_offset, dtype=np.float64)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    duplicate_index = 1
    while True:
        duplicate_marker = "" if duplicate_index == 1 else f"_{duplicate_index:02d}"
        record_path = PROJECT_ROOT / f"{OFFSET_RECORD_NAME_PREFIX}{duplicate_marker}_{timestamp}.txt"
        try:
            with record_path.open("x", encoding="utf-8", newline="\n") as record_file:
                record_file.write(
                    f'{_format_matrix_xyzrpy_mm_deg("T_charuco_off", charuco_matrix_m)}\n'
                )
        except FileExistsError:
            duplicate_index += 1
            continue
        logger.success("ChArUco offset 已立即写入 path={}", record_path)
        return record_path


def _append_three_ball_offset_and_delta(
    runtime: ReplayRuntime,
    three_ball_offset: tuple[tuple[float, float, float, float], ...],
    charuco_offset: tuple[tuple[float, float, float, float], ...],
) -> None:
    three_ball_matrix_m = np.asarray(three_ball_offset, dtype=np.float64)
    charuco_matrix_m = np.asarray(charuco_offset, dtype=np.float64)
    delta_matrix_m = charuco_matrix_m @ np.linalg.inv(three_ball_matrix_m)
    record_path = runtime.offset_record_path
    if record_path is None:
        record_path = _create_offset_record(charuco_offset)
        runtime.offset_record_path = record_path
    with record_path.open("a", encoding="utf-8", newline="\n") as record_file:
        record_file.write(
            f'{_format_matrix_xyzrpy_mm_deg("T_three_ball_off", three_ball_matrix_m)}\n'
        )
        record_file.write(f'{_format_matrix_xyzrpy_mm_deg("T_delta", delta_matrix_m)}\n')
    logger.info(
        "THREE_BALL_OFFSET_READY {}",
        _format_matrix_xyzrpy_mm_deg("T_three_ball_off", three_ball_matrix_m),
    )
    logger.info(
        "OFFSET_COMPARE T_delta=T_charuco_off@inv(T_three_ball_off) {}",
        _format_matrix_xyzrpy_mm_deg("T_delta", delta_matrix_m),
    )
    logger.success("三球 offset 与 delta 已追加写入 path={}", record_path)


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    return _rad_to_deg(list(_homogeneous_matrix_to_rpy(matrix.tolist())))


def _charuco_offset_xyzrpy_mm_deg(matrix: np.ndarray) -> np.ndarray:
    """将 ChArUco offset 齐次矩阵转换为统计使用的 xyz(mm)+rpy(deg)。"""

    matrix_np = np.asarray(matrix, dtype=np.float64)
    if matrix_np.shape != (4, 4) or not np.all(np.isfinite(matrix_np)):
        raise ValueError("ChArUco offset 必须是有限的 4x4 齐次矩阵")
    xyz_mm = matrix_np[:3, 3] * 1000.0
    rpy_deg = np.asarray(_rotation_matrix_to_rpy_deg(matrix_np[:3, :3]), dtype=np.float64)
    return np.concatenate((xyz_mm, rpy_deg))


def _format_matrix_xyzrpy_mm_deg(name: str, matrix: np.ndarray) -> str:
    matrix_np = np.asarray(matrix, dtype=np.float64)
    xyz_mm = [
        float(matrix_np[0, 3]) * 1000.0,
        float(matrix_np[1, 3]) * 1000.0,
        float(matrix_np[2, 3]) * 1000.0,
    ]
    rpy_deg = _rad_to_deg(list(_homogeneous_matrix_to_rpy(matrix_np)))
    return f"{name} xyzrpy(mm,deg)=[{_format_sequence(xyz_mm + rpy_deg)}]"


# endregion


# region 设备动作


def _execute_gripper_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    if runtime.gripper is None:
        raise RuntimeError("当前 runtime 未配置左手夹爪客户端")
    gripper = runtime.gripper
    if row.pose_value is None:
        raise RuntimeError(f"gripper pose 未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    target_value = int(round(row.pose_value))
    if not _retry_non_motion_call(
        f"gripper.set_pos({row.csv_name}:{row.row_index})",
        lambda: gripper.set_pos(target_value),
    ):
        raise RuntimeError("夹爪 set_pos 下发失败")
    logger.info("已下发夹爪目标 file={} row={} pos={}，当前策略不等待到位", row.csv_name, row.row_index, target_value)
    deadline_hint = _retry_non_motion_call(
        f"gripper.get_status({row.csv_name}:{row.row_index})",
        lambda: gripper.get_status(),
    )
    logger.info("夹爪当前状态 pos={} calibrated={}", deadline_hint.position, bool(deadline_hint.calibrated))


def _get_right_hand_positions(runtime: ReplayRuntime) -> list[float]:
    if runtime.right_hand is None:
        raise RuntimeError("当前 runtime 未配置右手 m11 客户端")
    right_hand = runtime.right_hand
    required_count = (
        max(
            *RIGHT_HAND_M11_ROOT_ACTUATOR_IDS,
            *RIGHT_HAND_M11_TIP_ACTUATOR_IDS,
        )
        + 1
    )
    deadline = time.monotonic() + M11_STATE_READ_TIMEOUT_S
    read_index = 0
    last_invalid_reason = "尚未读取"
    while True:
        read_index += 1
        positions: list[float] = []
        try:
            state = _retry_non_motion_call(
                f"right_hand.get_hand_state({runtime.connected_arm.arm_side})",
                lambda: right_hand.get_hand_state(include_tactile=False),
            )
        except Exception as exc:
            last_invalid_reason = f"RPC 失败：{exc}"
        else:
            if state is None:
                last_invalid_reason = "state=None"
            elif not isinstance(state, dict):
                last_invalid_reason = f"state 不是 dict actual={type(state).__name__}"
            else:
                actuators = state.get("actuators")
                if not isinstance(actuators, list):
                    last_invalid_reason = "actuators 不是 list"
                elif len(actuators) < required_count:
                    last_invalid_reason = f"执行器数量不足 required={required_count} actual={len(actuators)}"
                else:
                    for index, actuator in enumerate(actuators):
                        if not isinstance(actuator, dict):
                            last_invalid_reason = f"actuators[{index}] 不是 dict"
                            positions = []
                            break
                        position = actuator.get("position")
                        if not isinstance(position, int | float):
                            last_invalid_reason = f"actuators[{index}].position 非数值"
                            positions = []
                            break
                        positions.append(float(position))
                    if len(positions) >= required_count:
                        if read_index > 1:
                            logger.success(
                                "右手状态重试后恢复 read={} actuator_count={}",
                                read_index,
                                len(positions),
                            )
                        return positions
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f"读取有效右手状态超时 timeout={M11_STATE_READ_TIMEOUT_S:.1f} s "
                f"read={read_index} last_reason={last_invalid_reason}"
            )
        logger.warning(
            "右手状态无效，等待后重读 read={} reason={} remaining={:.1f} s",
            read_index,
            last_invalid_reason,
            remaining_s,
        )
        time.sleep(min(M11_STATE_READ_POLL_INTERVAL_S, remaining_s))


def _execute_m11_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    if runtime.right_hand is None:
        raise RuntimeError("当前 runtime 未配置右手 m11 客户端")
    right_hand = runtime.right_hand
    if row.joint_values is None:
        raise RuntimeError(f"m11 joints 未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    target_positions = list(row.joint_values)
    current_positions = _get_right_hand_positions(runtime)
    for actuator_id, target_value in enumerate(target_positions):
        current_positions[actuator_id] = float(target_value)
    if not _retry_non_motion_call(
        f"right_hand.set_hand_state({row.csv_name}:{row.row_index})",
        lambda: right_hand.set_hand_state(current_positions),
    ):
        raise RuntimeError("右手 m11 下发失败")
    logger.info(
        "已下发右手 m11 目标 file={} row={} root=[{}] tip=[{}]",
        row.csv_name,
        row.row_index,
        _format_sequence(
            [
                current_positions[actuator_id]
                for actuator_id in RIGHT_HAND_M11_ROOT_ACTUATOR_IDS
            ],
            decimals=4,
        ),
        _format_sequence(
            [
                current_positions[actuator_id]
                for actuator_id in RIGHT_HAND_M11_TIP_ACTUATOR_IDS
            ],
            decimals=4,
        ),
    )


def _read_lift_height_mm(result: object) -> float:
    if isinstance(result, tuple) and len(result) == 2:
        first_value = result[0]
        if isinstance(first_value, int | float):
            return float(first_value)
        raise TypeError(f"lift 返回值首元素类型无效：{type(first_value)!r}")
    if isinstance(result, int | float):
        return float(result)
    raise TypeError(f"lift 返回值类型无效：{type(result)!r}")


def _wait_replay_lift_until_near_target(body: WujiBodyClient, target_height_mm: int) -> float:
    lift = body.lift
    deadline = time.monotonic() + DEFAULT_REPLAY_LIFT_TIMEOUT_S
    command_attempt = 0
    valid_read_index = 0
    invalid_read_count = 0
    last_logged_height_mm: float | None = None
    last_height_mm = -1.0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"等待 lift 到位超时 target={target_height_mm} mm actual={last_height_mm:.1f} mm "
                f"timeout={DEFAULT_REPLAY_LIFT_TIMEOUT_S:.1f} s commands={command_attempt}"
            )
        command_attempt += 1
        try:
            command_result = _retry_non_motion_call(
                "lift.set_lift_physical_height(wait)",
                lambda: lift.set_lift_physical_height(target_height_mm),
            )
        except Exception as exc:
            command_result = f"调用异常：{exc}"
        logger.info(
            "lift 脉冲已下发，等待 {:.1f} s 后检测高度 target={} mm attempt={} command_return={}",
            DEFAULT_REPLAY_LIFT_PULSE_INTERVAL_S,
            target_height_mm,
            command_attempt,
            command_result,
        )
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f"等待 lift 到位超时 target={target_height_mm} mm actual={last_height_mm:.1f} mm "
                f"timeout={DEFAULT_REPLAY_LIFT_TIMEOUT_S:.1f} s commands={command_attempt}"
            )
        time.sleep(min(DEFAULT_REPLAY_LIFT_PULSE_INTERVAL_S, remaining_s))
        while True:
            current_height_mm = _read_lift_height_mm(
                _retry_non_motion_call(
                    "lift.get_lift_physical_height(wait)",
                    lift.get_lift_physical_height,
                )
            )
            last_height_mm = current_height_mm
            if current_height_mm < 0.0:
                invalid_read_count += 1
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"等待 lift 有效高度超时 target={target_height_mm} mm "
                        f"timeout={DEFAULT_REPLAY_LIFT_TIMEOUT_S:.1f} s "
                        f"commands={command_attempt} last={current_height_mm:.1f} mm"
                    )
                if invalid_read_count == 1 or invalid_read_count % 20 == 0:
                    logger.warning(
                        "lift 返回无效高度，判定为通信失败并立即重读 "
                        "invalid_read={} value={:.1f} mm",
                        invalid_read_count,
                        current_height_mm,
                    )
                continue
            valid_read_index += 1
            current_error_mm = abs(current_height_mm - float(target_height_mm))
            if last_logged_height_mm is None or abs(current_height_mm - last_logged_height_mm) >= 0.5:
                last_logged_height_mm = current_height_mm
                logger.info(
                    "lift 到位检查 valid_read={}: target={} mm actual={:.1f} mm error={:.1f} mm",
                    valid_read_index,
                    target_height_mm,
                    current_height_mm,
                    current_error_mm,
                )
            if current_error_mm <= DEFAULT_REPLAY_LIFT_HEIGHT_TOLERANCE_MM:
                logger.success(
                    "lift 已到位 target={} mm actual={:.1f} mm",
                    target_height_mm,
                    current_height_mm,
                )
                return current_height_mm
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待 lift 到位超时 target={target_height_mm} mm "
                    f"actual={current_height_mm:.1f} mm error={current_error_mm:.1f} mm "
                    f"timeout={DEFAULT_REPLAY_LIFT_TIMEOUT_S:.1f} s commands={command_attempt}"
                )
            break


def _execute_lift_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    if row.pose_value is None:
        raise RuntimeError(f"lift pose 未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    target_height_mm = int(round(row.pose_value))
    if target_height_mm < 0:
        raise ValueError(f"lift 目标高度非法：{target_height_mm}")
    _ensure_lift_enabled(runtime.body, f"lift.set_enable({row.csv_name}:{row.row_index})")
    actual_height_mm = _wait_replay_lift_until_near_target(runtime.body, target_height_mm)
    logger.info(
        "lift 已执行 file={} row={} target={} mm actual={:.1f} mm",
        row.csv_name,
        row.row_index,
        target_height_mm,
        actual_height_mm,
    )


def _execute_arm_move_abs_j_segment(
    runtime: ReplayRuntime,
    rows: list[ReplayRow],
) -> None:
    if not rows:
        return
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    targets = [_build_move_abs_j_target(runtime, row) for row in rows]
    commands: list[xCoreSDK_python.MoveAbsJCommand] = []
    for target_index, target in enumerate(targets):
        is_segment_end = target_index == len(targets) - 1
        end_linear_speed_mm_s = _resolve_replay_move_abs_j_end_linear_speed_mm_s(
            runtime,
            target.row.csv_name,
        )
        zone = 0.0 if is_segment_end else _resolve_replay_move_abs_j_zone_mm(runtime, target.row.csv_name)
        commands.append(
            xCoreSDK_python.MoveAbsJCommand(
                target.joint,
                end_linear_speed_mm_s,
                zone,
            )
        )
        logger.info(
            "已生成连续 MoveAbsJ 目标 file={} row={} source={} end_linear_speed_mm_s={:.2f} "
            "zone_mm={:.3f} joints(deg)=[{}]",
            target.row.csv_name,
            target.row.row_index,
            target.source,
            end_linear_speed_mm_s,
            zone,
            _format_sequence(target.joint.joints),
        )

    first_row = rows[0]
    last_row = rows[-1]
    cmd_id = xCoreSDK_python.PyString()
    _wait_until_reset_ready(
        robot,
        ec,
        f"moveReset(replay-arm-segment:{runtime.connected_arm.arm_side}:{first_row.csv_name}:{first_row.row_index}-{last_row.row_index})",
    )
    _retry_non_motion_call(
        f"moveReset(replay-arm-segment:{first_row.csv_name}:{first_row.row_index}-{last_row.row_index})",
        lambda: robot.moveReset(ec),
    )
    _print_sdk_result("moveReset(replay-arm-segment)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(
            "回放 arm 连续段 moveReset 失败 "
            f"arm_side={runtime.connected_arm.arm_side} file={first_row.csv_name} "
            f"rows={first_row.row_index}-{last_row.row_index} {_format_robot_runtime_state(robot, ec)}"
        )
    robot.moveAppend(commands, cmd_id, ec)
    _print_sdk_result("moveAppend(MoveAbsJ-segment)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放 arm 连续段 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(replay-arm-segment)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放 arm 连续段 moveStart 失败")
    logger.info(
        "已下发 arm 连续 MoveAbsJ 段 file={} rows={}-{} count={} cmd_id={}",
        first_row.csv_name,
        first_row.row_index,
        last_row.row_index,
        len(commands),
        cmd_id.content(),
    )
    _wait_until_motion_finished(robot, ec, "等待回放 arm 连续 MoveAbsJ 段")


def _flush_pending_arm_segment(runtime: ReplayRuntime) -> None:
    if not runtime.pending_arm_rows:
        return
    rows = runtime.pending_arm_rows
    runtime.pending_arm_rows = []
    _execute_arm_move_abs_j_segment(runtime, rows)


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
        runtime.pending_arm_rows.append(row)
        _flush_pending_arm_segment(runtime)
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
    raise ValueError(f"当前脚本暂不支持的记录类型：{row.action_type}")


def _cleanup_runtime(runtime: ReplayRuntime | None) -> None:
    if runtime is None:
        return
    _shutdown_robot(runtime.connected_arm.robot, runtime.connected_arm.ec)
    del runtime
    gc.collect()


# endregion


# region 自动回放编排


def _run_rows(runtime: ReplayRuntime, rows: tuple[ReplayRow, ...]) -> None:
    for row in rows:
        if runtime.stop_event is not None and runtime.stop_event.is_set():
            raise RuntimeError("检测到并行执行已请求停止，终止当前 CSV 后续动作")
        if row.action_type == "arm":
            runtime.pending_arm_rows.append(row)
            continue
        _flush_pending_arm_segment(runtime)
        _execute_row(runtime, row)


def _execute_single_csv(
    runtime: ReplayRuntime,
    csv_path: Path,
    flush_at_end: bool = False,
) -> None:
    rows = runtime.preloaded_rows_by_path.get(csv_path)
    if rows is None:
        raise RuntimeError(f"CSV 未在启动阶段预加载，拒绝在执行期读取文件：{csv_path}")
    if not rows:
        if flush_at_end:
            _flush_pending_arm_segment(runtime)
        logger.warning(
            "CSV 是零字节或只有表头的占位文件，跳过执行 arm_side={} file={}",
            runtime.connected_arm.arm_side,
            csv_path.name,
        )
        return
    is_offset_trigger_csv = _should_trigger_offset_calculation(runtime, csv_path.name)
    logger.info(
        "开始执行文件 {} arm_side={} 共 {} 行",
        csv_path.name,
        runtime.connected_arm.arm_side,
        len(rows),
    )
    if is_offset_trigger_csv:
        _flush_pending_arm_segment(runtime)
    _run_rows(runtime, rows)
    if is_offset_trigger_csv or flush_at_end:
        _flush_pending_arm_segment(runtime)
    if is_offset_trigger_csv:
        logger.info(
            "已到达 offset 触发 CSV，等待 {:.1f}s 后开始连续采集三球坐标 file={}",
            OFFSET_CAPTURE_SETTLE_DELAY_S,
            csv_path.name,
        )
        time.sleep(OFFSET_CAPTURE_SETTLE_DELAY_S)
        three_ball_offset = _calculate_global_cartesian_offset(runtime, csv_path)
        runtime.global_cartesian_offset = three_ball_offset
        if runtime.charuco_cartesian_offset is not None:
            _append_three_ball_offset_and_delta(
                runtime,
                three_ball_offset,
                runtime.charuco_cartesian_offset,
            )
    logger.success(
        "文件已加入执行流 {} arm_side={} pending_arm_rows={}",
        csv_path.name,
        runtime.connected_arm.arm_side,
        len(runtime.pending_arm_rows),
    )


def _execute_parallel_csv_pair(
    left_runtime: ReplayRuntime,
    left_csv_path: Path,
    right_runtime: ReplayRuntime,
    right_csv_path: Path,
    phase_label: str,
    stop_event: threading.Event,
) -> None:
    logger.info("开始并行执行 phase={} left={} right={}", phase_label, left_csv_path.name, right_csv_path.name)
    _flush_pending_arm_segment(left_runtime)
    _flush_pending_arm_segment(right_runtime)
    errors: list[BaseException] = []

    def _worker(runtime: ReplayRuntime, csv_path: Path) -> None:
        try:
            _execute_single_csv(
                runtime,
                csv_path,
                flush_at_end=True,
            )
        except BaseException as exc:  # noqa: BLE001
            stop_event.set()
            logger.exception(
                "并行执行线程失败 arm_side={} file={} phase={} exc={}",
                runtime.connected_arm.arm_side,
                csv_path.name,
                phase_label,
                exc,
            )
            errors.append(exc)

    left_thread = threading.Thread(
        target=_worker,
        args=(left_runtime, left_csv_path),
        name=f"left-sync-{left_csv_path.stem}",
        daemon=False,
    )
    right_thread = threading.Thread(
        target=_worker,
        args=(right_runtime, right_csv_path),
        name=f"right-sync-{right_csv_path.stem}",
        daemon=False,
    )
    left_thread.start()
    right_thread.start()
    left_thread.join()
    right_thread.join()
    if errors:
        raise RuntimeError(
            f"并行执行失败 phase={phase_label} left={left_csv_path.name} right={right_csv_path.name}: {errors[0]}"
        ) from errors[0]
    logger.success("并行执行完成 phase={} left={} right={}", phase_label, left_csv_path.name, right_csv_path.name)


def _execute_csv_execution_plan(
    left_runtime: ReplayRuntime,
    plan: CsvExecutionPlan,
    right_runtime: ReplayRuntime | None,
    stop_event: threading.Event,
) -> None:
    left_executed = False
    if plan.start_together:
        if right_runtime is None:
            raise RuntimeError(f"启动并行阶段缺少右臂 runtime：{plan.left_csv_path.name}")
        if plan.right_start_csv_path is None:
            raise RuntimeError(f"启动并行阶段缺少右臂首个同序号 CSV：{plan.left_csv_path.name}")
        _execute_parallel_csv_pair(
            left_runtime=left_runtime,
            left_csv_path=plan.left_csv_path,
            right_runtime=right_runtime,
            right_csv_path=plan.right_start_csv_path,
            phase_label="bootstrap",
            stop_event=stop_event,
        )
        left_executed = True
    if plan.right_pre_stage_csv_paths:
        _flush_pending_arm_segment(left_runtime)
    for right_csv_path in plan.right_pre_stage_csv_paths:
        if right_runtime is None:
            raise RuntimeError(f"右臂阶段执行缺少 runtime：{right_csv_path.name}")
        if stop_event.is_set():
            raise RuntimeError("检测到并行执行已请求停止，终止右臂后续阶段")
        _execute_single_csv(right_runtime, right_csv_path)
    if plan.right_sync_csv_path is not None:
        if right_runtime is None:
            raise RuntimeError(f"同步阶段缺少右臂 runtime：{plan.right_sync_csv_path.name}")
        _execute_parallel_csv_pair(
            left_runtime=left_runtime,
            left_csv_path=plan.left_csv_path,
            right_runtime=right_runtime,
            right_csv_path=plan.right_sync_csv_path,
            phase_label="sync",
            stop_event=stop_event,
        )
        left_executed = True
    if not left_executed:
        if stop_event.is_set():
            raise RuntimeError("检测到并行执行已请求停止，终止左臂后续阶段")
        if right_runtime is not None:
            _flush_pending_arm_segment(right_runtime)
        _execute_single_csv(
            left_runtime,
            plan.left_csv_path,
        )
    if right_runtime is None:
        return
    if plan.right_post_stage_csv_paths:
        _flush_pending_arm_segment(left_runtime)
    for right_csv_path in plan.right_post_stage_csv_paths:
        if stop_event.is_set():
            raise RuntimeError("检测到并行执行已请求停止，终止右臂后续阶段")
        _execute_single_csv(right_runtime, right_csv_path)


# endregion


# region CLI


def _toggle_arm_side(current_arm_side: str) -> str:
    if current_arm_side == "left":
        return "right"
    return "left"


def _confirm_runtime_config(
    execution_mode: str,
    arm_side: str,
    enable_agv_navigation: bool,
    record_dir: Path,
    max_files: int | None,
    offset_service_addr: str,
    offset_camera_name: str,
    offset_prior_capture_path: Path,
    hand_eye_result_path: Path,
    move_abs_j_end_linear_speed_mm_s_by_arm_side: dict[str, float],
) -> str:
    while True:
        print("")
        print("========== 回放配置 ==========")
        print(f"当前执行模式：{execution_mode}")
        print(f"当前机械臂侧别：{arm_side}")
        print(f"是否移动 AGV：{'是' if enable_agv_navigation else '否'}，目标点={DEFAULT_AGV_POINT}")
        print(f"当前 CSV 目录：{record_dir}")
        print(f"当前最大文件数：{'全部' if max_files is None else max_files}")
        if execution_mode == "parallel":
            print(
                "当前 MoveAbsJ 默认末端线速度："
                f"left={move_abs_j_end_linear_speed_mm_s_by_arm_side['left']:.2f} mm/s，"
                f"right={move_abs_j_end_linear_speed_mm_s_by_arm_side['right']:.2f} mm/s"
            )
        else:
            print(
                f"当前 {arm_side} 臂 MoveAbsJ 默认末端线速度："
                f"{move_abs_j_end_linear_speed_mm_s_by_arm_side[arm_side]:.2f} mm/s"
            )
        print(f"当前 offset 服务：{offset_service_addr}")
        print(f"当前 offset 相机：{offset_camera_name}")
        print(f"当前 offset 先验：{offset_prior_capture_path}")
        print(f"当前手眼结果：{hand_eye_result_path}")
        print("输入回车确认配置并开始自动回放")
        print("输入 a 切换 AGV 导航")
        print("输入 j 切换为双臂并行模式")
        print("输入 l 切换左右臂单臂模式")
        print("输入 s 调整初始速度")
        print("输入 q 退出")
        choice = input("请选择：").strip().lower()
        if choice == "q":
            return "quit"
        if choice == "a":
            return "toggle-agv"
        if choice == "j":
            return "parallel"
        if choice == "l":
            return "toggle-arm"
        if choice == "s":
            return "speed"
        if choice == "":
            return "confirm"
        print(f"未知输入：{choice}")


def _print_csv_summary(
    csv_paths: list[Path],
    move_abs_j_end_linear_speed_mm_s_by_arm_side: dict[str, float],
) -> None:
    print("本次将按以下顺序执行 CSV：")
    for index, csv_path in enumerate(csv_paths, start=1):
        print(f"  {index:02d}. {csv_path.name}")
    print(
        "本轮 MoveAbsJ 默认末端线速度："
        f"left={move_abs_j_end_linear_speed_mm_s_by_arm_side['left']:.2f} mm/s，"
        f"right={move_abs_j_end_linear_speed_mm_s_by_arm_side['right']:.2f} mm/s"
    )
    print(
        "左臂 MoveAbsJ CSV 速度配置："
        f"{LEFT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE}"
    )
    print(
        "右臂 MoveAbsJ CSV 速度配置："
        f"{RIGHT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE}"
    )
    print(f"左臂 MoveAbsJ CSV zone 配置：{LEFT_REPLAY_MOVE_ABS_J_ZONE_MM_BY_CSV_SEQUENCE}")
    print(f"右臂 MoveAbsJ CSV zone 配置：{RIGHT_REPLAY_MOVE_ABS_J_ZONE_MM_BY_CSV_SEQUENCE}")
    print(
        "手部笛卡尔纠偏配置（临时使用头部 ChArUco offset）："
        f"calculate_at={_format_optional_csv_sequence(CSV_CARTESIAN_OFFSET_CALCULATE_AT)}, "
        f"targets={[f'{value:02d}' for value in CSV_CARTESIAN_OFFSET_TARGETS]}"
    )


def _print_execution_plan_summary(plans: list[CsvExecutionPlan]) -> None:
    if not plans:
        return
    print("双臂执行计划：")
    for index, plan in enumerate(plans, start=1):
        right_parts: list[str] = []
        if plan.right_start_csv_path is not None:
            right_parts.append(f"start={plan.right_start_csv_path.name}")
        if plan.right_pre_stage_csv_paths:
            right_parts.append("pre=[" + ", ".join(csv_path.name for csv_path in plan.right_pre_stage_csv_paths) + "]")
        if plan.right_sync_csv_path is not None:
            right_parts.append(f"sync={plan.right_sync_csv_path.name}")
        if plan.right_post_stage_csv_paths:
            right_parts.append(
                "post=[" + ", ".join(csv_path.name for csv_path in plan.right_post_stage_csv_paths) + "]"
            )
        right_plan_text = " ".join(right_parts) if right_parts else "无右臂阶段"
        print(f"  {index:02d}. left={plan.left_csv_path.name} -> {right_plan_text}")


def _prompt_end_linear_speed_mm_s(current_value: float) -> float:
    while True:
        raw_text = (
            input(
                f"请输入新的 MoveAbsJ 末端线速度，范围 "
                f"[{MOVE_ABS_J_MIN_END_LINEAR_SPEED_MM_S:.0f}, {MOVE_ABS_J_MAX_END_LINEAR_SPEED_MM_S:.0f}] mm/s,"
                f"当前 {current_value:.2f} mm/s，输入 q 返回："
            )
            .strip()
            .lower()
        )
        if raw_text == "q":
            return current_value
        try:
            new_value = float(raw_text)
        except ValueError:
            print("末端线速度输入无效")
            continue
        if not MOVE_ABS_J_MIN_END_LINEAR_SPEED_MM_S <= new_value <= MOVE_ABS_J_MAX_END_LINEAR_SPEED_MM_S:
            print(
                "末端线速度必须在 "
                f"[{MOVE_ABS_J_MIN_END_LINEAR_SPEED_MM_S:.0f}, "
                f"{MOVE_ABS_J_MAX_END_LINEAR_SPEED_MM_S:.0f}] mm/s 范围内"
            )
            continue
        return new_value


def _configure_end_linear_speed_value(arm_side: str, current_end_linear_speed_mm_s: float) -> float:
    print(f"当前 {arm_side} 臂 MoveAbsJ 默认末端线速度：{current_end_linear_speed_mm_s:.2f} mm/s")
    new_end_linear_speed_mm_s = _prompt_end_linear_speed_mm_s(current_end_linear_speed_mm_s)
    logger.info(
        "{} 臂初始 MoveAbsJ 末端线速度已更新 speed={:.2f} mm/s",
        arm_side,
        new_end_linear_speed_mm_s,
    )
    return new_end_linear_speed_mm_s


def _print_replay_summary(
    csv_paths: list[Path],
    move_abs_j_end_linear_speed_mm_s_by_arm_side: dict[str, float],
) -> None:
    _print_csv_summary(csv_paths, move_abs_j_end_linear_speed_mm_s_by_arm_side)
    arm_side = "right" if "record_right" in str(csv_paths[0].parent) else "left"
    print(f"{arm_side} 臂基坐标固定为 tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    print("arm 动作策略：pose=NaN 直接使用 joints；否则优先 pose IK，失败后以 joints 最终兜底；统一 MoveAbsJ")
    if arm_side == "left":
        print("gripper 动作策略：仅下发，不等待到位")
    else:
        print("m11 动作策略：读取当前 11 轴状态后整体下发，不等待到位")
    print(
        "lift 动作策略：等待到位后才允许继续下一步，"
        f"pulse={DEFAULT_REPLAY_LIFT_PULSE_INTERVAL_S:.1f}s "
        f"total_timeout={DEFAULT_REPLAY_LIFT_TIMEOUT_S:.1f}s "
        f"tolerance={DEFAULT_REPLAY_LIFT_HEIGHT_TOLERANCE_MM:.1f}mm"
    )


def main(
    arm_side: str = DEFAULT_ARM_SIDE,
    record_dir: Path = DEFAULT_RECORD_DIR,
    max_files: int | None = DEFAULT_MAX_FILES,
    enable_agv_navigation: bool = DEFAULT_ENABLE_AGV_NAVIGATION,
    offset_service_addr: str = DEFAULT_OFFSET_SERVICE_ADDR,
    offset_camera_name: str = DEFAULT_OFFSET_CAMERA_NAME,
    offset_prior_capture_path: Path = DEFAULT_OFFSET_PRIOR_CAPTURE_PATH,
    hand_eye_result_path: Path = DEFAULT_HAND_EYE_RESULT_PATH,
) -> int:
    selected_execution_mode = DEFAULT_EXECUTION_MODE
    selected_arm_side = str(arm_side)
    selected_record_dir = Path(record_dir)
    selected_move_abs_j_end_linear_speed_mm_s_by_arm_side = {
        "left": LEFT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE[-1],
        "right": RIGHT_REPLAY_MOVE_ABS_J_END_LINEAR_SPEED_MM_S_BY_CSV_SEQUENCE[-1],
    }
    selected_enable_agv_navigation = bool(enable_agv_navigation)
    while True:
        resolved_record_dir = _resolve_record_dir(selected_arm_side, selected_record_dir)
        config_choice = _confirm_runtime_config(
            execution_mode=selected_execution_mode,
            arm_side=selected_arm_side,
            enable_agv_navigation=selected_enable_agv_navigation,
            record_dir=resolved_record_dir,
            max_files=max_files,
            offset_service_addr=offset_service_addr,
            offset_camera_name=offset_camera_name,
            offset_prior_capture_path=offset_prior_capture_path,
            hand_eye_result_path=hand_eye_result_path,
            move_abs_j_end_linear_speed_mm_s_by_arm_side=(
                selected_move_abs_j_end_linear_speed_mm_s_by_arm_side
            ),
        )
        if config_choice == "quit":
            logger.info("用户在配置阶段取消执行")
            return 0
        if config_choice == "toggle-agv":
            selected_enable_agv_navigation = not selected_enable_agv_navigation
            agv_state_text = "开启，将移动 AGV" if selected_enable_agv_navigation else "关闭，不移动 AGV"
            print(f"AGV 导航已切换：{agv_state_text}")
            logger.info("AGV 导航配置已切换 enable={}", selected_enable_agv_navigation)
            continue
        if config_choice == "parallel":
            selected_execution_mode = "parallel"
            selected_arm_side = "left"
            selected_record_dir = DEFAULT_RECORD_DIR
            continue
        if config_choice == "toggle-arm":
            selected_execution_mode = "single"
            selected_arm_side = _toggle_arm_side(selected_arm_side)
            selected_record_dir = DEFAULT_RECORD_DIR
            continue
        if config_choice == "speed":
            configured_arm_sides = (
                ("left", "right")
                if selected_execution_mode == "parallel"
                else (selected_arm_side,)
            )
            for configured_arm_side in configured_arm_sides:
                selected_move_abs_j_end_linear_speed_mm_s_by_arm_side[configured_arm_side] = (
                    _configure_end_linear_speed_value(
                        configured_arm_side,
                        selected_move_abs_j_end_linear_speed_mm_s_by_arm_side[configured_arm_side],
                    )
                )
            continue
        break
    logger.info("拖动示教自动回放 CLI 启动 arm_side={} record_dir={}", selected_arm_side, resolved_record_dir)
    csv_paths = _discover_csv_paths(resolved_record_dir, max_files)
    if not csv_paths:
        raise RuntimeError(f"没有在目录中发现 CSV: {record_dir}")
    execution_plans = [CsvExecutionPlan(left_csv_path=csv_path) for csv_path in csv_paths]
    if selected_execution_mode == "parallel":
        has_sync_csv = any(_extract_sync_csv_sequence(csv_path.name) is not None for csv_path in csv_paths)
        if has_sync_csv:
            right_csv_paths = _discover_csv_paths(DEFAULT_RIGHT_RECORD_DIR, max_files=None)
            execution_plans = _build_csv_execution_plans(csv_paths, right_csv_paths)
    execution_csv_paths = _collect_execution_csv_paths(execution_plans)
    preloaded_rows_by_path = _preload_replay_rows(execution_csv_paths)
    _print_replay_summary(
        csv_paths,
        selected_move_abs_j_end_linear_speed_mm_s_by_arm_side,
    )
    _print_execution_plan_summary(execution_plans)

    runtime: ReplayRuntime | None = None
    right_runtime: ReplayRuntime | None = None
    tunnel_group: ReplaySharedTunnelGroup | None = None
    stop_event = threading.Event()
    try:
        tunnel_group = _create_replay_shared_tunnel_group()
        if selected_enable_agv_navigation:
            _navigate_agv_before_replay(tunnel_group)
        else:
            logger.info("AGV 导航已关闭，直接开始自动回放")
        runtime = _create_runtime(selected_arm_side, tunnel_group)
        runtime.stop_event = stop_event
        runtime.preloaded_rows_by_path = preloaded_rows_by_path
        runtime.offset_service_addr = str(offset_service_addr)
        runtime.offset_camera_name = str(offset_camera_name)
        runtime.offset_prior_capture_path = Path(offset_prior_capture_path)
        runtime.hand_eye_result_path = Path(hand_eye_result_path)
        if runtime.gripper is not None:
            _prepare_gripper_before_replay(runtime.gripper)
        _prepare_runtime(runtime)
        if selected_execution_mode == "parallel" and any(
            plan.right_start_csv_path is not None
            or plan.right_pre_stage_csv_paths
            or plan.right_sync_csv_path is not None
            or plan.right_post_stage_csv_paths
            for plan in execution_plans
        ):
            right_runtime = _create_runtime("right", tunnel_group)
            right_runtime.stop_event = stop_event
            right_runtime.preloaded_rows_by_path = preloaded_rows_by_path
            _prepare_runtime(right_runtime)
        runtimes = [runtime] if right_runtime is None else [runtime, right_runtime]
        for configured_runtime in runtimes:
            configured_runtime.move_abs_j_end_linear_speed_mm_s = (
                selected_move_abs_j_end_linear_speed_mm_s_by_arm_side[
                    configured_runtime.connected_arm.arm_side
                ]
            )
        _initialize_charuco_cartesian_offsets(runtimes)
        for plan in execution_plans:
            _execute_csv_execution_plan(
                runtime,
                plan,
                right_runtime,
                stop_event,
            )
        _flush_pending_arm_segment(runtime)
        if right_runtime is not None:
            _flush_pending_arm_segment(right_runtime)
        logger.success("全部 CSV 执行完成")
        return 0
    finally:
        _cleanup_runtime(right_runtime)
        _cleanup_runtime(runtime)
        if tunnel_group is not None:
            tunnel_group.close()


# endregion


if __name__ == "__main__":
    raise SystemExit(main())
