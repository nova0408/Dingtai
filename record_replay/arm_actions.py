"""机械臂回放目标构建与连续 MoveAbsJ 执行。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from sdk.xcoresdk import xCoreSDK_python

from .arm_gateway import retry_non_motion_call
from .contracts import ReplayRow
from .runtime import ReplayRuntime

# region 数据结构
@dataclass(frozen=True, slots=True)
class ArmMoveTarget:
    """一条 arm 行最终使用的关节目标。"""

    row: ReplayRow
    "源 CSV 动作行。"
    joint: xCoreSDK_python.JointPosition
    "目标关节，单位 rad。"
    source: str
    "目标来源，csv-joints、tcp-ik 或 csv-joints-fallback。"


# endregion


# region 目标构建


def build_arm_target(runtime: ReplayRuntime, row: ReplayRow) -> ArmMoveTarget:
    """构建 CSV joints 或 pose IK 对应的关节目标。"""

    if row.arm_pose is None:
        if row.arm_joint_rad is None:
            raise RuntimeError(f"arm joints 未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
        return ArmMoveTarget(
            row,
            xCoreSDK_python.JointPosition(list(row.arm_joint_rad)),
            "csv-joints",
        )
    pose = row.arm_pose
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    current_pose = retry_non_motion_call(
        f"cartPosture(endInRef:{row.csv_name}:{row.row_index})",
        lambda: robot.cartPosture(xCoreSDK_python.endInRef, ec),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )
    if ec.get("ec", 0) != 0:
        raise RuntimeError("读取当前笛卡尔位姿失败")
    target_pose = xCoreSDK_python.CartesianPosition(_mm_to_m(list(pose.xyz_mm)) + _deg_to_rad(list(pose.rpy_deg)))
    target_pose.confData = list(current_pose.confData)
    target_pose.hasElbow = current_pose.hasElbow
    target_pose.elbow = current_pose.elbow
    if pose.has_elbow is not None:
        target_pose.hasElbow = pose.has_elbow
    if pose.elbow_deg is not None:
        target_pose.elbow = _deg_to_rad([pose.elbow_deg])[0]
    if pose.conf_data is not None:
        target_pose.confData = list(pose.conf_data)
    offset_matrix, offset_source = _resolve_cartesian_offset(runtime, row.csv_name)
    applies_offset = offset_matrix is not None
    if offset_matrix is not None:
        target_pose = _apply_global_offset(target_pose, offset_matrix)
    toolset = retry_non_motion_call(
        f"toolset(replay-arm-ik:{row.csv_name}:{row.row_index})",
        lambda: robot.toolset(ec),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放 arm 读取 toolset 失败")
    result = robot.model().calcIk(target_pose, toolset, ec)
    if ec.get("ec", 0) == 0:
        joint = _normalize_ik_result(result)
        source = "tcp-ik" if not applies_offset else f"tcp-ik-offset-{offset_source}"
        return ArmMoveTarget(row, joint, source)
    if row.arm_joint_rad is None:
        raise RuntimeError(f"CSV joints 兜底未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    ec["ec"] = 0
    fallback_reason = "offset 后 T_new_tcp 逆解失败，本行未实施 offset" if applies_offset else "原始 TCP 逆解失败"
    ec["message"] = f"{fallback_reason}，已改用 CSV 记录关节值"
    logger.warning(
        "{}，最终兜底为 CSV joints MoveAbsJ file={} row={}",
        fallback_reason,
        row.csv_name,
        row.row_index,
    )
    source = "csv-joints-fallback" if not applies_offset else "csv-joints-fallback-offset-ik"
    return ArmMoveTarget(row, xCoreSDK_python.JointPosition(list(row.arm_joint_rad)), source)


# endregion


# region 连续执行


def execute_arm_segment(runtime: ReplayRuntime, rows: list[ReplayRow]) -> None:
    """将连续 arm 行合并为一次 MoveAbsJ append/start。"""

    if not rows:
        return
    targets = [build_arm_target(runtime, row) for row in rows]
    commands = []
    for index, target in enumerate(targets):
        segment_end = index == len(targets) - 1
        csv_end = segment_end or targets[index + 1].row.csv_name != target.row.csv_name
        zone = 0.0 if csv_end else _resolve_replay_move_abs_j_zone_mm(runtime, target.row.csv_name)
        speed = _resolve_replay_move_abs_j_end_linear_speed_mm_s(runtime, target.row.csv_name)
        commands.append(xCoreSDK_python.MoveAbsJCommand(target.joint, speed, zone))
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    _wait_until_reset_ready(robot, ec, runtime, rows[0], rows[-1])
    retry_non_motion_call(
        f"moveReset(replay-arm-segment:{runtime.connected_arm.arm_side}:{rows[0].csv_name}:{rows[0].row_index}-{rows[-1].row_index})",
        lambda: robot.moveReset(ec),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放 arm 连续段 moveReset 失败")
    command_id = xCoreSDK_python.PyString()
    robot.moveAppend(commands, command_id, ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放 arm 连续段 moveAppend 失败")
    robot.moveStart(ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("回放 arm 连续段 moveStart 失败")
    _wait_until_motion_finished(robot, ec, runtime.settings.arm.motion_state_poll_interval_s)


def flush_pending_arm_segment(runtime: ReplayRuntime) -> None:
    """执行并清空 runtime 中积累的连续 arm 行。"""

    if not runtime.pending_arm_rows:
        return
    rows = list(runtime.pending_arm_rows)
    runtime.pending_arm_rows.clear()
    execute_arm_segment(runtime, rows)


def _wait_until_motion_finished(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    poll_interval_s: float,
) -> None:
    """轮询控制器直到运动进入 idle 或 unknown。"""

    while True:
        time.sleep(poll_interval_s)
        state = robot.operationState(ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"查询运动状态失败：{ec}")
        if state in (xCoreSDK_python.OperationState.idle, xCoreSDK_python.OperationState.unknown):
            return


def _wait_until_reset_ready(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    runtime: ReplayRuntime,
    first_row: ReplayRow,
    last_row: ReplayRow,
) -> None:
    """按旧回放语义等待连续段 reset 可用，超时后仍尝试 reset。"""

    arm_settings = runtime.settings.arm
    deadline = time.time() + arm_settings.reset_ready_timeout_s
    idle_count = 0
    last_state_text = ""
    while time.time() < deadline:
        operation_state = robot.operationState(ec)
        operate_mode = robot.operateMode(ec)
        power_state = robot.powerState(ec)
        last_state_text = f"operate_mode={operate_mode} operation_state={operation_state} power_state={power_state}"
        if operation_state == xCoreSDK_python.OperationState.idle:
            idle_count += 1
            if idle_count >= arm_settings.reset_ready_stable_idle_checks:
                return
        else:
            idle_count = 0
        time.sleep(arm_settings.reset_ready_poll_interval_s)
    logger.warning(
        "等待 moveReset 就绪超时，继续执行 arm_side={} file={} rows={}-{} {}",
        runtime.connected_arm.arm_side,
        first_row.csv_name,
        first_row.row_index,
        last_row.row_index,
        last_state_text,
    )


def _mm_to_m(values: list[float]) -> list[float]:
    """将 mm 坐标转换为 m。"""

    return [value / 1000.0 for value in values]


def _deg_to_rad(values: list[float]) -> list[float]:
    """将 deg 欧拉角或关节角转换为 rad。"""

    return [math.radians(value) for value in values]


def _extract_csv_sequence(csv_name: str) -> int:
    """解析 CSV 文件名前缀的整数阶段序号。"""

    return int(csv_name.split("_", maxsplit=1)[0])


def _resolve_cartesian_offset(
    runtime: ReplayRuntime,
    csv_name: str,
) -> tuple[np.ndarray | None, str]:
    """按测试 CLI 的优先级选择当前 CSV 的 ChArUco 或三球纠偏。"""

    sequence = _extract_csv_sequence(csv_name)
    charuco_sequences = (
        runtime.settings.offset.left_charuco_target_sequences
        if runtime.connected_arm.arm_side == "left"
        else runtime.settings.offset.right_charuco_target_sequences
    )
    if sequence in charuco_sequences:
        if runtime.charuco_cartesian_offset is None:
            raise RuntimeError(f"CSV {csv_name} 需要 ChArUco offset，但当前尚未完成目标板检测")
        return np.asarray(runtime.charuco_cartesian_offset, dtype=np.float64), "charuco"
    if sequence in runtime.offset_target_sequences:
        if runtime.global_cartesian_offset is None:
            raise RuntimeError(f"CSV {csv_name} 需要三球 offset，但当前尚未完成三球检测")
        return np.asarray(runtime.global_cartesian_offset, dtype=np.float64), "three-ball"
    return None, "none"


def _resolve_replay_move_abs_j_end_linear_speed_mm_s(runtime: ReplayRuntime, csv_name: str) -> float:
    """按机械臂侧别和 CSV 序号选择 MoveAbsJ 末端线速度。"""

    sequence = _extract_csv_sequence(csv_name)
    entries = (
        runtime.settings.arm.left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence
        if runtime.connected_arm.arm_side == "left"
        else runtime.settings.arm.right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence
    )
    values = dict(entries)
    return values.get(sequence, values[-1])


def _resolve_replay_move_abs_j_zone_mm(runtime: ReplayRuntime, csv_name: str) -> float:
    """按机械臂侧别和 CSV 序号选择连续 MoveAbsJ 中间点 zone。"""

    sequence = _extract_csv_sequence(csv_name)
    entries = (
        runtime.settings.arm.left_move_abs_j_zone_mm_by_csv_sequence
        if runtime.connected_arm.arm_side == "left"
        else runtime.settings.arm.right_move_abs_j_zone_mm_by_csv_sequence
    )
    values = dict(entries)
    return values.get(sequence, values[-1])


def _apply_global_offset(
    target_pose: xCoreSDK_python.CartesianPosition,
    offset_matrix: object,
) -> xCoreSDK_python.CartesianPosition:
    """对目标 TCP 应用 T_new_tcp = T_off @ T_tcp。"""

    matrix = np.asarray(offset_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"全局纠偏矩阵必须为 (4, 4)，实际为 {matrix.shape}")
    tcp_matrix = np.eye(4, dtype=np.float64)
    tcp_matrix[:3, :3] = Rotation.from_euler("xyz", target_pose.rpy, degrees=False).as_matrix()
    tcp_matrix[:3, 3] = np.asarray(target_pose.trans, dtype=np.float64)
    corrected = matrix @ tcp_matrix
    rpy_rad = Rotation.from_matrix(corrected[:3, :3]).as_euler("xyz", degrees=False)
    corrected_pose = xCoreSDK_python.CartesianPosition(corrected[:3, 3].tolist() + rpy_rad.tolist())
    corrected_pose.confData = list(target_pose.confData)
    corrected_pose.hasElbow = target_pose.hasElbow
    corrected_pose.elbow = target_pose.elbow
    return corrected_pose


def _normalize_ik_result(result: object) -> xCoreSDK_python.JointPosition:
    """将 SDK 的合法 IK 返回形式统一为 7 轴 JointPosition。"""

    if isinstance(result, xCoreSDK_python.JointPosition):
        values = [float(value) for value in result.joints]
    elif isinstance(result, np.ndarray):
        values = [float(value) for value in result.reshape(-1).tolist()]
    elif isinstance(result, (list, tuple)):
        values = [float(value) for value in result]
    else:
        raise RuntimeError(f"calcIk 成功，但返回值类型不支持：{type(result).__name__}")
    if len(values) != 7:
        raise RuntimeError(f"calcIk 成功，但返回关节数异常：len={len(values)}")
    return xCoreSDK_python.JointPosition(values)


# endregion
