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
from .motion_parsing import parse_joint_values, parse_pose_values
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

    if row.pose_text.lower() == "nan":
        return ArmMoveTarget(
            row, xCoreSDK_python.JointPosition(_deg_to_rad(parse_joint_values(row.joints_text))), "csv-joints"
        )
    pose = parse_pose_values(row.pose_text)
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
    applies_offset = (
        runtime.global_cartesian_offset is not None
        and _extract_csv_sequence(row.csv_name) in runtime.offset_target_sequences
    )
    if applies_offset:
        target_pose = _apply_global_offset(target_pose, runtime.global_cartesian_offset)
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
        return ArmMoveTarget(row, joint, "tcp-ik-offset" if applies_offset else "tcp-ik")
    if applies_offset:
        raise RuntimeError("offset 后 T_new_tcp 逆解失败，无法生成 MoveAbsJ 目标")
    fallback = parse_joint_values(row.joints_text)
    ec["ec"] = 0
    ec["message"] = "原始 TCP 逆解失败，已改用 CSV 记录关节值"
    return ArmMoveTarget(row, xCoreSDK_python.JointPosition(_deg_to_rad(fallback)), "csv-joints-fallback")


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
        force_zero_zone = (
            runtime.connected_arm.arm_side == "left"
            and csv_end
            and _extract_csv_sequence(target.row.csv_name) in runtime.settings.arm.left_zero_zone_sequences
        )
        zone = 0.0 if segment_end or force_zero_zone else runtime.settings.arm.move_abs_j_zone_mm
        commands.append(xCoreSDK_python.MoveAbsJCommand(target.joint, runtime.move_abs_j_end_linear_speed_mm_s, zone))
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


def _apply_global_offset(
    target_pose: xCoreSDK_python.CartesianPosition,
    offset_matrix: object,
) -> xCoreSDK_python.CartesianPosition:
    """对目标 TCP 应用 T_new_tcp = T_off @ T_tcp。"""

    matrix = np.asarray(offset_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"全局纠偏矩阵必须为 (4, 4)，实际为 {matrix.shape}")
    tcp_matrix = np.eye(4, dtype=np.float64)
    tcp_matrix[:3, :3] = Rotation.from_euler("XYZ", target_pose.rpy, degrees=False).as_matrix()
    tcp_matrix[:3, 3] = np.asarray(target_pose.trans, dtype=np.float64)
    corrected = matrix @ tcp_matrix
    rpy_rad = Rotation.from_matrix(corrected[:3, :3]).as_euler("XYZ", degrees=False)
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
