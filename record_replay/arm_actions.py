"""机械臂回放目标构建与连续 MoveAbsJ 执行。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from sdk.xcoresdk import xCoreSDK_python

from .action_sequence import NamedActionPlan
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
        runtime.stop_event,
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
        runtime.stop_event,
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


def execute_arm_segment(
    runtime: ReplayRuntime,
    rows: list[ReplayRow],
    final_arm_segment: bool = True,
) -> None:
    """将连续 arm 行合并为一次 MoveAbsJ append/start。

    ``final_arm_segment`` 指示本段是否包含该 CSV 最后一条 arm 记录；CSV 中间因夹爪、
    M11 或升降记录而切开的 arm 段不能因此误用 CaptureAction 的 ``final_speed``。
    """

    if not rows:
        return
    _raise_if_stop_requested(runtime)
    targets = [build_arm_target(runtime, row) for row in rows]
    _raise_if_stop_requested(runtime)
    commands = []
    for index, target in enumerate(targets):
        segment_end = index == len(targets) - 1
        action = runtime.current_action
        if action is None:
            raise RuntimeError("执行 arm 段时缺少当前命名动作")
        is_final_point = final_arm_segment and segment_end
        zone = _resolve_action_zone(action, is_final_point)
        speed = _resolve_action_speed(action, is_final_point)
        commands.append(xCoreSDK_python.MoveAbsJCommand(target.joint, speed, zone))
    robot = runtime.connected_arm.robot
    ec = runtime.connected_arm.ec
    _raise_if_stop_requested(runtime)
    _wait_until_reset_ready(robot, ec, runtime, rows[0], rows[-1])
    with runtime.connected_arm.command_lock:
        _raise_if_stop_requested(runtime)
        retry_non_motion_call(
            f"moveReset(replay-arm-segment:{runtime.connected_arm.arm_side}:{rows[0].csv_name}:{rows[0].row_index}-{rows[-1].row_index})",
            lambda: robot.moveReset(ec),
            runtime.settings.non_motion_retry_count,
            runtime.settings.non_motion_retry_delay_s,
            runtime.stop_event,
        )
        if ec.get("ec", 0) != 0:
            raise RuntimeError("回放 arm 连续段 moveReset 失败")
        _raise_if_stop_requested(runtime)
        command_id = xCoreSDK_python.PyString()
        robot.moveAppend(commands, command_id, ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError("回放 arm 连续段 moveAppend 失败")
        _raise_if_stop_requested(runtime)
        robot.moveStart(ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError("回放 arm 连续段 moveStart 失败")
    _wait_until_motion_finished(robot, ec, runtime)


def flush_pending_arm_segment(
    runtime: ReplayRuntime,
    final_arm_segment: bool = True,
) -> None:
    """执行并清空 runtime 中积累的连续 arm 行。"""

    if not runtime.pending_arm_rows:
        return
    rows = list(runtime.pending_arm_rows)
    runtime.pending_arm_rows.clear()
    execute_arm_segment(runtime, rows, final_arm_segment)


def _wait_until_motion_finished(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    runtime: ReplayRuntime,
) -> None:
    """轮询控制器直到运动进入 idle 或 unknown。"""

    while True:
        if runtime.stop_event.is_set():
            raise RuntimeError("检测到停止请求，终止等待机械臂运动")
        if runtime.stop_event.wait(timeout=runtime.settings.arm.motion_state_poll_interval_s):
            raise RuntimeError("检测到停止请求，终止等待机械臂运动")
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
        _raise_if_stop_requested(runtime)
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
        remaining_s = deadline - time.time()
        if remaining_s <= 0.0:
            break
        if runtime.stop_event.wait(timeout=min(arm_settings.reset_ready_poll_interval_s, remaining_s)):
            raise RuntimeError("检测到停止请求，终止等待机械臂复位")
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


def _resolve_cartesian_offset(
    runtime: ReplayRuntime,
    csv_name: str,
) -> tuple[np.ndarray | None, str]:
    """按当前命名动作的优先级选择 ChArUco 或三球纠偏。"""

    del csv_name
    action = runtime.current_action
    if action is None:
        raise RuntimeError("构建 arm 目标时缺少当前命名动作")
    action_name = action.item.function_name
    charuco_sequences = (
        runtime.settings.offset.left_charuco_target_action_names
        if runtime.connected_arm.arm_side == "left"
        else runtime.settings.offset.right_charuco_target_action_names
    )
    use_charuco = action_name in charuco_sequences
    use_three_ball = action_name in runtime.offset_target_action_names
    if use_charuco and use_three_ball:
        raise RuntimeError(f"动作 {action_name} 不能同时应用头部 offset 与三球 offset")
    if use_charuco:
        if runtime.charuco_cartesian_offset is None:
            raise RuntimeError(f"动作 {action_name} 需要 ChArUco offset，但当前尚未完成目标板检测")
        runtime.offset_source = "head"
        return np.asarray(runtime.charuco_cartesian_offset, dtype=np.float64), "charuco"
    if use_three_ball:
        if runtime.global_cartesian_offset is None:
            raise RuntimeError(f"动作 {action_name} 需要三球 offset，但当前尚未完成三球检测")
        runtime.offset_source = "three_ball"
        return np.asarray(runtime.global_cartesian_offset, dtype=np.float64), "three-ball"
    runtime.offset_source = "none"
    return None, "none"


def _resolve_action_speed(action: NamedActionPlan, is_final_point: bool) -> float:
    """读取动作级速度；capture 的最后一个 arm 点使用 final_speed。"""

    if is_final_point and action.item.action_type == "capture":
        if action.item.final_speed is None:
            raise RuntimeError(f"capture 动作缺少 final_speed：{action.item.function_name}")
        return action.item.final_speed
    return action.item.speed


def _resolve_action_zone(action: NamedActionPlan, is_final_point: bool) -> float:
    """读取动作级 zone；capture 最终拍摄点固定使用 zone=0。"""

    if action.item.action_type == "precise":
        return 0.0
    if is_final_point and action.item.action_type == "capture":
        return 0.0
    return action.item.zone


def _raise_if_stop_requested(runtime: ReplayRuntime) -> None:
    """在所有可能继续提交运动的边界检查停止锁存。"""

    if runtime.stop_event.is_set():
        raise RuntimeError("检测到停止请求，禁止继续发送机械臂指令")


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
