"""机械臂回放目标构建与连续 MoveAbsJ 执行。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from .action_sequence import NamedActionPlan
from .arm_gateway import (
    CartesianPose,
    calculate_ik,
    move_append_abs_j,
    move_reset,
    move_start,
    read_cart_posture,
    read_operate_mode,
    read_operation_state,
    read_power_state,
    retry_non_motion_call,
)
from .contracts import ReplayRow
from .runtime import ReplayRuntime

# region 数据结构
@dataclass(frozen=True, slots=True)
class ArmMoveTarget:
    """一条 arm 行最终使用的关节目标。"""

    row: ReplayRow
    "源 CSV 动作行。"
    joint_rad: tuple[float, ...]
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
            tuple(row.arm_joint_rad),
            "csv-joints",
        )
    pose = row.arm_pose
    current_pose = retry_non_motion_call(
        f"cartPosture(endInRef:{row.csv_name}:{row.row_index})",
        lambda: read_cart_posture(runtime.connected_arm),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
        runtime.stop_event,
    )
    target_pose = CartesianPose(
        tuple(_mm_to_m(list(pose.xyz_mm))),
        tuple(_deg_to_rad(list(pose.rpy_deg))),
        current_pose.has_elbow,
        current_pose.elbow_rad,
        current_pose.conf_data,
    )
    has_elbow = target_pose.has_elbow
    elbow_rad = target_pose.elbow_rad
    conf_data = target_pose.conf_data
    if pose.has_elbow is not None:
        has_elbow = pose.has_elbow
    if pose.elbow_deg is not None:
        elbow_rad = _deg_to_rad([pose.elbow_deg])[0]
    if pose.conf_data is not None:
        conf_data = tuple(pose.conf_data)
    target_pose = CartesianPose(
        target_pose.trans_m, target_pose.rpy_rad, has_elbow, elbow_rad, conf_data
    )
    offset_matrix, offset_source = _resolve_cartesian_offset(runtime, row.csv_name)
    applies_offset = offset_matrix is not None
    if offset_matrix is not None:
        target_pose = _apply_global_offset(target_pose, offset_matrix)
    try:
        joint_rad = calculate_ik(runtime.connected_arm, target_pose)
        source = "tcp-ik" if not applies_offset else f"tcp-ik-offset-{offset_source}"
        return ArmMoveTarget(row, joint_rad, source)
    except RuntimeError:
        logger.exception("RobotControl calcIk 失败 file={} row={}", row.csv_name, row.row_index)
    if row.arm_joint_rad is None:
        raise RuntimeError(f"CSV joints 兜底未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    fallback_reason = "offset 后 T_new_tcp 逆解失败，本行未实施 offset" if applies_offset else "原始 TCP 逆解失败"
    logger.warning(
        "{}，最终兜底为 CSV joints MoveAbsJ file={} row={}",
        fallback_reason,
        row.csv_name,
        row.row_index,
    )
    source = "csv-joints-fallback" if not applies_offset else "csv-joints-fallback-offset-ik"
    return ArmMoveTarget(row, tuple(row.arm_joint_rad), source)


# endregion


# region 连续执行


def execute_arm_segment(
    runtime: ReplayRuntime,
    rows: list[ReplayRow],
    final_arm_segment: bool = True,
) -> None:
    """将连续 arm 行合并为一次 MoveAbsJ append/start。

    ``final_arm_segment`` 指示本段是否包含该 CSV 最后一条 arm 记录；CSV 中间因夹爪、
    M6 或升降记录而切开的 arm 段不能因此误用 CaptureAction 的 ``final_speed``。
    """

    if not rows:
        return
    logger.info(
        "连续 MoveAbsJ 段开始 arm_side={} file={} rows={}-{} count={} final_arm_segment={}",
        runtime.connected_arm.arm_side,
        rows[0].csv_name,
        rows[0].row_index,
        rows[-1].row_index,
        len(rows),
        final_arm_segment,
    )
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
        commands.append((target.joint_rad, speed, zone))
    _raise_if_stop_requested(runtime)
    _wait_until_reset_ready(runtime, rows[0], rows[-1])
    with runtime.connected_arm.command_lock:
        _raise_if_stop_requested(runtime)
        retry_non_motion_call(
            f"moveReset(replay-arm-segment:{runtime.connected_arm.arm_side}:{rows[0].csv_name}:{rows[0].row_index}-{rows[-1].row_index})",
            lambda: move_reset(runtime.connected_arm),
            runtime.settings.non_motion_retry_count,
            runtime.settings.non_motion_retry_delay_s,
            runtime.stop_event,
        )
        _raise_if_stop_requested(runtime)
        move_append_abs_j(runtime.connected_arm, commands)
        _raise_if_stop_requested(runtime)
        _start_move_with_retry(runtime, rows)
    _wait_until_motion_finished(runtime)
    logger.info(
        "连续 MoveAbsJ 段完成 arm_side={} file={} rows={}-{} count={}",
        runtime.connected_arm.arm_side,
        rows[0].csv_name,
        rows[0].row_index,
        rows[-1].row_index,
        len(rows),
    )


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
    runtime: ReplayRuntime,
) -> None:
    """轮询控制器直到运动明确进入 idle。"""

    last_state: str | None = None
    poll_count = 0
    while True:
        if runtime.stop_event.is_set():
            raise RuntimeError("检测到停止请求，终止等待机械臂运动")
        if runtime.stop_event.wait(timeout=runtime.settings.arm.motion_state_poll_interval_s):
            raise RuntimeError("检测到停止请求，终止等待机械臂运动")
        state = read_operation_state(runtime.connected_arm)
        poll_count += 1
        if state != last_state:
            logger.info(
                "等待机械臂运动完成 arm_side={} state={} poll_count={}",
                runtime.connected_arm.arm_side,
                state,
                poll_count,
            )
            last_state = state
        if state == "idle":
            logger.info(
                "机械臂运动已完成 arm_side={} poll_count={}",
                runtime.connected_arm.arm_side,
                poll_count,
            )
            return
        if state == "unknown":
            raise RuntimeError(
                f"无法确认机械臂运动是否完成 arm_side={runtime.connected_arm.arm_side} "
                f"operation_state=unknown poll_count={poll_count}"
            )


def _start_move_with_retry(runtime: ReplayRuntime, rows: list[ReplayRow]) -> None:
    """处理控制器保存诊断期间拒绝 ``moveStart`` 的短暂窗口。"""

    retry_count = max(1, runtime.settings.non_motion_retry_count)
    retry_delay_s = runtime.settings.non_motion_retry_delay_s
    for attempt in range(1, retry_count + 1):
        try:
            move_start(runtime.connected_arm)
            return
        except RuntimeError as error:
            if not _is_diagnosis_save_busy(error) or attempt >= retry_count:
                raise RuntimeError(
                    f"moveStart 下发失败 arm_side={runtime.connected_arm.arm_side} "
                    f"file={rows[0].csv_name} rows={rows[0].row_index}-{rows[-1].row_index} "
                    f"attempt={attempt}/{retry_count} cause={error}"
                ) from error
            logger.warning(
                "moveStart 暂被控制器诊断保存占用，准备重试 arm_side={} file={} rows={}-{} "
                "attempt={}/{} retry_delay_s={} cause={}",
                runtime.connected_arm.arm_side,
                rows[0].csv_name,
                rows[0].row_index,
                rows[-1].row_index,
                attempt,
                retry_count,
                retry_delay_s,
                error,
            )
            if runtime.stop_event.wait(timeout=retry_delay_s):
                raise RuntimeError(
                    f"moveStart 重试期间收到停止请求 arm_side={runtime.connected_arm.arm_side} "
                    f"file={rows[0].csv_name} rows={rows[0].row_index}-{rows[-1].row_index}"
                ) from error


def _is_diagnosis_save_busy(error: RuntimeError) -> bool:
    """识别 xCoreSDK ``-60611`` 诊断保存临时占用错误。"""

    detail = str(error).lower()
    return "-60611" in detail or "saving diagnosis data" in detail


def _wait_until_reset_ready(
    runtime: ReplayRuntime,
    first_row: ReplayRow,
    last_row: ReplayRow,
) -> None:
    """等待连续段明确回到 idle 后再调用 ``moveReset``。"""

    arm_settings = runtime.settings.arm
    deadline = time.time() + arm_settings.reset_ready_timeout_s
    idle_count = 0
    last_state_text = ""
    while time.time() < deadline:
        _raise_if_stop_requested(runtime)
        operation_state = read_operation_state(runtime.connected_arm)
        operate_mode = read_operate_mode(runtime.connected_arm)
        power_state = read_power_state(runtime.connected_arm)
        last_state_text = f"operate_mode={operate_mode} operation_state={operation_state} power_state={power_state}"
        if operation_state == "idle":
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
    raise RuntimeError(
        "等待 moveReset 就绪超时，拒绝覆盖上一段运动 arm_side={} file={} rows={}-{} {}".format(
            runtime.connected_arm.arm_side,
            first_row.csv_name,
            first_row.row_index,
            last_row.row_index,
            last_state_text,
        )
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
    target_pose: CartesianPose,
    offset_matrix: object,
) -> CartesianPose:
    """对目标 TCP 应用 T_new_tcp = T_off @ T_tcp。"""

    matrix = np.asarray(offset_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"全局纠偏矩阵必须为 (4, 4)，实际为 {matrix.shape}")
    tcp_matrix = np.eye(4, dtype=np.float64)
    tcp_matrix[:3, :3] = Rotation.from_euler("xyz", target_pose.rpy_rad, degrees=False).as_matrix()
    tcp_matrix[:3, 3] = np.asarray(target_pose.trans_m, dtype=np.float64)
    corrected = matrix @ tcp_matrix
    rpy_rad = Rotation.from_matrix(corrected[:3, :3]).as_euler("xyz", degrees=False)
    return CartesianPose(
        tuple(float(value) for value in corrected[:3, 3]),
        tuple(float(value) for value in rpy_rad),
        target_pose.has_elbow,
        target_pose.elbow_rad,
        target_pose.conf_data,
    )


# endregion
