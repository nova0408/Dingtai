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
    restore_nrt_motion_state_locked,
    read_cart_posture,
    read_joint_position,
    read_operate_mode,
    read_operation_state,
    read_power_state,
    read_soft_limits,
    retry_non_motion_call,
)
from .contracts import ReplayRow
from .runtime import ReplayRuntime

SOFT_LIMIT_CLAMP_MARGIN_RAD = math.radians(1.0)
"软限位越界时将目标钳制到边界内 1 deg。"

IK_TCP_REPAIR_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
"IK 跳变时依次沿基坐标 TCP 的 X、Y、Z 轴微调。"

# region 数据结构
@dataclass(frozen=True, slots=True)
class ArmMoveTarget:
    """一条 arm 行最终使用的关节目标。"""

    row: ReplayRow
    "源 CSV 动作行。"
    joint_rad: tuple[float, ...]
    "目标关节，单位 rad。"
    source: str
    "目标来源，包含 csv-joints、tcp-ik、tcp-repair 或 fallback。"


# endregion


# region 目标构建


def build_arm_target(
    runtime: ReplayRuntime,
    row: ReplayRow,
) -> ArmMoveTarget:
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
    recorded_joint_rad = row.arm_joint_rad
    ik_jump_detected = False
    try:
        joint_rad = calculate_ik(runtime.connected_arm, target_pose)
        source = "tcp-ik" if not applies_offset else f"tcp-ik-offset-{offset_source}"
    except RuntimeError:
        logger.exception("RobotControl calcIk 失败 file={} row={}", row.csv_name, row.row_index)
    else:
        if recorded_joint_rad is None or not _is_ik_jump(
            recorded_joint_rad,
            joint_rad,
            runtime.settings.arm.ik_joint_jump_threshold_deg,
        ):
            return ArmMoveTarget(row, joint_rad, source)
        repaired_target = _repair_ik_jump(
            runtime,
            row,
            target_pose,
            recorded_joint_rad,
            joint_rad,
            source,
        )
        if repaired_target is not None:
            return repaired_target
        ik_jump_detected = True
        logger.warning(
            "IK 跳变修复失败，回退 CSV 原始 joints file={} row={} source={}",
            row.csv_name,
            row.row_index,
            source,
        )
    if row.arm_joint_rad is None:
        raise RuntimeError(f"CSV joints 兜底未在启动阶段完成预解析 file={row.csv_name} row={row.row_index}")
    if ik_jump_detected:
        fallback_reason = "IK 跳变经 TCP 微调仍未修复"
    else:
        fallback_reason = "offset 后 T_new_tcp 逆解失败，本行未实施 offset" if applies_offset else "原始 TCP 逆解失败"
    logger.warning(
        "{}，最终兜底为 CSV joints MoveAbsJ file={} row={}",
        fallback_reason,
        row.csv_name,
        row.row_index,
    )
    if ik_jump_detected:
        source = "csv-joints-fallback-ik-jump"
    else:
        source = "csv-joints-fallback" if not applies_offset else "csv-joints-fallback-offset-ik"
    return ArmMoveTarget(row, tuple(row.arm_joint_rad), source)


def _repair_ik_jump(
    runtime: ReplayRuntime,
    row: ReplayRow,
    target_pose: CartesianPose,
    recorded_joint_rad: tuple[float, ...],
    initial_ik_joint_rad: tuple[float, ...],
    source: str,
) -> ArmMoveTarget | None:
    """以当前记录 joints 为基准，做最多三次 1 mm TCP 微调并选择非跳变解。"""

    settings = runtime.settings.arm
    initial_jump_deg = _max_joint_jump_deg(recorded_joint_rad, initial_ik_joint_rad)
    attempt_count = min(
        max(0, settings.ik_tcp_repair_attempt_count),
        len(IK_TCP_REPAIR_DIRECTIONS),
    )
    logger.warning(
        "检测到 IK 关节跳变，开始 TCP 微调修复 arm_side={} file={} row={} "
        "threshold_deg={:.3f} attempt_count={} offset_mm={:.3f}",
        runtime.connected_arm.arm_side,
        row.csv_name,
        row.row_index,
        settings.ik_joint_jump_threshold_deg,
        attempt_count,
        settings.ik_tcp_repair_offset_mm,
    )
    logger.warning(
        "IK 与记录 joints 的原始最大单轴差 arm_side={} file={} row={} max_jump_deg={:.3f}",
        runtime.connected_arm.arm_side,
        row.csv_name,
        row.row_index,
        initial_jump_deg,
    )
    for attempt_index, direction in enumerate(
        IK_TCP_REPAIR_DIRECTIONS[:attempt_count],
        start=1,
    ):
        offset_m = tuple(
            direction_value * settings.ik_tcp_repair_offset_mm / 1000.0
            for direction_value in direction
        )
        repaired_pose = _translate_tcp(target_pose, offset_m)
        try:
            repaired_joint_rad = calculate_ik(runtime.connected_arm, repaired_pose)
        except RuntimeError:
            logger.exception(
                "IK TCP 微调重算失败 arm_side={} file={} row={} attempt={} "
                "offset_m={}",
                runtime.connected_arm.arm_side,
                row.csv_name,
                row.row_index,
                attempt_index,
                offset_m,
            )
            continue
        jump_deg = _max_joint_jump_deg(recorded_joint_rad, repaired_joint_rad)
        logger.info(
            "IK TCP 微调候选 arm_side={} file={} row={} attempt={} "
            "offset_m={} max_jump_deg={:.3f}",
            runtime.connected_arm.arm_side,
            row.csv_name,
            row.row_index,
            attempt_index,
            offset_m,
            jump_deg,
        )
        if jump_deg <= settings.ik_joint_jump_threshold_deg:
            return ArmMoveTarget(row, repaired_joint_rad, f"{source}-tcp-repair-{attempt_index}")
    return None


def _translate_tcp(target_pose: CartesianPose, offset_m: tuple[float, ...]) -> CartesianPose:
    """沿基坐标平移 TCP，保持姿态、肘角和构型数据不变。"""

    if len(offset_m) != 3:
        raise ValueError(f"TCP 平移量必须包含 3 个分量，实际为 {offset_m!r}")
    translated = tuple(
        position + delta
        for position, delta in zip(target_pose.trans_m, offset_m, strict=True)
    )
    return CartesianPose(
        translated,
        target_pose.rpy_rad,
        target_pose.has_elbow,
        target_pose.elbow_rad,
        target_pose.conf_data,
    )


def _is_ik_jump(
    baseline_joint_rad: tuple[float, ...],
    candidate_joint_rad: tuple[float, ...],
    threshold_deg: float,
) -> bool:
    """判断 IK 结果相对记录 joints 是否出现超过阈值的单轴跳变。"""

    return _max_joint_jump_deg(baseline_joint_rad, candidate_joint_rad) > threshold_deg


def _max_joint_jump_deg(
    baseline_joint_rad: tuple[float, ...],
    candidate_joint_rad: tuple[float, ...],
) -> float:
    """返回候选 IK 与记录 joints 的最大单轴差，单位 deg。"""

    if len(baseline_joint_rad) != len(candidate_joint_rad):
        raise ValueError(
            "IK 关节目标长度不一致："
            f"baseline={len(baseline_joint_rad)} candidate={len(candidate_joint_rad)}"
        )
    return max(
        math.degrees(abs(baseline - candidate))
        for baseline, candidate in zip(
            baseline_joint_rad,
            candidate_joint_rad,
            strict=True,
        )
    )


# endregion


# region 连续执行


def execute_arm_segment(
    runtime: ReplayRuntime,
    rows: list[ReplayRow],
    final_arm_segment: bool = True,
) -> tuple[float, ...] | None:
    """将连续 arm 行合并为一次 MoveAbsJ append/start。

    ``final_arm_segment`` 指示本段是否包含该 CSV 最后一条 arm 记录；CSV 中间因夹爪、
    M6 或升降记录而切开的 arm 段不能因此误用 CaptureAction 的 ``final_speed``。
    """

    if not rows:
        return None
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
    commands = _clamp_commands_to_soft_limits(runtime, targets, commands)
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
    _wait_until_motion_finished(runtime, commands[-1][0])
    logger.info(
        "连续 MoveAbsJ 段完成 arm_side={} file={} rows={}-{} count={}",
        runtime.connected_arm.arm_side,
        rows[0].csv_name,
        rows[0].row_index,
        rows[-1].row_index,
        len(rows),
    )
    return commands[-1][0]


def flush_pending_arm_segment(
    runtime: ReplayRuntime,
    final_arm_segment: bool = True,
) -> tuple[float, ...] | None:
    """执行并清空 runtime 中积累的连续 arm 行。"""

    if not runtime.pending_arm_rows:
        return None
    rows = list(runtime.pending_arm_rows)
    runtime.pending_arm_rows.clear()
    return execute_arm_segment(runtime, rows, final_arm_segment)


def ensure_arm_position_before_m6(
    runtime: ReplayRuntime,
    target_joint_rad: tuple[float, ...] | None,
) -> None:
    """确认 M6 前的机械臂点已到位，必要时用单点 MoveAbsJ 补位。"""

    if target_joint_rad is None:
        logger.warning(
            "M6 前没有可校验的机械臂轨迹点，跳过位置补位检查 arm_side={}",
            runtime.connected_arm.arm_side,
        )
        return
    tolerance_rad = runtime.settings.arm.motion_joint_position_tolerance_rad
    actual_joint_rad = read_joint_position(runtime.connected_arm)
    max_error_rad = _max_joint_position_error(actual_joint_rad, target_joint_rad)
    logger.info(
        "M6 前机械臂位置检查 arm_side={} max_error_rad={:.6f} tolerance_rad={:.6f}",
        runtime.connected_arm.arm_side,
        max_error_rad,
        tolerance_rad,
    )
    if max_error_rad <= tolerance_rad:
        logger.info(
            "M6 前机械臂位置已到位 arm_side={} max_error_rad={:.6f}",
            runtime.connected_arm.arm_side,
            max_error_rad,
        )
        return

    logger.warning(
        "M6 前机械臂位置未到位，开始单点 MoveAbsJ 补位 arm_side={} "
        "max_error_rad={:.6f} tolerance_rad={:.6f}",
        runtime.connected_arm.arm_side,
        max_error_rad,
        tolerance_rad,
    )
    _execute_single_arm_correction(runtime, target_joint_rad)
    actual_joint_rad = read_joint_position(runtime.connected_arm)
    max_error_rad = _max_joint_position_error(actual_joint_rad, target_joint_rad)
    logger.info(
        "M6 前单点 MoveAbsJ 补位后位置检查 arm_side={} max_error_rad={:.6f} "
        "tolerance_rad={:.6f}",
        runtime.connected_arm.arm_side,
        max_error_rad,
        tolerance_rad,
    )
    if max_error_rad > tolerance_rad:
        raise RuntimeError(
            "M6 前机械臂补位后仍未到位，拒绝下发 M6 "
            f"arm_side={runtime.connected_arm.arm_side} "
            f"max_error_rad={max_error_rad:.6f} tolerance_rad={tolerance_rad:.6f}"
        )


def _execute_single_arm_correction(
    runtime: ReplayRuntime,
    target_joint_rad: tuple[float, ...],
) -> None:
    """用单个 MoveAbsJ 目标修正 M6 前的机械臂位置。"""

    action = runtime.current_action
    if action is None:
        raise RuntimeError("M6 前机械臂补位时缺少当前命名动作")
    speed = _resolve_action_speed(action, False)
    correction_rows = [
        ReplayRow(
            csv_name="m6-precondition",
            row_index=0,
            action_type="arm",
            joints_text="",
            pose_text="",
            arm_joint_rad=target_joint_rad,
        )
    ]
    logger.info(
        "M6 前单点 MoveAbsJ 补位开始 arm_side={} speed={} zone=0.0",
        runtime.connected_arm.arm_side,
        speed,
    )
    with runtime.connected_arm.command_lock:
        _raise_if_stop_requested(runtime)
        retry_non_motion_call(
            f"moveReset(m6-precondition:{runtime.connected_arm.arm_side})",
            lambda: move_reset(runtime.connected_arm),
            runtime.settings.non_motion_retry_count,
            runtime.settings.non_motion_retry_delay_s,
            runtime.stop_event,
        )
        _raise_if_stop_requested(runtime)
        correction_target = ArmMoveTarget(correction_rows[0], target_joint_rad, "m6-precondition")
        commands = _clamp_commands_to_soft_limits(
            runtime,
            [correction_target],
            [(target_joint_rad, speed, 0.0)],
        )
        move_append_abs_j(runtime.connected_arm, commands)
        _raise_if_stop_requested(runtime)
        _start_move_with_retry(runtime, correction_rows)
    _wait_until_motion_finished(runtime, commands[0][0])
    logger.info(
        "M6 前单点 MoveAbsJ 补位完成 arm_side={}",
        runtime.connected_arm.arm_side,
    )


def _max_joint_position_error(
    actual_joint_rad: tuple[float, ...],
    target_joint_rad: tuple[float, ...],
) -> float:
    """返回实时关节位置与目标位置的最大轴误差，单位 rad。"""

    return max(
        abs(actual - target)
        for actual, target in zip(actual_joint_rad, target_joint_rad, strict=True)
    )


def _clamp_commands_to_soft_limits(
    runtime: ReplayRuntime,
    targets: list[ArmMoveTarget],
    commands: list[tuple[tuple[float, ...], float, float]],
) -> list[tuple[tuple[float, ...], float, float]]:
    """按 RobotControl 缓存软限位钳制越界目标，并记录最终 append 值。"""

    if len(targets) != len(commands):
        raise ValueError("MoveAbsJ 目标与命令数量不一致")
    soft_limits = retry_non_motion_call(
        f"read-soft-limits:{runtime.connected_arm.arm_side}",
        lambda: read_soft_limits(runtime.connected_arm),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
        runtime.stop_event,
    )
    clamped_commands: list[tuple[tuple[float, ...], float, float]] = []
    for target, (joints_rad, speed_mm_s, zone_mm) in zip(
        targets, commands, strict=True
    ):
        clamped_joints: list[float] = []
        clamped_axes: list[int] = []
        for axis_index, (joint_rad, limit_rad) in enumerate(
            zip(joints_rad, soft_limits.limits_rad, strict=True)
        ):
            lower_rad, upper_rad = limit_rad
            if not math.isfinite(joint_rad):
                raise RuntimeError(
                    "MoveAbsJ 目标关节值不是有限数："
                    f"file={target.row.csv_name} row={target.row.row_index} "
                    f"axis={axis_index + 1} value={joint_rad}"
                )
            safe_lower_rad = lower_rad + SOFT_LIMIT_CLAMP_MARGIN_RAD
            safe_upper_rad = upper_rad - SOFT_LIMIT_CLAMP_MARGIN_RAD
            if safe_lower_rad > safe_upper_rad:
                raise RuntimeError(
                    "机械臂软限位范围不足以保留 1 deg 安全边距："
                    f"axis={axis_index + 1} lower_rad={lower_rad} upper_rad={upper_rad}"
                )
            clamped_rad = joint_rad
            if joint_rad < lower_rad:
                clamped_rad = safe_lower_rad
            elif joint_rad > upper_rad:
                clamped_rad = safe_upper_rad
            if clamped_rad != joint_rad:
                clamped_axes.append(axis_index + 1)
                logger.warning(
                    "MoveAbsJ 目标超出软限位，已钳制到边界内 1 deg "
                    "arm_side={} file={} row={} source={} axis={} original_rad={:.9f} "
                    "original_deg={:.6f} lower_rad={:.9f} upper_rad={:.9f} "
                    "actual_rad={:.9f} actual_deg={:.6f} soft_limit_enabled={}",
                    runtime.connected_arm.arm_side,
                    target.row.csv_name,
                    target.row.row_index,
                    target.source,
                    axis_index + 1,
                    joint_rad,
                    math.degrees(joint_rad),
                    lower_rad,
                    upper_rad,
                    clamped_rad,
                    math.degrees(clamped_rad),
                    soft_limits.enabled,
                )
            clamped_joints.append(clamped_rad)
        actual_joints_rad = tuple(clamped_joints)
        logger.info(
            "MoveAbsJ append actual target arm_side={} file={} row={} source={} "
            "joints_rad={} joints_deg={} speed_mm_s={} zone_mm={} clamped_axes={} "
            "soft_limit_enabled={}",
            runtime.connected_arm.arm_side,
            target.row.csv_name,
            target.row.row_index,
            target.source,
            list(actual_joints_rad),
            [math.degrees(value) for value in actual_joints_rad],
            speed_mm_s,
            zone_mm,
            clamped_axes,
            soft_limits.enabled,
        )
        clamped_commands.append((actual_joints_rad, speed_mm_s, zone_mm))
    return clamped_commands


def _wait_until_motion_finished(
    runtime: ReplayRuntime,
    target_joint_rad: tuple[float, ...],
) -> None:
    """确认机械臂运动完成，兼容控制器短暂漏报 ``moving`` 的情况。"""

    wait_started_at = time.monotonic()
    moving_deadline = wait_started_at + runtime.settings.arm.motion_start_timeout_s
    position_check_deadline: float | None = None
    last_state: str | None = None
    poll_count = 0
    moving_seen = False
    while True:
        if runtime.stop_event.is_set():
            raise RuntimeError("检测到停止请求，终止等待机械臂运动")
        if runtime.stop_event.wait(timeout=runtime.settings.arm.motion_state_poll_interval_s):
            raise RuntimeError("检测到停止请求，终止等待机械臂运动")
        state = read_operation_state(runtime.connected_arm)
        poll_count += 1
        if state == "moving":
            moving_seen = True
        if state != last_state:
            logger.info(
                "等待机械臂运动完成 arm_side={} state={} poll_count={} elapsed_s={:.3f} moving_seen={}",
                runtime.connected_arm.arm_side,
                state,
                poll_count,
                time.monotonic() - wait_started_at,
                moving_seen,
            )
            last_state = state
        if state == "idle":
            if moving_seen:
                logger.info(
                    "机械臂运动已完成 arm_side={} poll_count={} elapsed_s={:.3f} "
                    "moving_seen=True",
                    runtime.connected_arm.arm_side,
                    poll_count,
                    time.monotonic() - wait_started_at,
                )
                return
            if time.monotonic() < moving_deadline:
                continue
            if position_check_deadline is None:
                logger.warning(
                    "机械臂在 {:.1f} s 内未观察到 moving，开始实时关节位置确认 "
                    "arm_side={} poll_count={} elapsed_s={:.3f}",
                    runtime.settings.arm.motion_start_timeout_s,
                    runtime.connected_arm.arm_side,
                    poll_count,
                    time.monotonic() - wait_started_at,
                )
                position_check_deadline = (
                    time.monotonic()
                    + runtime.settings.arm.motion_position_check_timeout_s
                )
            max_error_rad: float | None = None
            try:
                actual_joint_rad = read_joint_position(runtime.connected_arm)
                max_error_rad = max(
                    abs(actual - target)
                    for actual, target in zip(actual_joint_rad, target_joint_rad, strict=True)
                )
            except RuntimeError as error:
                logger.warning(
                    "实时关节位置读取失败，继续轮询 arm_side={} poll_count={} "
                    "elapsed_s={:.3f} error={}",
                    runtime.connected_arm.arm_side,
                    poll_count,
                    time.monotonic() - wait_started_at,
                    error,
                )
            if max_error_rad is not None:
                logger.info(
                    "实时关节位置确认 arm_side={} poll_count={} elapsed_s={:.3f} "
                    "max_error_rad={:.6f} tolerance_rad={:.6f}",
                    runtime.connected_arm.arm_side,
                    poll_count,
                    time.monotonic() - wait_started_at,
                    max_error_rad,
                    runtime.settings.arm.motion_joint_position_tolerance_rad,
                )
                if max_error_rad <= runtime.settings.arm.motion_joint_position_tolerance_rad:
                    logger.info(
                        "机械臂实时关节位置已到位 arm_side={} poll_count={} "
                        "elapsed_s={:.3f} max_error_rad={:.6f}",
                        runtime.connected_arm.arm_side,
                        poll_count,
                        time.monotonic() - wait_started_at,
                        max_error_rad,
                    )
                    return
            if time.monotonic() >= position_check_deadline:
                logger.warning(
                    "实时关节位置确认超时，继续后续流程 arm_side={} poll_count={} "
                    "elapsed_s={:.3f} max_error_rad={} timeout_s={:.1f}",
                    runtime.connected_arm.arm_side,
                    poll_count,
                    time.monotonic() - wait_started_at,
                    max_error_rad,
                    runtime.settings.arm.motion_position_check_timeout_s,
                )
                return
            continue
        if state == "unknown":
            raise RuntimeError(
                f"无法确认机械臂运动是否完成 arm_side={runtime.connected_arm.arm_side} "
                f"operation_state=unknown poll_count={poll_count}"
            )


def _start_move_with_retry(runtime: ReplayRuntime, rows: list[ReplayRow]) -> None:
    """下发 ``moveStart`` 并处理成功后仍为 idle 的控制器短暂漏启动。"""

    state_retry_count = max(0, runtime.settings.arm.motion_start_retry_count)
    total_attempt_count = state_retry_count + 1
    for start_attempt in range(1, total_attempt_count + 1):
        _send_move_start_with_retry(runtime, rows)
        state = _confirm_move_start_state(runtime, rows, start_attempt)
        if state != "idle":
            return
        if start_attempt >= total_attempt_count:
            logger.warning(
                "moveStart 重发次数耗尽，交由后续运动等待逻辑继续确认 "
                "arm_side={} file={} rows={}-{} attempts={} state=idle",
                runtime.connected_arm.arm_side,
                rows[0].csv_name,
                rows[0].row_index,
                rows[-1].row_index,
                total_attempt_count,
            )
            return
        logger.warning(
            "moveStart 成功后 {:.1f} s 仍为 idle，准备重发 "
            "arm_side={} file={} rows={}-{} retry={}/{}",
            runtime.settings.arm.motion_start_retry_interval_s,
            runtime.connected_arm.arm_side,
            rows[0].csv_name,
            rows[0].row_index,
            rows[-1].row_index,
            start_attempt,
            state_retry_count,
        )


def _confirm_move_start_state(
    runtime: ReplayRuntime,
    rows: list[ReplayRow],
    start_attempt: int,
) -> str:
    """等待固定间隔后读取 ``moveStart`` 的实际操作状态。"""

    wait_started_at = time.monotonic()
    if runtime.stop_event.wait(timeout=runtime.settings.arm.motion_start_retry_interval_s):
        raise RuntimeError(
            "moveStart 启动确认期间收到停止请求 "
            f"arm_side={runtime.connected_arm.arm_side} "
            f"file={rows[0].csv_name} rows={rows[0].row_index}-{rows[-1].row_index}"
        )
    state = read_operation_state(runtime.connected_arm)
    elapsed_s = time.monotonic() - wait_started_at
    logger.info(
        "moveStart 启动确认 arm_side={} file={} rows={}-{} attempt={} "
        "wait_s={:.3f} state={}",
        runtime.connected_arm.arm_side,
        rows[0].csv_name,
        rows[0].row_index,
        rows[-1].row_index,
        start_attempt,
        elapsed_s,
        state,
    )
    if state == "unknown":
        raise RuntimeError(
            "moveStart 后无法确认机械臂操作状态 "
            f"arm_side={runtime.connected_arm.arm_side} "
            f"file={rows[0].csv_name} rows={rows[0].row_index}-{rows[-1].row_index}"
        )
    return state


def _send_move_start_with_retry(runtime: ReplayRuntime, rows: list[ReplayRow]) -> None:
    """处理控制器保存诊断或电机状态拒绝 ``moveStart`` 的短暂窗口。"""

    retry_count = max(1, runtime.settings.non_motion_retry_count)
    retry_delay_s = runtime.settings.non_motion_retry_delay_s
    for attempt in range(1, retry_count + 1):
        try:
            move_start(runtime.connected_arm)
            return
        except RuntimeError as error:
            if _is_motor_power_state_unsupported(error):
                if attempt >= retry_count:
                    raise RuntimeError(
                        f"moveStart 下发失败 arm_side={runtime.connected_arm.arm_side} "
                        f"file={rows[0].csv_name} rows={rows[0].row_index}-{rows[-1].row_index} "
                        f"attempt={attempt}/{retry_count} cause={error}"
                    ) from error
                _raise_if_stop_requested(runtime)
                logger.warning(
                    "moveStart 被电机状态拒绝，准备恢复 NRT 与使能状态后重试 "
                    "arm_side={} file={} rows={} attempt={}/{} cause={}",
                    runtime.connected_arm.arm_side,
                    rows[0].csv_name,
                    f"{rows[0].row_index}-{rows[-1].row_index}",
                    attempt,
                    retry_count,
                    error,
                )
                restore_nrt_motion_state_locked(
                    runtime.connected_arm,
                    runtime.settings,
                    runtime.stop_event,
                )
                continue
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


def _is_motor_power_state_unsupported(error: RuntimeError) -> bool:
    """识别 xCoreSDK ``-17`` 电机使能状态不正确错误。"""

    detail = str(error).lower()
    return "ec=-17" in detail and "motor power state unsupported operation" in detail


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
