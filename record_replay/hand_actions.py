"""左夹爪、右 M11 与升降机构的回放动作。"""

from __future__ import annotations

import time

from loguru import logger

from .arm_gateway import retry_non_motion_call
from .contracts import ReplayRow
from .motion_parsing import parse_joint_values
from .runtime import ReplayRuntime

# region 左夹爪与右手


def execute_gripper_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    """下发左夹爪位置，不等待机械到位。"""

    gripper = runtime.hand_body.gripper
    if gripper is None:
        raise RuntimeError("当前 runtime 未配置左手夹爪客户端")
    target_value = int(round(float(row.pose_text)))
    if not retry_non_motion_call(
        f"gripper.set_pos({row.csv_name}:{row.row_index})",
        lambda: gripper.set_pos(target_value),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    ):
        raise RuntimeError("夹爪 set_pos 下发失败")
    status = retry_non_motion_call(
        f"gripper.get_status({row.csv_name}:{row.row_index})",
        gripper.get_status,
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )
    logger.info(
        "已下发夹爪目标 file={} row={} pos={} calibrated={}",
        row.csv_name,
        row.row_index,
        target_value,
        bool(status.calibrated),
    )


def execute_m11_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    """读取有效的右手 11 轴状态后整体下发目标。

    Parameters
    ----------
    runtime:
        当前单臂回放上下文，提供右手客户端及状态读取超时配置。
    row:
        M11 CSV 记录，``joints_text`` 包含 11 个归一化执行器目标值。

    Raises
    ------
    RuntimeError
        当前 runtime 未配置右手客户端，或目标下发失败。
    TimeoutError
        总超时内没有读取到至少 11 个带有效 ``position`` 的执行器状态。

    Notes
    -----
    空执行器列表属于瞬时通信无效状态，不会直接中止双臂并行执行。只有读取到完整状态后
    才会把 CSV 目标覆盖到当前状态并整体下发。
    """

    right_hand = runtime.hand_body.right_hand
    if right_hand is None:
        raise RuntimeError("当前 runtime 未配置右手 M11 客户端")
    positions = _read_valid_m11_positions(runtime)
    for actuator_id, target_value in enumerate(parse_joint_values(row.joints_text, expected_len=11)):
        positions[actuator_id] = target_value
    if not retry_non_motion_call(
        f"right_hand.set_hand_state({row.csv_name}:{row.row_index})",
        lambda: right_hand.set_hand_state(positions),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    ):
        raise RuntimeError("右手 M11 下发失败")


def _read_valid_m11_positions(runtime: ReplayRuntime) -> list[float]:
    """在总超时内持续读取结构完整的右手执行器位置。

    Parameters
    ----------
    runtime:
        当前单臂回放上下文，提供右手客户端、M11 索引及状态读取配置。

    Returns
    -------
    list[float]
        当前右手执行器位置，至少覆盖所有 M11 索引，数值为设备归一化位置。

    Raises
    ------
    RuntimeError
        当前 runtime 未配置右手客户端。
    TimeoutError
        总超时内持续返回空状态、数量不足状态、非法 position 或 RPC 异常。

    Notes
    -----
    重读次数不设上限，唯一边界是 ``m11_state_read_timeout_s``。该函数只验证通信状态，
    不下发执行器目标。
    """

    right_hand = runtime.hand_body.right_hand
    if right_hand is None:
        raise RuntimeError("当前 runtime 未配置右手 M11 客户端")
    hand_settings = runtime.settings.hand
    required_count = max(*hand_settings.m11_root_actuator_ids, *hand_settings.m11_tip_actuator_ids) + 1
    deadline = time.monotonic() + hand_settings.m11_state_read_timeout_s
    read_index = 0
    last_invalid_reason = "尚未读取"
    while True:
        read_index += 1
        positions: list[float] = []
        try:
            state = retry_non_motion_call(
                f"right_hand.get_hand_state({runtime.connected_arm.arm_side})",
                lambda: right_hand.get_hand_state(include_tactile=False),
                runtime.settings.non_motion_retry_count,
                runtime.settings.non_motion_retry_delay_s,
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
                        if not isinstance(actuator, dict) or not isinstance(actuator.get("position"), int | float):
                            last_invalid_reason = f"actuators[{index}].position 非数值"
                            positions = []
                            break
                        positions.append(float(actuator["position"]))
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
                f"读取有效右手状态超时 timeout={hand_settings.m11_state_read_timeout_s:.1f} s "
                f"read={read_index} last_reason={last_invalid_reason}"
            )
        logger.warning(
            "右手状态无效，等待后重读 read={} reason={} remaining={:.1f} s",
            read_index,
            last_invalid_reason,
            remaining_s,
        )
        time.sleep(min(hand_settings.m11_state_read_poll_interval_s, remaining_s))


# endregion


# region 升降机构


def execute_lift_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    """下发 lift 目标高度，并在确认到位后才允许回放继续。

    Parameters
    ----------
    runtime:
        当前单臂回放上下文，提供 lift 客户端及超时、轮询和容差配置。
    row:
        lift CSV 记录，``pose_text`` 表示目标物理高度，单位 mm。

    Raises
    ------
    ValueError
        目标高度小于 0 mm。
    TimeoutError
        总超时内始终没有有效高度，或有效高度没有进入到位容差。

    Notes
    -----
    本函数同步阻塞当前 CSV 执行流。等待函数成功返回前，不会放行后续机械臂轨迹。
    """

    target_height_mm = int(round(float(row.pose_text)))
    if target_height_mm < 0:
        raise ValueError(f"lift 目标高度非法：{target_height_mm}")
    _ensure_lift_enabled(runtime, f"lift.set_enable({row.csv_name}:{row.row_index})")
    actual_height_mm = wait_lift_until_near_target(runtime, target_height_mm)
    logger.info(
        "lift 已执行 file={} row={} target={} mm actual={:.1f} mm",
        row.csv_name,
        row.row_index,
        target_height_mm,
        actual_height_mm,
    )


def _ensure_lift_enabled(runtime: ReplayRuntime, label: str) -> None:
    """持续下发 enable，并以 ``get_enable()`` 状态判断是否生效。

    Parameters
    ----------
    runtime:
        当前单臂回放上下文，提供 lift 客户端和 enable 状态等待配置。
    label:
        日志与超时异常使用的指令标签。
    Raises
    ------
    TimeoutError
        总超时内 ``get_enable()`` 始终未返回 ``True``。

    Notes
    -----
    ``set_enable()`` 的返回值只记录用于诊断，不作为成功判据。
    """

    hand_settings = runtime.settings.hand
    lift = runtime.hand_body.body.lift
    deadline = time.monotonic() + hand_settings.lift_enable_state_timeout_s
    attempt = 0
    last_enable_state: object = None
    while True:
        attempt += 1
        try:
            command_result = retry_non_motion_call(
                label,
                lambda: lift.set_enable(True),
                runtime.settings.non_motion_retry_count,
                runtime.settings.non_motion_retry_delay_s,
            )
        except Exception as exc:
            command_result = f"调用异常：{exc}"
        try:
            last_enable_state = retry_non_motion_call(
                "lift.get_enable(wait)",
                lift.get_enable,
                runtime.settings.non_motion_retry_count,
                runtime.settings.non_motion_retry_delay_s,
            )
        except Exception as exc:
            last_enable_state = f"读取异常：{exc}"
        if last_enable_state is True:
            logger.success("lift enable 状态已生效 label={} attempt={}", label, attempt)
            return
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f"等待 lift enable 状态超时 label={label} "
                f"timeout={hand_settings.lift_enable_state_timeout_s:.1f} s "
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
        time.sleep(min(hand_settings.lift_enable_retry_interval_s, remaining_s))


def wait_lift_until_near_target(runtime: ReplayRuntime, target_height_mm: int) -> float:
    """在总超时内持续等待 lift 到达目标附近。

    Parameters
    ----------
    runtime:
        当前单臂回放上下文，提供 lift 客户端和等待配置。
    target_height_mm:
        lift 目标物理高度，单位 mm。

    Returns
    -------
    float
        已进入容差范围的有效 lift 物理高度，单位 mm。

    Raises
    ------
    TimeoutError
        读取始终返回负数通信无效值，或有效高度在总超时内未到位。

    Notes
    -----
    负数高度不具备物理意义，统一视为通信失败并立即重读，不计次数。有效高度未到位时
    按配置周期轮询。函数没有次数上限，唯一终止边界是到位或总超时。
    """

    lift = runtime.hand_body.body.lift
    hand_settings = runtime.settings.hand
    deadline = time.monotonic() + hand_settings.lift_motion_timeout_s
    next_command_time = 0.0
    valid_read_index = 0
    invalid_read_count = 0
    last_logged_height_mm: float | None = None
    while True:
        now = time.monotonic()
        if now >= next_command_time:
            try:
                command_result = retry_non_motion_call(
                    "lift.set_lift_physical_height(wait)",
                    lambda: lift.set_lift_physical_height(target_height_mm),
                    runtime.settings.non_motion_retry_count,
                    runtime.settings.non_motion_retry_delay_s,
                )
            except Exception as exc:
                command_result = f"调用异常：{exc}"
            logger.info(
                "lift 目标高度已下发，实际到位状态由物理高度判断 target={} mm command_return={}",
                target_height_mm,
                command_result,
            )
            next_command_time = now + hand_settings.lift_target_reissue_interval_s
        current_height_mm = _read_lift_height(
            retry_non_motion_call(
                "lift.get_lift_physical_height(wait)",
                lift.get_lift_physical_height,
                runtime.settings.non_motion_retry_count,
                runtime.settings.non_motion_retry_delay_s,
            )
        )
        if current_height_mm < 0.0:
            invalid_read_count += 1
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待 lift 有效高度超时 target={target_height_mm} mm "
                    f"timeout={hand_settings.lift_motion_timeout_s:.1f} s last={current_height_mm:.1f} mm"
                )
            if invalid_read_count == 1 or invalid_read_count % 20 == 0:
                logger.warning(
                    "lift 返回无效高度，判定为通信失败并立即重读 invalid_read={} value={:.1f} mm",
                    invalid_read_count,
                    current_height_mm,
                )
            continue
        valid_read_index += 1
        current_error_mm = abs(current_height_mm - target_height_mm)
        if last_logged_height_mm is None or abs(current_height_mm - last_logged_height_mm) >= 0.5:
            logger.info(
                "lift 到位状态 valid_read={} target={} mm actual={:.1f} mm error={:.1f} mm",
                valid_read_index,
                target_height_mm,
                current_height_mm,
                current_error_mm,
            )
            last_logged_height_mm = current_height_mm
        if current_error_mm <= hand_settings.lift_height_tolerance_mm:
            logger.success("lift 已到位 target={} mm actual={:.1f} mm", target_height_mm, current_height_mm)
            return current_height_mm
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f"等待 lift 到位超时 target={target_height_mm} mm actual={current_height_mm:.1f} mm "
                f"error={current_error_mm:.1f} mm timeout={hand_settings.lift_motion_timeout_s:.1f} s"
            )
        time.sleep(min(hand_settings.lift_poll_interval_s, remaining_s))


def _read_lift_height(result: object) -> float:
    """统一 qmlinker lift 的标量或二元返回格式。"""

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], int | float):
        return float(result[0])
    if isinstance(result, int | float):
        return float(result)
    raise TypeError(f"lift 返回值类型无效：{type(result)!r}")


# endregion
