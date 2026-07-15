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
    """读取当前右手 11 轴状态后整体下发目标。"""

    right_hand = runtime.hand_body.right_hand
    if right_hand is None:
        raise RuntimeError("当前 runtime 未配置右手 M11 客户端")
    state = retry_non_motion_call(
        f"right_hand.get_hand_state({runtime.connected_arm.arm_side})",
        lambda: right_hand.get_hand_state(include_tactile=False),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )
    if state is None:
        raise RuntimeError("右手状态不可用")
    actuators = state.get("actuators")
    if not isinstance(actuators, list):
        raise RuntimeError("右手状态格式异常：actuators 不是 list")
    positions: list[float] = []
    for index, actuator in enumerate(actuators):
        if not isinstance(actuator, dict) or not isinstance(actuator.get("position"), int | float):
            raise RuntimeError(f"右手状态格式异常：actuators[{index}].position 非数值")
        positions.append(float(actuator["position"]))
    required_max_id = max(*runtime.settings.hand.m11_root_actuator_ids, *runtime.settings.hand.m11_tip_actuator_ids)
    if len(positions) <= required_max_id:
        raise RuntimeError(f"右手状态执行器数量不足：required={required_max_id}, actual={len(positions)}")
    for actuator_id, target_value in enumerate(parse_joint_values(row.joints_text, expected_len=11)):
        positions[actuator_id] = target_value
    if not retry_non_motion_call(
        f"right_hand.set_hand_state({row.csv_name}:{row.row_index})",
        lambda: right_hand.set_hand_state(positions),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    ):
        raise RuntimeError("右手 M11 下发失败")


# endregion


# region 升降机构


def execute_lift_row(runtime: ReplayRuntime, row: ReplayRow) -> None:
    """下发 lift 目标高度并按旧策略等待到位。"""

    target_height_mm = int(round(float(row.pose_text)))
    if target_height_mm < 0:
        raise ValueError(f"lift 目标高度非法：{target_height_mm}")
    lift = runtime.hand_body.body.lift
    retry_non_motion_call(
        f"lift.set_lift_physical_height({row.csv_name}:{row.row_index})",
        lambda: lift.set_lift_physical_height(target_height_mm),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )
    actual_height_mm = wait_lift_until_near_target(runtime, target_height_mm)
    logger.info(
        "lift 已执行 file={} row={} target={} mm actual={:.1f} mm",
        row.csv_name,
        row.row_index,
        target_height_mm,
        actual_height_mm,
    )


def wait_lift_until_near_target(runtime: ReplayRuntime, target_height_mm: int) -> float:
    """等待 lift 到达目标附近，保留旧回放的无效值和重试规则。"""

    lift = runtime.hand_body.body.lift
    hand_settings = runtime.settings.hand
    time.sleep(hand_settings.lift_settle_delay_s)
    attempt = 0
    current_height_mm = _read_lift_height(
        retry_non_motion_call(
            "lift.get_lift_physical_height(initial)",
            lift.get_lift_physical_height,
            runtime.settings.non_motion_retry_count,
            runtime.settings.non_motion_retry_delay_s,
        )
    )
    while attempt < hand_settings.lift_retry_count:
        if current_height_mm == -1.0:
            current_height_mm = _read_lift_height(
                retry_non_motion_call(
                    "lift.get_lift_physical_height(invalid)",
                    lift.get_lift_physical_height,
                    runtime.settings.non_motion_retry_count,
                    runtime.settings.non_motion_retry_delay_s,
                )
            )
            continue
        if abs(current_height_mm - target_height_mm) <= hand_settings.lift_height_tolerance_mm:
            return current_height_mm
        attempt += 1
        if attempt < hand_settings.lift_retry_count:
            retry_non_motion_call(
                "lift.set_lift_physical_height(retry)",
                lambda: lift.set_lift_physical_height(target_height_mm),
                runtime.settings.non_motion_retry_count,
                runtime.settings.non_motion_retry_delay_s,
            )
            time.sleep(hand_settings.lift_settle_delay_s)
            current_height_mm = _read_lift_height(
                retry_non_motion_call(
                    "lift.get_lift_physical_height(retry)",
                    lift.get_lift_physical_height,
                    runtime.settings.non_motion_retry_count,
                    runtime.settings.non_motion_retry_delay_s,
                )
            )
    return current_height_mm


def _read_lift_height(result: object) -> float:
    """统一 qmlinker lift 的标量或二元返回格式。"""

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], int | float):
        return float(result[0])
    if isinstance(result, int | float):
        return float(result)
    raise TypeError(f"lift 返回值类型无效：{type(result)!r}")


# endregion
