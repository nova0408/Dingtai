"""双臂回放执行期 runtime 的创建、准备与释放。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .arm_gateway import (
    ConnectedArm,
    close_arm,
    connect_arm,
    prepare_nrt_motion,
    retry_non_motion_call,
)
from .contracts import ReplayRow
from .hand_gateway import (
    HandBodyRuntime,
    close_hand_body_runtime,
    create_hand_body_runtime,
)
from .settings import ReplayDeviceConnection, ReplayServiceSettings

# region 数据结构


@dataclass(slots=True)
class ReplayRuntime:
    """一侧机械臂完成一轮回放所需的全部运行资源。"""

    connected_arm: ConnectedArm
    "机械臂 xCoreSDK 连接。"
    hand_body: HandBodyRuntime
    "同侧手部与 body qmlinker 资源。"
    stop_event: threading.Event
    "双臂同步失败时共享的停止事件。"
    settings: ReplayServiceSettings
    "服务运行参数，由总配置页统一提供。"
    move_abs_j_end_linear_speed_mm_s: float
    "当前连续 MoveAbsJ 的末端线速度，单位 mm/s。"
    auto_execute_remaining: bool = True
    "自动服务固定为 True，保留执行期语义。"
    pending_arm_rows: list[ReplayRow] = field(default_factory=list)
    "尚未 flush 的连续 arm 行，具体类型由 motion 层约束。"
    global_cartesian_offset: object | None = None
    "当前轮次的全局笛卡尔纠偏矩阵。"
    offset_target_sequences: frozenset[int] = frozenset()
    "应用全局笛卡尔纠偏的 CSV 阶段序号，由 OffsetConfig 提供。"


# endregion


# region 生命周期


def create_runtime(
    arm_side: str,
    stop_event: threading.Event,
    device_connection: ReplayDeviceConnection,
    settings: ReplayServiceSettings,
) -> ReplayRuntime:
    """创建一侧机械臂及 hand/body runtime，不执行运动准备。"""

    robot_ip = device_connection.left_arm_ip if arm_side == "left" else device_connection.right_arm_ip
    connected_arm = connect_arm(arm_side, robot_ip, settings)
    try:
        hand_body = create_hand_body_runtime(
            arm_side,
            device_connection.qmlinker_host,
            device_connection.qmlinker_port,
            device_connection.gripper_port,
        )
    except Exception:
        close_arm(connected_arm)
        raise
    return ReplayRuntime(
        connected_arm,
        hand_body,
        stop_event,
        settings,
        settings.arm.move_abs_j_end_linear_speed_mm_s,
    )


def prepare_runtime(runtime: ReplayRuntime) -> None:
    """准备一侧机械臂回放所需的 NRT 与升降使能状态。"""

    arm_side = runtime.connected_arm.arm_side
    retry_non_motion_call(
        f"ensure_nrt_motion_ready({arm_side})",
        lambda: prepare_nrt_motion(runtime.connected_arm, runtime.settings),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )
    retry_non_motion_call(
        f"lift.set_enable({arm_side})",
        lambda: runtime.hand_body.body.lift.set_enable(True),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
    )


def close_runtime(runtime: ReplayRuntime | None) -> None:
    """按设备依赖方向释放一侧 runtime。"""

    if runtime is None:
        return
    close_hand_body_runtime(runtime.hand_body)
    close_arm(runtime.connected_arm)


# endregion
