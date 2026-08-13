"""双臂回放执行期 runtime 的创建、准备与释放。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .arm_gateway import (
    CartesianPose,
    ConnectedArm,
    close_arm,
    connect_arm,
    prepare_nrt_motion,
    retry_non_motion_call,
)
from .action_sequence import NamedActionPlan
from .contracts import ReplayRow
from .hand_gateway import (
    HandBodyRuntime,
    close_hand_body_runtime,
    create_hand_body_runtime,
)
from .settings import ReplayDeviceConnection, ReplayServiceSettings

# region 数据结构


@dataclass(frozen=True, slots=True)
class CompiledArmWaypoint:
    """offset 条件满足后冻结的一条 MoveAbsJ 轨迹点。"""

    row: ReplayRow
    "源 CSV 中已经预解析的 arm 行。"
    action: NamedActionPlan
    "轨迹点所属命名动作，用于保留该点的 offset、speed 和 zone 语义。"
    is_final_action_arm_point: bool
    "是否为所属动作 CSV 的最后一个 arm 点，用于保持 capture 末点参数语义。"
    requested_tcp: CartesianPose | None
    "应用 offset 后用于逆解的最终 TCP；纯 joints 行为空。"
    joint_rad: tuple[float, ...]
    "软限位处理后的最终关节目标，单位 rad。"
    speed_mm_s: float
    "该 waypoint 最终提交的速度，单位 mm/s。"
    zone_mm: float
    "该 waypoint 最终提交的 zone，单位 mm。"
    source: str
    "最终关节目标来源。"
    precompile_batch_id: int
    "本轮预编译批次编号，仅用于日志追踪。"


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
    auto_execute_remaining: bool = True
    "自动服务固定为 True，保留执行期语义。"
    compiled_arm_waypoints: dict[tuple[NamedActionPlan, int], CompiledArmWaypoint] = field(
        default_factory=dict
    )
    "按命名动作和 CSV 行号保存的已编译 arm 点。"
    pending_arm_waypoints: list[CompiledArmWaypoint] = field(default_factory=list)
    "尚未 flush 的连续已编译 arm 点；允许跨命名动作和 CSV 合并。"
    next_precompile_batch_id: int = 1
    "下一批 waypoint 预编译编号，从 1 开始，仅用于日志追踪。"
    next_motion_segment_id: int = 1
    "下一条 MoveAbsJ 物理轨迹段编号，从 1 开始，仅用于本轮日志追踪。"
    pending_motion_start_barrier: threading.Barrier | None = None
    "下一个物理轨迹段的双臂 moveStart 屏障；只允许同步动作首段消费一次。"
    pending_motion_start_label: str | None = None
    "待同步启动的命名动作标签，用于安全校验和日志关联。"
    global_cartesian_offset: object | None = None
    "当前轮次的全局笛卡尔纠偏矩阵。"
    charuco_cartesian_offset: object | None = None
    "当前轮次经历史安全门接受的 ChArUco 笛卡尔纠偏矩阵。"
    offset_record_path: object | None = None
    "本轮 offset 对比记录路径；仅用于诊断，不参与执行决策。"
    offset_target_action_names: frozenset[str] = frozenset()
    "应用全局笛卡尔纠偏的命名动作，由 OffsetConfig 提供。"
    offset_source: str = "none"
    "当前命名动作实际选用的 offset 来源：none、head 或 three_ball。"
    current_action: NamedActionPlan | None = None
    "当前正在执行的冻结命名动作。"
    preloaded_rows_by_path: dict[Path, tuple[ReplayRow, ...]] = field(default_factory=dict)
    "启动阶段按执行计划预解析的 CSV 行，执行期不再重新读取文件。"


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
    logger.info("runtime 创建：开始连接机械臂 arm_side={} ip={}", arm_side, robot_ip)
    connected_arm = connect_arm(arm_side, robot_ip, settings, stop_event)
    logger.info("runtime 创建：机械臂连接完成 arm_side={} ip={}", arm_side, robot_ip)
    try:
        logger.info(
            "runtime 创建：开始创建 qmlinker 附属设备 arm_side={} host={} body_port={} "
            "gripper_port={}",
            arm_side,
            device_connection.qmlinker_host,
            device_connection.qmlinker_port,
            device_connection.gripper_port,
        )
        hand_body = create_hand_body_runtime(
            arm_side,
            device_connection.qmlinker_host,
            device_connection.qmlinker_port,
            device_connection.gripper_port,
        )
        logger.info("runtime 创建：qmlinker 附属设备创建完成 arm_side={}", arm_side)
    except Exception:
        logger.exception("runtime 创建失败，准备关闭已连接机械臂 arm_side={}", arm_side)
        close_arm(connected_arm)
        raise
    return ReplayRuntime(
        connected_arm,
        hand_body,
        stop_event,
        settings,
    )


def prepare_runtime(runtime: ReplayRuntime) -> None:
    """准备一侧机械臂回放所需的 NRT 与升降使能状态。"""

    _raise_if_stopped(runtime)
    arm_side = runtime.connected_arm.arm_side
    logger.info("runtime 准备开始 arm_side={}", arm_side)
    if runtime.hand_body.gripper is not None:
        from .hand_actions import prepare_gripper_before_replay

        logger.info("runtime 准备：左侧夹爪准备开始 arm_side={}", arm_side)
        prepare_gripper_before_replay(runtime)
        logger.info("runtime 准备：左侧夹爪准备完成 arm_side={}", arm_side)
    _raise_if_stopped(runtime)
    logger.info("runtime 准备：机械臂 NRT 准备开始 arm_side={}", arm_side)
    retry_non_motion_call(
        f"ensure_nrt_motion_ready({arm_side})",
        lambda: prepare_nrt_motion(
            runtime.connected_arm,
            runtime.settings,
            runtime.stop_event,
        ),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
        runtime.stop_event,
    )
    logger.info("runtime 准备：机械臂 NRT 准备完成 arm_side={}", arm_side)
    _raise_if_stopped(runtime)
    logger.info("runtime 准备：升降使能开始 arm_side={}", arm_side)
    retry_non_motion_call(
        f"lift.set_enable({arm_side})",
        lambda: runtime.hand_body.lift.set_enable(True),
        runtime.settings.non_motion_retry_count,
        runtime.settings.non_motion_retry_delay_s,
        runtime.stop_event,
    )
    logger.info("runtime 准备：升降使能完成 arm_side={}", arm_side)
    _raise_if_stopped(runtime)
    logger.info("runtime 准备完成 arm_side={}", arm_side)


def _raise_if_stopped(runtime: ReplayRuntime) -> None:
    """阻止停止锁存后的准备阶段继续发送普通设备指令。"""

    if runtime.stop_event.is_set():
        raise RuntimeError("检测到停止请求，禁止继续准备设备")


def close_runtime(runtime: ReplayRuntime | None) -> None:
    """按设备依赖方向释放一侧 runtime。"""

    if runtime is None:
        return
    logger.info("runtime 释放开始 arm_side={}", runtime.connected_arm.arm_side)
    close_hand_body_runtime(runtime.hand_body)
    close_arm(runtime.connected_arm)
    logger.info("runtime 释放完成 arm_side={}", runtime.connected_arm.arm_side)


# endregion
