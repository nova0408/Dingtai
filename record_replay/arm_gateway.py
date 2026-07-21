"""双臂 xCoreSDK 连接与非实时运动准备。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from loguru import logger

from sdk.xcoresdk import xCoreSDK_python

from .settings import ReplayArmSettings, ReplayServiceSettings

_ResultT = TypeVar("_ResultT")


# region 数据结构
@dataclass(slots=True)
class ConnectedArm:
    """一台已连接机械臂的 SDK 运行资源。"""

    arm_side: str
    "机械臂侧别，left 或 right。"
    robot_ip: str
    "控制器 IPv4 地址。"
    robot: xCoreSDK_python.xMateErProRobot
    "xCoreSDK 机器人实例。"
    robot_type: str
    "控制器上报的型号。"
    robot_uid: str
    "控制器上报的唯一标识。"
    ec: dict[str, object]
    "SDK 调用共用错误上下文。"


# endregion


class _RobotInfoProtocol(Protocol):
    """本业务实际读取的 robotInfo 最小字段。"""

    type: str
    id: str


# region 基础调用


def retry_non_motion_call(
    label: str,
    operation: Callable[[], _ResultT],
    retry_count: int,
    retry_delay_s: float,
) -> _ResultT:
    """按旧回放策略重试非直接运动调用。"""

    last_error: BaseException | None = None
    for attempt in range(1, retry_count + 1):
        try:
            return operation()
        except BaseException as error:
            last_error = error
            logger.warning(
                "{} 失败，准备重试 attempt={}/{} delay={:.1f}s exc={}",
                label,
                attempt,
                retry_count,
                retry_delay_s,
                error,
            )
            if attempt < retry_count:
                time.sleep(retry_delay_s)
    raise RuntimeError(f"{label} 连续失败 {retry_count} 次") from last_error


def connect_arm(arm_side: str, robot_ip: str, settings: ReplayServiceSettings) -> ConnectedArm:
    """连接一台机械臂并设置固定 tool/wobj。"""

    if arm_side not in {"left", "right"}:
        raise ValueError(f"不支持的机械臂侧别：{arm_side}")
    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(robot_ip)
    info = cast(
        _RobotInfoProtocol,
        retry_non_motion_call(
            f"robotInfo({robot_ip})",
            lambda: robot.robotInfo(ec),
            settings.non_motion_retry_count,
            settings.non_motion_retry_delay_s,
        ),
    )
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"读取机械臂机器人信息失败：arm_side={arm_side}, ip={robot_ip}")
    apply_named_toolset(ConnectedArm(arm_side, robot_ip, robot, info.type, info.id, ec), settings)
    detected_arm_side = detect_arm_side(info.type, settings.arm)
    if detected_arm_side != arm_side:
        raise RuntimeError(f"连接到的机械臂侧别不匹配：expected={arm_side}, ip={robot_ip}, actual={detected_arm_side}")
    return ConnectedArm(detected_arm_side, robot_ip, robot, info.type, info.id, ec)


def prepare_nrt_motion(connected_arm: ConnectedArm, settings: ReplayServiceSettings) -> None:
    """按旧自动回放的控制器顺序进入可执行 NRT 状态。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    arm_settings = settings.arm
    robot.stop(ec)
    _raise_if_sdk_error(connected_arm, "stop")
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _raise_if_sdk_error(connected_arm, "setMotionControlMode(NrtCommandMode)")
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    _raise_if_sdk_error(connected_arm, "setOperateMode(automatic)")
    robot.setPowerState(True, ec)
    _raise_if_sdk_error(connected_arm, "setPowerState(True)")
    _wait_power_on(
        robot,
        ec,
        arm_settings.power_on_timeout_s,
        arm_settings.power_on_poll_interval_s,
    )
    robot.setDefaultConfOpt(False, ec)
    _raise_if_sdk_error(connected_arm, "setDefaultConfOpt(False)")
    apply_named_toolset(connected_arm, settings)
    robot.setDefaultSpeed(arm_settings.default_cartesian_speed_mm_s, ec)
    _raise_if_sdk_error(connected_arm, f"setDefaultSpeed({arm_settings.default_cartesian_speed_mm_s})")
    robot.setDefaultZone(arm_settings.default_cartesian_zone_mm, ec)
    _raise_if_sdk_error(connected_arm, f"setDefaultZone({arm_settings.default_cartesian_zone_mm})")
    if robot.powerState(ec) != xCoreSDK_python.PowerState.on:
        raise RuntimeError(f"{connected_arm.arm_side} 臂 NRT 准备完成后电机未处于 on 状态")
    _raise_if_sdk_error(connected_arm, "powerState")


def close_arm(connected_arm: ConnectedArm) -> None:
    """尽力停止、下电并断开机械臂。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    for operation in (
        robot.stop,
        robot.disableDrag,
        lambda context: robot.setPowerState(False, context),
        robot.disconnectFromRobot,
    ):
        try:
            operation(ec)
        except Exception as error:
            logger.warning("关闭 {} 臂时发生异常：{}", connected_arm.arm_side, error)


def apply_named_toolset(connected_arm: ConnectedArm, settings: ReplayServiceSettings) -> None:
    """以旧重试语义应用当前回放固定的 tool/wobj 坐标系。"""

    arm_settings = settings.arm
    result = retry_non_motion_call(
        f"setToolset({connected_arm.robot_ip})",
        lambda: connected_arm.robot.setToolset(arm_settings.tool_name, arm_settings.wobj_name, connected_arm.ec),
        settings.non_motion_retry_count,
        settings.non_motion_retry_delay_s,
    )
    if result is None or connected_arm.ec.get("ec", 0) != 0:
        raise RuntimeError(
            f"设置默认工具/工件失败：ip={connected_arm.robot_ip}, "
            f"tool={arm_settings.tool_name}, wobj={arm_settings.wobj_name}"
        )


def _wait_power_on(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    """等待 SDK 的电源状态变为 on，超时即按旧准备语义失败。"""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if robot.powerState(ec) == xCoreSDK_python.PowerState.on:
            return
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"读取机械臂电源状态失败：{ec}")
        time.sleep(poll_interval_s)
    raise RuntimeError("机械臂未在超时内确认上电")


def detect_arm_side(robot_type: str, settings: ReplayArmSettings) -> str:
    """按控制器上报型号识别左右臂，拒绝未知机型。"""

    for arm_side, expected_robot_type in (("left", settings.left_arm_type), ("right", settings.right_arm_type)):
        if robot_type == expected_robot_type:
            return arm_side
    raise ValueError(f"未识别的机器人型号：{robot_type}")


def _raise_if_sdk_error(connected_arm: ConnectedArm, operation: str) -> None:
    """在每一条 NRT 准备命令后立即保留旧流程的失败边界。"""

    if connected_arm.ec.get("ec", 0) != 0:
        raise RuntimeError(f"{connected_arm.arm_side} 臂 {operation} 失败：{connected_arm.ec}")


# endregion
