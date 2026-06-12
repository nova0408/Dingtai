from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# region 数据结构

ArmSide = Literal["left", "right"]
ArmDeviceName = Literal["left_arm", "right_arm"]
WujiBodyAxisName = Literal["body_z", "body_ry"]
WujiHeadAxisName = Literal["head_yaw"]


@dataclass(frozen=True, slots=True)
class WujiArmJointLimit:
    """qmlinker 机械臂单关节真实限位。
    """

    name: str
    "关节名，从 `j1` 开始递增。"

    minimum_deg: float
    "关节最小角度，单位 deg。"

    maximum_deg: float
    "关节最大角度，单位 deg。"



@dataclass(frozen=True, slots=True)
class WujiAxisLimit:
    """无际单轴运动范围定义。

    职责边界：
    - 只描述单个身体或头部轴在 GUI 与协议层共享的显示范围。
    - 不负责读取实时位置、发送控制命令或推断硬件限位。

    设计思想：
    - 机械臂关节范围优先来自 qmlinker FK/IK 模型；身体与头部当前 proto 未暴露动态限位接口，
      因此将文档确认过的静态范围集中定义，避免散落硬编码。
    - 保持字段命名与机械臂关节限位一致，便于 GUI 构造统一的 DoF 模型。

    生命周期：
    - 作为不可变配置常量长期存在，不持有网络连接或硬件资源。

    继承关系：
    - 不继承业务基类，作为协议与 GUI 共享的数据契约。
    """

    name: str
    "轴名，例如 `body_ry` 或 `head_yaw`。"

    minimum: float
    "轴最小值，单位由 `unit` 指定。"

    maximum: float
    "轴最大值，单位由 `unit` 指定。"

    unit: str
    "显示单位，例如 `deg` 或 `mm`。"


# endregion


# region 轴与设备映射

SUPPORTED_ARM_DEVICES: tuple[ArmDeviceName, ...] = ("left_arm", "right_arm")
"当前 arm 接口文档可真实控制的机械臂设备。"

WUJI_ARM_JOINT_LIMITS_DEG: dict[ArmDeviceName, tuple[WujiArmJointLimit, ...]] = {
    "left_arm": (
        WujiArmJointLimit("j1", -269.98, 269.98),
        WujiArmJointLimit("j2", -84.0, 84.8),
        WujiArmJointLimit("j3", -269.98, 269.98),
        WujiArmJointLimit("j4", -196.7, 35.01),
        WujiArmJointLimit("j5", -180.02, 180.02),
        WujiArmJointLimit("j6", -176.99, 0.52),
    ),
    "right_arm": (
        WujiArmJointLimit("j1", -449.77, 89.95),
        WujiArmJointLimit("j2", -95.0, 84.0),
        WujiArmJointLimit("j3", -269.98, 269.98),
        WujiArmJointLimit("j4", -214.97, 0.0),
        WujiArmJointLimit("j5", -180.02, 180.02),
        WujiArmJointLimit("j6", -180.02, 0.0),
    ),
}
"qmlinker SDK FK/IK 模型暴露的左右臂真实关节限位，角度单位 deg。"

WUJI_BODY_AXIS_LIMITS: dict[WujiBodyAxisName, WujiAxisLimit] = {
    "body_z": WujiAxisLimit("body_z", 0.0, 850.0, "mm"),
    "body_ry": WujiAxisLimit("body_ry", -30.0, 30.0, "deg"),
}
"无际本体轴显示范围。当前 qmlinker proto 未提供动态限位查询接口，因此使用文档确认值。"

WUJI_HEAD_AXIS_LIMITS: dict[WujiHeadAxisName, WujiAxisLimit] = {
    "head_yaw": WujiAxisLimit("head_yaw", -90.0, 90.0, "deg"),
}
"无际头部轴显示范围。当前 qmlinker proto 未提供动态限位查询接口，因此使用文档确认值。"


def parse_arm_axis_name(axis_name: str) -> tuple[ArmDeviceName, int] | None:
    """解析 GUI DoF 轴名为机械臂设备与关节索引。

    Parameters
    ----------
    axis_name:
        GUI 轴名，例如 `left_j1` 或 `right_j6`。

    Returns
    -------
    tuple[ArmDeviceName, int] | None
        成功时返回设备名与 1 基关节索引；非机械臂轴返回 `None`。
    """

    if axis_name.startswith("left_j"):
        index_text = axis_name.removeprefix("left_j")
        if index_text.isdigit() and 1 <= int(index_text) <= 6:
            return "left_arm", int(index_text)
        return None
    if axis_name.startswith("right_j"):
        index_text = axis_name.removeprefix("right_j")
        if index_text.isdigit() and 1 <= int(index_text) <= 6:
            return "right_arm", int(index_text)
        return None
    return None


def parse_body_axis_name(axis_name: str) -> WujiBodyAxisName | None:
    """解析身体轴名。

    Parameters
    ----------
    axis_name:
        GUI 轴名，当前支持 `body_z` 与 `body_ry`。

    Returns
    -------
    WujiBodyAxisName | None
        身体轴名；非身体轴返回 `None`。
    """

    if axis_name == "body_z":
        return "body_z"
    if axis_name == "body_ry":
        return "body_ry"
    return None


def parse_head_axis_name(axis_name: str) -> WujiHeadAxisName | None:
    """解析头部轴名。

    Parameters
    ----------
    axis_name:
        GUI 轴名，当前支持 `head_yaw`。

    Returns
    -------
    WujiHeadAxisName | None
        头部轴名；非头部轴返回 `None`。
    """

    if axis_name == "head_yaw":
        return "head_yaw"
    return None


def axis_names_for_device(device_name: ArmDeviceName, joint_count: int) -> tuple[str, ...]:
    """返回指定机械臂对应的 GUI 轴名序列。

    Parameters
    ----------
    device_name:
        机械臂设备名，取值为 `left_arm` 或 `right_arm`。

    Returns
    -------
    tuple[str, ...]
        GUI 轴名序列，长度由 `joint_count` 决定，单位语义为 deg。
    """

    prefix = "left" if device_name == "left_arm" else "right"
    return tuple(f"{prefix}_j{idx}" for idx in range(1, joint_count + 1))


# endregion
