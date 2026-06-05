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

    职责边界：
    - 只描述 qmlinker SDK 内置 FK/IK 模型暴露的关节角度限位。
    - 不负责 GUI 控件创建、运动命令发送或硬件状态刷新。

    设计思想：
    - 以 qmlinker 中 `fkik.joint_min/joint_max` 的真实值为准，避免 GUI 使用占位范围。
    - 左右臂分开保存，因为右臂安装与模型限位并不是左臂简单复制。

    生命周期：
    - 不持有网络连接，可作为配置常量跨线程读取。

    继承关系：
    - 不继承业务基类，作为机械臂 UI 与协议层共享的数据契约。
    """

    name: str
    "关节名，格式为 `j1` 到 `j6`。"

    minimum_deg: float
    "关节最小角度，单位 deg。"

    maximum_deg: float
    "关节最大角度，单位 deg。"

    unit: str = "deg"
    "角度单位，当前固定为 deg。"


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
