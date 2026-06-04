from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# region 数据结构

AgvAxisName = Literal["agv_x", "agv_y", "agv_yaw", "agv_battery"]


@dataclass(frozen=True, slots=True)
class WujiAgvStatusAxis:
    """无际 AGV 状态字段显示规格。

    职责边界：
    - 只描述底盘状态字段在 GUI 中的显示范围和单位。
    - 不负责底盘运动控制、导航、急停、状态查询或 qmlinker 连接。

    设计思想：
    - 按 `BaseService.GetBaseStatus` 响应字段组织，保持与接口文档一致。
    - 当前作为调试页展示规格，真实刷新后续由 `src.agv` 独立 adapter 接入。

    生命周期：
    - 不持有硬件连接，可作为只读配置跨线程传递。

    继承关系：
    - 不继承业务基类，避免配置层引入控制生命周期。
    """

    axis_name: AgvAxisName
    "GUI 状态轴名。"

    minimum: float
    "显示下限，单位由 `unit` 定义。"

    maximum: float
    "显示上限，单位由 `unit` 定义。"

    unit: str
    "显示单位，例如 m、deg 或 %。"


# endregion


# region 配置

WUJI_AGV_STATUS_AXES: tuple[WujiAgvStatusAxis, ...] = (
    WujiAgvStatusAxis("agv_x", -100.0, 100.0, "m"),
    WujiAgvStatusAxis("agv_y", -100.0, 100.0, "m"),
    WujiAgvStatusAxis("agv_yaw", -180.0, 180.0, "deg"),
    WujiAgvStatusAxis("agv_battery", 0.0, 100.0, "%"),
)
"BaseService.GetBaseStatus 的基础显示字段。"


# endregion


# region 轴与设备映射

def parse_agv_axis_name(axis_name: str) -> AgvAxisName | None:
    """解析 GUI AGV 状态轴名。

    Parameters
    ----------
    axis_name:
        GUI 轴名，例如 `agv_x`、`agv_y`、`agv_yaw` 或 `agv_battery`。

    Returns
    -------
    AgvAxisName | None
        成功时返回 AGV 状态轴名；非 AGV 轴返回 `None`。
    """

    if axis_name == "agv_x":
        return "agv_x"
    if axis_name == "agv_y":
        return "agv_y"
    if axis_name == "agv_yaw":
        return "agv_yaw"
    if axis_name == "agv_battery":
        return "agv_battery"
    return None


# endregion
