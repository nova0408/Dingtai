from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# region 数据结构

WujiQmlinkerEnableModuleName = Literal["body", "head", "left_arm", "right_arm"]


@dataclass(frozen=True, slots=True)
class WujiRuntimeAxisSpec:
    """运行时发现的单轴规格。"""

    axis_name: str
    minimum: float
    maximum: float
    unit: str
    control_supported: bool = True
    refresh_supported: bool = True


@dataclass(frozen=True, slots=True)
class WujiRuntimeModuleSpec:
    """运行时发现的模块规格。"""

    tab_name: str
    title: str
    device_name: str
    axes: tuple[WujiRuntimeAxisSpec, ...] = ()
    enable_supported: bool = True
    refresh_supported: bool = True


@dataclass(frozen=True, slots=True)
class WujiRobotRuntimeStructure:
    """qmlinker 运行时机器人结构快照。"""

    modules: tuple[WujiRuntimeModuleSpec, ...]


# endregion


# region 配置

SUPPORTED_WUJI_QMLINKER_ENABLE_MODULES: tuple[WujiQmlinkerEnableModuleName, ...] = (
    "body",
    "head",
    "left_arm",
    "right_arm",
)
"当前 qmlinker 可真实读写使能状态的整机模块。手与 AGV 状态由各自子模块描述。"


# endregion
