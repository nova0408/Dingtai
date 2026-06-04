from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# region 数据结构

HandDeviceName = Literal["left_hand", "right_hand"]
HandSpecName = Literal["hand_3", "hand_5"]


@dataclass(frozen=True, slots=True)
class WujiHandActuatorLimit:
    """无际手部单个执行器位置限制。

    职责边界：
    - 只描述手部 GUI 与 qmlinker `QMHand` 命令共享的执行器位置范围。
    - 不负责读取触觉、执行器电流、温度、卡滞状态或硬件连接。

    设计思想：
    - 使用接口文档示例中的 actuator 命令结构，保留 actuator_id 与位置范围。
    - 位置单位按 qmlinker 示例保持为归一化比例，不在协议层换算为角度或距离。

    生命周期：
    - 不持有网络连接，可跨线程只读共享。

    继承关系：
    - 不继承业务基类，作为手部协议配置数据使用。
    """

    name: str
    "执行器名称，用于 GUI 轴名后缀。"

    actuator_id: int
    "qmlinker 手部执行器 ID，按示例从 0 开始。"

    minimum: float
    "最小执行器位置，单位为归一化比例。"

    maximum: float
    "最大执行器位置，单位为归一化比例。"

    unit: str = ""
    "显示单位，归一化比例不额外显示单位。"


@dataclass(frozen=True, slots=True)
class WujiHandInstanceSpec:
    """无际手部硬件实例与规格模板的绑定。

    职责边界：
    - 只表达左手或右手当前挂载哪一种手部规格。
    - 不负责选择 qmlinker 常量、创建 `QMHand`、发送控制命令或读取状态。

    设计思想：
    - `device_name` 按硬件实例划分，对应 qmlinker 的 `HAND_LEFT` 与 `HAND_RIGHT`。
    - `spec_name` 按可替换手部规格划分，便于后续把左右手替换为不同执行器数量。

    生命周期：
    - 不持有外部资源，可作为 GUI 与协议层之间的只读配置。

    继承关系：
    - 不继承业务基类，避免在配置层引入硬件生命周期。
    """

    device_name: HandDeviceName
    "手部硬件实例名，取值为 `left_hand` 或 `right_hand`。"

    title: str
    "GUI 分组标题。"

    spec_name: HandSpecName
    "手部规格模板名，例如 `hand_3` 或 `hand_5`。"


# endregion


# region 配置

WUJI_HAND_SPECS: dict[HandSpecName, tuple[WujiHandActuatorLimit, ...]] = {
    "hand_3": tuple(WujiHandActuatorLimit(f"a{idx}", idx, 0.0, 0.95) for idx in range(7)),
    "hand_5": tuple(WujiHandActuatorLimit(f"a{idx}", idx, 0.0, 0.5) for idx in range(10)),
}
"接口文档示例中的手部规格模板，位置单位为归一化比例。"

DEFAULT_WUJI_HAND_INSTANCES: tuple[WujiHandInstanceSpec, ...] = (
    WujiHandInstanceSpec("left_hand", "left hand", "hand_3"),
    WujiHandInstanceSpec("right_hand", "right hand", "hand_3"),
)
"默认左右手实例配置；后续替换手型时只调整 `spec_name`。"


# endregion


# region 轴与设备映射

def parse_hand_axis_name(axis_name: str) -> tuple[HandDeviceName, int] | None:
    """解析 GUI 手部轴名为手部硬件实例与执行器 ID。

    Parameters
    ----------
    axis_name:
        GUI 轴名，例如 `left_hand_a0` 或 `right_hand_a6`。

    Returns
    -------
    tuple[HandDeviceName, int] | None
        成功时返回手部实例名与 0 基执行器 ID；非手部轴返回 `None`。
    """

    for instance in DEFAULT_WUJI_HAND_INSTANCES:
        prefix = f"{instance.device_name}_a"
        if not axis_name.startswith(prefix):
            continue
        index_text = axis_name.removeprefix(prefix)
        if index_text.isdigit():
            actuator_id = int(index_text)
            if 0 <= actuator_id < len(WUJI_HAND_SPECS[instance.spec_name]):
                return instance.device_name, actuator_id
        return None
    return None


def axis_names_for_hand(instance: WujiHandInstanceSpec) -> tuple[str, ...]:
    """返回指定手部实例对应的 GUI 轴名序列。

    Parameters
    ----------
    instance:
        手部硬件实例与规格模板绑定。

    Returns
    -------
    tuple[str, ...]
        GUI 轴名序列，顺序与 qmlinker actuator_id 一致。
    """

    return tuple(f"{instance.device_name}_{limit.name}" for limit in WUJI_HAND_SPECS[instance.spec_name])


# endregion
