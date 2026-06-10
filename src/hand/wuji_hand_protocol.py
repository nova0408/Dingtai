from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# region 数据结构

HandDeviceName = Literal["right_hand"]
HandAxisName = str


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
    - `device_name` 仅表示右手灵巧手，对应 qmlinker 的 `HAND_RIGHT`。
    - 左手语义不再纳入通用 hand 协议，直接由 gripper 客户端处理。

    生命周期：
    - 不持有外部资源，可作为 GUI 与协议层之间的只读配置。

    继承关系：
    - 不继承业务基类，避免在配置层引入硬件生命周期。
    """

    device_name: HandDeviceName
    "手部硬件实例名，当前仅取值为 `right_hand`。"

    title: str
    "GUI 分组标题。"

    actuator_count: int
    "当前手部可用执行器数量，由 qmlinker 运行时读取。"


# endregion


# region 配置

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

    prefix = "right_hand_a"
    if axis_name.startswith(prefix):
        index_text = axis_name.removeprefix(prefix)
        if index_text.isdigit():
            return "right_hand", int(index_text)
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

    return tuple(f"{instance.device_name}_a{idx}" for idx in range(instance.actuator_count))


# endregion
