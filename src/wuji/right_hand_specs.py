from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# region 右手数据结构


@dataclass(frozen=True, slots=True)
class WujiRightHandActuatorSpec:
    """右手单个执行器的运行时轴定义。

    职责边界：
    - 只描述右手灵巧手的固定轴号、显示名和控制范围。
    - 不负责读取状态、不负责发送控制命令。

    设计思想：
    - 轴数量与名称来自 qmlinker `hand_info`，避免在客户端重复维护设备轴数。
    - 轴顺序与 `qmlinker` actuator_id 一致，供 GUI 和订阅缓存直接使用。

    生命周期：
    - 纯只读数据，可跨线程共享。

    继承关系：
    - 不继承业务基类。
    """

    axis_name: str
    "GUI 轴名，例如 `right_hand_a0`。"

    actuator_id: int
    "qmlinker 执行器 ID，从 0 开始连续编号。"

    label: str
    "界面显示名称，直接对应右手执行器中文名称。"

    minimum: float = 0.0
    "最小位置，单位为归一化比例。"

    maximum: float = 1.0
    "最大位置，单位为归一化比例。"


def build_right_hand_actuator_specs(
    actuator_count: int,
    actuator_names: Sequence[str],
) -> tuple[WujiRightHandActuatorSpec, ...]:
    """根据 qmlinker 手部信息生成右手执行器规格。

    Parameters
    ----------
    actuator_count:
        qmlinker 返回的执行器数量，单位 个。
    actuator_names:
        qmlinker 返回的执行器名称序列，顺序与 0 基执行器 ID 一致。

    Returns
    -------
    tuple[WujiRightHandActuatorSpec, ...]
        右手执行器规格，数量与 `actuator_count` 一致。

    Raises
    ------
    ValueError
        执行器数量不是正数，或名称数量与执行器数量不一致。
    """

    if actuator_count <= 0:
        raise ValueError(f"右手执行器数量必须为正数，当前为 {actuator_count}")
    if len(actuator_names) != actuator_count:
        raise ValueError(
            f"右手执行器名称数量与 actuator_count 不一致: "
            f"names={len(actuator_names)} count={actuator_count}"
        )
    return tuple(
        WujiRightHandActuatorSpec(
            axis_name=f"right_hand_a{actuator_id}",
            actuator_id=actuator_id,
            label=label,
        )
        for actuator_id, label in enumerate(actuator_names)
    )

# endregion
