from __future__ import annotations

from dataclasses import dataclass


# region 右手数据结构


@dataclass(frozen=True, slots=True)
class WujiRightHandActuatorSpec:
    """右手单个执行器的固定轴定义。

    职责边界：
    - 只描述右手灵巧手的固定轴号、显示名和控制范围。
    - 不负责读取状态、不负责发送控制命令。

    设计思想：
    - 右手轴数已由现场读取结果确认，直接用硬编码结构体固定，避免再从运行时动态枚举。
    - 轴顺序与 `qmlinker` actuator_id 一致，供 GUI 和订阅缓存直接使用。

    生命周期：
    - 纯只读数据，可跨线程共享。

    继承关系：
    - 不继承业务基类。
    """

    axis_name: str
    "GUI 轴名，例如 `right_hand_a0`。"

    actuator_id: int
    "qmlinker 执行器 ID，固定为 0 到 10。"

    label: str
    "界面显示名称，直接对应右手执行器中文名称。"

    minimum: float = 0.0
    "最小位置，单位为归一化比例。"

    maximum: float = 1.0
    "最大位置，单位为归一化比例。"


RIGHT_HAND_ACTUATOR_SPECS: tuple[WujiRightHandActuatorSpec, ...] = (
    WujiRightHandActuatorSpec("right_hand_a0", 0, "大拇指偏摆"),
    WujiRightHandActuatorSpec("right_hand_a1", 1, "大拇指根"),
    WujiRightHandActuatorSpec("right_hand_a2", 2, "大拇指末端"),
    WujiRightHandActuatorSpec("right_hand_a3", 3, "食指根部"),
    WujiRightHandActuatorSpec("right_hand_a4", 4, "食指末端"),
    WujiRightHandActuatorSpec("right_hand_a5", 5, "中指根部"),
    WujiRightHandActuatorSpec("right_hand_a6", 6, "中指末端"),
    WujiRightHandActuatorSpec("right_hand_a7", 7, "无名指根部"),
    WujiRightHandActuatorSpec("right_hand_a8", 8, "无名指末端"),
    WujiRightHandActuatorSpec("right_hand_a9", 9, "小指根部"),
    WujiRightHandActuatorSpec("right_hand_a10", 10, "小指末端"),
)

# 方便按“同一组四根手指一起动”的方式控制右手。
# 第一组对应食指 / 中指 / 无名指 / 小指的近端关节。
RIGHT_HAND_FOUR_FINGER_GROUP_A: tuple[int, ...] = (4, 6, 8, 10)
"四根手指同组 A，按同一归一化目标值同时控制。"

# 第二组对应食指 / 中指 / 无名指 / 小指的远端关节。
RIGHT_HAND_FOUR_FINGER_GROUP_B: tuple[int, ...] = (3, 5, 7, 9)
"四根手指同组 B，按同一归一化目标值同时控制。"


# endregion
