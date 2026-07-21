"""双臂回放的 qmlinker 手部与升降设备网关。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from qmlinker import QMGripper, QMHand, QMLift, create_channel

# region 数据结构


@dataclass(slots=True)
class HandBodyRuntime:
    """一个机械臂侧别关联的 qmlinker 设备对象。"""

    lift: QMLift
    "qmlinker 升降对象。"
    gripper: QMGripper | None = None
    "左侧大寰夹爪，仅 left runtime 有效。"
    right_hand: QMHand | None = None
    "右侧 M11 灵巧手，仅 right runtime 有效。"


# endregion


# region 生命周期


def create_hand_body_runtime(
    arm_side: str,
    qmlinker_host: str,
    qmlinker_port: int,
    gripper_port: int,
) -> HandBodyRuntime:
    """创建指定侧别的 hand/body 设备 runtime。"""

    if arm_side not in {"left", "right"}:
        raise ValueError(f"不支持的机械臂侧别：{arm_side}")
    body_channel = create_channel(f"{qmlinker_host}:{qmlinker_port}")
    lift = QMLift(body_channel)
    if arm_side == "left":
        gripper_channel = create_channel(f"{qmlinker_host}:{gripper_port}")
        return HandBodyRuntime(lift, gripper=QMGripper(gripper_channel))
    return HandBodyRuntime(lift, right_hand=QMHand(body_channel, cast(str, QMHand.HAND_RIGHT)))


def close_hand_body_runtime(runtime: HandBodyRuntime | None) -> None:
    """释放手部 runtime 引用；qmlinker channel 由对象自身管理。"""

    del runtime


# endregion
