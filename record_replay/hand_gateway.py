"""双臂回放的 qmlinker 手部与升降设备网关。"""

from __future__ import annotations

from dataclasses import dataclass

from src.wuji.body_client import WujiBodyClient
from src.wuji.dahuan_gripper_client import DahuanGripperClient
from src.wuji.qmlinker_session import create_qmlinker_channel
from src.wuji.right_hand_client import WujiRightHandClient

# region 数据结构


@dataclass(slots=True)
class HandBodyRuntime:
    """一个机械臂侧别关联的手部与 body 客户端。"""

    body: WujiBodyClient
    "升降与腰部客户端，连接由 service 的本地 qmlinker 地址提供。"
    gripper: DahuanGripperClient | None = None
    "左侧大寰夹爪，仅 left runtime 有效。"
    right_hand: WujiRightHandClient | None = None
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
    body_channel = create_qmlinker_channel(f"{qmlinker_host}:{qmlinker_port}")
    body = WujiBodyClient(body_channel)
    if arm_side == "left":
        gripper_channel = create_qmlinker_channel(f"{qmlinker_host}:{gripper_port}")
        return HandBodyRuntime(body, gripper=DahuanGripperClient(gripper_channel))
    return HandBodyRuntime(body, right_hand=WujiRightHandClient(body_channel))


def close_hand_body_runtime(runtime: HandBodyRuntime | None) -> None:
    """释放手部 runtime 引用；SSH 隧道由测试入口负责关闭。"""

    del runtime


# endregion
