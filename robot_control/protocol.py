"""RobotControl HTTP 协议数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from . import ROBOT_CONTROL_VERSION

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

API_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DeviceState:
    """单个设备的统一状态结果。

    ``data`` 是已经完成单位转换和 JSON 边界收窄的状态字段，不暴露第三方 SDK 对象。
    """

    name: str
    "设备稳定名称，例如 ``ar5_left`` 或 ``qmlinker_head``。"

    backend: str
    "底层实现名称，例如 ``xcoresdk`` 或 ``qmlinker``。"

    connected: bool
    "本次只读状态是否成功读取。"

    error: str | None
    "读取失败时的错误摘要；成功时为 None。"

    data: dict[str, JsonValue]
    "只读状态字段，单位由字段名和 API 文档明确表达。"


@dataclass(frozen=True, slots=True)
class RobotControlStatus:
    """一次机器人统一状态读取结果。"""

    service_version: str
    "RobotControl 服务版本。"

    api_version: str
    "对外 HTTP API 主版本。"

    devices: tuple[DeviceState, ...]
    "本次读取的全部设备状态，单项失败不遮蔽其它设备。"


@dataclass(frozen=True, slots=True)
class ActionResponse:
    """控制请求已被服务接受后的统一响应。

    该结构只描述服务端接受结果，不声称动作已经物理完成。
    """

    service_version: str
    "RobotControl 服务版本。"

    api_version: str
    "对外 HTTP API 主版本。"

    accepted: bool
    "服务端是否接受本次控制请求。"

    data: dict[str, JsonValue]
    "控制请求结果摘要。"


def health_payload() -> dict[str, JsonValue]:
    """构造不连接硬件的健康检查响应。"""

    return {
        "service_version": ROBOT_CONTROL_VERSION,
        "api_version": API_VERSION,
        "hardware_access": "lazy",
    }
