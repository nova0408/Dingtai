"""RobotControl 服务配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RobotControlSettings:
    """机器人控制服务的现场连接配置。

    职责边界：
    - 只保存服务监听地址和现场设备端点。
    - 不创建 qmlinker channel、xCoreSDK 对象或 SSH 隧道。

    生命周期：
    - 服务启动时创建一次，作为不可变配置跨请求共享。
    """

    http_host: str = "127.0.0.1"
    "HTTP 监听地址；默认仅允许本机或 SSH 转发访问。"

    http_port: int = 6500
    "HTTP 监听端口。"

    status_stream_interval_s: float = 0.2
    "状态 SSE 事件流默认推送间隔，单位为秒。"

    qmlinker_host: str = "192.168.100.60"
    "qmlinker 默认服务地址。"

    qmlinker_port: int = 50062
    "qmlinker 默认服务端口。"

    qmlinker_waist_available: bool = True
    "是否声明当前机型支持 qmlinker 腰部状态能力；不支持的机型设为 False。"

    agv_host: str = "192.168.100.70"
    "AGV qmlinker 服务地址。"

    gripper_port: int = 50066
    "大寰夹爪 qmlinker 服务端口。"

    left_ar5_ip: str = "192.168.100.161"
    "左侧 AR5 控制器地址。"

    right_ar5_ip: str = "192.168.100.160"
    "右侧 AR5 控制器地址。"
