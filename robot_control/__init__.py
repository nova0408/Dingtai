"""机器人统一控制服务。

本包把 qmlinker 设备客户端与 AR5 xCoreSDK 客户端收口到同一个 HTTP 服务边界。
服务运行时才创建硬件对象，导入本包不会连接或控制现场设备。
"""

from .config import RobotControlSettings

ROBOT_CONTROL_VERSION = "0.10.1"

__all__ = ["ROBOT_CONTROL_VERSION", "RobotControlSettings"]
