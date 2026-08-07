"""统一入口配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """统一入口监听和业务服务目标配置。"""

    host: str = "0.0.0.0"
    "统一入口监听地址；默认接受外部网络客户端请求。"

    port: int = 443
    "统一入口监听端口。"

    camera_http_host: str = "127.0.0.1"
    "CameraPipeline HTTP 后端地址。"

    camera_http_port: int = 6400
    "CameraPipeline HTTP 后端端口。"

    camera_websocket_host: str = "127.0.0.1"
    "CameraPipeline WebSocket 后端地址。"

    camera_websocket_port: int = 6401
    "CameraPipeline WebSocket 后端端口。"

    record_replay_host: str = "127.0.0.1"
    "RecordReplay HTTP 后端地址。"

    record_replay_port: int = 6300
    "RecordReplay HTTP 后端端口。"

    record_replay_websocket_host: str = "127.0.0.1"
    "RecordReplay 状态 WebSocket 后端地址。"

    record_replay_websocket_port: int = 6301
    "RecordReplay 状态 WebSocket 后端端口。"

    robot_control_host: str = "127.0.0.1"
    "RobotControl HTTP 后端地址。"

    robot_control_port: int = 6500
    "RobotControl HTTP 后端端口。"

    calibration_host: str = "127.0.0.1"
    "手眼标定与先验记录服务地址。"

    calibration_port: int = 6600
    "手眼标定与先验记录服务 HTTP 端口。"
