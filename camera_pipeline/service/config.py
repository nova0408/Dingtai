from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraPipelineServiceConfig:
    """统一服务监听地址、外部 API 地址和请求循环超时。"""

    service_bind_addr: str = "tcp://0.0.0.0:6200"
    "统一 REQ/REP 服务监听地址。"
    frame_bind_addr: str = "tcp://0.0.0.0:6201"
    "完整 RGBD 帧发布地址。"
    color_bind_addr: str = "tcp://0.0.0.0:6202"
    "彩色帧发布地址。"
    depth_bind_addr: str = "tcp://0.0.0.0:6203"
    "深度帧发布地址。"
    http_host: str = "0.0.0.0"
    "外部 HTTP/JSON API 监听地址。"
    http_port: int = 6400
    "外部 HTTP/JSON API 监听端口。"
    websocket_host: str = "0.0.0.0"
    "外部 WebSocket 图像流监听地址。"
    websocket_port: int = 6401
    "外部 WebSocket 图像流监听端口。"
    request_receive_timeout_ms: int = 500
    "REP 循环接收超时，单位 ms，用于及时响应停止事件。"
    response_send_timeout_ms: int = 30_000
    "REP 响应发送超时，单位 ms。"
    log_path: str = "logs/camera_pipeline_service.log"
    "长期运行日志文件路径；相对路径基于服务工作目录。"
    log_rotation: str = "1 hour"
    "日志文件每小时轮转一次。"
    log_retention: str = "7 days"
    "轮转日志保留时间。"
