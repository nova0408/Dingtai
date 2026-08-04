"""统一入口的显式 URL 路由。"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from .config import GatewaySettings


@dataclass(frozen=True, slots=True)
class GatewayRoute:
    """一个外部 URL 前缀到内部服务的映射。"""

    public_prefix: str
    "统一入口上的 URL 前缀。"

    upstream_host: str
    "后端服务地址。"

    upstream_port: int
    "后端服务端口。"

    upstream_prefix: str
    "转发到后端时替换外部前缀后的路径前缀。"

    websocket: bool = False
    "是否为 WebSocket 长连接路由。"

    def matches(self, path: str) -> bool:
        """判断路径是否属于本路由。"""

        return path == self.public_prefix or path.startswith(self.public_prefix + "/")

    def rewrite_target(self, target: str) -> str:
        """将外部 request-target 改写为后端 request-target。"""

        parsed = urlsplit(target)
        suffix = parsed.path[len(self.public_prefix) :]
        upstream_path = self.upstream_prefix.rstrip("/") + suffix
        if not upstream_path:
            upstream_path = "/"
        return urlunsplit(("", "", upstream_path, parsed.query, parsed.fragment))


def build_routes(settings: GatewaySettings) -> tuple[GatewayRoute, ...]:
    """按固定顺序构造统一入口路由。"""

    return (
        GatewayRoute(
            public_prefix="/api/v1/camera-ws",
            upstream_host=settings.camera_websocket_host,
            upstream_port=settings.camera_websocket_port,
            upstream_prefix="/api/v1/ws",
            websocket=True,
        ),
        GatewayRoute(
            public_prefix="/api/v1/camera",
            upstream_host=settings.camera_http_host,
            upstream_port=settings.camera_http_port,
            upstream_prefix="/api/v1",
        ),
        GatewayRoute(
            public_prefix="/api/v1/record-replay",
            upstream_host=settings.record_replay_host,
            upstream_port=settings.record_replay_port,
            upstream_prefix="",
        ),
        GatewayRoute(
            public_prefix="/api/v1/robot-control",
            upstream_host=settings.robot_control_host,
            upstream_port=settings.robot_control_port,
            upstream_prefix="/api/v1",
        ),
    )
