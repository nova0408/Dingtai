"""基于 aiohttp 的 HTTP/WebSocket 反向代理。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

from . import API_GATEWAY_VERSION
from .config import GatewaySettings
from .routes import GatewayRoute, build_routes

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_WEBSOCKET_MAX_MESSAGE_SIZE = 16 * 1024 * 1024
_CORS_ALLOWED_ORIGINS = frozenset({"https://wujibrain-desktop"})
_CORS_ALLOWED_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CORS_ALLOWED_HEADERS = "Accept, Content-Type"
_CORS_ALLOWED_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS"


def create_app(settings: GatewaySettings | None = None) -> web.Application:
    """创建统一入口应用，不连接后端或硬件。"""

    resolved_settings = settings or GatewaySettings()
    routes = build_routes(resolved_settings)
    app = web.Application(middlewares=(_cors_middleware,))
    app["gateway_settings"] = resolved_settings
    app["gateway_routes"] = routes
    app.router.add_get("/api/v1/gateway/health", _gateway_health)
    app.router.add_route(
        "*",
        "/api/v1/camera-ws/{tail:.*}",
        _make_websocket_handler(routes[0]),
    )
    app.router.add_route("*", "/api/v1/camera-ws", _make_websocket_handler(routes[0]))
    app.router.add_route(
        "*",
        "/api/v1/camera/{tail:.*}",
        _make_http_handler(routes[1]),
    )
    app.router.add_route("*", "/api/v1/camera", _make_http_handler(routes[1]))
    app.router.add_route(
        "*",
        "/api/v1/record-replay/{tail:.*}",
        _make_http_handler(routes[2]),
    )
    app.router.add_route("*", "/api/v1/record-replay", _make_http_handler(routes[2]))
    app.router.add_route(
        "*",
        "/api/v1/record-replay-ws/{tail:.*}",
        _make_websocket_handler(routes[3]),
    )
    app.router.add_route("*", "/api/v1/record-replay-ws", _make_websocket_handler(routes[3]))
    app.router.add_route(
        "*",
        "/api/v1/robot-control/{tail:.*}",
        _make_http_handler(routes[4]),
    )
    app.router.add_route("*", "/api/v1/robot-control", _make_http_handler(routes[4]))
    app.router.add_route(
        "*",
        "/api/v1/calibration/{tail:.*}",
        _make_http_handler(routes[5]),
    )
    app.router.add_route("*", "/api/v1/calibration", _make_http_handler(routes[5]))
    return app


async def _gateway_health(request: web.Request) -> web.Response:
    """返回 Gateway 自身信息，不探测后端。"""

    routes: tuple[GatewayRoute, ...] = request.app["gateway_routes"]
    return web.json_response(
        {
            "gateway_version": API_GATEWAY_VERSION,
            "backend_ports": {
                "camera_http": routes[1].upstream_port,
                "camera_websocket": routes[0].upstream_port,
                "record_replay": routes[2].upstream_port,
                "record_replay_websocket": routes[3].upstream_port,
                "robot_control": routes[4].upstream_port,
                "calibration": routes[5].upstream_port,
            },
            "backend_probe": False,
        }
    )


def _make_http_handler(route: GatewayRoute) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """为一个 HTTP 后端构造显式转发处理器。"""

    async def _handle(request: web.Request) -> web.StreamResponse:
        session: ClientSession = request.app.get("client_session")
        if session is None:
            raise web.HTTPServiceUnavailable(text="gateway client session is not ready")
        target = _target_url(route, request)
        headers = _upstream_headers(request, route)
        body = await request.read()
        try:
            upstream = await session.request(
                request.method,
                target,
                headers=headers,
                data=body,
                allow_redirects=False,
            )
        except OSError as exc:
            raise web.HTTPBadGateway(text=f"gateway upstream failure: {exc}") from exc

        response = web.StreamResponse(
            status=upstream.status,
            reason=upstream.reason,
            headers=_downstream_headers(upstream.headers),
        )
        origin = _allowed_cors_origin(request.headers.get("Origin"))
        if origin is not None:
            _add_cors_headers(response, origin)
        await response.prepare(request)
        try:
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await response.write(chunk)
        finally:
            upstream.close()
        await response.write_eof()
        return response

    return _handle


def _make_websocket_handler(
    route: GatewayRoute,
) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """为 CameraPipeline CPWS1 构造 WebSocket 双向转发处理器。"""

    async def _handle(request: web.Request) -> web.StreamResponse:
        session: ClientSession = request.app.get("client_session")
        if session is None:
            raise web.HTTPServiceUnavailable(text="gateway client session is not ready")
        client_ws = web.WebSocketResponse(
            autoping=True,
            max_msg_size=_WEBSOCKET_MAX_MESSAGE_SIZE,
        )
        await client_ws.prepare(request)
        upstream = None
        try:
            upstream = await session.ws_connect(
                _target_url(route, request, websocket=True),
                headers=_upstream_headers(request, route),
                autoping=True,
                heartbeat=30,
                max_msg_size=_WEBSOCKET_MAX_MESSAGE_SIZE,
            )
            await _relay_websocket(client_ws, upstream)
        except OSError as exc:
            await client_ws.close(code=1011, message=str(exc).encode("utf-8"))
        finally:
            if upstream is not None:
                await upstream.close()
            await client_ws.close()
        return client_ws

    return _handle


async def _relay_websocket(
    client_ws: web.WebSocketResponse,
    upstream: Any,
) -> None:
    """在客户端和 CameraPipeline WebSocket 之间转发消息。"""

    async def _client_to_upstream() -> None:
        async for message in client_ws:
            if message.type is WSMsgType.TEXT:
                await upstream.send_str(message.data)
            elif message.type is WSMsgType.BINARY:
                await upstream.send_bytes(message.data)
            elif message.type is WSMsgType.PING:
                await upstream.ping(message.data)
            elif message.type is WSMsgType.PONG:
                await upstream.pong(message.data)
            elif message.type in {WSMsgType.CLOSE, WSMsgType.ERROR}:
                break

    async def _upstream_to_client() -> None:
        async for message in upstream:
            if message.type is WSMsgType.TEXT:
                await client_ws.send_str(message.data)
            elif message.type is WSMsgType.BINARY:
                await client_ws.send_bytes(message.data)
            elif message.type is WSMsgType.PING:
                await client_ws.ping(message.data)
            elif message.type is WSMsgType.PONG:
                await client_ws.pong(message.data)
            elif message.type in {WSMsgType.CLOSE, WSMsgType.ERROR}:
                break

    tasks = {
        asyncio.create_task(_client_to_upstream()),
        asyncio.create_task(_upstream_to_client()),
    }
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


def _target_url(
    route: GatewayRoute,
    request: web.Request,
    *,
    websocket: bool = False,
) -> str:
    """构造后端 HTTP 或 WebSocket 地址。"""

    scheme = "ws" if websocket else "http"
    target = route.rewrite_target(request.raw_path)
    parsed = urlsplit(target)
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{route.upstream_host}:{route.upstream_port}{parsed.path}{query}"


def _upstream_headers(request: web.Request, route: GatewayRoute) -> dict[str, str]:
    """构造后端请求头，去除客户端连接级头部。"""

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }
    headers["Host"] = f"{route.upstream_host}:{route.upstream_port}"
    return headers


def _downstream_headers(headers: Any) -> dict[str, str]:
    """构造客户端响应头，去除后端连接级头部。"""

    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }


async def on_startup(app: web.Application) -> None:
    """创建共享后端 HTTP 客户端会话。"""

    app["client_session"] = ClientSession(
        timeout=ClientTimeout(total=None, connect=10.0, sock_connect=10.0)
    )


async def on_cleanup(app: web.Application) -> None:
    """关闭共享后端 HTTP 客户端会话。"""

    session: ClientSession | None = app.get("client_session")
    if session is not None:
        await session.close()


@web.middleware
async def _cors_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """为浏览器 GUI 提供受限的跨域响应和预检处理。"""

    origin = _allowed_cors_origin(request.headers.get("Origin"))
    if request.method == "OPTIONS" and request.headers.get("Origin"):
        if origin is None:
            raise web.HTTPForbidden(text="CORS origin is not allowed")
        response: web.StreamResponse = web.Response(status=204)
    else:
        response = await handler(request)

    if origin is not None:
        _add_cors_headers(response, origin)
    return response


def _allowed_cors_origin(origin: str | None) -> str | None:
    """只允许本机开发 GUI 和正式 hostname 来源，拒绝任意来源反射。"""

    if not origin:
        return None
    if origin in _CORS_ALLOWED_ORIGINS:
        return origin

    parsed = urlsplit(origin)
    if (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in _CORS_ALLOWED_LOCAL_HOSTS
        and not parsed.username
        and not parsed.password
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    ):
        return origin
    return None


def _add_cors_headers(response: web.StreamResponse, origin: str) -> None:
    """给响应添加不带凭据的 CORS 头。"""

    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = _CORS_ALLOWED_METHODS
    response.headers["Access-Control-Allow-Headers"] = _CORS_ALLOWED_HEADERS
    response.headers["Access-Control-Max-Age"] = "600"
    response.headers["Vary"] = "Origin"
