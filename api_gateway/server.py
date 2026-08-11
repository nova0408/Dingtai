"""基于 aiohttp 的 HTTP/WebSocket 反向代理。"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

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
_CORS_ALLOWED_ORIGINS = frozenset(
    {
        "https://wujibrain-desktop",
        "https://192.168.100.70",
        *(f"https://192.168.1.{last_octet}" for last_octet in range(1, 255)),
    }
)
_CORS_ALLOWED_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CORS_ALLOWED_HEADERS = "Accept, Content-Type"
_CORS_ALLOWED_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
_LOGGER = logging.getLogger(__name__)
_REQUEST_ID_KEY = "gateway_request_id"


def create_app(settings: GatewaySettings | None = None) -> web.Application:
    """创建统一入口应用，不连接后端或硬件。"""

    resolved_settings = settings or GatewaySettings()
    routes = build_routes(resolved_settings)
    app = web.Application(middlewares=(_error_middleware, _cors_middleware))
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
            return _gateway_error_response(
                request,
                status=503,
                error_code="gateway_not_ready",
                error_text="API Gateway 后端客户端会话尚未初始化，请稍后重试",
            )
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
        except (ClientError, TimeoutError, OSError) as error:
            _LOGGER.exception(
                "Gateway 上游连接失败 request_id=%s method=%s path=%s upstream=%s:%s "
                "type=%s",
                _request_id(request),
                request.method,
                request.path_qs,
                route.upstream_host,
                route.upstream_port,
                type(error).__name__,
            )
            return _gateway_error_response(
                request,
                status=502,
                error_code="gateway_upstream_unavailable",
                error_text=(
                    f"API Gateway 无法连接上游 {route.public_prefix} "
                    f"({route.upstream_host}:{route.upstream_port})；"
                    f"type={type(error).__name__} detail={_safe_error_detail(error)}"
                ),
            )

        if upstream.status >= 400:
            _LOGGER.warning(
                "Gateway 上游返回错误 request_id=%s method=%s path=%s upstream=%s:%s "
                "status=%s upstream_request_id=%s",
                _request_id(request),
                request.method,
                request.path_qs,
                route.upstream_host,
                route.upstream_port,
                upstream.status,
                upstream.headers.get("X-Request-ID"),
            )

        downstream_headers = _downstream_headers(upstream.headers)
        upstream_request_id = upstream.headers.get("X-Request-ID")
        if upstream_request_id is not None:
            downstream_headers["X-Upstream-Request-ID"] = upstream_request_id
        response = web.StreamResponse(
            status=upstream.status,
            reason=upstream.reason,
            headers=downstream_headers,
        )
        origin = _allowed_cors_origin(request.headers.get("Origin"))
        if origin is not None:
            _add_cors_headers(response, origin)
        response.headers["X-Request-ID"] = _request_id(request)
        await response.prepare(request)
        try:
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await response.write(chunk)
        except Exception as error:
            _LOGGER.exception(
                "Gateway 响应流转发失败 request_id=%s method=%s path=%s status=%s type=%s",
                _request_id(request),
                request.method,
                request.path_qs,
                upstream.status,
                type(error).__name__,
            )
            return response
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
            return _gateway_error_response(
                request,
                status=503,
                error_code="gateway_not_ready",
                error_text="API Gateway WebSocket 后端客户端会话尚未初始化，请稍后重试",
            )
        client_ws = web.WebSocketResponse(
            autoping=True,
            max_msg_size=_WEBSOCKET_MAX_MESSAGE_SIZE,
        )
        client_ws.headers["X-Request-ID"] = _request_id(request)
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
        except Exception as error:
            _LOGGER.exception(
                "Gateway WebSocket 转发失败 request_id=%s path=%s upstream=%s:%s type=%s",
                _request_id(request),
                request.path_qs,
                route.upstream_host,
                route.upstream_port,
                type(error).__name__,
            )
            close_message = (
                f"request_id={_request_id(request)} type={type(error).__name__} "
                f"detail={_safe_error_detail(error)}"
            )
            await client_ws.close(code=1011, message=close_message[:120].encode("utf-8"))
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
    _LOGGER.info("API Gateway 后端 HTTP/WebSocket 客户端会话已创建")


async def on_cleanup(app: web.Application) -> None:
    """关闭共享后端 HTTP 客户端会话。"""

    session: ClientSession | None = app.get("client_session")
    if session is not None:
        await session.close()
        _LOGGER.info("API Gateway 后端 HTTP/WebSocket 客户端会话已关闭")


@web.middleware
async def _error_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """统一记录请求，并把 Gateway 自身异常转换为可定位的 JSON。"""

    request[_REQUEST_ID_KEY] = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    _LOGGER.info(
        "Gateway 请求开始 request_id=%s method=%s path=%s remote=%s",
        _request_id(request),
        request.method,
        request.path_qs,
        request.remote,
    )
    try:
        response = await handler(request)
    except web.HTTPException as error:
        _LOGGER.warning(
            "Gateway HTTP 拒绝 request_id=%s method=%s path=%s status=%s reason=%s",
            _request_id(request),
            request.method,
            request.path_qs,
            error.status,
            error.text,
        )
        response = _gateway_error_response(
            request,
            status=error.status,
            error_code=_gateway_http_error_code(error.status),
            error_text=(
                f"API Gateway 拒绝请求 method={request.method} path={request.path_qs}；"
                f"reason={_safe_error_detail(error.text or error.reason)}"
            ),
        )
    except Exception as error:
        _LOGGER.exception(
            "Gateway 未处理异常 request_id=%s method=%s path=%s type=%s",
            _request_id(request),
            request.method,
            request.path_qs,
            type(error).__name__,
        )
        response = _gateway_error_response(
            request,
            status=500,
            error_code="gateway_internal_error",
            error_text=(
                f"API Gateway 处理请求时发生未预期异常；type={type(error).__name__} "
                f"detail={_safe_error_detail(error)}"
            ),
        )
    if not response.prepared:
        response.headers["X-Request-ID"] = _request_id(request)
    _LOGGER.info(
        "Gateway 请求结束 request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
        _request_id(request),
        request.method,
        request.path_qs,
        response.status,
        (time.perf_counter() - started_at) * 1000.0,
    )
    return response


@web.middleware
async def _cors_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """为浏览器 GUI 提供受限的跨域响应和预检处理。"""

    origin = _allowed_cors_origin(request.headers.get("Origin"))
    if request.method == "OPTIONS" and request.headers.get("Origin"):
        if origin is None:
            raise web.HTTPForbidden(text=f"CORS 来源不在允许列表：{request.headers.get('Origin')}")
        response: web.StreamResponse = web.Response(status=204)
    else:
        response = await handler(request)

    if origin is not None:
        _add_cors_headers(response, origin)
    return response


def _allowed_cors_origin(origin: str | None) -> str | None:
    """只允许本机开发 GUI 和证书覆盖的正式来源，拒绝任意来源反射。"""

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


def _gateway_error_response(
    request: web.Request,
    *,
    status: int,
    error_code: str,
    error_text: str,
) -> web.Response:
    """构造 Gateway 自身的稳定 JSON 错误响应。"""

    request_id = _request_id(request)
    return web.json_response(
        {
            "error_code": error_code,
            "error_text": f"{error_text}；request_id={request_id}",
        },
        status=status,
        headers={"X-Request-ID": request_id},
    )


def _request_id(request: web.Request) -> str:
    """读取错误中间件生成的请求标识。"""

    return request[_REQUEST_ID_KEY]


def _gateway_http_error_code(status: int) -> str:
    """按 HTTP 状态生成 Gateway 自身的稳定错误码。"""

    if status == 404:
        return "gateway_not_found"
    if status == 405:
        return "gateway_method_not_allowed"
    if status == 403:
        return "gateway_forbidden"
    return "gateway_request_rejected"


def _safe_error_detail(error: BaseException | str) -> str:
    """把异常原因压成单行并限长；完整堆栈保留在服务日志。"""

    detail = " ".join(str(error).splitlines()).strip()
    if not detail:
        detail = "异常未提供文本说明"
    return detail[:1000]
