from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import zmq

from .protocol import CameraPipelineServiceRequest, CameraPipelineServiceResponse
from .wire_codec import decode_wire, encode_wire


@dataclass(frozen=True, slots=True)
class ZmqSocketOptions:
    """统一请求响应 socket 配置。"""

    receive_high_water_mark: int = 1
    send_high_water_mark: int = 1
    receive_timeout_ms: int = 30_000
    send_timeout_ms: int = 30_000


class CameraPipelineRpcServer:
    """只负责统一 CameraPipeline REP socket 的收发与关闭。"""

    def __init__(
        self,
        bind_addr: str,
        *,
        context: zmq.Context | None = None,
        options: ZmqSocketOptions | None = None,
    ) -> None:
        self._context = zmq.Context.instance() if context is None else context
        self._options = ZmqSocketOptions() if options is None else options
        self._socket = self._context.socket(zmq.REP)
        _configure_socket(self._socket, self._options)
        self._socket.bind(bind_addr)

    def close(self) -> None:
        """立即关闭 REP socket。"""

        self._socket.close(linger=0)

    def receive(self) -> CameraPipelineServiceRequest:
        """接收并校验一个统一服务请求。"""

        return decode_wire(self._socket.recv(), CameraPipelineServiceRequest)

    def send(self, response: CameraPipelineServiceResponse) -> None:
        """发送一个统一服务响应。"""

        self._socket.send(encode_wire(response))


class CameraPipelineRpcClient:
    """只负责统一 CameraPipeline REQ socket 的收发与关闭。"""

    def __init__(
        self,
        connect_addr: str,
        *,
        context: zmq.Context | None = None,
        options: ZmqSocketOptions | None = None,
    ) -> None:
        self._context = zmq.Context.instance() if context is None else context
        self._options = ZmqSocketOptions() if options is None else options
        self._connect_addr = connect_addr
        self._socket: zmq.Socket | None = None
        self._lock = Lock()
        self._closed = False

    def close(self) -> None:
        """关闭持久 REQ socket，并禁止继续发起请求。"""

        with self._lock:
            self._closed = True
            self._close_socket()

    def call(
        self, request: CameraPipelineServiceRequest
    ) -> CameraPipelineServiceResponse:
        """复用持久 REQ socket 完成一次请求响应并校验返回类型。

        正常请求复用同一条连接，避免连续采样时反复建立 TCP/SSH 转发流。
        REQ/REP 必须严格一问一答，因此使用锁串行化调用；发送、接收或解码失败时
        立即丢弃已失效 socket，下一次调用再建立新连接。
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("camera pipeline RPC client is closed")
            socket = self._get_socket()
            try:
                socket.send(encode_wire(request))
                return decode_wire(socket.recv(), CameraPipelineServiceResponse)
            except Exception:
                self._close_socket()
                raise

    def _get_socket(self) -> zmq.Socket:
        """返回当前持久 socket；首次调用或故障恢复时创建。"""

        socket = self._socket
        if socket is None:
            socket = self._context.socket(zmq.REQ)
            _configure_socket(socket, self._options)
            socket.connect(self._connect_addr)
            self._socket = socket
        return socket

    def _close_socket(self) -> None:
        """立即关闭当前 socket，并清空连接状态。"""

        socket = self._socket
        self._socket = None
        if socket is not None:
            socket.close(linger=0)


def _configure_socket(socket: zmq.Socket, options: ZmqSocketOptions) -> None:
    socket.setsockopt(zmq.RCVHWM, options.receive_high_water_mark)
    socket.setsockopt(zmq.SNDHWM, options.send_high_water_mark)
    socket.setsockopt(zmq.RCVTIMEO, options.receive_timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, options.send_timeout_ms)
