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
    """使用调用级 REQ socket 完成 CameraPipeline RPC 收发。"""

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
        self._lock = Lock()
        self._closed = False

    def close(self) -> None:
        """关闭持久 REQ socket，并禁止继续发起请求。"""

        with self._lock:
            self._closed = True

    def call(
        self, request: CameraPipelineServiceRequest
    ) -> CameraPipelineServiceResponse:
        """使用独立 REQ socket 完成一次请求响应并校验返回类型。

        REQ/REP 必须严格一问一答。部分上游 ZMQ 组合在同一 REQ socket 跨越
        短查询与长等待操作时会丢失后一响应，因此每次调用建立独立 socket，并在
        成功或失败后立即关闭。锁仍用于串行化同一客户端实例并协调关闭状态。
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("camera pipeline RPC client is closed")
            socket = self._context.socket(zmq.REQ)
            _configure_socket(socket, self._options)
            socket.connect(self._connect_addr)
            try:
                socket.send(encode_wire(request))
                return decode_wire(socket.recv(), CameraPipelineServiceResponse)
            finally:
                socket.close(linger=0)


def _configure_socket(socket: zmq.Socket, options: ZmqSocketOptions) -> None:
    socket.setsockopt(zmq.RCVHWM, options.receive_high_water_mark)
    socket.setsockopt(zmq.SNDHWM, options.send_high_water_mark)
    socket.setsockopt(zmq.RCVTIMEO, options.receive_timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, options.send_timeout_ms)
