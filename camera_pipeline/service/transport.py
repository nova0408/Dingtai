from __future__ import annotations

import pickle
from dataclasses import dataclass

import zmq

from .protocol import CameraPipelineServiceRequest, CameraPipelineServiceResponse


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

        request = pickle.loads(self._socket.recv())
        if not isinstance(request, CameraPipelineServiceRequest):
            raise RuntimeError("invalid camera pipeline service request")
        return request

    def send(self, response: CameraPipelineServiceResponse) -> None:
        """发送一个统一服务响应。"""

        self._socket.send(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))


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
        self._socket = self._context.socket(zmq.REQ)
        _configure_socket(self._socket, self._options)
        self._socket.connect(connect_addr)

    def close(self) -> None:
        """立即关闭 REQ socket。"""

        self._socket.close(linger=0)

    def call(
        self, request: CameraPipelineServiceRequest
    ) -> CameraPipelineServiceResponse:
        """同步发送请求并接收经过类型校验的响应。"""

        self._socket.send(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(self._socket.recv())
        if not isinstance(response, CameraPipelineServiceResponse):
            raise RuntimeError("invalid camera pipeline service response")
        return response


def _configure_socket(socket: zmq.Socket, options: ZmqSocketOptions) -> None:
    socket.setsockopt(zmq.RCVHWM, options.receive_high_water_mark)
    socket.setsockopt(zmq.SNDHWM, options.send_high_water_mark)
    socket.setsockopt(zmq.RCVTIMEO, options.receive_timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, options.send_timeout_ms)
