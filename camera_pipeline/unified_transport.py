from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import zmq

from .unified_protocol import CameraPipelineServiceRequest, CameraPipelineServiceResponse


@dataclass
class ZmqSocketOptions:
    rcv_hwm: int = 1
    snd_hwm: int = 1
    recv_timeout_ms: int = 30_000
    send_timeout_ms: int = 30_000


def _encode_message(message: object) -> bytes:
    return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_request(payload: bytes) -> CameraPipelineServiceRequest:
    request = pickle.loads(payload)
    if not isinstance(request, CameraPipelineServiceRequest):
        raise RuntimeError("invalid camera pipeline service request")
    return request


def _decode_response(payload: bytes) -> CameraPipelineServiceResponse:
    response = pickle.loads(payload)
    if not isinstance(response, CameraPipelineServiceResponse):
        raise RuntimeError("invalid camera pipeline service response")
    return response


class CameraPipelineRpcServer:
    """统一 camera pipeline 服务端收发封装。"""

    def __init__(self, bind_addr: str, context: Optional[zmq.Context] = None, options: Optional[ZmqSocketOptions] = None) -> None:
        self._context = zmq.Context.instance() if context is None else context
        self._options = ZmqSocketOptions() if options is None else options
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.RCVHWM, int(self._options.rcv_hwm))
        self._socket.setsockopt(zmq.SNDHWM, int(self._options.snd_hwm))
        self._socket.setsockopt(zmq.RCVTIMEO, int(self._options.recv_timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(self._options.send_timeout_ms))
        self._socket.bind(bind_addr)

    def close(self) -> None:
        self._socket.close(linger=0)

    def recv_request(self) -> CameraPipelineServiceRequest:
        return _decode_request(self._socket.recv())

    def send_response(self, response: CameraPipelineServiceResponse) -> None:
        self._socket.send(_encode_message(response))


class CameraPipelineRpcClient:
    """统一 camera pipeline 客户端底层收发封装。"""

    def __init__(self, connect_addr: str, context: Optional[zmq.Context] = None, options: Optional[ZmqSocketOptions] = None) -> None:
        self._context = zmq.Context.instance() if context is None else context
        self._options = ZmqSocketOptions() if options is None else options
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVHWM, int(self._options.rcv_hwm))
        self._socket.setsockopt(zmq.SNDHWM, int(self._options.snd_hwm))
        self._socket.setsockopt(zmq.RCVTIMEO, int(self._options.recv_timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(self._options.send_timeout_ms))
        self._socket.connect(connect_addr)

    def close(self) -> None:
        self._socket.close(linger=0)

    def call(self, request: CameraPipelineServiceRequest) -> CameraPipelineServiceResponse:
        self._socket.send(_encode_message(request))
        return _decode_response(self._socket.recv())
