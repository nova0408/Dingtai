from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import zmq

from .codec import decode_request, decode_response, encode_request, encode_response
from .protocol import OpeningDetectionPipelineRequest, OpeningDetectionPipelineResponse


@dataclass
class ZmqSocketOptions:
    rcv_hwm: int = 1
    snd_hwm: int = 1
    recv_timeout_ms: int = 10000
    send_timeout_ms: int = 10000


class OpeningDetectionPipelineRpcServer:
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

    def recv_request(self) -> OpeningDetectionPipelineRequest:
        return decode_request(self._socket.recv_multipart())

    def send_response(self, response: OpeningDetectionPipelineResponse) -> None:
        self._socket.send_multipart(encode_response(response))


class OpeningDetectionPipelineRpcClient:
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

    def call(self, request: OpeningDetectionPipelineRequest) -> OpeningDetectionPipelineResponse:
        self._socket.send_multipart(encode_request(request))
        return decode_response(self._socket.recv_multipart())
