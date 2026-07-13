from __future__ import annotations

import logging
import threading

import zmq

from .application import CameraPipelineApplication
from .protocol import CameraPipelineServiceResponse
from .transport import CameraPipelineRpcServer

LOGGER = logging.getLogger(__name__)


class CameraPipelineServer:
    """运行统一 REP 请求循环，并把业务处理委托给 application。"""

    def __init__(
        self,
        transport: CameraPipelineRpcServer,
        application: CameraPipelineApplication,
    ) -> None:
        self._transport = transport
        self._application = application

    def serve(self, stop_event: threading.Event) -> None:
        """处理请求直到外部停止事件被设置。"""

        while not stop_event.is_set():
            try:
                request = self._transport.receive()
            except zmq.error.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("camera pipeline request decode failed: %s", exc)
                self._transport.send(
                    CameraPipelineServiceResponse(
                        operation="camera_status",
                        error=f"invalid request: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            try:
                response = self._application.handle(request)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("camera pipeline request failed: %s", exc)
                response = CameraPipelineServiceResponse(
                    operation=request.operation,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._transport.send(response)
