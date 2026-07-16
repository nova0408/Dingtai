from __future__ import annotations

import threading
import time

import zmq
from loguru import logger

from .application import CameraPipelineApplication
from .protocol import CameraPipelineServiceResponse
from .transport import CameraPipelineRpcServer

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
                logger.exception("camera pipeline request decode failed: {}", exc)
                self._transport.send(
                    CameraPipelineServiceResponse(
                        operation="camera_status",
                        error=f"invalid request: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            started_at = time.perf_counter()
            try:
                response = self._application.handle(request)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "camera pipeline request failed operation={} error={}",
                    request.operation,
                    exc,
                )
                response = CameraPipelineServiceResponse(
                    operation=request.operation,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._transport.send(response)
            logger.info(
                "camera pipeline request completed operation={} success={} elapsed_ms={:.3f}",
                request.operation,
                response.error is None,
                (time.perf_counter() - started_at) * 1000.0,
            )
