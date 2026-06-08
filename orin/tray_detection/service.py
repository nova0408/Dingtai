from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Optional

import zmq

from orin.camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig

from .engine import OrinTrayDetectionExecutor, OrinTrayDetectionExecutorConfig
from .protocol import OrinTrayDetectionResponse, TrayDetectionServiceEndpointConfig
from .transport import OrinTrayDetectionRpcServer, ZmqSocketOptions


LOGGER = logging.getLogger("orin.tray_detection.service")


class OrinTrayDetectionService:
    """可被本机或远端调用的独立托盘检测服务。"""

    def __init__(
        self,
        endpoint_config: TrayDetectionServiceEndpointConfig,
        frame_runtime_config: CameraStreamRuntimeConfig,
        executor_config: Optional[OrinTrayDetectionExecutorConfig] = None,
        socket_options: Optional[ZmqSocketOptions] = None,
    ) -> None:
        self._frame_runtime = CameraStreamRuntime(frame_runtime_config)
        self._frame_runtime.start()
        self._server = OrinTrayDetectionRpcServer(endpoint_config.request_bind_addr, options=socket_options)
        self._executor = OrinTrayDetectionExecutor(frame_runtime=self._frame_runtime, config=executor_config)
        self._running = True

    def close(self) -> None:
        self._running = False
        self._server.close()
        self._frame_runtime.stop()

    def run_forever(self) -> None:
        LOGGER.info("orin tray detection rpc service started")
        while self._running:
            try:
                request = self._server.recv_request()
            except zmq.error.Again:
                continue
            try:
                response = self._executor.process_request(request)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("orin tray detection service failed: %s", exc)
                response = OrinTrayDetectionResponse(
                    request_id=int(request.request_id),
                    frame_id=-1,
                    camera_name=str(request.camera_name),
                    timestamp_ms=0.0,
                    source_meta={},
                    tray_count=0,
                    tray_results=tuple(),
                    error="{0}: {1}".format(type(exc).__name__, exc),
                )
            self._server.send_response(response)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Orin tray detection RPC service")
    parser.add_argument("--bind-addr", type=str, default="tcp://0.0.0.0:6210")
    parser.add_argument("--host", type=str, default="192.168.100.60")
    parser.add_argument("--control-port", type=int, default=5570)
    parser.add_argument("--stream-port", type=int, default=5562)
    parser.add_argument("--camera-id", type=str, default="LEFT")
    parser.add_argument("--camera-name", type=str, default="left_hand_camera")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = OrinTrayDetectionService(
        endpoint_config=TrayDetectionServiceEndpointConfig(request_bind_addr=str(args.bind_addr)),
        frame_runtime_config=CameraStreamRuntimeConfig(
            host=str(args.host),
            control_port=int(args.control_port),
            stream_port=int(args.stream_port),
            camera_id=str(args.camera_id),
            camera_name=str(args.camera_name),
        ),
        executor_config=OrinTrayDetectionExecutorConfig(),
        socket_options=ZmqSocketOptions(),
    )
    if not service._frame_runtime.wait_until_ready(timeout_s=8.0):  # noqa: SLF001
        LOGGER.warning("camera stream not ready within timeout")

    def _handle_signal(signum, _frame) -> None:  # noqa: ANN001
        LOGGER.info("received stop signal %s", signum)
        service.close()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        service.run_forever()
    finally:
        service.close()
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
