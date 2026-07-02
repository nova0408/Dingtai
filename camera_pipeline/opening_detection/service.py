from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Optional

import zmq

from ..camera_stream import CameraStreamRuntimeConfig
from ..pipeline_context import PipelineContext, PipelineContextConfig
from ..ports import (
    DEFAULT_CAMERA_HOST,
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_NAME,
    DEFAULT_CONTROL_PORT,
    DEFAULT_STREAM_PORT,
    OPENING_DETECTION_BIND_ADDR,
)
from .engine import OpeningDetectionPipelineExecutor, OpeningDetectionPipelineExecutorConfig
from .protocol import OpeningDetectionPipelineResponse, OpeningDetectionPipelineServiceEndpointConfig
from .transport import OpeningDetectionPipelineRpcServer, ZmqSocketOptions


LOGGER = logging.getLogger("..opening_detection.service")

class OpeningDetectionPipelineService:
    """抓取位姿主服务。"""

    def __init__(
        self,
        endpoint_config: OpeningDetectionPipelineServiceEndpointConfig,
        frame_runtime_config: CameraStreamRuntimeConfig,
        executor_config: Optional[OpeningDetectionPipelineExecutorConfig] = None,
        socket_options: Optional[ZmqSocketOptions] = None,
    ) -> None:
        self._context = PipelineContext(
            PipelineContextConfig(camera_runtime=frame_runtime_config),
        )
        self._context.start()
        self._server = OpeningDetectionPipelineRpcServer(endpoint_config.request_bind_addr, options=socket_options)
        self._executor = OpeningDetectionPipelineExecutor(config=executor_config)
        self._running = True

    def close(self) -> None:
        self._running = False
        self._server.close()
        self._context.close()

    def run_forever(self) -> None:
        LOGGER.info("opening detection pipeline rpc service started")
        while self._running:
            try:
                request = self._server.recv_request()
            except zmq.error.Again:
                continue
            try:
                response = self._process_request(request)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("opening detection pipeline service failed: %s", exc)
                response = OpeningDetectionPipelineResponse(
                    request_id=int(request.request_id),
                    frame_id=-1,
                    camera_name=str(request.camera_name),
                    timestamp_ms=0.0,
                    source_meta={},
                    selected_tray_index=int(request.target_tray_index),
                    error="{0}: {1}".format(type(exc).__name__, exc),
                )
            self._server.send_response(response)

    def _process_request(self, request) -> OpeningDetectionPipelineResponse:
        frame = self._context.resolve_frame(request.frame_id)
        tray_response = self._context.request_tray_detection_for_frame(
            request_id=int(request.request_id),
            camera_name=str(request.camera_name),
            frame_id=int(request.frame_id),
            enable_debug=bool(request.enable_debug),
        )
        if tray_response.error is not None:
            raise RuntimeError(tray_response.error)
        if request.target_tray_index < 0 or request.target_tray_index >= len(tray_response.tray_results):
            raise RuntimeError(f"目标托盘索引越界：{request.target_tray_index}")
        tray_mask = self._extract_tray_mask(tray_response, int(request.target_tray_index))
        tray_pose, debug = self._executor.compute(
            frame=frame,
            tray_mask=tray_mask,
            request_id=int(request.request_id),
            target_tray_index=int(request.target_tray_index),
            enable_debug=bool(request.enable_debug),
        )
        return OpeningDetectionPipelineResponse(
            request_id=int(request.request_id),
            frame_id=int(tray_response.frame_id),
            camera_name=str(tray_response.camera_name),
            timestamp_ms=float(tray_response.timestamp_ms),
            source_meta=dict(tray_response.source_meta),
            elapsed_ms=float(tray_response.elapsed_ms),
            tray_count=int(tray_response.tray_count),
            tray_results=tray_response.tray_results,
            selected_tray_index=int(request.target_tray_index),
            selected_result=tray_pose,
            all_tray_results=(tray_pose,),
            debug=debug,
            error=None,
        )

    @staticmethod
    def _extract_tray_mask(tray_response, target_tray_index: int):
        debug = tray_response.debug
        if debug is None or len(debug.tray_masks) <= target_tray_index:
            raise RuntimeError("tray detection debug masks unavailable")
        return debug.tray_masks[int(target_tray_index)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Orin opening detection pipeline RPC service")
    parser.add_argument("--bind-addr", type=str, default=OPENING_DETECTION_BIND_ADDR)
    parser.add_argument("--host", type=str, default=DEFAULT_CAMERA_HOST)
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT)
    parser.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT)
    parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = OpeningDetectionPipelineService(
        endpoint_config=OpeningDetectionPipelineServiceEndpointConfig(request_bind_addr=str(args.bind_addr)),
        frame_runtime_config=CameraStreamRuntimeConfig(
            host=str(args.host),
            control_port=int(args.control_port),
            stream_port=int(args.stream_port),
            camera_id=str(args.camera_id),
            camera_name=str(args.camera_name),
        ),
        executor_config=OpeningDetectionPipelineExecutorConfig(),
        socket_options=ZmqSocketOptions(),
    )
    if not service._context.wait_until_ready(timeout_s=8.0):
        LOGGER.warning("camera stream not ready within timeout")

    def _handle_signal(signum, _frame) -> None:
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
