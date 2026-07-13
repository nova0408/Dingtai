from __future__ import annotations

from pathlib import Path
import sys
import threading
import uuid

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient
from camera_pipeline.pipeline_context import PipelineContext
from camera_pipeline.protocol import CameraFramePacket
from camera_pipeline.service.application import CameraPipelineApplication
from camera_pipeline.service.frame_publisher import CameraFramePublisher
from camera_pipeline.service.protocol import CameraPipelineServiceRequest
from camera_pipeline.service.server import CameraPipelineServer
from camera_pipeline.service.transport import (
    CameraPipelineRpcClient,
    CameraPipelineRpcServer,
    ZmqSocketOptions,
)

DEFAULT_TIMEOUT_MS = 1_000
"离线回环请求超时，单位 ms。"


class _TestPipelineContext(PipelineContext):
    """不连接真实相机的固定帧上下文。"""

    def __init__(self) -> None:
        self._test_frame = CameraFramePacket(
            frame_id=42,
            camera_name="test_camera",
            timestamp_ms=1234.0,
            color_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
            depth_mm=np.full((48, 64), 1000, dtype=np.uint16),
            fx=600.0,
            fy=600.0,
            cx=32.0,
            cy=24.0,
        )

    def wait_until_ready(self, timeout_s: float = 8.0) -> bool:
        return timeout_s > 0.0

    def get_latest_frame(self) -> CameraFramePacket:
        return self._test_frame

    def get_camera_id(self) -> str:
        return "TEST"


def test_public_client_calls_in_process_service() -> None:
    resources = _start_service()
    client = CameraPipelineClient(
        service_addr=resources.address, timeout_ms=DEFAULT_TIMEOUT_MS
    )
    try:
        summary = client.get_camera_summary(timeout_s=0.5)
        assert summary.frame_id == 42
        assert summary.camera_name == "test_camera"
        assert summary.depth_shape == (48, 64)
    finally:
        client.close()
        resources.close()


def test_protocol_version_mismatch_returns_error_response() -> None:
    resources = _start_service()
    client = CameraPipelineRpcClient(
        resources.address,
        options=ZmqSocketOptions(
            receive_timeout_ms=DEFAULT_TIMEOUT_MS,
            send_timeout_ms=DEFAULT_TIMEOUT_MS,
        ),
    )
    try:
        response = client.call(
            CameraPipelineServiceRequest(
                operation="camera_summary",
                protocol_version=999,
            )
        )
        assert response.error is not None
        assert "unsupported protocol version" in response.error
    finally:
        client.close()
        resources.close()


class _ServiceResources:
    """离线回环测试资源。"""

    def __init__(
        self,
        address: str,
        stop_event: threading.Event,
        thread: threading.Thread,
        transport: CameraPipelineRpcServer,
        publisher: CameraFramePublisher,
    ) -> None:
        self.address = address
        self._stop_event = stop_event
        self._thread = thread
        self._transport = transport
        self._publisher = publisher

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._publisher.close()
        self._transport.close()
        if self._thread.is_alive():
            raise RuntimeError("offline camera pipeline service did not stop")


def _start_service() -> _ServiceResources:
    address = f"inproc://camera-pipeline-{uuid.uuid4().hex}"
    pipeline_context = _TestPipelineContext()
    publisher = CameraFramePublisher(pipeline_context)
    application = CameraPipelineApplication(pipeline_context, publisher)
    transport = CameraPipelineRpcServer(
        address,
        options=ZmqSocketOptions(
            receive_timeout_ms=50,
            send_timeout_ms=DEFAULT_TIMEOUT_MS,
        ),
    )
    server = CameraPipelineServer(transport, application)
    stop_event = threading.Event()
    thread = threading.Thread(target=server.serve, args=(stop_event,), daemon=True)
    thread.start()
    return _ServiceResources(address, stop_event, thread, transport, publisher)


def main() -> None:
    """在 IDE 中运行全部离线服务回环测试。"""

    test_public_client_calls_in_process_service()
    test_protocol_version_mismatch_returns_error_response()
    logger.success("CameraPipeline service 离线回环测试通过")
    logger.warning("本测试未连接真实相机、Orin 或 CUDA 模型")


if __name__ == "__main__":
    main()
