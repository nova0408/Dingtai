from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
import sys
import threading
import uuid
from typing import cast

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraName, CameraPipelineClient
from camera_pipeline.camera_stream import CameraStreamRuntime
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
    """不连接真实相机的多安装位固定帧上下文。"""

    def __init__(self) -> None:
        self._frame_id = 41
        self._test_frames = {
            "head_camera": self._build_frame("head_camera", 500.0),
            "chest_camera": self._build_frame("chest_camera", 550.0),
            "left_hand_camera": self._build_frame("left_hand_camera", 600.0),
        }

    def wait_until_ready(
        self,
        timeout_s: float = 8.0,
        camera_name: str | None = None,
    ) -> bool:
        self._resolve_camera_name(camera_name)
        return timeout_s > 0.0

    def get_latest_frame(
        self,
        camera_name: str | None = None,
    ) -> CameraFramePacket:
        self._frame_id += 1
        frame = self._test_frames[self._resolve_camera_name(camera_name)]
        return replace(frame, frame_id=self._frame_id)

    def get_camera_id(self, camera_name: str | None = None) -> str:
        return self._resolve_camera_name(camera_name).upper()

    def get_camera_runtime(
        self,
        camera_name: str | None = None,
    ) -> CameraStreamRuntime:
        self._resolve_camera_name(camera_name)
        return cast(CameraStreamRuntime, self)

    def get_connected_camera_names(self) -> tuple[str, ...]:
        return tuple(self._test_frames)

    def _resolve_camera_name(self, camera_name: str | None) -> str:
        resolved_name = "left_hand_camera" if camera_name is None else camera_name
        if resolved_name not in self._test_frames:
            raise RuntimeError(f"camera {resolved_name} is configured but not connected")
        return resolved_name

    @staticmethod
    def _build_frame(camera_name: str, focal_px: float) -> CameraFramePacket:
        return CameraFramePacket(
            frame_id=42,
            camera_name=camera_name,
            timestamp_ms=1234.0,
            color_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
            depth_mm=np.full((48, 64), 1000, dtype=np.uint16),
            fx=focal_px,
            fy=focal_px,
            cx=32.0,
            cy=24.0,
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        )


def test_public_client_calls_in_process_service() -> None:
    resources = _start_service()
    client = CameraPipelineClient(
        service_addr=resources.address, timeout_ms=DEFAULT_TIMEOUT_MS
    )
    try:
        summary = client.get_camera_summary(CameraName.LEFT_ARM, timeout_s=0.5)
        assert summary.frame_id >= 42
        assert summary.camera_name == "left_hand_camera"
        assert summary.depth_shape == (48, 64)
    finally:
        client.close()
        resources.close()


def test_named_head_subscription_filters_multiplexed_stream() -> None:
    resources = _start_service()
    client = CameraPipelineClient(
        service_addr=resources.address,
        timeout_ms=DEFAULT_TIMEOUT_MS,
    )
    stream = cast(
        Generator[CameraFramePacket, None, None],
        client.subscribe_camera_frames(CameraName.HEAD),
    )
    try:
        frame = next(stream)
        assert frame.camera_name == "head_camera"
        assert frame.fx == 500.0
    finally:
        stream.close()
        client.close()
        resources.close()


def test_named_camera_intrinsics_are_routed_to_requested_camera() -> None:
    resources = _start_service()
    client = CameraPipelineClient(
        service_addr=resources.address,
        timeout_ms=DEFAULT_TIMEOUT_MS,
    )
    try:
        head = client.get_camera_intrinsics(CameraName.HEAD, timeout_s=0.5)
        chest = client.get_camera_intrinsics(CameraName.CHEST, timeout_s=0.5)
        left = client.get_camera_intrinsics(CameraName.LEFT_ARM, timeout_s=0.5)
        assert (head.camera_name, head.fx) == ("head_camera", 500.0)
        assert (chest.camera_name, chest.fx) == ("chest_camera", 550.0)
        assert (left.camera_name, left.fx) == ("left_hand_camera", 600.0)
    finally:
        client.close()
        resources.close()


def test_right_arm_intrinsics_api_is_retained_but_reports_disconnected() -> None:
    resources = _start_service()
    client = CameraPipelineClient(
        service_addr=resources.address,
        timeout_ms=DEFAULT_TIMEOUT_MS,
    )
    try:
        try:
            client.get_camera_intrinsics(CameraName.RIGHT_ARM, timeout_s=0.5)
        except RuntimeError as exc:
            assert "not connected" in str(exc)
        else:
            raise AssertionError("disconnected right arm camera must fail explicitly")
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
                camera_name=CameraName.LEFT_ARM,
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
    resource_id = uuid.uuid4().hex
    address = f"inproc://camera-pipeline-{resource_id}"
    pipeline_context = _TestPipelineContext()
    publisher = CameraFramePublisher(
        pipeline_context,
        frame_bind_addr=f"inproc://camera-pipeline-frame-{resource_id}",
        color_bind_addr=f"inproc://camera-pipeline-color-{resource_id}",
        depth_bind_addr=f"inproc://camera-pipeline-depth-{resource_id}",
    )
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
    test_named_head_subscription_filters_multiplexed_stream()
    test_named_camera_intrinsics_are_routed_to_requested_camera()
    test_right_arm_intrinsics_api_is_retained_but_reports_disconnected()
    test_protocol_version_mismatch_returns_error_response()
    logger.success("CameraPipeline service 离线回环测试通过")
    logger.warning("本测试未连接真实相机、Orin 或 CUDA 模型")


if __name__ == "__main__":
    main()
