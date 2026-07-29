from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.pipeline_context import PipelineContext, PipelineContextConfig
from camera_pipeline.protocol import CameraFramePacket

DEFAULT_FRAME_INTERVAL_MS = 100.0
"模拟相机帧间隔，单位 ms。"


class _TestPipelineContext(PipelineContext):
    """使用内存帧序列验证稳定帧编排的上下文。"""

    def __init__(self, frames: tuple[CameraFramePacket, ...]) -> None:
        self._config = PipelineContextConfig()
        self._camera_endpoints = self._resolve_camera_endpoints(self._config)
        self._test_frames = frames
        self._test_frame_index = 0
        self._test_frame_by_id = {frame.frame_id: frame for frame in frames}

    def get_latest_frame(
        self,
        camera_name: str | None = None,
    ) -> CameraFramePacket:
        del camera_name
        frame = self._test_frames[self._test_frame_index]
        if self._test_frame_index < len(self._test_frames) - 1:
            self._test_frame_index += 1
        return frame

    def get_frame_by_id(
        self,
        frame_id: int,
        camera_name: str | None = None,
    ) -> CameraFramePacket | None:
        del camera_name
        return self._test_frame_by_id.get(frame_id)

    def get_latest_color_frame(
        self,
        camera_name: str | None = None,
    ) -> CameraFramePacket:
        """返回下一帧作为纯彩色稳定检查输入。"""

        return self.get_latest_frame(camera_name)

    def get_color_frame_by_id(
        self,
        frame_id: int,
        camera_name: str | None = None,
    ) -> CameraFramePacket | None:
        """按帧号返回纯彩色稳定窗口的证据帧。"""

        return self.get_frame_by_id(frame_id, camera_name)


def test_resolve_frame_defaults_to_stable_window_midpoint() -> None:
    frames = tuple(_build_frame(frame_id) for frame_id in range(11))
    context = _TestPipelineContext(frames)

    stable_frame = context.resolve_frame(-1)

    assert stable_frame.frame_id == 5
    assert stable_frame.timestamp_ms == 500.0


def test_resolve_frame_keeps_explicit_frame_id_behavior() -> None:
    frames = tuple(_build_frame(frame_id) for frame_id in range(11))
    context = _TestPipelineContext(frames)

    selected_frame = context.resolve_frame(3)

    assert selected_frame.frame_id == 3
    assert selected_frame.timestamp_ms == 300.0


def test_color_stability_does_not_require_constant_depth() -> None:
    frames = tuple(
        _build_frame(frame_id, depth_mm=1000 + frame_id * 100)
        for frame_id in range(11)
    )
    context = _TestPipelineContext(frames)

    stable_frame = context.wait_for_stable_color_frame()

    assert stable_frame.frame_id == 5


def main() -> None:
    """在 IDE 中直接执行 PipelineContext 稳定帧验证。"""

    test_resolve_frame_defaults_to_stable_window_midpoint()
    test_resolve_frame_keeps_explicit_frame_id_behavior()
    test_color_stability_does_not_require_constant_depth()
    logger.success("PipelineContext 原 resolve_frame API 稳定帧验证通过")
    logger.warning("本结果使用内存合成帧，未验证真实相机缓存与现场稳定性阈值")


def _build_frame(
    frame_id: int,
    *,
    depth_mm: int = 1000,
) -> CameraFramePacket:
    return CameraFramePacket(
        frame_id=frame_id,
        camera_name="test_camera",
        timestamp_ms=frame_id * DEFAULT_FRAME_INTERVAL_MS,
        color_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
        depth_mm=np.full((48, 64), depth_mm, dtype=np.uint16),
        fx=600.0,
        fy=600.0,
        cx=32.0,
        cy=24.0,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


if __name__ == "__main__":
    main()
