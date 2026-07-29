from __future__ import annotations

from typing import Protocol

from .protocol import ColorFrameProtocol, RgbdFrameProtocol


class CameraRuntimeProtocol(Protocol):
    """`PipelineContext` 管理的相机采集运行时协议。

    ZMQ 和本机 USB 实现均遵循本协议。协议只描述生命周期和帧访问，不暴露
    socket 或 pyorbbecsdk 对象，避免上层依赖具体采集方式。
    """

    @property
    def keeps_frame_history(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool: ...

    def get_latest_frame(self) -> RgbdFrameProtocol | None: ...

    def get_frame_by_id(self, frame_id: int) -> RgbdFrameProtocol | None: ...

    def get_latest_color_frame(self) -> ColorFrameProtocol | None: ...

    def get_color_frame_by_id(self, frame_id: int) -> ColorFrameProtocol | None: ...
