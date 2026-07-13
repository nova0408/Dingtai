from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


class StableRgbdFrameProtocol(Protocol):
    """时序稳定性算法所需的最小 RGBD 帧协议。"""

    @property
    def frame_id(self) -> int: ...

    @property
    def timestamp_ms(self) -> float: ...

    @property
    def color_bgr(self) -> npt.NDArray[np.uint8]: ...

    @property
    def depth_mm(self) -> npt.NDArray[np.uint16]: ...


class RgbdFrameProtocol(StableRgbdFrameProtocol, Protocol):
    """几何算法和业务编排共同依赖的完整只读 RGBD 帧协议。"""

    @property
    def camera_name(self) -> str: ...

    @property
    def fx(self) -> float: ...

    @property
    def fy(self) -> float: ...

    @property
    def cx(self) -> float: ...

    @property
    def cy(self) -> float: ...


@dataclass(frozen=True, slots=True)
class CameraFramePacket:
    """相机运行时和帧订阅使用的完整 RGBD 数据包。"""

    frame_id: int
    "上游相机 sequence 帧号。"
    camera_name: str
    "项目内逻辑相机名。"
    timestamp_ms: float
    "采集时间戳，单位 ms。"
    color_bgr: npt.NDArray[np.uint8]
    "彩色图像，形状 `(H, W, 3)`，通道顺序 BGR。"
    depth_mm: npt.NDArray[np.uint16]
    "深度图，形状 `(H, W)`，单位 mm，零值表示无效。"
    fx: float
    "X 方向焦距，单位 pixel。"
    fy: float
    "Y 方向焦距，单位 pixel。"
    cx: float
    "主点 X 坐标，单位 pixel。"
    cy: float
    "主点 Y 坐标，单位 pixel。"


@dataclass(frozen=True, slots=True)
class CameraColorFramePacket:
    """彩色帧订阅使用的轻量数据包。"""

    frame_id: int
    "上游相机 sequence 帧号。"
    camera_name: str
    "项目内逻辑相机名。"
    timestamp_ms: float
    "采集时间戳，单位 ms。"
    color_bgr: npt.NDArray[np.uint8]
    "彩色图像，形状 `(H, W, 3)`，通道顺序 BGR。"
    fx: float
    "X 方向焦距，单位 pixel。"
    fy: float
    "Y 方向焦距，单位 pixel。"
    cx: float
    "主点 X 坐标，单位 pixel。"
    cy: float
    "主点 Y 坐标，单位 pixel。"


@dataclass(frozen=True, slots=True)
class CameraDepthFramePacket:
    """深度帧订阅使用的轻量数据包。"""

    frame_id: int
    "上游相机 sequence 帧号。"
    camera_name: str
    "项目内逻辑相机名。"
    timestamp_ms: float
    "采集时间戳，单位 ms。"
    depth_mm: npt.NDArray[np.uint16]
    "深度图，形状 `(H, W)`，单位 mm，零值表示无效。"
    fx: float
    "X 方向焦距，单位 pixel。"
    fy: float
    "Y 方向焦距，单位 pixel。"
    cx: float
    "主点 X 坐标，单位 pixel。"
    cy: float
    "主点 Y 坐标，单位 pixel。"
