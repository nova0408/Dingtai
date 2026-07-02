from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..ports import TRAY_DETECTION_BIND_ADDR


# region 数据结构
@dataclass(frozen=True)
class TrayDetectionServiceEndpointConfig:
    """托盘检测服务端点配置。"""

    request_bind_addr: str = TRAY_DETECTION_BIND_ADDR
    "托盘检测服务 `REP` 绑定地址。"


@dataclass(frozen=True)
class OrinTrayDetectionRequest:
    """托盘检测请求。"""

    request_id: int = 0
    "调用侧请求编号。"

    camera_name: str = "left_hand_camera"
    "逻辑相机名。"

    frame_id: int = -1
    "请求帧号。`-1` 表示使用最新缓存帧。"

    enable_debug: bool = True
    "是否返回调试图和掩码。"


@dataclass(frozen=True)
class OrinTrayDetectionInfo:
    """单个托盘检测结果。"""

    tray_id: int
    "托盘编号。按当前帧中目标中心 `x` 从小到大编号。"

    label_text: str
    "托盘标签文本。"

    confidence_2d: float
    "2D 检测置信度。"

    bbox_xywh: Tuple[int, int, int, int]
    "托盘包围框 `(x, y, w, h)`，单位 像素。"

    center_uv: Tuple[float, float]
    "托盘中心像素坐标，单位 像素。"

    mask_area_px: int
    "托盘掩码面积，单位 像素。"

    source: str
    "托盘结果来源。"


@dataclass(frozen=True)
class OrinTrayDetectionDebugArtifacts:
    """托盘检测调试图与掩码。"""

    overlay_bgr: Optional[np.ndarray] = None
    "叠加预览图，形状 `(H, W, 3)`。"

    mask_bgr: Optional[np.ndarray] = None
    "掩码预览图，形状 `(H, W, 3)`。"

    tray_masks: Tuple[np.ndarray, ...] = field(default_factory=tuple)
    "托盘掩码序列，顺序与 `tray_results` 保持一致。"


@dataclass(frozen=True)
class OrinTrayDetectionResponse:
    """托盘检测响应。"""

    request_id: int
    "请求编号透传。"

    frame_id: int
    "真实相机帧号。"

    camera_name: str
    "逻辑相机名。"

    timestamp_ms: float
    "真实帧时间戳，单位 ms。"

    source_meta: Dict[str, Any] = field(default_factory=dict)
    "来源元信息。"

    elapsed_ms: float = 0.0
    "服务端检测耗时，单位 ms。"

    tray_count: int = 0
    "托盘数量。"

    tray_results: Tuple[OrinTrayDetectionInfo, ...] = field(default_factory=tuple)
    "托盘结果序列。"

    debug: Optional[OrinTrayDetectionDebugArtifacts] = None
    "调试图与掩码。"

    error: Optional[str] = None
    "失败信息。"


# endregion
