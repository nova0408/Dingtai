from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..grasp_pose.protocol import DebugArtifacts, TrayPoseInfo
from ..tray_detection.protocol import OrinTrayDetectionInfo


@dataclass(frozen=True)
class GraspPosePipelineServiceEndpointConfig:
    """抓取位姿主服务端点配置。"""

    request_bind_addr: str = "tcp://0.0.0.0:6220"
    "抓取位姿主服务 `REP` 绑定地址。"


@dataclass(frozen=True)
class GraspPosePipelineRequest:
    """抓取位姿主服务请求。"""

    request_id: int = 0
    "请求编号。"

    camera_name: str = "left_hand_camera"
    "逻辑相机名。"

    frame_id: int = -1
    "请求帧号。`-1` 表示最新缓存帧。"

    target_tray_index: int = 0
    "目标托盘编号，按图像从左到右编号。"

    enable_debug: bool = True
    "是否返回调试图与调试遮罩。"


@dataclass(frozen=True)
class GraspPosePipelineResponse:
    """抓取位姿主服务响应。"""

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
    "总执行耗时，单位 ms。"

    tray_count: int = 0
    "托盘数量。"

    tray_results: Tuple[OrinTrayDetectionInfo, ...] = field(default_factory=tuple)
    "阶段 1 托盘检测结果。"

    selected_tray_index: int = 0
    "目标托盘编号。"

    selected_result: Optional[TrayPoseInfo] = None
    "目标托盘最终抓取结果。"

    all_tray_results: Tuple[TrayPoseInfo, ...] = field(default_factory=tuple)
    "所有托盘的抓取结果。"

    debug: Optional[DebugArtifacts] = None
    "启用 debug 时返回的 RGB、深度、掩码和调试图。"

    error: Optional[str] = None
    "失败信息。"
