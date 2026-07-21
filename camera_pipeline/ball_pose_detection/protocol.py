from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..protocol import CameraName


@dataclass(frozen=True, slots=True)
class BallPosePriorInfo:
    """单个小球的先验信息。"""

    color_hex: str
    "目标球颜色，使用带 `#` 的六位 RGB 十六进制字符串。"
    radius_mm: float
    "模型先验半径，单位 mm。"
    model_center_mm: tuple[float, float, float]
    "模型中心在业务参考坐标系中的位置，单位 mm，顺序为 `(x, y, z)`。"


@dataclass(frozen=True, slots=True)
class BallPoseDetectionRequest:
    """球位姿检测请求。"""

    request_id: int
    "调用方请求编号，原样透传到响应。"
    camera_name: CameraName
    "逻辑相机名称。"
    frame_id: int = -1
    "请求帧号；正数表示精确缓存帧，非正数表示由业务层选择稳定帧。"
    enable_debug: bool = False
    "是否生成彩色图、深度图和检测叠加图等调试产物。"
    priors: tuple[BallPosePriorInfo, ...] = field(default_factory=tuple)
    "待检测球的颜色、尺寸和模型中心先验，顺序决定结果顺序。"


@dataclass(frozen=True, slots=True)
class BallDetectionInfo:
    """单个球的明确检测结果；未检测到时坐标元组为空。"""

    color_hex: str
    "目标球颜色，与请求先验中的颜色保持一致。"
    detected: bool
    "是否获得有效的二维候选和三维深度估计。"
    center_px: tuple[float, ...]
    "球心像素坐标 `(u, v)`；未检测到时为空元组，单位 pixel。"
    center_mm: tuple[float, ...]
    "球心相机坐标 `(x, y, z)`；深度不足时为空元组，单位 mm。"
    radius_mm: float
    "根据深度和投影估计的球半径，单位 mm。"
    radius_px: float
    "图像平面拟合半径，单位 pixel。"
    center_norm: tuple[float, ...]
    "归一化图像球心 `(x, y)`；无二维候选时为空元组。"
    radius_norm: float
    "相对于焦距归一化后的图像半径，无量纲。"
    point_count: int
    "参与深度估计的有效深度点数量。"
    status: str
    "检测状态文本，例如 `detected`、`depth_weak` 或 `missing`。"


@dataclass(frozen=True, slots=True)
class BallPoseDetectionDebugArtifacts:
    """球位姿检测调试信息。"""

    color_bgr: np.ndarray
    "输入彩色图副本，形状 `(H, W, 3)`，dtype `uint8`，通道顺序 BGR。"
    depth_mm: np.ndarray
    "输入深度图，形状 `(H, W)`，单位 mm，零值表示无效深度。"
    camera_intrinsics: tuple[float, float, float, float]
    "针孔内参 `(fx, fy, cx, cy)`，焦距单位 pixel，主点单位 pixel。"
    overlay_bgr: np.ndarray
    "包含检测结果标注的彩色叠加图，形状 `(H, W, 3)`。"
    detection_overlay_bgr: np.ndarray
    "检测轮廓叠加图；当前与 `overlay_bgr` 使用同一结果。"
    detections: tuple[BallDetectionInfo, ...] = field(default_factory=tuple)
    "调试用明确检测结果；关闭 debug 时整个 debug_artifacts 为空元组。"


@dataclass(frozen=True, slots=True)
class BallPoseDetectionResponse:
    """球位姿检测响应。"""

    request_id: int
    "调用方请求编号。"
    frame_id: int
    "实际参与计算的相机帧号。"
    camera_name: str
    "实际使用的逻辑相机名称。"
    timestamp_ms: float
    "实际帧采集时间戳，单位 ms。"
    elapsed_ms: float = 0.0
    "球检测和位姿估计耗时，单位 ms。"
    matched_count: int = 0
    "同时获得有效二维候选和三维中心的球数量。"
    detections: tuple[BallDetectionInfo, ...] = field(default_factory=tuple)
    "按请求先验顺序排列的检测结果；无先验时为空元组。"
    debug_artifacts: tuple[BallPoseDetectionDebugArtifacts, ...] = field(
        default_factory=tuple
    )
    "调试产物集合；成功启用 debug 时包含一个元素，否则为空元组。"
