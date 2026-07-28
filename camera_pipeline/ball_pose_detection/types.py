from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, slots=True)
class BallObservation:
    """单个球的内部二维、三维和 debug 观测。"""

    color_hex: str
    "目标球颜色，使用六位 RGB 十六进制字符串。"
    detected: bool
    "是否通过二维候选和深度有效性检查。"
    center_px: np.ndarray | None
    "球心像素坐标 `(u, v)`；内部未检测到时为空。"
    center_mm: np.ndarray | None
    "球心相机坐标 `(x, y, z)`，单位 mm；深度不足时为空。"
    diameter_mm: float
    "估计球直径，单位 mm。"
    radius_px: float
    "图像平面拟合半径，单位 pixel。"
    contour: np.ndarray | None
    "候选轮廓，形状 `(N, 2)`；没有二维候选时为空。"
    mask: np.ndarray | None
    "候选二值掩码，形状 `(H, W)`；没有二维候选时为空。"
    center_norm: np.ndarray | None
    "归一化图像球心；没有二维候选时为空。"
    radius_norm: float
    "相对于焦距归一化的图像半径。"
    point_count: int
    "有效深度采样点数量。"
    debug_bgr: np.ndarray
    "当前颜色对应的 BGR 调试颜色。"
    status: str
    "检测状态文本。"
    observed_hsv: np.ndarray | None
    "候选颜色像素的实测 HSV 中心，形状 `(3,)`；颜色样本不足或未检测到时为空。"


@dataclass(frozen=True, slots=True)
class BallPoseDetectionResult:
    """一次多球检测的内部汇总结果。"""

    detections: list[BallObservation]
    "按输入先验顺序排列的内部观测。"
    matched_count: int
    "获得有效三维中心的观测数量。"
    debug_ball_colors_bgr: dict[str, np.ndarray]
    "按颜色索引的 BGR 调试颜色。"
    debug_ball_diameters_mm: dict[str, float]
    "按颜色索引的先验直径，单位 mm。"
    debug_ball_positions_mm: dict[str, np.ndarray]
    "按颜色索引的检测中心，位于相机坐标系，单位 mm。"
    debug_ball_model_positions_mm: dict[str, np.ndarray]
    "按颜色索引的模型先验中心。"
    status: str
    "整体状态：至少一个有效检测时为 `detected`，否则为 `missing`。"
    timings_ms: dict[str, float] = field(default_factory=dict)
    "阶段耗时字典，键为阶段名，值单位 ms。"


@dataclass(frozen=True, slots=True)
class BallPoseDetectionConfig:
    """颜色连通域、圆形筛选、深度估计和三球先验校验参数。"""

    min_component_area_px: int = 28
    "颜色连通域最小面积，单位 pixel。"
    max_color_components: int = 6
    "每种颜色参与联合评分的最大候选数量。"
    min_circularity: float = 0.46
    "候选轮廓最小圆形度，范围 0-1。"
    min_fill_ratio: float = 0.34
    "候选轮廓相对最小外接圆的最小填充比例，范围 0-1。"
    depth_trim_ratio: float = 0.18
    "深度样本两端各自裁剪的比例，范围 0-0.5。"
    min_depth_points: int = 18
    "球面拟合所需的最少有效深度点数量。"
    min_color_sample_pixels: int = 20
    "估计候选实测 HSV 中心所需的最少颜色像素数量。"
    min_center_distance_ratio: float = 1.35
    "启用三球相对位置先验所需的最小球心距离，相对最大先验直径的倍数。"
    max_diameter_error_ratio: float = 0.35
    "估计直径相对先验直径的最大允许误差，超过时判定该候选未检出。"
    max_relative_distance_error_ratio: float = 0.30
    "检测球间距相对模型球间距的最大允许误差，超过时整组三球判定未检出。"
    reference_hue_tolerance: int = 10
    "未提供标定 HSV 时，由输入 HEX 颜色生成的 Hue 半宽，OpenCV Hue 单位。"
    reference_saturation_tolerance: int = 115
    "未提供标定 HSV 时，相对输入 HEX 饱和度向下放宽的幅度。"
    reference_value_tolerance: int = 135
    "未提供标定 HSV 时，相对输入 HEX 明度向下放宽的幅度。"
