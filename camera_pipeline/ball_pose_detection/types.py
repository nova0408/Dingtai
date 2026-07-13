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
    radius_mm: float
    "估计球半径，单位 mm。"
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


@dataclass(frozen=True, slots=True)
class BallPoseDetectionResult:
    """一次多球检测的内部汇总结果。"""

    detections: list[BallObservation]
    "按输入先验顺序排列的内部观测。"
    matched_count: int
    "获得有效三维中心的观测数量。"
    debug_ball_colors_bgr: dict[str, np.ndarray]
    "按颜色索引的 BGR 调试颜色。"
    debug_ball_radii_mm: dict[str, float]
    "按颜色索引的先验半径，单位 mm。"
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
    """颜色连通域、圆形筛选和深度估计参数。"""

    min_component_area_px: int = 28
    "颜色连通域最小面积，单位 pixel。"
    max_color_components: int = 6
    min_circularity: float = 0.46
    min_fill_ratio: float = 0.34
    depth_trim_ratio: float = 0.18
    min_depth_points: int = 18
    min_center_distance_ratio: float = 1.35
    color_ranges: dict[str, tuple[tuple[int, int, int, int, int, int], ...]] = field(
        default_factory=lambda: {
            "#ff0000": ((0, 75, 55, 10, 255, 255), (170, 75, 55, 179, 255, 255)),
            "#0000ff": ((90, 55, 35, 130, 255, 255),),
            "#ffff00": ((18, 60, 70, 42, 255, 255),),
            "#ff00ff": ((130, 55, 35, 170, 255, 255),),
        }
    )
