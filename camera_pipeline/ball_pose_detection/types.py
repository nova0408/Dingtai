from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, slots=True)
class BallObservation:
    """单个球的内部二维、三维和 debug 观测。"""

    color_hex: str
    detected: bool
    center_px: np.ndarray | None
    center_mm: np.ndarray | None
    radius_mm: float
    radius_px: float
    contour: np.ndarray | None
    mask: np.ndarray | None
    center_norm: np.ndarray | None
    radius_norm: float
    point_count: int
    debug_bgr: np.ndarray
    status: str


@dataclass(frozen=True, slots=True)
class BallPoseDetectionResult:
    """一次多球检测的内部汇总结果。"""

    detections: list[BallObservation]
    matched_count: int
    debug_ball_colors_bgr: dict[str, np.ndarray]
    debug_ball_radii_mm: dict[str, float]
    debug_ball_positions_mm: dict[str, np.ndarray]
    debug_ball_model_positions_mm: dict[str, np.ndarray]
    status: str
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BallPoseDetectionConfig:
    """颜色连通域、圆形筛选和深度估计参数。"""

    min_component_area_px: int = 28
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
