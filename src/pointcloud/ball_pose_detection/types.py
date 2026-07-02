from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class BallPoseFrame(Protocol):
    """球位姿检测输入协议。"""

    color_bgr: np.ndarray
    depth_mm: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class BallObservation:
    name: str
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


@dataclass(frozen=True)
class BallPoseDetectionResult:
    detections: list[BallObservation]
    pose_translation_mm: np.ndarray | None
    pose_rotation: np.ndarray | None
    pose_transform: np.ndarray | None
    residual_mm: float | None
    matched_count: int
    debug_ball_colors_bgr: dict[str, np.ndarray]
    debug_ball_radii_mm: dict[str, float]
    debug_ball_positions_mm: dict[str, np.ndarray]
    debug_ball_model_positions_mm: dict[str, np.ndarray]
    status: str
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BallPoseDetectionConfig:
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
