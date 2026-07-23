from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class BallPosePrior:
    """单个小球的先验信息。

    Parameters
    ----------
    color_hex:
        颜色 HEX，例如 `#ff0000`。
    diameter_mm:
        小球物理直径，单位为毫米。
    model_center_mm:
        该小球在先验模型中的三维中心坐标，单位为毫米。
    hsv_ranges:
        标定得到的专属 HSV 范围；为空时由参考颜色选择全局宽范围。
    """

    color_hex: str
    diameter_mm: float
    model_center_mm: np.ndarray
    hsv_ranges: tuple[tuple[int, int, int, int, int, int], ...] = field(
        default_factory=tuple
    )


@dataclass(frozen=True, slots=True)
class BallPoseReferencePose:
    """用于采集先验时的参考位姿。"""

    rotation: np.ndarray
    translation_mm: np.ndarray
