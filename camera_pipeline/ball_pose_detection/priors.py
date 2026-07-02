from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallPosePrior:
    """单个小球的先验信息。

    Parameters
    ----------
    color_hex:
        颜色 HEX，例如 `#ff0000`。
    radius_mm:
        小球物理半径，单位为毫米。
    model_center_mm:
        该小球在先验模型中的三维中心坐标，单位为毫米。
    """

    color_hex: str
    radius_mm: float
    model_center_mm: np.ndarray


@dataclass(frozen=True)
class BallPoseReferencePose:
    """用于采集先验时的参考位姿。"""

    rotation: np.ndarray
    translation_mm: np.ndarray
