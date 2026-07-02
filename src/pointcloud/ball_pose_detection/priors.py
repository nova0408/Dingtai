from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallPosePrior:
    """单个小球先验描述。"""

    name: str
    color_hex: str
    radius_mm: float
    model_center_mm: np.ndarray


@dataclass(frozen=True)
class BallPoseReferencePose:
    """多球目标物体的位姿先验。"""

    translation_mm: np.ndarray
    rotation: np.ndarray
