"""连续 RGBD 画面稳定性判定模块。"""

from .detector import StableFrameDetector
from .types import StableFrameConfig

__all__ = ["StableFrameConfig", "StableFrameDetector"]
