"""多球位姿检测子模块。

该子模块聚合 RGBD 多球识别、三维球心估计与刚体位姿求解所需的数据结构和核心流程。
当前实现只承载可复用的公共能力，不包含相机采集、GUI 预览或先验标定脚本。
"""

from .pipeline import BallPoseDetectionPipeline
from .priors import BallPosePrior, BallPoseReferencePose
from .types import (
    BallObservation,
    BallPoseDetectionConfig,
    BallPoseDetectionResult,
    BallPoseFrame,
)

__all__ = [
    "BallObservation",
    "BallPoseDetectionConfig",
    "BallPoseDetectionPipeline",
    "BallPoseDetectionResult",
    "BallPoseFrame",
    "BallPoseReferencePose",
    "BallPosePrior",
]
