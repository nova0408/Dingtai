"""Orin 抓取位姿主服务模块。"""

from .protocol import (
    DebugArtifacts,
    GraspPoseInfo,
    OpeningDetectionPipelineRequest,
    OpeningDetectionPipelineResponse,
    TrayPoseInfo,
)

__all__ = [
    "DebugArtifacts",
    "GraspPoseInfo",
    "OpeningDetectionPipelineRequest",
    "OpeningDetectionPipelineResponse",
    "TrayPoseInfo",
]
