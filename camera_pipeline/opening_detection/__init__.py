"""Orin 抓取位姿主服务模块。"""

from .protocol import (
    DebugArtifacts,
    GraspPoseInfo,
    GraspPoseRequest,
    GraspPoseResponse,
    OpeningDetectionPipelineRequest,
    OpeningDetectionPipelineResponse,
    TrayPoseInfo,
)

__all__ = [
    "DebugArtifacts",
    "GraspPoseInfo",
    "GraspPoseRequest",
    "GraspPoseResponse",
    "OpeningDetectionPipelineRequest",
    "OpeningDetectionPipelineResponse",
    "TrayPoseInfo",
]
