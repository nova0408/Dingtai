"""Orin 侧独立托盘检测服务。"""

from .protocol import (
    OrinTrayDetectionDebugArtifacts,
    OrinTrayDetectionInfo,
    OrinTrayDetectionRequest,
    OrinTrayDetectionResponse,
)
from .transport import OrinTrayDetectionRpcClient, OrinTrayDetectionRpcServer, ZmqSocketOptions

__all__ = [
    "OrinTrayDetectionDebugArtifacts",
    "OrinTrayDetectionInfo",
    "OrinTrayDetectionRequest",
    "OrinTrayDetectionResponse",
    "OrinTrayDetectionRpcClient",
    "OrinTrayDetectionRpcServer",
    "ZmqSocketOptions",
]
