"""CameraPipeline 统一网络服务实现。"""

from .client import CameraPipelineClient
from .protocol import CameraPipelineServiceRequest, CameraPipelineServiceResponse

__all__ = [
    "CameraPipelineClient",
    "CameraPipelineServiceRequest",
    "CameraPipelineServiceResponse",
]
