"""CameraPipeline 公共客户端入口。"""

from .protocol import CameraName
from .service.client import CameraPipelineClient

__all__ = ["CameraName", "CameraPipelineClient"]
