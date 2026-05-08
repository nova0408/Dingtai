"""托盘检测子模块。

该子模块聚合托盘检测相关的数据结构、模型缓存、2D 检测、投影映射与实时流程，
用于替代散落在 `src/pointcloud` 根目录下的 `tray_detection_*` 前缀文件。
"""

from .detector import TrayPointExcluder
from .pipeline import TrayDetectionPipeline, TrayPipelineConfig, TrayRuntimeState
from .projection import collect_indices_in_mask, project_points_to_image
from .types import TrayDetection, TrayDetectionConfig, TrayExclusionResult

__all__ = [
    "TrayDetection",
    "TrayDetectionConfig",
    "TrayDetectionPipeline",
    "TrayExclusionResult",
    "TrayPipelineConfig",
    "TrayPointExcluder",
    "TrayRuntimeState",
    "collect_indices_in_mask",
    "project_points_to_image",
]
